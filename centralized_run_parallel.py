from tqdm import tqdm
import argparse
from models.model import GPT
import random
import numpy as np
import torch 
import os
import ast
from collab_utils.collaboration_strategies import to_collaboration_strategy
from collab_utils.aggregation_strategies import to_aggregation_strategy
import wandb
from models.lora import get_ft_model
from torch import nn
from torch.distributed import init_process_group, destroy_process_group

from contextlib import nullcontext
import os
import copy
import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.optim import lr_scheduler
from models import expert_info, reset_expert_info
from torch.nn.parallel import DistributedDataParallel as DDP

backend = 'nccl'

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed

# Function to detect device and number of GPUs
def get_device():
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            print(f"Using {num_gpu} GPUs for data parallelism.")
            return 'cuda'
        else:
            print("Using a single GPU.")
            return 'cuda'
    else:
        print("CUDA not available. Using CPU.")
        return 'cpu'

# Custom get_batch function
def get_batch(data, seq_length, batch_size, is_shifted, device='cpu'):
    '''
    Returns a batch of size ([batch_size, seq_length])
    '''
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
    if is_shifted:
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
    else:
        y = x.clone()
    if "cuda" in device:
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    return x, y

def parse_list(value):
    return ast.literal_eval(value)

def train(model, expert_optimizer, expert_scheduler, args, acc_steps, local_data_train, local_data_valid, local_data_test, is_shifted, is_train_on_val=False, seq_length=128):
    type_ctx = nullcontext() if get_device() == 'cpu' else torch.amp.autocast(device_type=get_device(), dtype=torch.bfloat16)
    model.train()
    step = 0
    for step in tqdm(range(args.num_local_steps)):  
        ##### Update Expert Params on Training Dataset
        loss = 0.0
        loss_report = 0.0
        for _ in range(acc_steps): # gradient accumulation
            if is_train_on_val:
                x, y = get_batch(local_data_valid, seq_length, batch_size=args.micro_batch_size, is_shifted=is_shifted, device=get_device())
            else:
                x, y = get_batch(local_data_train, seq_length, batch_size=args.micro_batch_size, is_shifted=is_shifted, device=get_device())
            with type_ctx:
                _, loss_bp , loss_r = model(x, targets=y)
                loss += loss_bp
                loss_report += loss_r
        loss = loss / acc_steps
        loss.backward()
        expert_optimizer.step()
        loss_report = loss_report/acc_steps
        expert_optimizer.zero_grad()
        expert_scheduler.step()
        
        step +=1
            
        if args.wandb_log == True and step % args.num_log_steps == 0 and master_process:
            metrics = {
                f"client_{0}/train_loss": loss_report,
                "iter": step,
                "expert_lr": expert_scheduler.get_lr()[0],
            }
            wandb.log(metrics)
        if args.wandb_log == True and step % args.num_log_steps == 0 and step % args.num_eval_steps == 0 and master_process:
            _, val_loss, val_perplexity = eval(model, seq_length, local_data_test, is_shifted)
            metrics = {
                f"client_{0}/val_loss": val_loss,
                f"client_{0}/val_ppl": val_perplexity,
                "iter": step,
            }
            reset_expert_info(model)
            wandb.log(metrics)
    
@torch.no_grad()
def eval(model, seq_length, local_data_test, is_shifted, max_num_batches = 100 ):
    type_ctx = nullcontext() if get_device() == 'cpu' else torch.amp.autocast(device_type=get_device(), dtype=torch.bfloat16)
    model.eval()
    loss_list_val, acc_list = [], []
    for _ in tqdm(range(max_num_batches)): 
        x, y = get_batch(local_data_test, seq_length, batch_size=args.micro_batch_size, is_shifted=is_shifted, device=get_device())
        with type_ctx:
            logits, _, loss = model(x, targets=y)
        loss_list_val.append(loss)
        acc_list.append((logits.argmax(-1) == y).float().mean())
    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss
    print('val_loss:',val_loss, '\n',
          'val_perplexity:',val_perplexity)
    return val_acc, val_loss, val_perplexity

parser = argparse.ArgumentParser()
parser.add_argument("-gr", "--num_global_rounds", default = 20, type=int)
parser.add_argument("-num_steps", "--num_local_steps", default = 25, type=int)
parser.add_argument("-model_path", "--model_path", type = str)
parser.add_argument('-lr',"--learning_rate",default=2e-3,type=float)
parser.add_argument('-wd',"--weight_decay",default=1e-2,type=float)
parser.add_argument('-ds','--dataset',default='agnews',type=str)
parser.add_argument('-data_path','--data_path',type=str)
parser.add_argument('-nc','--num_clients',default=4,type=int)
parser.add_argument('-device','--device', default=get_device(), type=str)
parser.add_argument('-el', '--expert_lora_ranks', default='[8,8,8,8]', type=parse_list, help='Comma-separated list of LoRA ranks')
parser.add_argument('-en', '--expert_numbers', default='[2,2,2,2]', type=parse_list, help='Comma-separated list of number of experiments')
parser.add_argument('-k', '--topk', default=2,type=int)
parser.add_argument('-as','--collaboration_strategy',default="all", type=str)
parser.add_argument('-aggregation_strategy','--aggregation_strategy',default="default", type=str)
parser.add_argument('-bs','--batch_size',default=64,type=int)
parser.add_argument('-micro_bs','--micro_batch_size',default=64,type=int)
parser.add_argument('-wandb','--wandb_log',action='store_true')
parser.add_argument('-wandb_proj','--wandb_project',default="CoMoLE", type=str)
parser.add_argument('-wandb_run_name','--wandb_run_name',default="test", type=str)
parser.add_argument('-out_dir','--output_dir',default="../out", type=str)
parser.add_argument('-log_every','--num_log_steps',default=1, type=int)
parser.add_argument('-eval_every','--num_eval_steps',default=1, type=int)
parser.add_argument('-update_router_every','--num_router_update_steps',default=1, type=int)
parser.add_argument('-seed','--seed',default=1, type=int)
parser.add_argument('-scheduler','--scheduler', default="cosine", type=str)
parser.add_argument('-lb_lam','--lb_lambda', default=0.01, type=float)
parser.add_argument('-p_lam','--p_lambda', default=0.01, type=float)
parser.add_argument('-p_strength','--pruning_strength', default=0.99, type=float)
parser.add_argument('-is_pruning', '--is_pruning', action='store_true', help='Enable pruning if set')
parser.add_argument('-exp0_importance','--expert0_importance', default=0.9, type=float)
parser.add_argument('-gating_update_iters','--gating_update_iters', default=1, type=int)
parser.add_argument('-save_model','--save_model', action='store_true')
parser.add_argument('-lora_do','--lora_dropout', default=0.0, type=float)
parser.add_argument('-alter_on_train','--alter_gate_update_on_train', action='store_true')
parser.add_argument('-bm','--base_model', default="gpt2", type=str)
parser.add_argument('-is_alter','--is_alternating', action='store_true')
parser.add_argument('-is_no_router','--is_no_router', action='store_true')
parser.add_argument('-learning_rate_scale','--learning_rate_scale', default=1.0, type=float)
args = parser.parse_args()

collaboration_strategy = to_collaboration_strategy(args.collaboration_strategy)
aggregation_strategy = to_aggregation_strategy(args.aggregation_strategy)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(args.seed)

if args.wandb_log and master_process:
    import wandb
    wandb.init(project=args.wandb_project, entity='ec-llm', name=args.wandb_run_name, config=vars(args))

acc_steps = args.batch_size // args.micro_batch_size
override_args = dict(
    expert_num = args.expert_numbers[0],
    lora_rank = args.expert_lora_ranks[0],
    lora_dropout = args.lora_dropout,
    topk_exp = min(args.topk,args.expert_numbers[0]),
    load_balancing_lambda = args.lb_lambda,
    pruning_lambda = args.p_lambda,
    pruning_strength = args.pruning_strength,
    pruning = args.is_pruning,
    expert0_importance = args.expert0_importance,
    is_no_router = args.is_no_router,
    device=get_device())


def init_client_model(override_args):
    if args.base_model.startswith("gpt"):
        model = GPT.from_pretrained(args.base_model, override_args)
        model = get_ft_model(model, collaboration_strategy)
    elif "llama" in args.base_model:
        from models.modeling_llama_moe_hf import LlamaMoEForCausalLM
        from models.configuration_llama_moe import LlamaMoEConfig
        model = LlamaMoEForCausalLM.from_pretrained(args.base_model, LlamaMoEConfig(**override_args))
        model = get_ft_model(model, collaboration_strategy)     
    else:   
        raise ValueError("Unknown model type")
    model.to(get_device())

    expert_optimizer, _ = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=get_device(),
        is_alternating=args.is_alternating,
        learning_rate_scaling=args.learning_rate_scale)
    expert_scheduler = lr_scheduler.CosineAnnealingLR(expert_optimizer, T_max=args.num_local_steps*args.num_global_rounds, eta_min=args.learning_rate/10)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    return model, expert_optimizer, expert_scheduler

centralized_model, expert_optimizer, expert_scheduler = init_client_model(override_args)

local_train_data_path = os.path.join(args.data_path, "1", "train_{}.bin".format(0))
local_test_data_path = os.path.join(args.data_path, "1", "test_{}.bin".format(0))
local_valid_data_path = os.path.join(args.data_path, "1", "valid_{}.bin".format(0))
local_data_train = np.memmap(local_train_data_path, dtype=np.uint32, mode='r')
local_data_test = np.memmap(local_test_data_path, dtype=np.uint32, mode='r')
local_data_valid = np.memmap(local_valid_data_path, dtype=np.uint32, mode='r')

for epoch in tqdm(range(args.num_global_rounds)):
    print(f"Starting training of epoch: {epoch}")
    train(centralized_model, expert_optimizer, expert_scheduler, args, acc_steps, local_data_train, local_data_valid, local_data_test, is_shifted=args.base_model.startswith("gpt"))
    print(f"Locally trained client: {id}")
if ddp:
    destroy_process_group()