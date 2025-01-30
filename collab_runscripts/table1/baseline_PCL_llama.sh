#!/usr/bin/env zsh

ground=20
num_steps=10
seed_list=(42 45 47)
exp0_importance=0.5
r=0
update_router_every=30
lb_lam=0.0

for seed in ${seed_list[@]}; do
  CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=1 collab_run_client_level.py \
  -gr $ground \
  -num_steps $num_steps \
  -wandb \
  -nc 4 \
  -out_dir "" \
  -data_path "" \
  -wandb_run_name "" \
  -seed $seed \
  -eval_every 20 \
  -log_every 5 \
  -update_router_every $update_router_every \
  -exp0_importance $exp0_importance \
  --collaboration_strategy "all"  \
  -gating_update_iters $r \
  -lb_lam $lb_lam \
  -bm "meta-llama/Llama-3.2-1B" \
  -wandb_proj "icml-2025"
  -lr 2e-4 \
  -is_no_router \
  -as "all"
done
