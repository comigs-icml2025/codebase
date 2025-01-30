#!/usr/bin/env zsh

ground=1
num_steps=200
seed_list=(42 45 47)
exp0_importance=0.5
r=0
update_router_every=30



for seed in ${seed_list[@]}; do
  python3 collab_run.py \
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
  -gating_update_iters $r \
  -is_no_router \
  -wandb_proj "CoMoLE-Agnews-Formal-Distribution-Shift"
done
