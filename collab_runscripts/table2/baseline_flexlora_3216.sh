#!/usr/bin/env zsh

ground=20
num_steps=10
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
    -el '[32, 16, 16, 16]' \
    -en '[1, 1, 1, 1]' \
    -k 1 \
    -out_dir "" \
    -data_path "" \
    -wandb_run_name "" \
    -seed $seed \
    -eval_every 20 \
    -log_every 5 \
    -update_router_every $update_router_every \
    -exp0_importance $exp0_importance \
    --aggregation_strategy "flexlora" \
    -gating_update_iters $r \
    -wandb_proj "icml-2025"
done

