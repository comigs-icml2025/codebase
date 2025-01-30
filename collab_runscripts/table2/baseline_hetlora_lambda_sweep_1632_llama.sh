#!/usr/bin/env zsh

ground=20
num_steps=10
seed_list=(42 45 47)
exp0_importance=0.5
r=0
update_router_every=30
p_lam_list=( 0.05 0.005 0.0005 )
pruning_strengths=( 0.99 )

iteration=0

for pruning_strength in ${pruning_strengths[@]}; do
  for seed in ${seed_list[@]}; do
    for p_lam in ${p_lam_list[@]}; do

      iteration=$((iteration + 1))
      python3 collab_run.py \
      -gr $ground \
      -num_steps $num_steps \
      -wandb \
      -nc 4 \
      -el '[16, 32, 32, 16]' \
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
      --aggregation_strategy "hetlora" \
      --is_pruning \
      --pruning_strength $pruning_strength \
      -gating_update_iters $r \
      -p_lam $p_lam \
      -bm "meta-llama/Llama-3.2-1B" \
      -wandb_proj "icml-2025" \
      -lr 2e-4
    done
  done
done
