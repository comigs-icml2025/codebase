#!/usr/bin/env zsh

# Check if the correct number of arguments are passed
if [ $# -ne 2 ]; then
  echo "Usage: $0 <start_run> <end_run>"
  exit 1
fi

# Read the arguments for start_run and end_run
start_run=$1
end_run=$2

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

      # Skip until we reach iteration 25
      if [ $iteration -lt $start_run ]; then
        echo "Skipping iteration $iteration..."
        continue
      fi

      # Stop after iteration 50
      if [ $iteration -gt $end_run ]; then
        echo "Reached iteration $iteration. Exiting loop..."
        exit 0
      fi

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
      --aggregation_strategy "hetlora" \
      --is_pruning \
      --pruning_strength $pruning_strength \
      -gating_update_iters $r \
      -p_lam $p_lam \
      -wandb_proj "icml-2025"
    done
  done
done
