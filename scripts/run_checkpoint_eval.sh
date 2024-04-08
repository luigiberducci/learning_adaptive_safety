#!/bin/bash

# create list of exp arguments
args=(
  # n-eps env-id min-gamma max-gamma gamma-n cbf-type use-decay seed grid-params
  #
  # particle-env-v0
  "3 particle-env-v1 checkpoints/PPOPID-{particle-env-v1}/checkpoints/model_final.pt 42 human n_agents=3,5,7"
  # f110-multi-agent-v0
  "10 f110-multi-agent-v1 checkpoints/PPOPID-{f110-multi-agent-v1}/checkpoints/model_final.pt 42 human vgain=0.5,0.6,0.7"
)

logdir=logs/only_adaptive/evaluations
script=evaluation/test_different_gammas.py

# check the only input is exp-id
if [ $# -ne 1 ]; then
  echo "Usage: $0 exp-id"
  exit 1
fi

# parse args based on exp-id
exp_id=$1

if [ $exp_id -lt 0 ] || [ $exp_id -ge ${#args[@]} ]; then
  echo "Invalid exp-id: $exp_id"
  exit 1
fi

# extract args
exp_args=${args[$exp_id]}

neps=$(echo $exp_args | cut -d' ' -f1)
env_id=$(echo $exp_args | cut -d' ' -f2)
checkpoint=$(echo $exp_args | cut -d' ' -f3)
seed=$(echo $exp_args | cut -d' ' -f4)
render_mode=$(echo $exp_args | cut -d' ' -f5)
grid_params=$(echo $exp_args | cut -d' ' -f6-)	# -f*- take all the remaining fields from the *th
grid_params_str=$(echo $grid_params | sed 's/ //g')

exp=${env_id}_static_eval_n${neps}_gammas${ming}-${maxg}-${ng}_cbf${cbf_type}_usedecay${use_decay}_seed${seed}_param${grid_params_str}

cmd="
python ${script} --outdir ${logdir} --n-episodes ${neps} --checkpoints ${checkpoint} \
            --grid-params ${grid_params} --seed ${seed} ${env_id} --plot default \
            --render-mode ${render_mode}
"

echo $cmd
eval $cmd