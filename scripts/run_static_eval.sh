#!/bin/bash

# create list of exp arguments
args=(
  # n-eps env-id min-gamma max-gamma gamma-n cbf-type use-decay seed grid-params
  #
  # particle-env-v0
  "100 particle-env-v0 0.0 1.0 10 simple False 42 n_agents=3,5,7"
  "100 particle-env-v0 0.0 1.0 10 simple True 42 n_agents=3,5,7"
  # f110-multi-agent-v0
  "100 f110-multi-agent-v0 0.0 1.0 10 simple False 42 vgain=0.5,0.6,0.7"
  "100 f110-multi-agent-v0 0.0 1.0 10 simple True 42 vgain=0.5,0.6,0.7"
)

logdir=logs/static_eval
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
ming=$(echo $exp_args | cut -d' ' -f3)
maxg=$(echo $exp_args | cut -d' ' -f4)
ng=$(echo $exp_args | cut -d' ' -f5)
cbf_type=$(echo $exp_args | cut -d' ' -f6)
use_decay=$(echo $exp_args | cut -d' ' -f7)
seed=$(echo $exp_args | cut -d' ' -f8)
grid_params=$(echo $exp_args | cut -d' ' -f9-)	# -f*- take all the remaining fields from the *th
grid_params_str=$(echo $grid_params | sed 's/ //g')

exp=${env_id}_static_eval_n${neps}_gammas${ming}-${maxg}-${ng}_cbf${cbf_type}_usedecay${use_decay}_seed${seed}_param${grid_params_str}

cmd="
python ${script} --outdir ${logdir} --exp-id ${exp} --n-episodes ${neps} \
                --grid-min-gamma ${ming} --grid-max-gamma ${maxg} --grid-gamma-n ${ng} \
                --cbf-type ${cbf_type} --use-decay=${use_decay} \
                --grid-params ${grid_params} --seed ${seed} ${env_id}
"

echo $cmd
eval $cmd