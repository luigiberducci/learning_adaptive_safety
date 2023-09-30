#!/bin/bash

n_steps=1024000

# create list of exp arguments
args=(
  # n-runs total-timesteps env-id num-envs use-cbf use-ctrl cost-limit
  #
  "1 ${n_steps} f110-multi-agent-v1 2 True True 0.25"
  "1 ${n_steps} particle-env-v1 4 True True 0.25"
)

script=run_training_only_adaptive.py

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

nrepeats=$(echo $exp_args | cut -d' ' -f1)
ns=$(echo $exp_args | cut -d' ' -f2)
env_id=$(echo $exp_args | cut -d' ' -f3)
num_envs=$(echo $exp_args | cut -d' ' -f4)
use_cbf=$(echo $exp_args | cut -d' ' -f5)
use_ctrl=$(echo $exp_args | cut -d' ' -f6)
cost_limit=$(echo $exp_args | cut -d' ' -f7)

logdir=logs/only_adaptive/PPOPID-{${env_id}}/

exp=run_ns${ns}_climit${cl}_$(date +'%Y%m%d')

cmd="
python ${script} --env-id ${env_id} --log-dir ${logdir} --exp-name ${exp} --total-timesteps ${ns} \
                 --cost-limit ${cost_limit} --num-envs ${num_envs}
"

for ((i=1;i<${nrepeats}+1;i++)); do
  echo -e "\n\n"
  echo "***************************************************************************************************************"
  echo "Repeat $i/$nrepeats for exp ${exp}"
  echo "***************************************************************************************************************"
  echo $cmd
  eval $cmd
done
