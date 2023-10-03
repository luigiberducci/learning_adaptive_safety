#!/bin/bash

n_steps=1024000

# create list of exp arguments
args=(
  # n-runs total-timesteps algo env-id num-envs use-cbf use-ctrl cost-limit
  # on policy
  "1 ${n_steps} PPOPID f110-multi-agent-v0 2 True False 0.25"
  "1 ${n_steps} PPOLag f110-multi-agent-v0 2 False False 0.25"
  "1 ${n_steps} CPO f110-multi-agent-v0 2 False False 0.25"
  "1 ${n_steps} IPO f110-multi-agent-v0 2 False False 0.25"
  # off policy
  "1 ${n_steps} DDPGLag f110-multi-agent-v0 2 False False 0.25"
  "1 ${n_steps} TD3Lag f110-multi-agent-v0 2 False False 0.25"
  # other formulations
  "1 ${n_steps} PPOSaute f110-multi-agent-v0 2 False False 0.25"

  # particle-env
  # on policy
  "1 ${n_steps} PPOPID particle-env-v0 4 True False 0.25"
  "1 ${n_steps} PPOLag particle-env-v0 4 False False 0.25"
  "1 ${n_steps} CPO particle-env-v0 4 False False 0.25"
  "1 ${n_steps} IPO particle-env-v0 4 False False 0.25"
  # off policy
  "1 ${n_steps} DDPGLag particle-env-v0 4 False False 0.25"
  "1 ${n_steps} TD3Lag particle-env-v0 4 False False 0.25"
  # other formulations
  "1 ${n_steps} PPOSaute particle-env-v0 4 False False 0.25"
)

logdir=logs/baselines/
script=run_training_baselines.py

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
total_timesteps=$(echo $exp_args | cut -d' ' -f2)
algo=$(echo $exp_args | cut -d' ' -f3)
env_id=$(echo $exp_args | cut -d' ' -f4)
numenvs=$(echo $exp_args | cut -d' ' -f5)
use_cbf=$(echo $exp_args | cut -d' ' -f6)
use_ctrl=$(echo $exp_args | cut -d' ' -f7)
cost_limit=$(echo $exp_args | cut -d' ' -f8)

cmd="
python ${script} --algo ${algo} --env-id ${env_id} --num-envs ${numenvs} --total-timesteps ${total_timesteps} \
       --use-cbf ${use_cbf} --use-ctrl ${use_ctrl} --cost-limit ${cost_limit} \
       --log-dir ${logdir}
"

for ((i=1;i<${nrepeats}+1;i++)); do
  echo -e "\n\n"
  echo "***************************************************************************************************************"
  echo "Repeat $i/$nrepeats for exp ${exp}"
  echo "***************************************************************************************************************"
  echo $cmd
  eval $cmd
done
