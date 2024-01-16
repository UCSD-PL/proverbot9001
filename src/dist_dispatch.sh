#!/usr/bin/env bash
cd $SLURM_SUBMIT_DIR
if (( $SLURM_JOBID == 0 )) ; then
  python src/distributed_rl_learning_server.py $(eval echo $2) &> $1/learner.out
else
  python src/distributed_rl_acting_worker.py $(eval echo $3) &> $1/actor-$((${SLURM_NODEID}-1)).out
fi
