#!/bin/bash
runner=$1; shift
run_name=$1; shift
n_seeds=$1; shift
results_dir=$1; shift
optimizer=$1; shift
rt=$1; shift
cdf=$1; shift
variance_weight=$1; shift
linear_schedule=$1; shift
train_horizons=$1; shift
test_horizon=$1; shift
cuda=$1; shift

if [ "$cdf" == "True" ]; then
  run_name=$run_name"_rr"
elif [ "$rt" == "True" ]; then
  run_name=$run_name"_ss"
elif [ "$rt" == "False" ] && [ "$cdf" == "False" ] && [ "$linear_schedule" == "False" ]; then
  run_name=$run_name"_full"
elif [ "$rt" == "False" ] && [ "$cdf" == "False" ] && [ "$linear_schedule" == "True" ]; then
  run_name=$run_name"_lin"
else
  echo "Error! Invalid args" 1>&2
  exit 64
fi
for train_horizon in $(echo $train_horizons | sed "s/,/ /g")
do
  this_run_name=$run_name"_"$optimizer$variance_weight$train_horizon
  command="./run_experiments.sh $runner $this_run_name $n_seeds \
  --train_horizon=$train_horizon --test_horizon=$test_horizon \
  --optimizer=$optimizer --rt=$rt --cdf=$cdf --results_dir=$results_dir \
  --cuda=$cuda --variance_weight=$variance_weight --linear_schedule=$linear_schedule"
  echo $command
  $command
done
