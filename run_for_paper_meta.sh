#!/bin/bash
experiment=$1
results_dir_main=$2
if [ "$experiment" == "lv" ]; then
  ./run_for_paper.sh lv_runner.py lv 5 $results_dir_main"_lv" 0,1,2,3,4,5,6,7,8,9 9 True
elif [ "$experiment" == "mnist" ]; then
  command="./run_for_paper.sh mnist_runner.py mnist 5 $results_dir_main"_mnist" 1,3,5,7,9 9 True"
  echo $command
  $command
elif [ "$experiment" == "enwik" ]; then
  ./run_for_paper.sh enwik_runner.py enwik 5 $results_dir_main"_enwik"  1,3,5,7,9 9 True
elif [ "$experiment" == "nq" ]; then
  ./run_for_paper.sh nq_runner.py nq 5 $results_dir_main"_nq" 5,9 9 False
else
  echo "Error! Invalid experiment name!" 1>&2
  exit 64
fi
#run_for_paper.sh accpts
#runner=$1
#name=$2
#n_seeds=$3
#results_dir=$4
#train_horizons=$5
#test_horizon=$6
#cuda=$7
