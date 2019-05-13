#!/bin/bash
runner=$1
name=$2
n_seeds=$3
results_dir=$4
train_horizons=$5
test_horizon=$6
cuda=$7

# SGD runs
./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "sgd" "False" "False" 1.0 "False" $train_horizons $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "sgd" "False" "False" 1.0 "True" $test_horizon $test_horizon $cuda
./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "sgd" "True" "False" 1.0 "False" $train_horizons $test_horizon $cuda
./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "sgd" "True" "True" 1.0 "False" $train_horizons $test_horizon $cuda
# Agg sgd runs
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "sgd" "True" "False" 0.5 "False" $train_horizons $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "sgd" "True" "True" 0.5 "False" $train_horizons $test_horizon $cuda

# Mom runs
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "mom" "False" "False" 1.0 "False" $train_horizons $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "mom" "False" "False" 1.0 "True" $test_horizon $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "mom" "True" "False" 1.0 "False" $train_horizons $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "mom" "True" "True" 1.0 "False" $train_horizons $test_horizon $cuda
# Agg mom runs
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "mom" "True" "False" 0.5 "False" $train_horizons $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "mom" "True" "True" 0.5 "False" $train_horizons $test_horizon $cuda

# Adam runs
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "adam" "False" "False" 1.0 "False" $train_horizons $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "adam" "False" "False" 1.0 "True" $test_horizon $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "adam" "True" "False" 1.0 "False" $train_horizons $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "adam" "True" "True" 1.0 "False" $train_horizons $test_horizon $cuda
# Agg Adam runs
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "adam" "True" "False" 0.5 "False" $train_horizons $test_horizon $cuda
#./run_for_train_horizons.sh $runner $name $n_seeds $results_dir "adam" "True" "True" 0.5 "False" $train_horizons $test_horizon $cuda
