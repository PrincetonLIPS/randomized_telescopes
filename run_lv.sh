#!/bin/bash
CUDA_VISIBLE_DEVICES=0 ./run_experiments.sh lv_runner.py lv_ss_9 $1 --rt --train_horizon 9 --test_horizon 10 --optimizer sgd
#./run_experiments.sh lv_runner.py lv_ss_10_avg $1 --rt --averaged_test=True --train_horizon 10 --test_horizon 11 --optimizer sgd
CUDA_VISIBLE_DEVICES=1 ./run_experiments.sh lv_runner.py lv_full_9 $1 --train_horizon 9 --test_horizon 10 --optimizer sgd
CUDA_VISIBLE_DEVICES=1 ./run_experiments.sh lv_runner.py lv_ss_regress_9 $1 --rt --regress_grads --train_horizon 9 --test_horizon 10 --optimizer sgd
#./run_experiments.sh lv_runner.py lv_full_10_avg $1 --averaged_test=True --train_horizon 10 --test_horizon 11 --optimizer sgd
