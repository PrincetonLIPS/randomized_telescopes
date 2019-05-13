#!/bin/bash
#sgd
declare -a exps=("1" "2" "3")
declare -a pres=("1" "2.2" "5")

for exp in "${exps[@]}"
do
  for pre in "${pres[@]}"
  do
    en="en"
    em="e-"
    postname="$pre$en$exp"
    lr="$pre$em$exp"
    ./run_experiments.sh $1 $2_sgd_full_9_$4lr$postname $3 --train_horizon 9 --test_horizon 9 --optimizer sgd --meta_lr $lr --results_dir /tigress/abeatson/sgd_$2 --use_cuda
    ./run_experiments.sh $1 $2_mom_full_9_lr$4$postname $3 --train_horizon 9 --test_horizon 9 --optimizer mom --meta_lr $lr --results_dir /tigress/abeatson/mom_$2 --use_cuda
    ./run_experiments.sh $1 $2_adam_full_9_lr$4$postname $3 --train_horizon 9 --test_horizon 9 --optimizer adam --meta_lr $lr --results_dir /tigress/abeatson/adam_$2 --use_cuda
  done
done
