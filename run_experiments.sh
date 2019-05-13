#!/bin/bash
# Runs $1 with name $2, number of consecutive seeds $3, with args $4:
runner="$1"
name="$2"
seeds="$3"
shift
shift
shift
flags="$@"
for (( i=0; i<$seeds; i++))
do
  f="--name $name "$flags" --seed $i"
  command="python "$runner" "$f
  echo $command
  if hash sbatch 2>/dev/null; then
    cat run_proc.sh | sed -e "s@COMMAND@$command@g" | sbatch
  else
    nohup $command &
  fi
done
