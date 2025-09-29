#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <task_name> <data_name> "
  exit 1
fi

TASK_NAME=$1
DATASET=$2
export TASK_NAME

TMP_SCRIPT=$(mktemp $(pwd)/slurm_job_XXXXXX.slurm)
cat <<EOL > $TMP_SCRIPT
#!/bin/bash

#SBATCH --job-name=${DATASET}_${TASK_NAME}
#SBATCH --output=$(pwd)/slurm_output/${DATASET}_${TASK_NAME}.log
#SBATCH --error=$(pwd)/slurm_output/${DATASET}_${TASK_NAME}.log
#SBATCH --time=48:00:00
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

cd $(pwd)

EOL

if [ ! -d "$(pwd)/slurm_output" ]; then
  mkdir -p "$(pwd)/slurm_output"
fi


grep -v "^#" shell_scripts/run_script_param.sh \
  | sed -e "s/DATASET=.*/DATASET=${DATASET}/" \
        -e "s/TASK_NAME=.*/TASK_NAME=${TASK_NAME}/" \
  >> $TMP_SCRIPT

# sbatch $TMP_SCRIPT

# rm "$TMP_SCRIPT" 
