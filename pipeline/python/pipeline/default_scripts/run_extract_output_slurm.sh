#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=extr
#SBATCH --time=02:00:00
#SBATCH -o slurm_logs/extr_%A.out
#SBATCH -e slurm_logs/extr_%A.err

# sourcing
source $scripts_dir/source_file.sh
source $scripts_dir/extract_execution_model_bash.sh

date
start=`date +%s`

folder=$MAINDIRECTORY/outputs/$2/$1

echo "extracting output for $folder"
echo $scripts_dir/extract_output.py -f $folder -p $prot_file
python $scripts_dir/extract_output.py -f $folder -p $prot_file

echo "done."

end=`date +%s`
runtime=$((end-start))
echo "runtime was $runtime seconds, which is $((runtime/60)) minutes"