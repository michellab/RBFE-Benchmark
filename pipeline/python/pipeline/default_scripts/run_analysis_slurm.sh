#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ana
#SBATCH --time=01:00:00
#SBATCH -o slurm_logs/ana_%A.out
#SBATCH -e slurm_logs/ana_%A.err

# sourcing
source $scripts_dir/source_file.sh
source $scripts_dir/extract_execution_model_bash.sh

date
start=`date +%s`

echo "Folder for these runs is : $MAINDIRECTORY"
echo "Analysis for the transformation $1, $2."

echo $scripts_dir/analysis.py -pert $1 -eng $2 -mf $MAINDIRECTORY -a $ana_file -p $prot_file # -ra True
python $scripts_dir/analysis.py -pert $1 -eng $2 -mf $MAINDIRECTORY -a $ana_file -p $prot_file # -ra True

end=`date +%s`
runtime=$((end-start))
echo "runtime was $runtime seconds, which is $((runtime/60)) minutes"
