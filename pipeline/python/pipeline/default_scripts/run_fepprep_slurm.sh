#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=fepprep
#SBATCH --time=03:00:00
#SBATCH -o slurm_logs/fepprep_%A_%a.out
#SBATCH -e slurm_logs/fepprep_%A_%a.err

# sourcing
source $scripts_dir/source_file.sh
source $scripts_dir/extract_execution_model_bash.sh

date
start=`date +%s`

echo "Folder for these runs is : $MAINDIRECTORY"
echo "Network file is : $net_file"

trans=${trans_array[$SLURM_ARRAY_TASK_ID]}
eng=${eng_array[$SLURM_ARRAY_TASK_ID]}
win=${win_array[$SLURM_ARRAY_TASK_ID]}

echo "fepprep for $trans using $eng"
echo $scripts_dir/fepprep.py -pert $trans -eng $eng -lam $win -mf $MAINDIRECTORY -p $prot_file -prep $prep_folder
python $scripts_dir/fepprep.py -pert $trans -eng $eng -lam $win -mf $MAINDIRECTORY -p $prot_file -prep $prep_folder

end=`date +%s`
runtime=$((end-start))
echo "runtime was $runtime seconds, which is $((runtime/60)) minutes"
