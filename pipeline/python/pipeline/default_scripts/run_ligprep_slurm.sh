#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ligprep
#SBATCH --time=10:00:00
#SBATCH -o slurm_logs/ligprep_%A_%a.out
#SBATCH -e slurm_logs/ligprep_%A_%a.err

# sourcing
source $scripts_dir/source_file.sh
source $scripts_dir/extract_execution_model_bash.sh

date
start=`date +%s`

echo "Folder for these runs is : $MAINDIRECTORY"
echo "Ligands file is : $lig_file"

lig=${lig_array[$SLURM_ARRAY_TASK_ID]}

echo "prep for $lig..."
echo $scripts_dir/ligprep.py -lig $lig -lf $ligands_folder -prot $protein_file -mf $MAINDIRECTORY -p $prot_file
python $scripts_dir/ligprep.py -lig $lig -lf $ligands_folder -prot $protein_file -mf $MAINDIRECTORY -p $prot_file

end=`date +%s`
runtime=$((end-start))
echo "runtime was $runtime seconds, which is $((runtime/60)) minutes"
