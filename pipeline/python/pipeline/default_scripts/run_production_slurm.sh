#!/bin/bash
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=prod
#SBATCH --time=24:00:00
#SBATCH -o slurm_logs/prod_%A_%a.out
#SBATCH -e slurm_logs/prod_%A_%a.err

# specifying the number of threads (needed for gromacs)
export OMP_NUM_THREADS=8

# sourcing
source $scripts_dir/source_file.sh
source $scripts_dir/extract_execution_model_bash.sh

date
start=`date +%s`

echo "Folder for these runs is : $MAINDIRECTORY"
echo "The transformation is $1 using $3 windows and $2 as the MD engine with $repeats repeats"

# define no of windows based on the assosciative array generated from reading the net_file in extract execution model
IFS=","
read -r -a lamvals <<< "${wins_array[$3]}"

# define lambda based on slurm array
lam=${lamvals[SLURM_ARRAY_TASK_ID]}

# change to the trans dir, abort and message if not there
cd $MAINDIRECTORY/outputs/$2/$1
if [[ ! -d $MAINDIRECTORY/outputs/$2/$1 ]]; then
    echo "$MAINDIRECTORY/outputs/$2/$1 does not exist. Production run aborted..."
    exit
fi
trans_dir=$(pwd)

# iterate over dir (for each leg) based on no of repeats
for dir in 'free' 'bound' ; do
for rep in "${repeats_array[@]}" ; do
repeat=${dir}_${rep}
cd $repeat
repeat_dir=$(pwd)

echo "Running in $repeat_dir"
echo "Lambda is $lam"

# in amber22 ?
# sed -i '5 a\   gti_add_sc=5,' min/lambda_$lam/amber.cfg
# sed -i '5 a\   gti_add_sc=5,' heat/lambda_$lam/amber.cfg
# sed -i '5 a\   gti_add_sc=5,' eq/lambda_$lam/amber.cfg
# sed -i '5 a\   gti_add_sc=5,' prod/lambda_$lam/amber.cfg

# run the runs based on which engine
if [ $2 = "AMBER" ]; then
echo "min"

if [[ ! -f min/lambda_$lam/initial_amber.rst7 ]]; then
    cp min/lambda_$lam/amber.rst7 min/lambda_$lam/initial_amber.rst7
fi
    
    pmemd.cuda -i min/lambda_$lam/amber.cfg -c min/lambda_$lam/initial_amber.rst7 -ref min/lambda_$lam/initial_amber.rst7 -p min/lambda_$lam/amber.prm7 -O -o min/lambda_$lam/amber.out -inf min/lambda_$lam/amber.info -e min/lambda_$lam/amber.en -r min/lambda_$lam/amber.rst7 -x min/lambda_$lam/amber.nc -l min/lambda_$lam/amber.log ;
echo "heat"
    pmemd.cuda -i heat/lambda_$lam/amber.cfg -c min/lambda_$lam/amber.rst7 -ref min/lambda_$lam/amber.rst7 -p heat/lambda_$lam/amber.prm7 -O -o heat/lambda_$lam/amber.out -inf heat/lambda_$lam/amber.info -e heat/lambda_$lam/amber.en -r heat/lambda_$lam/amber.rst7 -x heat/lambda_$lam/amber.nc -l heat/lambda_$lam/amber.log ;
echo "eq"
    pmemd.cuda -i eq/lambda_$lam/amber.cfg -c heat/lambda_$lam/amber.rst7 -ref heat/lambda_$lam/amber.rst7 -p eq/lambda_$lam/amber.prm7 -O -o eq/lambda_$lam/amber.out -inf eq/lambda_$lam/amber.info -e eq/lambda_$lam/amber.en -r eq/lambda_$lam/amber.rst7 -x eq/lambda_$lam/amber.nc -l eq/lambda_$lam/amber.log ;
echo "prod"
    pmemd.cuda -i prod/lambda_$lam/amber.cfg -c eq/lambda_$lam/amber.rst7 -ref eq/lambda_$lam/amber.rst7 -p prod/lambda_$lam/amber.prm7 -O -o prod/lambda_$lam/amber.out -inf prod/lambda_$lam/amber.info -e prod/lambda_$lam/amber.en -r prod/lambda_$lam/amber.rst7 -x prod/lambda_$lam/amber.nc -l prod/lambda_$lam/amber.log ; # -AllowSmallBox ;

# delete simulation data 
if [[ $keep_traj == "None" ]]; then
    echo "Removing all trajectories..."
    rm min/lambda_$lam/*.nc heat/lambda_$lam/*.nc eq/lambda_$lam/*.nc
    rm prod/lambda_$lam/*.nc
fi

fi

if [ $2 = "GROMACS" ]; then

cp min/lambda_$lam/gromacs.gro min/lambda_$lam/initial_gromacs.gro

echo "min"
gmx23 grompp -maxwarn 1 -f min/lambda_$lam/gromacs.mdp -c min/lambda_$lam/initial_gromacs.gro -p min/lambda_$lam/gromacs.top -o min/lambda_$lam/gromacs.tpr
gmx23 mdrun -ntmpi 1 -deffnm min/lambda_$lam/gromacs ;

echo "heat"
gmx23 grompp -maxwarn 1 -f heat/lambda_$lam/gromacs.mdp -c min/lambda_$lam/gromacs.gro -p heat/lambda_$lam/gromacs.top -o heat/lambda_$lam/gromacs.tpr
gmx23 mdrun -ntmpi 1 -deffnm heat/lambda_$lam/gromacs ;

echo "eq"
gmx23 grompp -maxwarn 1 -f eq/lambda_$lam/gromacs.mdp -c heat/lambda_$lam/gromacs.gro -p eq/lambda_$lam/gromacs.top -t heat/lambda_$lam/gromacs.cpt  -o eq/lambda_$lam/gromacs.tpr
gmx23 mdrun -ntmpi 1 -deffnm eq/lambda_$lam/gromacs ;

echo "prod"
gmx23 grompp -maxwarn 1 -f prod/lambda_$lam/gromacs.mdp -c eq/lambda_$lam/gromacs.gro -p prod/lambda_$lam/gromacs.top -t eq/lambda_$lam/gromacs.cpt -o prod/lambda_$lam/gromacs.tpr
gmx23 mdrun -ntmpi 1 -deffnm prod/lambda_$lam/gromacs ;

# delete simulation data 
if [[ $keep_traj == "None" ]]; then
    echo "Removing all trajectories..."
    rm min/lambda_$lam/*.trr heat/lambda_$lam/*.trr eq/lambda_$lam/*trr
    rm prod/lambda_$lam/*.trr
fi

fi

if [ $2 = "SOMD" ]; then
echo "min + eq"
cd eq/lambda_$lam
cp somd.rst7 initial_somd.rst7
sed '/gpu/d' -i somd.cfg
somd-freenrg -c somd.rst7 -t somd.prm7 -m somd.pert -C somd.cfg -p CUDA

cd $repeat_dir
#copy the files over for restart
echo "copying restart files for production..."
cp eq/lambda_$lam/sim_restart.s3 prod/lambda_$lam/sim_restart.s3
cp eq/lambda_$lam/sim_restart.s3.previous prod/lambda_$lam/sim_restart.s3.previous
cp eq/lambda_$lam/SYSTEM.s3 prod/lambda_$lam/SYSTEM.s3

echo "prod"
cd prod/lambda_$lam
sed '/gpu/d' -i somd.cfg
somd-freenrg -c somd.rst7 -t somd.prm7 -m somd.pert -C somd.cfg -p CUDA
cd $repeat_dir

# delete simulation data 
if [[ $keep_traj == "None" ]]; then
    echo "Removing all trajectories..."
    rm eq/lambda_$lam/*.dcd eq/lambda_$lam/*.s3* eq/lambda_$lam/latest*
    rm prod/lambda_$lam/*.dcd prod/lambda_$lam/*.s3* prod/lambda_$lam/latest*
fi

fi

cd $trans_dir

done
done

echo "done."

end=`date +%s`
runtime=$((end-start))
echo "runtime was $runtime seconds, which is $((runtime/60)) minutes"


# delete trajectory files

#removing all traj
# remove amber trajectories
if [ "$2" = "AMBER" ] || [ "$2" = "ALL" ] && [ "$keep_traj" = "None" ]; then
for i in $(find $trans_dir -name 'amber.nc');
do
    rm -rf $i
done
fi

# remove gromacs trajectories
if [ "$2" = "GROMACS" ] || [ "$2" = "ALL" ] && [ "$keep_traj" = "None" ]; then
for i in $(find $trans_dir -name 'gromacs.trr');
do
    rm -rf $i
done
fi

# remove somd trajectories
if [ "$2" = "SOMD" ] || [ "$2" = "ALL" ] && [ "$keep_traj" = "None" ]; then
for i in $(find $trans_dir -name 'traj*.dcd');
do
    rm -rf $i
done
for i in $(find $trans_dir -name '*.s3*');
do
    rm -rf $i
done
fi

#removing all except lambda 0 and 1
# remove amber trajectories
if [ "$2" = "AMBER" ] || [ "$2" = "ALL" ] && [ "$keep_traj" = "0,1" ]; then
for i in $(find $trans_dir -name 'amber.nc');

do
if [[ "$i" != *"lambda_0.0000"* ]] && [[ "$i" != *"lambda_1.0000"* ]]; then
    rm -rf $i
fi
done

fi

# remove gromacs trajectories
if [ "$2" = "GROMACS" ] || [ "$2" = "ALL" ]&& [ "$keep_traj" = "0,1" ]; then
for i in $(find $trans_dir -name 'gromacs.trr');

do
if [[ "$i" != *"lambda_0.0000"* ]] && [[ "$i" != *"lambda_1.0000"* ]]; then
    rm -rf $i
fi
done

fi

# remove somd trajectories
if [ "$2" = "SOMD" ] || [ "$2" = "ALL" ]&& [ "$keep_traj" = "0,1" ]; then
for i in $(find $trans_dir -name 'traj*.dcd');

do
if [[ "$i" != *"lambda_0.0000"* ]] && [[ "$i" != *"lambda_1.0000"* ]]; then
    rm -rf $i
fi
done

for i in $(find $trans_dir -name '*.s3*');

do
if [[ "$i" != *"lambda_0.0000"* ]] && [[ "$i" != *"lambda_1.0000"* ]]; then
    rm -rf $i
fi
done

fi

#removing all except lambda 0, 0.5 and 1
# remove amber trajectories
if [ "$2" = "AMBER" ] || [ "$2" = "ALL" ] && [ "$keep_traj" = "0,0.5,1" ]; then
for i in $(find $trans_dir -name 'amber.nc');

do
if [[ "$i" != *"lambda_0.0000"* ]] && [[ "$i" != *"lambda_0.5000"* ]] && [[ "$i" != *"lambda_1.0000"* ]]; then
    rm -rf $i
fi
done

fi

# remove gromacs trajectories
if [ "$2" = "GROMACS" ] || [ "$2" = "ALL" ]&& [ "$keep_traj" = "0,0.5,1" ]; then
for i in $(find $trans_dir -name 'gromacs.trr');

do
if [[ "$i" != *"lambda_0.0000"* ]] && [[ "$i" != *"lambda_0.5000"* ]] && [[ "$i" != *"lambda_1.0000"* ]]; then
    rm -rf $i
fi
done

fi

# remove somd trajectories
if [ "$2" = "SOMD" ] || [ "$2" = "ALL" ]&& [ "$keep_traj" = "0,0.5,1" ]; then
for i in $(find $trans_dir -name 'traj*.dcd');

do
if [[ "$i" != *"lambda_0.0000"* ]] && [[ "$i" != *"lambda_0.5000"* ]] && [[ "$i" != *"lambda_1.0000"* ]]; then
    rm -rf $i
fi
done

for i in $(find $trans_dir -name '*.s3*');

do
if [[ "$i" != *"lambda_0.0000"* ]] && [[ "$i" != *"lambda_0.5000"* ]] && [[ "$i" != *"lambda_1.0000"* ]]; then
    rm -rf $i
fi
done

fi

echo "Done deleting, keeping $keep_traj trajectories."
