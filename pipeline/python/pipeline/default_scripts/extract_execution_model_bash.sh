#!/bin/bash 
# script to turn the execution model files into bash suitable arrays and variables.

# define the number of repeats in an array
repeats=$(awk '/repeats/{print $NF}' $prot_file)

re_repeats=$(awk '/rerun start repeat/{print $NF}' $prot_file)
if [ -z "$re_repeats" ]; then
repeats_array=($(seq 0 1 $(($repeats-1))))
else
repeats_array=($(seq $re_repeats 1 $(($repeats-1))))
fi

# get the name of the run
nam=$(awk '/name/{print $NF}' $prot_file)
if [ -z "$nam" ]; then

name=""

else

name="_$nam"

if [ $nam == "None" ]; then
name=""
fi

fi

# keeping the trajectory?
keep_traj=$(awk '/trajectories/{print $NF}' $prot_file)

# For all the ligands in ligands.dat, make an array
lig_array=()
while read lig; do 
lig_clean=$(sed 's/\r$//' <<< $lig); lig_array+=("${lig_clean}");
done < $lig_file

# Make a list of the transformations from the network.dat file
trans_array=()
eng_array=()
win_array=()
declare -A wins_array # assosciative array
IFS=' '
while read trans; do
while read -a tra; do tran=${tra[0]}~${tra[1]}; eng=${tra[-1]}; win=${tra[2]}; wins=(${tra[3]}); #wins=(${tra[@]:3:${#tra[@]}-2})
trans_array+=("$tran"); eng_array+=("$eng"); win_array+=("$win"); wins_array[$win]=$wins; done <<< $trans
done < $net_file

# check that the trans, eng, and win arrays are the same length.
