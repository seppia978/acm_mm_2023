#!/bin/bash
#SBATCH --job-name=unlearning
#SBATCH --output=job_logs/zero_3way_fixed_l1_%a.log
#SBATCH --error=job_logs/zero_3way_fixed_l1_%a.log
#SBATCH --gpus=1
#SBATCH --partition=prod
#SBATCH --array=0-3
#SBATCH --time=5:00:00

lambdas=(0.000025 0.00005 0.0001 0.00025)
bss=(256)
lrs=(0.00005)

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate acm_mm_2023

for i in {0..3}
do
    for j in {0..0}
    do
        for k in {0..0}
        do
            id=$(($i+$j*4+$k*4*1))
            if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
            then
                python -u zero_shot_normal.py -u ${lrs[$k]} -n vit_small_16224 -N vit_small_zero_3way_fixed_l1 -P acm23-gsearch-unlhyp -0 ${lambdas[$i]} -1 1 -2 0 -L zero_3way_fixed_l1 -b ${bss[$j]} -z 1 -T 9999999 -D cifar10 --patience=50
            fi
        done
    done
done