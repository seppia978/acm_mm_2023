#!/bin/bash
#SBATCH --job-name=unlearning
#SBATCH --output=job_logs/unlearning_sup_%a.log
#SBATCH --error=job_logs/unlearning_sup_%a.log
#SBATCH --gpus=1
#SBATCH --partition=prod
#SBATCH --array=0-1

lambdas=(0.01 0.005)
bss=(256)
lrs=(0.0001)

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate acm_mm_2023

for i in {0..1}
do
    for j in {0..0}
    do
        for k in {0..0}
        do
            id=$(($i+$j*2+$k*2*1))
            if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
            then
                python -u zero_shot_normal.py -u ${lrs[$k]} -n vit_small_16224 -N vit_small_neggrad -P acm23-gsearch-unlhyp -0 1 -1 ${lambdas[$i]} -2 0 -L difference -b ${bss[$j]} -z 1 -T 9999999 -D cifar10 --patience=50
            fi
        done
    done
done