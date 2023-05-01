#!/bin/bash
#SBATCH --job-name=unlearning
#SBATCH --output=job_logs/lora_zero_3way_fixed_l1_%a.log
#SBATCH --error=job_logs/lora_zero_3way_fixed_l1_%a.log
#SBATCH --gpus=1
#SBATCH --partition=prod
#SBATCH --array=0-4
#SBATCH --time=5:00:00

lambdas=(0.05 0.075 0.001 0.0025 0.005)
bss=(256)
lrs=(0.01)

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate acm_mm_2023

for i in {0..4}
do
    for j in {0..0}
    do
        for k in {0..0}
        do
            id=$(($i+$j*5+$k*5*1))
            if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
            then
                python -u zero_shot_normal_lora.py -u ${lrs[$k]} -n vit_small_16224 -N vit_small_lora_zero_3way_fixed_l1 -P acm23-gsearch-unlhyp -0 ${lambdas[$i]} -1 1 -2 0 -L zero_3way_fixed_l1 -b ${bss[$j]} -z 1 -T 9999999 -D cifar10 --patience=50
            fi
        done
    done
done