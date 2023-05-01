#!/bin/bash
#SBATCH --job-name=unlearning
#SBATCH --output=job_logs/lora_zero_3way_fixed_l1_r_%a.log
#SBATCH --error=job_logs/lora_zero_3way_fixed_l1_r_%a.log
#SBATCH --gpus=1
#SBATCH --partition=prod
#SBATCH --array=0-5
#SBATCH --time=5:00:00

rs=(2 4 8 16 32 64)
lambdas=(0.0025)
bss=(256)
lrs=(0.01)

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate acm_mm_2023

for i in {0..5}
do
    for j in {0..0}
    do
        for k in {0..0}
        do
            for h in {0..0}
            do
                id=$(($i+$j*5+$k*5*1))
                if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
                then
                    python -u zero_shot_normal_lora.py -u ${lrs[$k]} -n vit_small_16224 -N vit_small_lora_zero_3way_fixed_l1 -P acm23-gsearch-unlhyp -0 ${lambdas[$h]} -1 1 -2 0 -L zero_3way_fixed_l1 -b ${bss[$j]} -z 1 -T 9999999 -D cifar10 --patience=50 --lora_r ${rs[$i]}
                fi
            done
        done
    done
done