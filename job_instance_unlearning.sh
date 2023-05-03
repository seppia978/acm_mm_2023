#!/bin/bash
#SBATCH --job-name=vits_diff_n-imgs
#SBATCH --output=/nas/softechict-nas-2/spoppi/pycharm_projects/inspecting_twin_models/outs/test/vit_small_16224/difference_zero_fixed/%a.test
#SBATCH --error=/nas/softechict-nas-2/spoppi/pycharm_projects/inspecting_twin_models/outs/test/vit_small_16224/difference_zero_fixed/%a.err
#SBATCH --gpus=1
#SBATCH --partition=prod
#SBATCH --array=0-2
#SBATCH --time=1:00:00
#SBATCH --exclude=huber

lambdas=(0.7)
# lambdas=(0.4 0.3 0.5)
bss=256
n_imgs=(128 256 512)
lrs=(0.0001)
model="vit_tiny_16224"

# . /usr/local/anaconda3/etc/profile.d/conda.sh
cd /homes/spoppi/pycharm_projects/acm_2023
source activate prova0

for i in {0..2}
do
    for j in {0..2}
    do
        for k in {0..0}
        do
            id=$(($i+$j*4+$k*4*1))
            if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
            then
                python -u instance_unlearning.py -u ${lrs[$k]} -n $model -N $model-diff -P acm23-gsearch-unlhyp -0 1 -1 ${lambdas[$i]} -2 0 -L difference -b $bss -z 1 -T 9999999 -D cifar10 --patience=10 --num-imgs-4-instance-unlearning ${n_imgs[$j]}
            fi
        done
    done
done