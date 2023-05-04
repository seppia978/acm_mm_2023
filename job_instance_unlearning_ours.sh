#!/bin/bash
#SBATCH --job-name=max-vits-ours_instance
#SBATCH --output=/nas/softechict-nas-2/spoppi/pycharm_projects/inspecting_twin_models/outs/test/vit_small_16224/difference_zero_fixed/%a.test
#SBATCH --error=/nas/softechict-nas-2/spoppi/pycharm_projects/inspecting_twin_models/outs/test/vit_small_16224/difference_zero_fixed/%a.err
#SBATCH --gpus=1
#SBATCH --partition=prod
#SBATCH --array=0-47
#SBATCH --time=2:00:00
#SBATCH --exclude=huber

lambdas=(1.5 1.25 1 0.9)
# lambdas=(0.4 0.3 0.5)
bss=256
n_imgs=(64 128 256 512)
lrs=(0.005 0.0005 0.00005)
model="vit_small_16224"

# . /usr/local/anaconda3/etc/profile.d/conda.sh
cd /homes/spoppi/pycharm_projects/acm_2023
source activate prova0

for i in {0..3}
do
    for j in {0..3}
    do
        for k in {0..2}
        do
            id=$(($i+$j*4+$k*4*3))
            if [ "$id" -eq "$SLURM_ARRAY_TASK_ID" ]
            then
                python -u instance_unlearning_lora.py -u ${lrs[$k]} -n $model -N $model-zero -P acm23-gsearch-unlhyp -0 ${lambdas[$i]} -1 1 -2 0 -L 3way_zero_fixed -b $bss -z 1 -T 9999999 -D cifar10 --patience=50 --num-imgs-4-instance-unlearning ${n_imgs[$j]}
            fi
        done
    done
done