#!/bin/bash
#SBATCH --gres=gpu:2

source /cs/labs/adiyoss/ortal1602/lab/bin/activate
module load cuda/10.1

python /cs/labs/adiyoss/ortal1602/projects/BS/train.py dset=valentini  experiment.experiment_name=demucs_hifi_3 experiment=demucs_hifi experiment.hifi.l1_factor=5 experiment.hifi.gen_factor=5

