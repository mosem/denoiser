#!/bin/bash
#SBATCH --gres=gpu:1 --mem=12G
source /cs/labs/adiyoss/ortal1602/lab/bin/activate
python /cs/usr/ortal1602/Desktop/MSc/ortal1602/projects/tmp_testing_BS/test_denoiser.py