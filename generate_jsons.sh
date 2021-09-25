#!/bin/bash

#noisy_train='C:/ortal1602/valentini/generated_valentini_16k/noisy'
#clean_train='C:/ortal1602/valentini/generated_valentini_16k/clean'
#noisy_test='C:/ortal1602/valentini/generated_valentini_16k/noisy'
#clean_test='C:/ortal1602/valentini/generated_valentini_16k/clean'
#noisy_dev='C:/ortal1602/valentini/generated_valentini_16k/noisy'
#clean_dev='C:/ortal1602/valentini/generated_valentini_16k/clean'
noisy_train='/cs/labs/adiyoss/ortal1602/generated_valentini_noisy/train/generated16000'
clean_train='/cs/labs/adiyoss/ortal1602/generated_valentini/train/generated16000'
noisy_test='/cs/labs/adiyoss/ortal1602/generated_valentini_noisy/test/generated16000'
clean_test='/cs/labs/adiyoss/ortal1602/generated_valentini/test/generated16000'
noisy_dev='/cs/labs/adiyoss/ortal1602/generated_valentini_noisy/val/generated16000'
clean_dev='/cs/labs/adiyoss/ortal1602/generated_valentini/val/generated16000'

mkdir -p egs/val/tr
mkdir -p egs/val/cv
mkdir -p egs/val/tt

python -m denoiser.audio $noisy_train > egs/val/tr/noisy.json
python -m denoiser.audio $clean_train > egs/val/tr/clean.json

python -m denoiser.audio $noisy_test > egs/val/tt/noisy.json
python -m denoiser.audio $clean_test > egs/val/tt/clean.json

python -m denoiser.audio $noisy_dev > egs/val/cv/noisy.json
python -m denoiser.audio $clean_dev > egs/val/cv/clean.json
