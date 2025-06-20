#!/bin/bash
#SBATCH --job-name=GraphRNN_MIS_P3_500_N9_5000_lr000003_sd2
#SBATCH --output=./emaitzak/GraphRNN_MIS_bnd_P3_500_N100_5000_lr000003_sd2_%j.txt
#SBATCH --error=./emaitzak/GraphRNN_MIS_bnd_P3_500_N100_5000_lr000003_sd2_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=16G

#SBATCH --partition=GPU
#SBATCH --gpus=1

#SBATCH --array=1-1



bnd -exec python3 "main_iker.py --datasets random_erdos-renyi_500_M100_P3 --modelsToTrain GraphRNN_RNN  --maximumEpochs 8000 --learningrate 0.000003 --learningrateDecay 1.0 --milestones 1 --getValidationLoss --epochTestStep 100 --epochsTestStart 0 --zeroEpoch 1 --seed 2 --emaitzenCsvAnGehitu _lr000003_sd2 --executeEvaluate"
