#!/bin/bash
#SBATCH --job-name=GraphRNN_MIS_P3_500_N200_5000_realNY_road_lr000003_sd5  
#SBATCH --output=./emaitzak/GraphRNN_MIS_bnd_P3_500_N200_5000_realNY_road_lr000003_sd5_%j.txt
#SBATCH --error=./emaitzak/GraphRNN_MIS_bnd_P3_500_N200_5000_realNY_road_lr000003_sd5_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=16G

#SBATCH --partition=GPU
#SBATCH --gpus=1

#SBATCH --array=1-1



bnd -exec python3 "main_iker.py --datasets USA-road_d_500_N200_NY --modelsToTrain GraphRNN_RNN  --maximumEpochs 8000 --learningrate 0.000003 --learningrateDecay 1.0 --milestones 1 --getValidationLoss --epochTestStep 100 --epochsTestStart 0 --zeroEpoch 1 --seed 5 --emaitzenCsvAnGehitu _lr000003_sd5 --executeEvaluate"
