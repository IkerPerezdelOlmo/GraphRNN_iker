#!/bin/bash
#SBATCH --job-name=GraphRNN_MIS_P3_500_N100_5000_realNY_road_lr000003_sd3_eval
#SBATCH --output=./emaitzak/GraphRNN_MIS_bnd_P3_500_N100_5000_realNY_road_lr000003_sd3_%j.txt
#SBATCH --error=./emaitzak/GraphRNN_MIS_bnd_P3_500_N100_5000_realNY_road_lr000003_sd3_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=16G

#SBATCH --partition=GPU
#SBATCH --gpus=1

#SBATCH --array=1-1



bnd -exec python3 "main_iker.py --datasets USA-road_d_500_N100_NY --modelsToTrain GraphRNN_RNN  --maximumEpochs 8000 --learningrate 0.000003 --learningrateDecay 1.0 --milestones 1 --getValidationLoss --epochTestStep 1000 --epochsTestStart 0 --zeroEpoch 1 --seed 3 --emaitzenCsvAnGehitu _lr000003_sd3 --executeEvaluate --onlyEvaluate"
