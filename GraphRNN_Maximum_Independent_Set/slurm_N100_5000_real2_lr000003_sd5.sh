#!/bin/bash
#SBATCH --time=8-00:00:00     # Walltime  COMPUTATIONAL TIME: ONCE ELAPSED IT WILL STOP
#SBATCH --mem-per-cpu=16G  # memory/cpu
#SBATCH --job-name=GraphRNN_MIS_P3_500_N100_5000_realTopology_lr000003_sd5  # CAREFUL TO CHANGE IT ALSO IN THE RUN LINE
#SBATCH --partition=general  # This is the default partition
#SBATCH --gres=gpu:1          # Request 2 GPUs
#SBATCH --output=/scratch/ikperez/GraphRNN_MIS/GraphRNN_Maximum_Independent_Set/emaitzak/GraphRNN_MIS_P3_500_N100_5000_realTopology_lr000003_sd5_%j.txt
#SBATCH --error=/scratch/ikperez/GraphRNN_MIS/GraphRNN_Maximum_Independent_Set/emaitzak/GraphRNN_MIS_P3_500_N100_5000_realTopology_lr000003_sd5_%j_error.txt
#SBATCH --cpus-per-task=1
#SBATCH --qos=xlong

module load Python/Python-3.10.9-Anaconda3-2023.03-1
module load CUDA
module load GCC/13.2.0

ENV_PYTHON=/scratch/ikperez/GraphRNN_MIS/GraphRNN5/bin
$ENV_PYTHON/python main_iker.py --datasets Internet_Topology_500_N100_skitter --modelsToTrain GraphRNN_RNN  --maximumEpochs 8000 --learningrate 0.000003 --learningrateDecay 1.0 --milestones 1 --getValidationLoss --epochTestStep 100 --epochsTestStart 0 --zeroEpoch 1 --seed 5 --emaitzenCsvAnGehitu _lr000003_sd5
