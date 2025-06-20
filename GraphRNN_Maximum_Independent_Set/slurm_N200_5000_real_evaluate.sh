#!/bin/bash
#SBATCH --time=8-00:00:00     # Walltime  COMPUTATIONAL TIME: ONCE ELAPSED IT WILL STOP
#SBATCH --mem-per-cpu=25G  # memory/cpu
#SBATCH --job-name=GraphRNN_MIS_P3_500_N200_5000_realNY_road_eval  # CAREFUL TO CHANGE IT ALSO IN THE RUN LINE
#SBATCH --partition=general  # This is the default partition
#SBATCH --gres=gpu:2          # Request 2 GPUs
#SBATCH --output=/scratch/ikperez/GraphRNN_MIS/GraphRNN_Maximum_Independent_Set/GraphRNN_MIS_P3_500_N200_5000_realNY_road_eval_%j.txt
#SBATCH --error=/scratch/ikperez/GraphRNN_MIS/GraphRNN_Maximum_Independent_Set/GraphRNN_MIS_P3_500_N200_5000_realNY_road_eval_%j_error.txt
#SBATCH --cpus-per-task=1
#SBATCH --qos=xlong

module load Python/Python-3.10.9-Anaconda3-2023.03-1
module load CUDA
module load GCC/13.2.0

ENV_PYTHON=/scratch/ikperez/GraphRNN_MIS/GraphRNN5/bin
$ENV_PYTHON/python main_iker.py --datasets  USA-road_d_500_N200_NY --executeEvaluate --onlyEvaluate --modelsToTrain GraphRNN_RNN  
