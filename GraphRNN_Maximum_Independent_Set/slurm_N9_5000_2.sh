#!/bin/bash
#SBATCH --job-name=GraphRNN_MIS_P3_500_N9_5000
#SBATCH --output=./emaitzak/GraphRNN_MIS_P3_500_N9_5000_%j.txt
#SBATCH --error=./emaitzak/GraphRNN_MIS_P3_500_N9_5000_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=16G

#SBATCH --partition=GPU
#SBATCH --gpus=1

#SBATCH --array=1-1



bnd -exec python3 main_iker.py --datasets random_erdos-renyi_500_M9_P3 --executeEvaluate --modelsToTrain GraphRNN_RNN



