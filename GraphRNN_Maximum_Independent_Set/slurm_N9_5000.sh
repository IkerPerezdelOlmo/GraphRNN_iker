#!/bin/bash
##SBATCH --time=0-10:00:00     # Walltime  COMPUTATIONAL TIME: ONCE ELAPSED IT WILL STOP
#SBATCH --mem-per-cpu=16G  # memory/cpu
#SBATCH --job-name=GraphRNN_MIS_P3_500_N9_5000  # CAREFUL TO CHANGE IT ALSO IN THE RUN LINE
#SBATCH --partition=GPU  # This is the default partition
#SBATCH --gpus=2         # Request 2 GPUs
#SBATCH --output=/home/iperez/GraphRNN/GraphRNN_MIS/GraphRNN_Maximum_Independent_Set/emaitzak/GraphRNN_MIS_P3_500_N9_5000_%j.txt
#SBATCH --error=/home/iperez/GraphRNN/GraphRNN_MIS/GraphRNN_Maximum_Independent_Set/emaitzak/GraphRNN_MIS_P3_500_N9_5000_%j_error.txt





bnd -exec python main_iker.py --datasets random_erdos-renyi_500_M9_P3 --executeEvaluate --modelsToTrain GraphRNN_RNN  
