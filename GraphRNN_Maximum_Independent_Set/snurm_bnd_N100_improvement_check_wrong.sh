#!/bin/bash

#SBATCH --job-name=GraphRNN_MIS_multiple
#SBATCH --output=./emaitzak/GraphRNN_MIS_multiple_%j.txt
#SBATCH --error=./emaitzak/GraphRNN_MIS_multiple_%j_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
# #SBATCH --ntasks-per-core=2

#SBATCH --mem-per-cpu=10G

#SBATCH --partition=GPU
#SBATCH --gpus=1

# Use SLURM array jobs to run in parallel
#SBATCH --array=0-4




# Define arrays of learning rate decay values and datasets
DATASETS=('random_Community_500_M100_P3' 'Internet_Topology_500_N100_skitter' 'road_d_500_N100_NY' 'random_erdos-renyi_500_M100_P3')

# Fixed parameters
LEARNING_RATE=(0.003 0.0003)
SEED=(1 2)
MODEL_SIZE=(1 2)

# Loop through each learning rate decay and dataset combination
for LR in "${LEARNING_RATE[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for SD in "${SEED[@]}"; do
            for M_SIZE in "${MODEL_SIZE[@]}"; do
                # Create a descriptive suffix for output files
                SUFFIX="improvement_${DATASET}_lr000003_decay${LR}_sd${SD}_model-sz${M_SIZE}"
                
                echo "Running with dataset: ${DATASET}, learning rate decay: ${DECAY}"
                
                # Execute the command with the current parameters
                bnd -exec python3 "main_iker.py \
                --datasets ${DATASET} \
                --modelsToTrain GraphRNN_RNN \
                --maximumEpochs 3000 \
                --learningrate ${LR} \
                --getValidationLoss \
                --epochTestStep 100 \
                --epochsTestStart 0 \
                --zeroEpoch 1 \
                --seed ${SD} \
                --emaitzenCsvAnGehitu ${SUFFIX} \
                --executeEvaluate \
                --modelSize ${M_SIZE}"
                
                # Optional: add a sleep command if you want to pause between runs
                # sleep 10
            done
        done
    done
done