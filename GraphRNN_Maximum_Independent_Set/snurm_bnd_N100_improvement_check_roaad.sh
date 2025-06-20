#!/bin/bash

#SBATCH --job-name=Proba2
#SBATCH --output=./emaitzak/ZProba_%A_%a.txt
#SBATCH --error=./emaitzak/ZProba_%A_%a_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1

#SBATCH --mem-per-cpu=10G

#SBATCH --partition=GPU
#SBATCH --gpus=1

# Limitar a 4 trabajos simultáneos máximo
#SBATCH --array=0-7%3

# Define arrays of parameters
DATASETS=('USA-road_d_500_N100_NY')
LEARNING_RATE=(0.003 0.0003)
SEED=(1 2)
MODEL_SIZE=(1 2)

# Usar el ID de tarea para determinar qué combinación ejecutar
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calcular índices para cada parámetro
MS_INDEX=$(( TASK_ID % ${#MODEL_SIZE[@]} ))
SEED_INDEX=$(( (TASK_ID / ${#MODEL_SIZE[@]}) % ${#SEED[@]} ))
LR_INDEX=$(( (TASK_ID / (${#MODEL_SIZE[@]} * ${#SEED[@]})) % ${#LEARNING_RATE[@]} ))
DS_INDEX=$(( TASK_ID / (${#MODEL_SIZE[@]} * ${#SEED[@]} * ${#LEARNING_RATE[@]}) ))

# Comprobar si el ID de tarea es válido
if [ $DS_INDEX -ge ${#DATASETS[@]} ]; then
    echo "ID de tarea $TASK_ID excede el número de combinaciones. Saliendo."
    exit 0
fi

# Extraer los parámetros específicos para esta tarea
DATASET=${DATASETS[$DS_INDEX]}
LR=${LEARNING_RATE[$LR_INDEX]}
SD=${SEED[$SEED_INDEX]}
M_SIZE=${MODEL_SIZE[$MS_INDEX]}

# Crear un sufijo descriptivo para archivos de salida
LR_FORMATTED="${LR//./_}"
SUFFIX="_improvement_${DATASET}_lr${LR_FORMATTED}_sd${SD}_model-sz${M_SIZE}"

echo "Ejecutando con dataset: ${DATASET}, learning rate: ${LR}, seed: ${SD}, model size: ${M_SIZE}"
echo "Sufijo de salida: ${SUFFIX}"

# Ejecutar el comando con los parámetros actuales

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
