#!/bin/bash

#SBATCH --job-name=lr5-6_eval
#SBATCH --output=./emaitzak/zzz_lr5-6_eval_%A_%a.txt
#SBATCH --error=./emaitzak/zzz_lr5-6_eval_%A_%a_error.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1

#SBATCH --mem-per-cpu=10G

#SBATCH --partition=GPU
#SBATCH --gpus=1

# Limitar a 4 trabajos simultáneos máximo
#SBATCH --array=0

# Define arrays of parameters
DATASETS=('Internet_Topology_500_N100_skitter' 'road_d_500_N100_NY' 'random_erdos-renyi_500_M100_P3')
LEARNING_RATE=(0.00003 0.000003)
SEED=(5 7)
LR_TEXT=(00003 000003)
MODEL_SIZE=(1)

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
LR_TEXT=${LR_TEXT[$LR_INDEX]}
SD=${SEED[$SEED_INDEX]}
M_SIZE=${MODEL_SIZE[$MS_INDEX]}

# Crear un sufijo descriptivo para archivos de salida
SUFFIX="_lr${LR_TEXT}_sd${SD}_${M_SIZE}"

echo "Ejecutando con dataset: ${DATASET}, learning rate: ${LR}, seed: ${SD}, model size: ${M_SIZE}"
echo "Sufijo de salida: ${SUFFIX}"

# Ejecutar el comando con los parámetros actuales

bnd -exec python3 "main_iker.py \
    --datasets ${DATASET} \
    --modelsToTrain GraphRNN_RNN \
    --maximumEpochs 8000 \
    --learningrate ${LR} \
    --getValidationLoss \
    --epochTestStep 100 \
    --epochsTestStart 0 \
    --zeroEpoch 1 \
    --seed ${SD} \
    --emaitzenCsvAnGehitu ${SUFFIX} \
    --executeEvaluate \
    --onlyEvaluate \
    --modelSize ${M_SIZE}"
