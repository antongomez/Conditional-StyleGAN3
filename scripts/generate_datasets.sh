#!/bin/bash
set -euo pipefail

# Definir array de datasets
datasets=(eiras ermidas ferreiras mera mestas oitaven ulla xesta)
datasets=(ulla)

# Definir seeds (incluíndo 0 como caso especial)
seeds=(0 42 43 44 45 46)

# Función para procesar un dataset completo
process_dataset () {
    local ds=$1

    echo "Procesando dataset: $ds"

    # Escoller batch size segundo dataset
    if [ "$ds" == "ulla" ]; then
        batch_size=32
    else
        batch_size=64
    fi

    # Crear directorio de destino se non existe
    INPUT_PATH="data"
    DATASET_PATH="$INPUT_PATH/${ds}"
    mkdir -p "$DATASET_PATH"

    for seed in "${seeds[@]}"; do
        echo "  Usando seed: $seed (batch_size=$batch_size)"

        # Extraer patches co seed correspondente
        python extract_patches.py --input-path="$INPUT_PATH" --filename="$ds" --batch-size="$batch_size" --seed="$seed"

        # Definir sufixo para o nome do zip
        suffix=""
        if [ "$seed" -ne 0 ]; then
            suffix="_${seed}"
        fi

        # Crear zips de train, validation e test
        python dataset_tool.py --source="$DATASET_PATH/patches/train" --dest="$DATASET_PATH/${ds}_train${suffix}.zip"
        python dataset_tool.py --source="$DATASET_PATH/patches/validation" --dest="$DATASET_PATH/${ds}_val${suffix}.zip"
        python dataset_tool.py --source="$DATASET_PATH/patches/test" --dest="$DATASET_PATH/${ds}_test${suffix}.zip"

        # Eliminar directorios temporais
        rm -r "$DATASET_PATH/patches/train/"
        rm -r "$DATASET_PATH/patches/validation/"
        rm -r "$DATASET_PATH/patches/test"

        echo "  Seed $seed completada para $ds"
    done

    echo "Rematado dataset: $ds"
    echo "-----------------------------------"
}

# Lanzar cada dataset en paralelo
for ds in "${datasets[@]}"; do
    process_dataset "$ds"
done

# Esperar a que rematen todos
wait
echo "Todos os datasets procesados!"
