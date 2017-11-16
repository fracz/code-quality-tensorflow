#!/bin/bash -l
## Liczba alokowanych węzłów
#SBATCH -N 1
## Liczba zadań per węzeł (domyślnie jest to liczba alokowanych rdzeni na węźle)
#SBATCH --ntasks-per-node=1
## Ilość pamięci przypadającej na jeden rdzeń obliczeniowy (domyślnie 5GB na rdzeń)
#SBATCH --mem-per-cpu=4092mb
## Maksymalny czas trwania zlecenia (format HH:MM:SS)
#SBATCH --time=0:05:00
## Nazwa grantu do rozliczenia zużycia zasobów
#SBATCH -A scqfracz
## Specyfikacja partycji
#SBATCH -p plgrid-gpu
## Parametr wyznaczający indeksy zadania tablicowego
#SBATCH --array=1,2
#SBATCH --gres=gpu:1

export DATASET1=100-diff10
export DATASET2=200-diff10

export CURRENT_DATASET_VARIABLE=DATASET$SLURM_ARRAY_TASK_ID
export DATASET=${!CURRENT_DATASET_VARIABLE}

cd /net/people/plgfracz/quality

unset PYTHONPATH
module load plgrid/apps/cuda/8.0
module load plgrid/tools/python/3.6.0
source venv/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/people/plgfracz/cudnn/cuda/lib64

stdbuf -oL python model2.py $DATASET --steps 50000 --numHidden 128 &> logs/model2-$DATASET.log
