#!/bin/bash

#SBATCH --job-name=contriever_seeds
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=120

source activate /home/seraphina/.conda/envs/contriever


port=$(shuf -i 15000-16000 -n 1)
TDIR="contriever/encoded-data/original_contriever"
TRAINDATASETS="${TDIR}/wikisub/ ${TDIR}/cc-netsub/"

rmin=0.05
rmax=0.5
T=0.05
QSIZE=131072
MOM=0.9995
POOL=average
AUG=delete
PAUG=0.1
LC=0
mo=$1
mp=none

name=$SLURM_JOB_ID-$POOL-rmin$rmin-rmax$rmax-T$T-$QSIZE-$MOM-$mo-$AUG-$PAUG
outdir=contriever/$name

python train.py \
        --model_path $mp \
        --sampling_coefficient $LC \
        --retriever_model_id $mo --pooling $POOL \
        --augmentation $AUG --prob_augmentation $PAUG \
        --train_data $TRAINDATASETS --loading_mode split \
        --ratio_min $rmin --ratio_max $rmax --chunk_length 256 \
        --momentum $MOM --queue_size $QSIZE --temperature $T \
        --warmup_steps 20000 --total_steps 500000 --lr 0.00005 \
        --name $name \
        --scheduler linear \
        --optim adamw \
        --per_gpu_batch_size 64 \
        --output_dir ${outdir} \
        --main_port $port \
