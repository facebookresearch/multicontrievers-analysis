#!/bin/bash

source activate /home/seraphina/.conda/envs/sgt

model=$1 # bert, contriever_new
seed=$2
dataset=biasbios
base_dir="/home/seraphina/sgt/data/${dataset}/vectors_extracted_from_trained_models/"

if [ ${model} == "bert" ]
then
  data_dir="${base_dir}google_multiberts-seed_${seed}/seed_${seed}/"
else # contriever_new
   data_dir="${base_dir}contriever/seed_${seed}/"
fi

time python run_inlp.py --model_type ${model} --data_dir ${data_dir} --display > ${data_dir}inlp.log
