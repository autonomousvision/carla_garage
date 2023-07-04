#!/usr/bin/env bash

cd ..
mkdir data
cd data

down_load_unzip() {
  wget https://s3.eu-central-1.amazonaws.com/avg-projects-2/jaeger2023arxiv/dataset/$1.zip
  unzip -q $1.zip
  rm $1.zip
}

# Download 2022 dataset
for scenario in ll_dataset_2023_05_10 rr_dataset_2023_05_10 lr_dataset_2023_05_10 rl_dataset_2023_05_10 s1_dataset_2023_05_10 s3_dataset_2023_05_10 s7_dataset_2023_05_10 s10_dataset_2023_05_10 s4_dataset_2023_05_10 s8_dataset_2023_05_10 s9_dataset_2023_05_10
do
  down_load_unzip "${scenario}" &
done

