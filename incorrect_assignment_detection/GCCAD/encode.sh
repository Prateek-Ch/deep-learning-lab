#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=0-02:00:00
#SBATCH --job-name=gccad_encode
#SBATCH --partition=accelerated-h100
#SBATCH --gres=gpu:4

source ../GCN/pytorch/bin/activate
accelerate launch encoding.py --path ../dataset/pid_to_info_all.json --save_path ../dataset/roberta_embeddings.pkl > encode.txt
python build_graph.py --author_dir ../dataset/train_author.json  --save_dir ../dataset/train.pkl > build_graph.txt
python build_graph.py > build_graph_again.txt
python train.py  --train_dir ../dataset/train.pkl  --test_dir ../dataset/valid.pkl > train.txt
