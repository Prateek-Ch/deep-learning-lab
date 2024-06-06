import json as js
import torch
import os
import pickle as pk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse

#Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="dataset/pid_to_info_all.json")
parser.add_argument("--save_path", type = str, default = "dataset/roberta_embeddings.pkl")
args = parser.parse_args()

#Load paper data from the specified JSON file
with open(args.path, "r", encoding="utf-8") as f:
    papers = js.load(f)

batch_size = 5000
device = torch.device("cuda:0") #Set the device to the first CUDA-capable GPU

# Initialize RoBERTa tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('roberta-base').to(device)

#Initialize an empty dictionary to store paper embeddings
dic_paper_embedding = {}

#Convert the papers dictionary to a list of key-value pairs
papers = [[key, value] for key,value in papers.items()]

#Process the papers in batches
for ii in tqdm(range(0, len(papers), batch_size), total=len(papers)//batch_size):
    
    batch_papers = papers[ii: ii + batch_size] #Get the current batch of papers
    texts = [paper[1]["title"] for paper in batch_papers] #Extract the titles of the papers in the current batch

    #Tokenize the titles using the RoBERTa tokenizer
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=30)
    inputs = {key: value.to(device) for key, value in inputs.items()} #Move the inputs to the GPU
    with torch.no_grad():    #Disable gradient calculation for inference
        outputs = model(**inputs)  #Pass the inputs through the RoBERTa model
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy() #Get the embeddings for the [CLS] token and move them to the CPU
    tt = 0     #Initialize a counter

    #Store the embeddings in the dictionary
    for jj in range(ii, ii+len(batch_papers)):
        paper_id = papers[jj][0]    #Get the paper ID
        paper_vec = embedding[tt]    #Get the corresponding embedding
        tt+=1
        dic_paper_embedding[paper_id] = paper_vec  #Store the embedding in the dictionary

#Save the embeddings dictionary to a file using pickle
with open(args.save_path, "wb") as f:
    pk.dump(dic_paper_embedding, f)
