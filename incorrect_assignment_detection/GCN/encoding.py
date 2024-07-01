import json as js
import torch
import os
import pickle as pk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
from accelerate import Accelerator
import torch.nn as nn

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings):
        weights = torch.softmax(self.attn(embeddings), dim=1)
        weighted_embeddings = embeddings * weights
        return weighted_embeddings.sum(dim=1)

def main(args):
    accelerator = Accelerator()
    
    with open(args.path, "r", encoding="utf-8") as f:
        papers = js.load(f)

    batch_size = 5000

    # Initialize RoBERTa tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base')
    
    hidden_size = model.config.hidden_size
    print("HIDDEN SIZE: ", hidden_size)
    model, tokenizer = accelerator.prepare(model, tokenizer)

    # Initialize the attention layer
    attention_layer = AttentionLayer(hidden_dim=hidden_size).to(accelerator.device)
    attention_layer = accelerator.prepare(attention_layer)
    
    dic_paper_embedding = {}
    papers = [[key, value] for key, value in papers.items()]

    for ii in tqdm(range(0, len(papers), batch_size), total=len(papers)//batch_size):
        batch_papers = papers[ii: ii + batch_size]
        
        # Extract titles, abstracts, and keywords
        titles = [paper[1].get("title", "") for paper in batch_papers]
        abstracts = [paper[1].get("abstract", "") for paper in batch_papers]
        keywords = [','.join(paper[1].get("keywords", [])) for paper in batch_papers]

        # Tokenize separately
        inputs_titles = tokenizer(titles, return_tensors="pt", padding=True, truncation=True, max_length=100)
        inputs_abstracts = tokenizer(abstracts, return_tensors="pt", padding=True, truncation=True, max_length=200)
        inputs_keywords = tokenizer(keywords, return_tensors="pt", padding=True, truncation=True, max_length=50)

        # Move inputs to accelerator device
        inputs_titles = {key: value.to(accelerator.device) for key, value in inputs_titles.items()}
        inputs_abstracts = {key: value.to(accelerator.device) for key, value in inputs_abstracts.items()}
        inputs_keywords = {key: value.to(accelerator.device) for key, value in inputs_keywords.items()}

        with torch.no_grad():
            outputs_titles = model(**inputs_titles)
            outputs_abstracts = model(**inputs_abstracts)
            outputs_keywords = model(**inputs_keywords)

        # Extract CLS token embeddings
        embeddings_titles = outputs_titles.last_hidden_state[:, 0, :]
        embeddings_abstracts = outputs_abstracts.last_hidden_state[:, 0, :]
        embeddings_keywords = outputs_keywords.last_hidden_state[:, 0, :]

        # Stack embeddings and apply attention
        embeddings_stack = torch.stack((embeddings_titles, embeddings_abstracts, embeddings_keywords), dim=1)
        embeddings_combined = attention_layer(embeddings_stack).detach().cpu().numpy()
        
        for jj in range(ii, ii + len(batch_papers)):
            paper_id = papers[jj][0]
            paper_vec = embeddings_combined[jj - ii]
            dic_paper_embedding[paper_id] = paper_vec

    if accelerator.is_main_process:
        with open(args.save_path, "wb") as f:
            pk.dump(dic_paper_embedding, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="dataset/pid_to_info_all.json")
    parser.add_argument("--save_path", type=str, default="dataset/roberta_embeddings.pkl")
    args = parser.parse_args()

    main(args)
