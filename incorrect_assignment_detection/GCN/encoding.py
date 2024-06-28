import json as js
import torch
import os
import pickle as pk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
from accelerate import Accelerator

def main(args):
    accelerator = Accelerator()
    
    with open(args.path, "r", encoding="utf-8") as f:
        papers = js.load(f)

    batch_size = 5000

    # Initialize RoBERTa tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base')
    
    model, tokenizer = accelerator.prepare(model, tokenizer)
    
    dic_paper_embedding = {}
    papers = [[key, value] for key, value in papers.items()]

    for ii in tqdm(range(0, len(papers), batch_size), total=len(papers)//batch_size):
        batch_papers = papers[ii: ii + batch_size]
        
        # Extract titles, abstracts, and keywords
        titles = [paper[1].get("title", "") for paper in batch_papers]
        abstracts = [paper[1].get("abstract", "") for paper in batch_papers]
        keywords = [','.join(paper[1].get("keywords", [])) for paper in batch_papers]
        
        # Combine titles, abstracts, and keywords into a single text
        texts = [f"{title}. {abstract}. {keywords}." for title, abstract, keywords in zip(titles, abstracts, keywords)]

        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=300)
        inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

        for i in range(3):  # Print first 3 examples
            print(tokenizer.decode(inputs['input_ids'][i].tolist()))

        # Print the shape of the tokenized inputs
        for key, value in inputs.items():
            print(f"{key} shape: {value.shape}")
        
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        for jj in range(ii, ii + len(batch_papers)):
            paper_id = papers[jj][0]
            paper_vec = embedding[jj - ii]
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
