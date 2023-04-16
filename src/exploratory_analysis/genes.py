from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from torch.utils.data import Dataset
import multiprocessing as mp
from torch import cuda
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import fuzz
from typing import List
import itertools
import json
import time
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Read annotations
gene_metadata_df = pd.read_csv("../data/LKP_Gene_AKA.csv")
gene_metadata_df.loc[:, "GeneName"] = gene_metadata_df.loc[:, "GeneMasterName"].astype(str) + " - " + gene_metadata_df.loc[:, "AltNameGene"].astype(str)
gene_metadata_df.loc[:, "GeneName"] = gene_metadata_df["GeneName"].apply(lambda x: " ".join(set(x.split(' '))))

ignore_words1 = ["emg", "crc", "t300a", "hpgds", "glutathione", "nan", "fap", "ttr", "dispersin", "hba1c",
                "lynch", "plasmid", "catenin", "growth", "heterotrimeric", "gnas1", "chd1", "rarbeta", "abl", 
                "scf", "human", "factor", "g20210a", "prostate", "hfe"]

ignore_words2 = ["pca", "hgpin", "ho - 1", "se", "t", "at", "lp", "p", "associated", "ls", "alt",
                "bard1", "protein c", "c", "fibrinogen", "hemoglobin", "deletion", "l", "h",
                "gs - g", "gs - e", "x - linked g6pd g202a variant", "aj", "end", "nhl", "hbeag",
                "aj founder mutations", "g", "mipa - d159", "cb", "rarb", "zeb family members",
                "glucocorticoid - induced leucine zipper", "txnip", "i", "r", "ms", "t", "el", 
                "in", "gene", "as", "rh", "o", "-", ".", "z"]

dict_genes = json.load(open("../data/common_genes_dict.json"))

# Read abstracts
abstracts_df = pd.read_csv("../data/pubmed_abstracts.csv")
abstracts_df = abstracts_df[abstracts_df["abstract_text"].notnull() & abstracts_df["title"].notnull()]

class MyDataset(Dataset):
    def __init__(self, text): 
        self.text=text

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx: int):
        return self.text[idx]

def entity_table(outputs, tokenizer, entity_type: str, pubmed_id: str, sentence_num: str, sentence, is_title):
    results = []
    current = []
    last_idx = 0
    # make to sub group by position
    for output in outputs:
        if output["index"]-1==last_idx:
            current.append(output)
        else:
            results.append(current)
            current = [output, ]
        last_idx = output["index"]
    if len(current)>0:
        results.append(current)
    
    # from tokens to string
    strings = []
    for c in results:
        tokens = []
        starts = []
        ends = []
        for o in c:
            tokens.append(o['word'])
            starts.append(o['start'])
            ends.append(o['end'])
        new_str = tokenizer.convert_tokens_to_string(tokens)
        if entity_type == 'gene':
            norm_word, score = normalize_genes(new_str, ignore_words1, ignore_words2, dict_genes, gene_metadata_df)
            if new_str!='' and norm_word!=None:
                strings.append(dict(
                    pubmed_id = pubmed_id,
                    sentence_num = sentence_num,
                    sentence = sentence,
                    is_title = is_title,
                    word = new_str,
                    normalized_word = norm_word,
                    score = score
                    #start = min(starts),
                    #end = max(ends),
                    #entity = c[0]['entity']
                ))
    return strings

# Breakdown abstract into sentences
def tokenize(abstracts_df):
    sentences = []
    for idx, row in abstracts_df.iterrows():
        tokens = nltk.sent_tokenize(row.abstract_text)
        sentences.append([row.pubmed_id, 0, row.title, 1])
        for idx, sentence in enumerate(tokens, 1):
            sentences.append([row.pubmed_id, idx, sentence, 0])
    return sentences


def tokenize_worker(pubmed_id, title, abstract_text):
   output = [[pubmed_id, 0, title, 1]] + [[pubmed_id, idx, sentence, 0] for idx, sentence in enumerate(nltk.sent_tokenize(abstract_text), 1)]
   return output

def parallel_tokenize(abstracts_df):
    nprocs = mp.cpu_count() - 1
    pool = mp.Pool(processes=nprocs)
    sentences = pool.starmap(tokenize_worker, abstracts_df.loc[:, ["pubmed_id", "title", "abstract_text"]].values.tolist())
    sentences = list(itertools.chain(*sentences)) 

    return sentences

def normalize_genes(entity, ignore_words1, ignore_words2, dict_genes, gene_metadata_df):
    gene = None
    score = None
    word = str(entity).lower()
    word_match = [dw for w, dw in dict_genes.items() if w in word]
    if len(word_match) > 0:
        gene = word_match[0]
        score = 100
    elif any([True if i in word else False for i in ignore_words1]):
        pass
    elif word not in ignore_words2:

        gene_match_scores = {row.GeneMasterName: fuzz.ratio(word, str(row.GeneMasterName).lower()) for _, row in gene_metadata_df.iterrows()}
        sorted_gene_match_scores = sorted(gene_match_scores.items(), key=lambda x:x[1], reverse=True)
        gene_name = sorted_gene_match_scores[0][0]
        gene_score = sorted_gene_match_scores[0][1]
        if gene_score > 15:
            gene = gene_name
            score = gene_score
        else:
            gene_match_scores = {row.GeneMasterName: fuzz.ratio(word, str(row.AltNameGene).lower()) for _, row in gene_metadata_df.iterrows()}
            sorted_gene_match_scores = sorted(gene_match_scores.items(), key=lambda x:x[1], reverse=True)
            gene_name = sorted_gene_match_scores[0][0]
            gene_score = sorted_gene_match_scores[0][1]
            if gene_score > 15:
                gene = gene_name
                score = gene_score

    return  gene, score


start = time.time()
if abstracts_df.shape[0] < 10000:
    sentences = tokenize(abstracts_df)
else:
    sentences = parallel_tokenize(abstracts_df)
end = time.time()
print("The time of execution of above program is :", (end-start), "seconds")

# Convert into DataFrame
data = pd.DataFrame(sentences, columns=["pubmed_id", "sentence_num", "sentence", "is_title"])
dataset = MyDataset(text=data["sentence"].tolist())

device = 1 if cuda.is_available() else -1

print(f"device: {device}")

# Genes: Named entity recognition pipeline, passing in a specific model and tokenizer
model_gene = AutoModelForTokenClassification.from_pretrained("drAbreu/bioBERT-NER-BC2GM_corpus")
tokenizer_gene = AutoTokenizer.from_pretrained("drAbreu/bioBERT-NER-BC2GM_corpus", model_max_length=512)
ner_gene = pipeline('token-classification', model=model_gene, tokenizer=tokenizer_gene, device=device)

start = time.time()
print(f"Inference Gene")
gene_output = ner_gene(dataset)
end = time.time()
print("The time of execution of above program is :", (end-start) * 10**3, "ms")

start = time.time()
print(f"Create entity table")
disease_entities = []
gene_entities = []
for idx, genes in enumerate(gene_output):
    gene_entities.extend(entity_table(genes, tokenizer=tokenizer_gene, entity_type="gene", sentence=data.loc[idx, "sentence"], is_title=data.loc[idx, "is_title"], pubmed_id=data.loc[idx, "pubmed_id"], sentence_num=data.loc[idx, "sentence_num"]))
end = time.time()
print("The time of execution of above program is :", (end-start), "seconds")

start = time.time()
print(f"Write gene entity table")
gene_entity_df = pd.DataFrame(gene_entities)
gene_entity_df.to_csv(f"genes.csv", index=None)
end = time.time()
print("The time of execution of above program is :", (end-start), "seconds")
print("DONE!")
