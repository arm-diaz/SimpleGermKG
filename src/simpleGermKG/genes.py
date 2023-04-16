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
import re

nltk.download('punkt')
nltk.download('stopwords')

def mapper(word):
    source_truth = {
            'BRCA2': [r'(brca+)(?=\d)|\b2\b'],
            'BRCA1': ['brca'],
            'TP53': ['p53'],
            'VHL': ['vhl'],
            'PALB2': ['palb2'],
            'CDH1': ['cadherin|cdh1'],
            'BAP1': ['bap1'],
            'CASR': ['casr'],
            'CDC73': ['cdc73'],
            'CDK4': ['cdk4'],
            'CDKN1B': ['cdkn1b'],
            'CDKN2A': ['cdkn2a'],
            'CHEK2': ['chek2|chk2'],
            'CDKN1C': ['cdkn1c'],
            'CTNNA1': ['ctnna1'],
            'DICER1': ['dicer1|dicer 1'],
            'MUTYH': ['mutyh'],
            'FH': ['fh'],
            'FLCN': ['flcn'],
            'GPC3': ['gpc3'],
            'GREM1': ['grem1'],
            'MLH1': ['mlh1'],
            'MSH2': ['msh2'],
            'MSH6': ['msh6'],
            'NF1': ['nf1'],
            'HOXB13': ['hoxb13|hoxb 13'],
            'MEN1': ['men1'],
            'MITF': ['mitf'],
            'MSH3': ['msh3'],
            'NBN': ['nbn'],
            'NF2': ['nf2'],
            'NTHL1': ['nthl1'],
            'PDGFRA': ['pdgfra'],
            'PMS2': ['pms2'],
            'POLD1': ['pold1'],
            'POLE': ['pole'],
            'POT1': ['pot1'],
            'PRKAR1A': ['prkar1a'],
            'PTCH1': ['ptch1'],
            'PTEN': ['pten'],
            'RAD50': ['rad50'],
            'RAD51C': ['rad51c'],
            'RAD51D': ['rad51d'],
            'RB1': ['rb1'],
            'RUNX1': ['runx1'],
            'SDHA': ['sdha'],
            'SDHAF2': ['sdhaf2'],
            'SDHB': ['sdhb'],
            'SDHC': ['sdhc'],
            'SDHD': ['sdhd'],
            'SMAD4': ['smad4'],
            'SMARCA4': ['smarca4'],
            'SMARCB1': ['smarcb1'],
            'SMARCE1': ['smarce1'],
            'STK11': ['stk11'],
            'SUFU': ['sufu'],
            'TERC': ['terc'],
            'TERT': ['tert'],
            'TMEM127': ['tmem127'],
            'TSC1': ['tsc1'],
            'TSC2': ['tsc2'],
            'HRAS': ['hras'],
            'BRIP1': ['brip2'],
            'APC': ['apc|adenomatous polyposis coli'],
            'EGFR': ['egfr'],
            'ATM': [r"(atm+)(?=\d)|\batm\b"],
            'ALK': [r"(alk+)(?=\d)|\balk\b"],
            'AIP': [r"(alk+)(?=\d)|\balk\b"],
            'AXIN2': [r"(axin2+)(?=\d)|\baxin2\b"],
            'BARD1': [r"(bard1+)(?=\d)|\bbard1\b"],
            'BLM': [r"(blm+)(?=\d)|\bblm\b"],
            'BMPR1A': [r"(bmpr1a+)(?=\d)|\bbmpr1a\b"],
            'CEBPA': [r"(cebpa+)(?=\d)|\bcebpa\b"],
            'DIS3L2': [r"(dis3l2+)(?=\d)|\bdis3l2\b"],
            'EPCAM': [r"(epcam+)(?=\d)|\bepcam\b"],
            'GATA2': [r"(gata2+)(?=\d)|\bgata2\b"],
            'KIT': [r"(kit+)(?=\d)|\bkit\b"],
            'MAX': [r"(max+)(?=\d)|\bmax\b"],
            'MET': [r"(met+)(?=\d)|\bmet\b"],
            'PHOX2B': [r"(phox2b+)(?=\d)|\bphox2b\b"],
            'RECQL4': [r"(recql4+)(?=\d)|\brecql4\b"],
            'RET': [r"(ret+)(?=\d)|\bret\b"],
            'WRN': [r"(wrn+)(?=\d)|\bwrn\b"],
            'WT1': [r"(wt1+)(?=\d)|\bwt1\b"]

    }

    best_match = "0"
    if type(word) == str:
        for s in source_truth.keys():
            regex = source_truth[s][0]
            gene_match = re.findall(regex, word)
            if len(gene_match) >= 1:
                best_match = s
                break
            else:
                pass

    return [best_match]

def parallel_nen(genes):
    nprocs = mp.cpu_count() - 1
    pool = mp.Pool(processes=nprocs)
    new_genes = pool.starmap(mapper, genes.loc[:, ["word"]].values.tolist())
    new_genes = list(itertools.chain(*new_genes))
    return new_genes

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
            if new_str!='':
                strings.append(dict(
                    pubmed_id = pubmed_id,
                    sentence_num = sentence_num,
                    sentence = sentence,
                    is_title = is_title,
                    word = new_str
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

print(f"Named Entity Normalization")

gene_entity_df = pd.DataFrame(gene_entities)
new_genes = parallel_nen(gene_entity_df)
gene_entity_df.loc[:, "nlp_genes"] = new_genes

print(f"Write gene entity table")
gene_entity_df.to_csv("genes.csv", index=False)

end = time.time()
print("The time of execution of above program is :", (end-start), "seconds")
print("DONE!")
