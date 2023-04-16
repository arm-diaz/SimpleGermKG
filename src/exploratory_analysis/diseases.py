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
disease_metadata_df = pd.read_csv("../../data/LKP_Cancer_AKA.csv")
disease_metadata_df.loc[:, "CancerName"] = disease_metadata_df.loc[:, "CancerMasterName"].astype(str) + " - " + disease_metadata_df.loc[:, "CancerAltName"].astype(str) + " - " + disease_metadata_df.loc[:, "Organ"].astype(str)
disease_metadata_df.loc[:, "CancerName"] = disease_metadata_df["CancerName"].apply(lambda x: " ".join(set(x.split(' '))))

# NER Disambiguation: Diseases
dict_exact_macth_diseases = json.load(open("../data/disease_abbr_dict.json"))
dict_diseases = json.load(open("../data/common_diseases_dict.json"))

skip_diseases = ["cancer", "tumor", "h", "cancers", "nan cancer", "malignancy", "carcinoma",
                "ps", "e.", "malignant tumours", "s", "sa", "and", "all", "p", "c", "b", "neo",
                "t cell malignancy", "fhit", "ap", "em",
                "malignant tumors", "non - malignant disease", "non - cns solid tumors", "neop",
                "dysplasia", "malignancies", "second malignancies", "bilateral", "tumour", "asm",
                "adenoma", "moderate adenoma", "hpc", "non", "poor", "pm", "hh", "e", "afad", "os",
                "scc", "meta", "advanced cancers", "o", "ii",
                "ma", "cfs", "bi", "ngs tumor", "vhl", "bhd", "tsc", "amls", "ant", "fa", "ch", "dp",
                "chole", "-", "- induced malignancies", "g", "pbi", "the", "mps", "mp", "duodenal",
                "tumors", "mi", "ben", "benign", "hands and", "gene", "complex", "general",
                "dys", "im", "cag", "sg", "laryngeal",  "main cancers", "cll", "f", "ds", "rh d",
                "duo", "pro", "primary tumors", "u", "cvid", "or", "cy", "npc", "n", "fdrs", "ec",
                "es", "acc", "accs", "dcis", "hhc", "lc", "escc", "uc", "rp", "npcs",
                "eoc", "mm", "vus", "hl", "ad", "als", "ald", "at", "a", "di", "lipid", "fap"
                                  ]

ignore_words1 = ["emg", "crc", "t300a", "hpgds", "glutathione", "nan", "fap", "ttr", "dispersin", "hba1c",
                "lynch", "plasmid", "catenin", "growth", "heterotrimeric", "gnas1", "chd1", "rarbeta", "abl", 
                "scf", "human", "factor", "g20210a", "prostate", "hfe"]

ignore_words2 = ["pca", "hgpin", "ho - 1", "se", "t", "at", "lp", "p", "associated", "ls", "alt",
                "bard1", "protein c", "c", "fibrinogen", "hemoglobin", "deletion", "l", "h",
                "gs - g", "gs - e", "x - linked g6pd g202a variant", "aj", "end", "nhl", "hbeag",
                "aj founder mutations", "g", "mipa - d159", "cb", "rarb", "zeb family members",
                "glucocorticoid - induced leucine zipper", "txnip", "i", "r", "ms", "t", "el", 
                "in", "gene", "as", "rh", "o", "-", ".", "z"]

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
        if entity_type == 'disease':
            norm_word, score = normalize_disease(new_str, skip_diseases, dict_diseases, dict_exact_macth_diseases, disease_metadata_df)
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

def normalize_disease(entity, skip_diseases, dict_diseases, dict_exact_macth_diseases, disease_metadata_df):
    disease = None
    score = None
    word = str(entity).lower()
    if word in dict_exact_macth_diseases:
        disease = dict_exact_macth_diseases[word]
        score = 100
    else:
        word_match = [dw for w, dw in dict_diseases.items() if w in word]
        if len(word_match) > 0:
            disease = word_match[0]
            score = 100
        elif word not in skip_diseases:

            disease_match_scores = {row.CancerMasterName: fuzz.token_sort_ratio(word, str(row.CancerName).lower()) for _, row in disease_metadata_df.iterrows()}
            sorted_disease_match_scores = sorted(disease_match_scores.items(), key=lambda x:x[1], reverse=True)
            disease_name = sorted_disease_match_scores[0][0]
            disease_score = sorted_disease_match_scores[0][1]

            if (disease_score > 15):
                disease = disease_name
                score= disease_score

    return  disease, score


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

# Diseases: Named entity recognition pipeline, passing in a specific model and tokenizer
tokenizer_disease = AutoTokenizer.from_pretrained("drAbreu/bioBERT-NER-NCBI_disease", model_max_length=512) # dmis-lab/biobert-v1.1, drAbreu/bioBERT-NER-NCBI_disease", drAbreu/bioBERT-NER-BC2GM_corpus
model_disease = AutoModelForTokenClassification.from_pretrained("drAbreu/bioBERT-NER-NCBI_disease") 
ner_disease = pipeline(task="token-classification", model=model_disease, tokenizer=tokenizer_disease, device=device) # pass device=0 if using gpu

start = time.time()
print(f"Inference Disease")
disease_output = ner_disease(dataset)
end = time.time()
print("The time of execution of above program is :", (end-start) * 10**3, "ms")

start = time.time()
print(f"Create entity table")
disease_entities = []
for idx, diseases in enumerate(disease_output):
    disease_entities.extend(entity_table(diseases, tokenizer=tokenizer_disease, entity_type="disease", sentence=data.loc[idx, "sentence"], is_title=data.loc[idx, "is_title"], pubmed_id=data.loc[idx, "pubmed_id"], sentence_num=data.loc[idx, "sentence_num"]))
end = time.time()
print("The time of execution of above program is :", (end-start), "seconds")

start = time.time()
print(f"Write disease entity table")
disease_entity_df = pd.DataFrame(disease_entities)
disease_entity_df.to_csv(f"diseases.csv", index=None)
end = time.time()
print("The time of execution of above program is :", (end-start), "seconds")
