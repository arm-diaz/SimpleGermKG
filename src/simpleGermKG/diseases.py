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
                "Pancreas ('Neuroendocrine')": ["pancreatic|pancreas", "paraganglioma|ochromocytoma|vhl|VHL|Von Hippel-Lindau|von hippel-lindau|gastrinoma|glucagonoma|gastrinoma|gastero|insulin|endocrine|islet|gastroentero|entero|neurogenic|carcinoid|pituitary|parathyroid|ulcerogenic|function|men1|stromal|secret|a - cell|beta|alpha|somatostatin|polypeptide|glucagon|apudoma|vip|adrenal|smooth muscle|hormon|brain"],
                "Colorectal ('Neuroendocrine')": ["appendix", "endocrine"],
                "Gastric ('Neuroendocrine')": ["gastric", "carcinoid"],
                "Lung ('Neuroendocrine')": ["lung", "carcinoid"],
                "Bladder ('Neuroendocrine')": ["bladder|urinary", "endocrine"],
                "Pitutitary ('Neuroendocrine')": ["pitutitary", "endocrine"],
                "Thymus ('Neuroendocrine')": ["thymus", "carcinoid"],
                "Uterine ('Neuroendocrine')": ["uteri", "endocrine"],
                "Uterus ('Neuroendocrine')": ["uterus", "endocrine"],
                "Adrenal Medulla Cancer": ["adrenal", "medulla"],
                "Adrenal Neoplasm": ["adrenal", "neoplasm|hyperplasia|adenoma|tumor|tumour|cyst"],
                "Adrenal Cortical (Benign)": ["adrenal", "benign|disease"],
                "Adrenal Cortical Carcinoma": ["adrenal", "cancer|carcinoma|malign"],
                "Adrenal Hypoplasia": ["adrenal", "hypoplasia"],
                "Upper Urinary Tract Cancer": ["kidney/ureter|renal pelvis|upper urinary tract", "cancer|carcinoma|malign"],
                "Appendiceal Neoplasm": ["appendiceal", "neoplasm|adenoma|tumor|tumour|cyst"],
                "Appendiceal Cancer": ["appendiceal", "cancer|carcinoma|malign"],
                "Hepatobiliary Cancer": ["bile duct|biliary|gall bladder|gallbladder|hepatobiliary|liver", "cancer|carcinoma|malign|neoplasm"],
                "Bladder (Benign)": ["bladder", "benign"],
                "Bladder Cancer": ["bladder", "cancer|carcinoma|malign"],
                "Bladder Neoplasm": ["bladder", "neoplasm|adenoma|tumor|tumour|cyst|polyp"],
                "Blood (Benign)": ["blood|anemia|erythrocytosis|polycythemia|thrombocythaemia|thrombocytopenia", "benign|aplastic|erythrocytosis|polycythemia|thrombocythaemia|thrombocytopenia"],
                "Blood Vessel (Benign)": ["blood|vessel|hemangioma|telangiectasia|angioma", "vessel|hemangioma|telangiectasia|angioma"],
                "Bone (Benign)": ["bone|craniosynostosis|jaw|osteoarthritis|osteopetrosis|rheumatoid|skeletal|spine", "craniosynostosis|keratocyst|tumor|tumour|lesion|cyst|dysplasia|osteoarthritis|osteopetrosis|rheumatoid"],
                "Bone Cancer": ["bone", "cancer|carcinoma|malign"],
                "Bone Neoplasm": ["bone|osteochondromyxoma|osteoma", "neoplasm|adenoma|osteochondromyxoma|osteoma"],
                "Brain Tumor": ["glioblastoma|glioma|choroid|oligodendroglioma|medulloblastoma|gangliocytoma|cerebellar|nervous tissue|meninges|chorea|meningioma|neuroma|medulloblastoma|astrocytoma|meningioma|astroblastoma|astrocytic|astrocytoma|astroglioma|brain", "meningioma|neuroma|medulloblastoma|astrocytoma|meningioma|astroblastoma|astrocytic|astrocytoma|astroglioma|brain|cancer|carcinoma|tumor|tumour|glioma|malign|neoplasm|gangliocytoma|hemangioblastoma|medulloblastoma|oligodendroglioma|glioblastoma"],
                "Breast (Benign)": ["breast|gynecomastia|in situ|phyllodes", "benign|fibroadenoma|myxoma|hyperplasia|disorders|disease|gynecomastia"],
                "Breast Cancer (Contralateral)": ["breast|mammary", "second|contralateral|primary"],
                "Breast Cancer (Male)": ["breast", "male"],
                "Breast Neoplasm": ["breast|atypical ductal|atypical lobular", "neoplasm|adenoma|hyperplasia"],
                "Breast Cancer": ["breast|in situ|invasive ductal|invasive lobular|phyllodes|mammary", "cancer|carcinoma|malign|tumor|tumour"],
                "Buccal Cancer": ["buccal", "cancer|carcinoma|malign|tumor|tumour"],
                "Cardiovascular (Benign)": ["congenital|cardiac|heart|septal", "fibromas|defect|disease"],
                "Cardiovascular Neoplasm": ["congenital|cardiac|heart|septal", "cancer|myxoma|myopathy"],
                "Central Nervous System (Benign)": ["nervous system|epilepsy", "benign"],
                "Cervical Cancer": ["cervix|cervical", "cancer|carcinoma|malign|neoplasm"],
                "Childhood Cancer": ["childhood", "cancer|carcinoma|malign|neoplasm"],
                "CNS Tumor": ["cns|nervous system", "cancer|carcinoma|malign|neoplasm|tumor|tumour"],
                "Colorectal Cancer": ["colon|colorectal|colorectum|rectum|sigmoid|large intestine", "cancer|carcinoma|malign"],
                "Colorectal Neoplasm": ["colon|colorectal|colorectum|rectum|sigmoid|large intestine", "neoplasm|adenoma|hyperplasia|polyp|tumor|tumour"],
                "Diaphragmatic Hernia": ["diaphragmatic", "hernia"],
                "Duodenum Cancer": ["duodenum", "cancer|carcinoma|malign"],
                "Duodenum Neoplasm": ["duodenum", "neoplasm|adenoma|hyperplasia|polyp|tumor|tumour"],
                "Duodenal Cancer": ["duodenal", "cancer|carcinoma|malign|carcinoid"],
                "Duodenal Neoplasm": ["duodenal", "neoplasm|adenoma|hyperplasia|polyp|tumor|tumour"],
                "Dyskeratosis Congenita": ["dyskeratosis", "congenita"],
                "Dysmorphic Facises": ["Polysyndactyly|Prognathism", "Polysyndactyly|Prognathism"],
                "Dyskeratosis Feature": ["facies|macrocephaly|microcephaly|cicrophthalmia", "abnormal|deform|microcephaly|cicrophthalmia"],
                "Dysplastic": ["dysplastic", "dysplastic"],
                "Dystonia": ["dystonia", "dystonia"],
                "Ear (Benign)": ["ear|deafness|hearing loss", "benign|deafness|hearing loss"],
                "Endocrine Abnormalities": ["diabetes|hyperparathyroid|hyperparathyroidism|hypocalcemia|hypocalciuric|hypoparathyroidism", "diabetes|hyperparathyroid|hyperparathyroidism|hypocalcemia|hypocalciuric|hypoparathyroidism"],
                "Uterus Cancer": [r"\b(uterus|uterine)\b.*\b(body|corpus)\b|\b(body|corpus)\b.*\b(uterus|uterine)\b", "cancer|carcinoma|malign|carcinofibroma"],
                "Endometrial Cancer": ["endometrial|endometrium|uterine|uterus", "cancer|carcinoma|malign"],
                "Endometrial Neoplasm": ["endometrial|uterus", "neoplasm|neoplasia|adenoma|hyperplasia|polyp|tumor|tumour"],
                "Esophageal Cancer": ["esophagus|oesophageal|esophagus", "cancer|carcinoma|malign"],
                "Extracolonic Cancer": ["extracolonic|extra-colonic|extraintestinal|extrocolonic", "cancer|carcinoma|malign"],
                "Extrapancreatic Cancer": ["extrapancreatic", "neoplasm"],
                "Eye (Benign)": ["anirida|coloboma|epicanthal|eye|ocular|retinal|iris", "anirida|coloboma|epicanthal|benign|demorm|anomal|lesion|hamartoma"],
                "Eye Cancer": ["eye|ocular|retinal|iris", "cancer|carcinoma|malign"],
                "Eye Neoplasm": ["eye|ocular|retinal|iris", "myxoma|neoplasm|neoplasia|adenoma|polyp|tumor|tumour"],
                "Facial Palsy": ["facial", "palsy"],
                "Fibrohistiocytoma": ["fibrohistiocytoma", "fibrohistiocytoma"],
                "Gastric Cancer": ["stomach|gastric", "cancer|carcinoma|malign"],
                "Gastric Neoplasm": ["stomach|gastric", "neoplasm|adenoma|hyperplasia|polyp|tumor|tumour"],
                "Gastrointestinal (Benign)": ["gastrointestinal|hirschsprung|bowel|megacolon|meckel|tracheoesophageal", "benign|disease|inflammatory|diverticulum|fistula"],
                "Gastrointestinal Cancer": [r"digestive|gastrointestinal|intestinal|\bgi\b", "cancer|carcinoma|malign"],
                "Genitourinary (Benign)": ["genital|genitourinary|vaginal|vulvar", "cancer|carcinoma|malign"],
                "Genitourinary Cancer": ["genital|vaginal|vulvar", "deficiency"],
                "Fumarase Deficiency": ["fumarase", "deficiency"],
                "GI Neoplasm": [r"digestive|gastrointestinal|intestinal|\bgi\b|hamartomatous", "neoplasm|adenoma|polyp|cyst"],
                "GI Hamartoma": [r"digestive|gastrointestinal|intestinal|\bgi\b", "hamartoma"],
                "GIST": [r"digestive|gastrointestinal|intestinal|\bgi\b", "stroma"],
                "Gynecologic Cancer": ["gynecologic|female reproductive", "cancer|carcinoma|malign"],
                "Hair (Benign)": ["alopecia", "areata|mucinosa|benign"],
                "Hand (Benign)": ["hand", "benign"],
                "Head and Neck (Benign)": ["head and neck", "benign"],
                "Head and Neck Cancer": ["head and neck", "cancer|carcinoma|malign"],
                "Hematologic Cancer": ["hematologic|hematolymphoid", "cancer|carcinoma|malign"],
                "Hematologic Neoplasm": ["hematologic|hematopoietic", "neoplasm|adenoma|polyp|cyst"],
                "Hidradenoma": ["hidradenoma", "hidradenoma"],
                "Hypereosinophilic Syndrome": ["hypereosinophilic", "syndrome"],
                "Hypotonia": ["hypotonia", "hypotonia"],
                "Immunodeficiency": ["immunodeficiency", "immunodeficiency"],
                "Inflammatory Myofibroblastic Tumor": ["myofibroblastic", "inflammatory|tumor"],
                "Juvenile Polyps": ["juvenile", "polyps"],
                "Kidney (Benign)": ["nephrotic|nephroma|hydronephrosis|kidney|nephroblastomatosis|renal", "agenesis|malformation|cyst|syndrome|hydronephrosis|stone|benign|nephroblastomatosis"],
                "Kidney Cancer": ["kidney|renal", "sarcoma|cancer|carcinoma|malign|tumor|tumour"],
                "Kidney Neoplasia": ["kidney|renal", "plasia"],
                "Kidney Neoplasm": ["kidney|renal", "neoplasm|adenoma|polyp|cyst|angioma|angiomyolipoma"],
                "Laryngotracheal Stenosis": ["laryngotracheal", "stenosis"],
                "Larynx Cancer": ["laryngeal|larynx", "cancer|carcinoma|malign|tumor|tumour"],
                "Leiomyoma": ["leiomyoma", "leiomyoma"],
                "Leukemia": ["leukemia|leukaemia", "leukemia|leukaemia"],
                "Leukodystrophy": ["leukodystrophy", "leukodystrophy"],
                "Lipodystrophy": ["lipodystrophy", "lipodystrophy"],
                "Lipoma": ["lipoma", "lipoma"],
                "Liver (Benign)": ["liver|hepatomegaly", "benign|disease|hamartoma|hepatomegaly"],
                "Lung (Benign)": ["bronchial|lung|respiratory|pleural|pulmonary|pneumothorax", "polyp|benign|disease|cyst|distress|blastoma|pneumothorax"],
                "Lung Cancer": ["bronchial|lung|respiratory|pulmonary|bronchus", "cancer|carcinoma|malign|tumor|tumour"],
                "Lung Neoplasm": ["bronchial|lung|respiratory|pulmonary|bronchus", "neoplasm|adenoma|chondroma|nodule"],
                "Lymphadenopathy": ["lymphadenopathy", "lymphadenopathy"],
                "Lymphangiomyomatosis": ["lymphangioleiomyomatosis|lymphangiomyomatosis", "lymphangioleiomyomatosis|lymphangiomyomatosis"],
                "Lymphedema": ["lymphedema", "lymphedema"],
                "Lymphoma": ["lymphoma|hodgkin|lymphosarcoma", "lymphoma|hodgkin|lymphosarcoma"],
                "Lymphomesentric Cyst": ["lymphomesentric", "cyst"],
                "Mandibular Hypoplasia": ["mandibular", "hypoplasia"],
                "Mandibuloacral Dysostosis": ["mandibuloacral", "dysostosis"],
                "Mastocytosis": ["mastocytosis", "mastocytosis"],
                "MDPL Syndrome": ["MDPL", "syndrome"],
                "Melanoma (Uveal)": ["melanoma", "uveal"],
                "Melanoma": ["melanoma", "melanoma"],
                "Mesothelioma": ["mesothelium|mesothelioma", "mesothelium|mesothelioma"],
                "Metaphyseal Dysplasia": ["metaphyseal", "dysplasia"],
                "Mirror Movements": ["mirror", "movements"],
                "Mitochondrial Complex II Deficiency": ["mitochondrial complex", "mitochondrial complex"],
                "Multiple Congenital Exostosis": ["exostosis", "exostosis"],
                "Multiple Fibrofolliculomas": ["fibrofolliculomas", "fibrofolliculomas"],
                "Myelodysplasia": ["myelodysplasia", "myelodysplasia"],
                "Myopathy": ["myopathy", "myopathy"],
                "Nails (Benign)": ["nails", "benign"],
                "Nasopharyngeal Cancer": ["nasopharyngeal|angiofibroma|nasal", "hamartoma|angiofibroma|cancer|carcinoma|malign|tumor|tumour|neoplasm"],
                "Neurofibroma": ["neurofibroma|neuroepithelial", "neurofibroma|neoplasm"],
                "Nose (Benign)": ["choanal|nasal", "atresia|polyp"],
                "Oral (Benign)": ["oral", "leukoplakia|papillomatosis"],
                "Oral Cancer": ["oral", "cancer"],
                "Orolaryngeal Carcinoma": ["orolaryngeal", "cancer|carcinoma|malign"],
                "Oropharynx (Benign)": ["oropharynx|bifid epiglottis|cleft|tongue|microstomia|papillomatous papule", "benign|lip|palate|hamartoma|microstomia|bifid epiglottis|orofacial|papillomatous papule"],
                "Oropharynx Cancer": ["pharynx|throat", "cancer|carcinoma|malign"],
                "Osteofibrous Dysplasia": ["osteofibrous", "dysplasia"],
                "Myelodysplasia": ["myelodysplasia", "myelodysplasia"],
                "Myelodysplasia": ["myelodysplasia", "myelodysplasia"],
                "Myelodysplasia": ["myelodysplasia", "myelodysplasia"],
                "Ovarian (Benign)": ["ovarian|ovaries|ovary|fallopian|peritoneal|peritoneum|sex cord", 
                                   "benign|fibroma|premature|immature|streak"],
                "Ovarian (Borderline)": ["ovarian|ovaries|ovary|fallopian|peritoneal|peritoneum|sex cord", 
                                   "borderline"],
                "Ovarian Cancer": ["ovarian|ovaries|ovary|fallopian|peritoneal|peritoneum|sex cord", 
                                   "cancer|carcinoma|malign|endometrioid|small cell|primary"], 
                "Ovarian Neoplasm": ["ovarian|ovaries|ovary|fallopian|peritoneal|peritoneum|sex cord", 
                                     "neoplasm|non epithelial|adenoma|tumor|cyst"],
                "Pancreatitis": ["pancreatitis|pancreas", "pancreatitis"],
                "Pancreatic Schwannoma": ["pancreatic|pancreas", "schwannoma"], 
                "Pancreatic Neoplasm": ["pancreatic|pancreas", "neoplas|adenoma|demoid|benign|pseudopapillary"],
                "Pancreatic Cancer": ["pancreatic|pancreas", "carcinoma|cancer|malign|melanoma|head|neck|body|destruction|cyst|solid|meningeal|primary|hepatobiliary|pseudopapillary|gastrointestinal|tumor|tumour"],
                "Parathyroid (Benign)": ["parathyroid|hyperparathyroidism", "benign|primary"],
                "Parathyroid Carcinoma": ["parathyroid|hyperparathyroidism", "cancer|carcinoma|malign"],
                "Parathyroid Neoplasm": ["parathyroid|hyperparathyroidism", "neoplasm|adenoma|tumor|cyst"],
                "Parkinson": ["parkinson", "parkinson"],
                "Parotid Gland Cancer": ["parotid gland", "cancer"],
                "Peripheral Nervous System Benign": ["ganglioneuroma|melanotic|psammomatous|vestibular", "ganglioneuroma|schwannoma"],
                "Peripheral Nervous System Cancer": ["peripheral nervous|neuroblastoma|mpnst|neurosarcoma", "neuroblastoma|mpnst|neurosarcoma|cancer|carcinoma|malign"],
                "Peripheral Nervous System Neoplasm": ["peripheral nervous|nerve sheath|neuroectodermal", "tumor|neoplasm|neuroectodermal"],
                "Pheochromocytoma/Paraganglioma": ["paraganglioma|pheochromocytoma", "paraganglioma|pheochromocytoma"],
                "Pituitary Cancer": ["pituitary", "cancer|carcinoma|malign"],
                "Pituitary Neoplasm": ["pituitary|prolactinoma|somatotroph", "prolactinoma|tumor|neoplasm|adenoma|blastoma"],
                "Plasma Cell Cancer": ["myeloma", "myeloma"],
                "Plasma Cell Neoplasm": ["histiocytosis|macroglobulinemia|monoclonal|myelodysplastic|plasma cell|plasmacytoma", "plasmacytoma|histiocytosis|macroglobulinemia|monoclonal|myelodysplastic|neoplasm|adenoma"],
                "Prostate Benign": ["prostatic|prostate", "benign|nodular|hyperplasia"],
                "Prostate Cancer": ["prostatic|prostate", "cancer|carcinoma|malign"],
                "Prostate Neoplasm": ["prostatic|prostate", "neoplasm|adenoma"],
                "Psychiatric": ["autism|schizophrenia|schizophrenic|psychiatric", "autism|schizophrenia|schizophrenic|psychiatric"],
                "Pulmonary Fibrosis": ["pulmonary", "fibrosis"],
                "Retinoblastoma": ["retinoblastoma", "retinoblastoma"],
                "Retroperitoneum And Peritoneum Cancer": ["retroperitoneum|peritoneum", "retroperitoneum|peritoneum"],
                "Sclerosteosis": ["sclerosteosis", "sclerosteosis"],
                "Sebaceous Adenoma": ["sebaceous", "adenoma"],
                "Sebaceous Cancer": ["sebaceous", "cancer|carcinoma|malign"],
                "Skin (Benign)": ["acral keratoses|albinism|spitz|cutaneous|cafe|nevus|ephelides|epidermoid|epidermolysis|facial|keratoacanthoma|keratosis|lentigines|livedo|nevi|skin|trichilemmoma", "keratosis|lentigines|livedo|nevi|keratoacanthoma|papules|epidermolysis|ephelides|lichen|amyloidosis|myxoma|cyst|nevus|cafe|acral keratoses|albinism|spitz|benign|freckling|papilloma|pigment|rash|tag|trichilemmoma"],
                "Skin Cancer (Non-Melanoma)": ["basal cell|skin", "cancer|carcinoma|malign"],
                "Skin Neoplasm": ["basal cell|skin|cutaneous|cylindromatosis", "neoplasm|cylindromatosis|leiomyoma"],
                "Small Intestine Benign": ["intussusception", "intussusception"],
                "Small Intestine Cancer": ["small intestine|small bowel", "cancer|carcinoma|malign"],
                "Small Intestine Neoplasm": ["small intestine|small bowel", "neoplasm|aadenoma|polyp|tumor"],
                "Soft Tissue Cancer": ["soft tissue", "cancer|carcinoma|malign"],
                "Soft Tissue Neoplasm": ["soft tissue", "neoplasm|neoplasia|leiomyoma|tumor"],
                "Spleen (Benign)": ["spleen|splenomegaly", "benign|splenomegaly"],
                "Teeth (Benign)": ["teeth|dental|malocclusion|odontogenic|odontoma|oligodontia|tooth", "teeth|dental|malocclusion|odontogenic|odontoma|oligodontia|tooth"],
                "Testicular Benign": ["hydrocele|varicocele", "hydrocele|varicocele"],
                "Testicular Cancer": ["testicular|gonadoblastoma|seminoma", "cancer|carcinoma|malign|gonadoblastoma|seminoma"],
                "Testicular Neoplasm": ["testicular|epididymal|sertoli cell", "cyst|tumor|adenoma|neoplasm"],
                "Thymoma": ["thymoma", "thymoma"],
                "Thyroid (Benign)": ["thyroid|goiter|hypothyroid", "benign|hashimoto|goiter|hypothyroid|hypoplasia"],
                "Thyroid Cancer": ["thyroid", "cancer|carcinoma|malign"],
                "Thyroid Neoplasm": ["thyroid", "cyst|tumor|adenoma|neoplasm"],
                "Trypsinogen Deficiency": ["trypsinogen", "deficiency"],
                "UGI Cancer": ["ugi|upper gastrointestinal|upper gi", "cancer|carcinoma|malign"],
                "Ureter Cancer": ["ureter", "cancer|carcinoma|malign"],
                "Urinary Tract Cancer": ["urinary|urothelial|uroepithelial|urologic", "cancer|carcinoma|malign"],
                "Urinary Tract Neoplasm": ["urinary|urothelial|uroepithelial|urologic", "neoplasm|adenoma"],
                "Uterus (Benign)": ["uterine|uterus", "fibroid|benign"],
                "Uterus Neoplasm": ["uterine|uterus", "neoplasm|adenoma|tumor|tumour"],
                "Yolk Sac Tumor": ["yolk sac", "yolk sac|tumor"],
                "Neuroendocrine": ["endocrine|neural", "ochromocytoma|gastrinoma|glucagonoma|insulinoma|tumor"], 
                "Sarcoma": ["sarcoma|adamantinoma|rhabdoid|connective tissue", "rhabdoid|adamantinoma|malig|cancer|carcinoma|sarcoma"],
               }

    skip_source_truth = {
                "pancreatic": "pseudotum|inflammatory|extrapancreatic tumor|extra - pancreatic tumor|solitary fibrous tumor|secondary pancreatic tumor|pancreatic subcutaneous tumor|nonpancreatic tumor|metastatic tumor|subcutaneous tumor|mia paca - 2"
    }

    best_match = "0"
    if type(word) == str:
        for s in source_truth.keys():
            organs = source_truth[s][0]
            organ_match = re.findall(organs, word)
            if len(organ_match) >= 1:
                cancer_match = re.findall(source_truth[s][1], word)
                if len(cancer_match) >= 1:
                    skip_match = len(re.findall(skip_source_truth[organ_match[0]], word)) == 0 if organ_match[0] in skip_source_truth else True
                    if skip_match:
                        best_match = s
                        break
                    else:
                        pass
                else:
                    pass
    return [best_match]

def parallel_nen(diseases):
    nprocs = mp.cpu_count() - 1
    pool = mp.Pool(processes=nprocs)
    new_diseases = pool.starmap(mapper, diseases.loc[:, ["word"]].values.tolist())
    new_diseases = list(itertools.chain(*new_diseases))
    return new_diseases

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
print(f"Named Entity Normalization")
disease_entity_df = pd.DataFrame(disease_entities)
new_diseases = parallel_nen(disease_entity_df)
disease_entity_df.loc[:, "nlp_cancers"] = new_diseases

print(f"Write disease entity table")

disease_entity_df.to_csv("diseases.csv", index=False)

end = time.time()
print("The time of execution of above program is :", (end-start), "seconds")
print("DONE!")
