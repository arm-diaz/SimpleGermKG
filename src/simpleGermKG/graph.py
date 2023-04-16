import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


# Read abstracts
abstracts_df = pd.read_csv("../data/tblPubMedAbstractsGermline.csv")
print("pubmedIDs:", abstracts_df.shape)

# Read sentences
sentences_df = pd.read_csv("../results/pubmed_sentences2.csv")
print("sentences:", sentences_df.shape)


diseases_source_truth = {
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
print("diseases dict", len(diseases_source_truth.keys()))


genes_source_truth = {
                "BRCA2": ["brca", "2"],
                "BRCA1": ["brca"],
                "SMCRB1": ["smcrb1"],
                "TP53": ["p53"],
                "VHL": ["vhl"],
                "PALB2": ["palb2"],
                "WNT": ["wnt"],
                "KRAS": ["kras"],
                "MPG": ["mpg"],
                "CDH1": ["cadherin|cdh1"],
                "BAP1": ["bap1"],
                "CASR": ["casr"],
                "CCND1": ["ccnd1|cyclin d1"],
                "CD3": ["cd3"],
                "CDC20": ["cdc20"],
                "CDC73": ["cdc73"],
                "CDK12": ["cdk12"],
                "CDK4": ["cdk4"],
                "CDKN1B": ["cdkn1b"],
                "CDKN2A": ["cdkn2a"],
                "CFTR": ["cftr"],
                "CHEK1": ["chek1|chk1"],
                "CHEK2": ["chek2|chk2"],
                "CDKN1C": ["cdkn1c"],
                "CNF1": ["cnf1"],
                "CPA1": ["cpa1"],
                "CTNNA1": ["ctnna1"],
                "CTRC": ["ctrc"],
                "DICER1": ["dicer1|dicer 1"],
                "DMC1": ["dmc1"],
                "MUTYH": ["mutyh"],
                "DOT1L": ["dot1"],
                "DRB1": ["drb1"],
                "DSC2": ["dsc2"],
                "ERCC1": ["ercc1"],
                "ERCC2": ["ercc2"],
                "EZH2": ["ezh2"],
                "FANCA": ["fanca"],
                "FANCC": ["fancc"],
                "FANCD2": ["fancd2"],
                "FANCF": ["fancf"],
                "FANCG": ["fancg"],
                "FANCM": ["fancm"],
                "FAT4": ["fat4"],
                "FGFR3": ["fgfr3"],
                "FH": ["fh"],
                "FLCN": ["flcn"],
                "FLT3": ["flt3"],
                "FOXE1": ["foxe1"],
                "FOXP3": ["foxp3"],
                "GPC3": ["gpc3"],
                "GREM1": ["grem1"],
                "GSTP1": ["gstp1"],
                "HER2": ["her2"],
                "HER3": ["her3"],
                "MLH1": ["mlh1"],
                "MSH2": ["msh2"],
                "MSH6": ["msh6"],
                "NF1": ["nf1"],
                "OGG1": ["ogg1"],
                "HOXB13": ["hoxb13|hoxb 13"],
                "IGF2": ["igf2"],
                "KIR": ["kir"],
                "LAMP2": ["lamp2"],
                "LDLR": ["ldlr"],
                "LDLRAP1": ["ldlrap1"],
                "LKB1": ["lkb1"],
                "LMNA": ["lmna"],
                "MC1R": ["mc1r"],
                "MCPH1": ["mcph1"],
                "MEN1": ["men1"],
                "MITF": ["mitf"],
                "MRE11": ["mre11"],
                "MSH3": ["msh3"],
                "MSI": ["msi"],
                "MUC2": ["muc2"],
                "NBN": ["nbn"],
                "NDRG1": ["ndrg1"],
                "NF2": ["nf2"],
                "NTHL1": ["nthl1"],
                "NTRK": ["ntrk"],
                "PCSK9": ["pcsk9"],
                "PDGFRA": ["pdgfra"],
                "PIK3CA": ["pik3ca"],
                "PLN": ["pln"],
                "PMS1": ["pms1"],
                "PMS2": ["pms2"],
                "PNPLA3": ["pnpla3"],
                "POLD1": ["pold1"],
                "POLE": ["pole"],
                "POT1": ["pot1"],
                "PRKAR1A": ["prkar1a"],
                "PRSS1": ["prss1"],
                "PTCH1": ["ptch1"],
                "PTCH2": ["ptch2"],
                "PTEN": ["pten"],
                "RAD50": ["rad50"],
                "RAD51B": ["rad51b"],
                "RAD51C": ["rad51c"],
                "RAD51D": ["rad51d"],
                "RAD51": ["rad51"],
                "RAD54L": ["rad54l"],
                "RB1": ["rb1"],
                "RECQL": ["recql"],
                "RNASEL": ["rnasel"],
                "RNF43": ["rnf43"],
                "RPS20": ["rps20"],
                "RUNX1": ["runx1"],
                "SBDS": ["sbds"],
                "SDHA": ["sdha"],
                "SDHAF2": ["sdhaf2"],
                "SDHB": ["sdhb"],
                "SDHC": ["sdhc"],
                "SDHD": ["sdhd"],
                "SMAD4": ["smad4"],
                "SMARCA4": ["smarca4"],
                "SMARCB1": ["smarcb1"],
                "SMARCE1": ["smarce1"],
                "SPINK1": ["spink1|spink 1|spink"],
                "STK11": ["stk11"],
                "SUFU": ["sufu"],
                "TAZ": ["taz"],
                "TERC": ["terc"],
                "TERT": ["tert"],
                "TMEM127": ["tmem127"],
                "TRRAP": ["trrap"],
                "TSC1": ["tsc1"],
                "TSC2": ["tsc2"],
                "TYR": ["tyr"],
                "XRCC2": ["xrcc2"],
                "XRCC3": ["xrcc3"],
                "SURF1": ["smurf1"],
                "MLH3": ["mlh3"],
                "HRAS": ["hras"],
                "BRIP1": ["brip2"],
                "APC": ["apc|adenomatous polyposis coli"],
                "EGFR": ["egfr"],
                "ATR": [r"(atr+)(?=\d)|\batr\b"],
                "ATM": [r"(atm+)(?=\d)|\batm\b"],
                "AKT": [r"(akt+)(?=\d)|\bakt\b"],
                "ALK": [r"(alk+)(?=\d)|\balk\b"],

    }

print("genes dict", len(genes_source_truth.keys()))

gene_df = pd.read_csv("new_genes.csv")
diseases_df = pd.read_csv("new_diseases.csv")

unique_ner_genes = list(gene_df[gene_df["word"].notnull()]["word"].unique())
unique_ner_diseases = list(diseases_df[diseases_df["word"].notnull()]["word"].unique())
gene_ner_words = Counter(gene_df[gene_df["word"].notnull()]["word"])
diseases_ner_words = Counter(diseases_df[diseases_df["word"].notnull()]["word"])

gene_df = gene_df[~gene_df["nlp_genes"].isin(["0", 0])]
gene_df.loc[:, "nlp_genes"] = gene_df["nlp_genes"].str.replace("'", "")
diseases_df = diseases_df[~diseases_df["nlp_cancers"].isin(["0", 0])]
diseases_df.loc[:, "nlp_cancers"] = diseases_df["nlp_cancers"].str.replace("'", "")

gene_ner_dis_words = Counter(gene_df["nlp_genes"])
diseases_ner_dis_words = Counter(diseases_df["nlp_cancers"])

print("genes match", len(unique_ner_genes))
print("diseases match", len(unique_ner_diseases))

gene_pubmed = gene_df.groupby(["nlp_genes"])["pubmed_id"].unique().reset_index()
disease_pubmed = diseases_df.groupby(["nlp_cancers"])["pubmed_id"].unique().reset_index()
pubmed_genes = gene_df.groupby(["pubmed_id"])["nlp_genes"].unique().reset_index()
pubmed_diseases= diseases_df.groupby(["pubmed_id"])["nlp_cancers"].unique().reset_index()

pubmed_data = pubmed_genes.merge(pubmed_diseases, on=["pubmed_id"], how="left")
pubmed_data.head()

genes = list(gene_pubmed[gene_pubmed["nlp_genes"].notnull()]["nlp_genes"].unique())
diseases = list(disease_pubmed[disease_pubmed["nlp_cancers"].notnull()]["nlp_cancers"].unique())
pubmed_ids = list(pubmed_data.pubmed_id.values)

nl = "\r\n"
sl = "\'"
disease_relationship = "diseases_in"
gene_relationship = "genes_in"

pubmed_gene_triples_ls = []
pubmed_disease_triples_ls = []
pubmed_triples_ls = []
for _, rows in pubmed_data.iterrows():
    tmp_gene = rows.nlp_genes
    tmp_pubmed_id = rows.pubmed_id
    tmp_disease = rows.nlp_cancers
    if (tmp_disease is not None) and (type(tmp_disease) is not float):
        pubmed_disease_triples_ls.append(", ".join([f'(pubmed_{tmp_pubmed_id})-[:{disease_relationship}]->({d.replace("-", "_").replace(" ", "_").replace("/", "_").replace(nl, "_").replace(")", "").replace("(", "_").replace(sl, "")})' for d in tmp_disease]))
        pubmed_gene_triples_ls.append(", ".join([f'(pubmed_{tmp_pubmed_id})-[:{gene_relationship}]->({g.replace("-", "_").replace(" ", "_").replace("/", "_").replace(nl, "_").replace(")", "").replace("(", "_").replace(sl, "")})' for g in tmp_gene]))

pubmed_triples_ls.extend(pubmed_gene_triples_ls)
pubmed_triples_ls.extend(pubmed_disease_triples_ls)
pubmed_gene_triples = ", ".join(pubmed_gene_triples_ls)
pubmed_disease_triples = ", ".join(pubmed_disease_triples_ls)
pubmed_triples = ", ".join(pubmed_triples_ls)

print(f"Total number of pubmed nodes: {len(pubmed_ids)}")
print(f"Total number of genes nodes: {len(genes)}")
print(f"Total number of diseases nodes: {len(diseases)}")
print(f"Total number of gene triples: {len(pubmed_gene_triples.split(','))}")
print(f"Total number of disease triples: {len(pubmed_disease_triples.split(','))}")
print(f"Total number of triples: {len(pubmed_triples.split(','))}")
print(f"Total number of entities: {len(genes) + len(diseases) + len(pubmed_ids)}")

cqlCreate = f""" 
CREATE
{", ".join([f'(pubmed_{p}:pubmed_id {{name: "{p}"}})' for p in pubmed_ids])},
{", ".join([f'({g.replace("-", "_").replace(" ", "_").replace("/", "_").replace(nl, "_").replace(")", "").replace("(", "_").replace(sl, "") }:gene {{name: "{g.replace(nl, " ").replace(sl, "")}"}})' for g in genes])},
{", ".join([f'({d.replace("-", "_").replace(" ", "_").replace("/", "_").replace(nl, "_").replace(")", "").replace("(", "_").replace(sl, "") }:disease {{name: "{d.replace(nl, " ").replace(sl, "")}"}})' for d in diseases])}
"""


with open("Neo4J.txt", "w") as text_file:
    print(cqlCreate, file=text_file)

pubmed_gene_triples_ls = []
pubmed_disease_triples_ls = []
pubmed_triples_ls = []
for _, rows in pubmed_data[pubmed_data["nlp_cancers"].isin(["Pancreatitis", "Pancreatic Schwannoma", "Pancreatic Neoplasm", "Pancreas ('Neuroendocrine')", "Pancreatic Cancer", "Breast (Benign)", "Breast Cancer (Contralateral)", "Breast Cancer (Male)", "Breast Neoplasm", "Breast Cancer"])].iterrows():
    tmp_gene = rows.nlp_genes
    tmp_pubmed_id = rows.pubmed_id
    tmp_disease = rows.nlp_cancers
    if (tmp_disease is not None) and (type(tmp_disease) is not float):
        pubmed_disease_triples_ls.append(", ".join([f'(pubmed_{tmp_pubmed_id})-[:{disease_relationship}]->({d.replace("-", "_").replace(" ", "_").replace("/", "_").replace(nl, "_").replace(")", "").replace("(", "_").replace(sl, "")})' for d in tmp_disease]))
        pubmed_gene_triples_ls.append(", ".join([f'(pubmed_{tmp_pubmed_id})-[:{gene_relationship}]->({g.replace("-", "_").replace(" ", "_").replace("/", "_").replace(nl, "_").replace(")", "").replace("(", "_").replace(sl, "")})' for g in tmp_gene]))

pubmed_triples_ls.extend(pubmed_gene_triples_ls)
pubmed_triples_ls.extend(pubmed_disease_triples_ls)
pubmed_gene_triples = ", ".join(pubmed_gene_triples_ls)
pubmed_disease_triples = ", ".join(pubmed_disease_triples_ls)
pubmed_triples = ", ".join(pubmed_triples_ls)


cqlCreate = f""" 
CREATE
{", ".join([f'(pubmed_{p}:pubmed_id {{name: "{p}"}})' for p in pubmed_ids])},
{", ".join([f'({g.replace("-", "_").replace(" ", "_").replace("/", "_").replace(nl, "_").replace(")", "").replace("(", "_").replace(sl, "") }:gene {{name: "{g.replace(nl, " ").replace(sl, "")}"}})' for g in genes])},
{", ".join([f'({d.replace("-", "_").replace(" ", "_").replace("/", "_").replace(nl, "_").replace(")", "").replace("(", "_").replace(sl, "") }:disease {{name: "{d.replace(nl, " ").replace(sl, "")}"}})' for d in diseases])}
"""


with open("PancreasBreastNeo4J.txt", "w") as text_file:
    print(cqlCreate, file=text_file)

