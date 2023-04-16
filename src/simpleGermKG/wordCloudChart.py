import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt

# READ FILE FILTERED WITH PANCREAS
diseases_df = pd.read_csv("new_pancreas.csv")

diseases_df = diseases_df[~diseases_df["nlp_cancers"].isin(["0", 0])]

diseases_ner_words = Counter(diseases_df[diseases_df["word"].notnull()]["word"])
diseases_ner_dis_words = Counter(diseases_df[diseases_df["nlp_cancers"].notnull()]["nlp_cancers"])

wordcloud = WordCloud(background_color="#e2e1eb", max_words=100).generate_from_frequencies(diseases_ner_words)

#plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")

wordcloud.to_file("wordCloudDisease.png")

wordcloud = WordCloud(background_color="#e2e1eb", max_words=100).generate_from_frequencies(diseases_ner_dis_words)

#plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")

wordcloud.to_file("wordCloudDisambiguatedDisease.png")
