from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

df = pd.read_csv('articles.csv')
df = df[df['title'].notna()]
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['title'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
df = df.reset_index()
indices = pd.Series(df.index, index=df['contentId'])

def get_recommendations(contentId):
    idx = indices[int(contentId)]
    simi_scr = list(enumerate(cosine_sim2[idx]))
    simi_scr = sorted(simi_scr, key=lambda x: x[1], reverse=True)
    simi_scr = simi_scr[1:11]
    article_indices = [i[0] for i in simi_scr]
    return df[["url", "title", "text", "lang", "total_events"]].iloc[article_indices].values.tolist()