import numpy as np
import pandas as pd
from numpy import linalg as LA
from tqdm import tqdm


def sending_sentiment_adj_mat(mat, df, source_field='from', target_field='to'):
    for index, row in tqdm(df.iterrows()):
        source_index = nodes.index(getattr(row, source_field))
        target_index = nodes.index(getattr(row, target_field))
        roles[source_index] = getattr(row, 'from_title')
        roles[target_index] = getattr(row, 'to_title')
        mat[source_index][target_index] += row.sentiment


def normalise_matrix(G):
    sum_rows = G.sum(axis=1)
    for index, row_sum in enumerate(sum_rows):
        for col in range(G.shape[0]):
            G[index][col] = G[index][col] / row_sum
    return G


def add_dangling(G):
    sum_rows = G.sum(axis=1)
    for index, row_sum in enumerate(sum_rows):
        if row_sum == 0:
            for col in range(G.shape[0]):
                G[index][col] = 1 / G.shape[0]
    return G


enron_links = pd.read_csv('data/enron_links_with_sentiment_roles.csv')
enron_links = enron_links[enron_links.sentiment > 0]  # get positive sentiment
nodes = pd.unique(enron_links[['from', 'to']].values.ravel('K')).tolist()
n = len(nodes)
roles = [None] * n
mat = np.zeros([n,n])
sending_sentiment_adj_mat(mat, enron_links)
mat = add_dangling(mat)
mat = normalise_matrix(mat)
mat2 = np.array([[1 / mat.shape[0]] * mat.shape[0]] * mat.shape[0])
alpha = 7 / 8
G = alpha * mat + (1 - alpha) * mat2
w, v = LA.eig(G.T)
rank_vec = v[:, np.argmax(w)]
rank_vec = rank_vec / rank_vec.sum()
scores = [(nodes[i],roles[i], v) for i, v in enumerate(rank_vec)]
for i in sorted(scores, key=lambda x: x[2], reverse=True):
    print(i)