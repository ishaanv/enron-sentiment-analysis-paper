import numpy as np
import pandas as pd
import pagerank as pr
from tqdm import tqdm


def sending_sentiment_adj_mat(mat, df, source_field='from', target_field='to'):
    #     set_trace()
    for index, row in tqdm(df.iterrows()):
        mat[nodes.index(getattr(row, source_field))][nodes.index(
            getattr(row, target_field))] += row.sentiment


enron_links = pd.read_csv('data/enron_links_with_sentiment_roles.csv')
enron_links = enron_links[enron_links.sentiment < 0] # get positive sentiment
nodes = pd.unique(enron_links[['from', 'to']].values.ravel('K')).tolist()
n = len(nodes)
mat = np.zeros([n,n])
sending_sentiment_adj_mat(mat, enron_links)
# filter out the negatiave links
#
mat = pr.add_dangling(mat)
mat = pr.normalise_matrix(mat)
mat2 = np.array([[1 / mat.shape[0]] * mat.shape[0]] * mat.shape[0])
alpha = 7 / 8
G = alpha * mat + (1 - alpha) * mat2
w, v = pr.LA.eig(G.T)
rank_vec = v[:, np.argmax(w)]
rank_vec = rank_vec / rank_vec.sum()
scores = [(nodes[i], v) for i, v in enumerate(rank_vec)]
print(sorted(scores, key=lambda x: x[1], reverse=True))
print(w[0])

# import pdb; pdb.set_trace()
