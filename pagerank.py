import numpy as np
from numpy import linalg as LA


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


if __name__ == '__main__':
    alpha = 7/8
    G1 = np.array(
        [[0, 1, 1, 1, 0, 0, 0, 1], 
         [1, 0, 1, 0, 1, 0, 1, 0], 
         [0, 1, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 0, 1, 0, 0, 0]])  # all the links yet to be normalised, rows made stocastic
    G1 = G1.astype(float)  # convert to floats
    G1 = add_dangling(G1)
    G1 = normalise_matrix(G1)
    G2 = np.array([[1 / G1.shape[0]] * G1.shape[0]] * G1.shape[0])
    '''
    [[ 0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125]
    [ 0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125]
    [ 0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125]
    [ 0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125]
    [ 0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125]
    [ 0.125  0.125  0.125  0.125  0.125  0.125  0.125  0.125]]
    '''
    G = alpha * G1 + (1 - alpha) * G2
    w, v = LA.eig(G.T)
    # get the eigenvec corresp to the principle eig value
    rank_vec = v[:, np.argmax(w)]
    rank_vec = rank_vec / rank_vec.sum()
    scores = [(i + 1, v) for i, v in enumerate(rank_vec)]
    print(sorted(scores, key=lambda x: x[1], reverse=True))
