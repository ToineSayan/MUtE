from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import spearmanr, pearsonr

def utility_MSE(x, fx, max_iter=500):
    mlpreg = MLPRegressor(max_iter=max_iter, random_state=0)
    x_pred = mlpreg.fit(fx, x).predict(fx)
    return ((x-x_pred)**2).mean() 


def neighborhood_overlap(x, fx, k=0.5):
    n_neighbors = int(k * len(x))
    nbrs_x = NearestNeighbors(n_neighbors=n_neighbors).fit(x)
    nbrs_fx = NearestNeighbors(n_neighbors=n_neighbors).fit(fx)
    _, indices_x = nbrs_x.kneighbors(x)
    _, indices_fx = nbrs_fx.kneighbors(fx)
    overlap = []
    for i in range(len(x)):
        overlap.append(len(set(indices_x[i]) & set(indices_fx[i])) / n_neighbors)
    return np.clip(sum(overlap) / len(overlap), 0.0, 1.0)



def similarity_correlation(x1, x2, y_sim):
    cosine = np.sum(x1 * x2, axis=1)/(np.linalg.norm(x1, axis=1)*np.linalg.norm(x2, axis=1))
    return spearmanr(cosine, y_sim).statistic, pearsonr(cosine, y_sim).statistic