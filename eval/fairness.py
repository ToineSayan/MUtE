import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
from statistics import mean, stdev



def train_clfs(x, y, x_test, y_test, ntrain=1, max_iter=20, ignore_warnings=True, return_clfs=False, classification = True, random_state=None):
    scores_train, scores_test = [], []
    clfs = []
    for i in range(ntrain):
        seed = random_state + i if random_state is not None else None
        if classification:
            with warnings.catch_warnings(action="ignore" if ignore_warnings else "all"):
                clf = MLPClassifier(max_iter=max_iter, early_stopping=True, random_state=seed).fit(x, y)
            scores_train.append(clf.score(x, y))
            scores_test.append(clf.score(x_test, y_test))
            clfs.append(clf) if return_clfs else None
        else:
            with warnings.catch_warnings(action="ignore" if ignore_warnings else "all"):
                clf = MLPRegressor(max_iter=max_iter, early_stopping=True, random_state=seed).fit(x, y)
            scores_train.append(clf.score(x, y))
            scores_test.append(clf.score(x_test, y_test))
            clfs.append(clf) if return_clfs else None
    if return_clfs:
        if ntrain > 1:
            return (mean(scores_train), mean(scores_test), stdev(scores_train), stdev(scores_test)), clfs
        else:
            return (scores_train[0], scores_test[0], 0.0, 0.0), clfs
    return (mean(scores_train), mean(scores_test), stdev(scores_train), stdev(scores_test)) if ntrain > 1 else (scores_train[0], scores_test[0], 0.0, 0.0)


def demographic_parity(X, Y, Z, clfs, z1=0, z2=1):
        
    def dp_single(y_hat, y, z):
        return  sum([abs(np.mean(y_hat[z == z1] == i) - np.mean(y_hat[z == z2] == i)) for i in np.unique(y)])

    dps = [dp_single(clf.predict(X), Y, Z) for clf in clfs]
    return np.mean(dps), np.std(dps)





def TPR_Gaps(y, Z, Y, ref_z_val, compared_z_val):
    """ 
    Calculate the TPR-Gaps for a z, compared to the counterfactual value z'
    for each Y-value   
    """
    filter_z_val = lambda z_val: Z == z_val # filter on Z values
    filter_y_val = lambda y_val: Y == y_val # filter on True Y predictions

    TPR_Gaps = dict()
    for y_val in np.unique(Y):
        TPR_ref = np.mean(
                    y[filter_z_val(ref_z_val)][filter_y_val(y_val)[filter_z_val(ref_z_val)]] == y_val
                )
        TPR_compared = np.mean(
                    y[filter_z_val(compared_z_val)][filter_y_val(y_val)[filter_z_val(compared_z_val)]] == y_val
                )
        TPR_Gaps[y_val] = TPR_ref - TPR_compared
    return TPR_Gaps

def Z_proportions(z_val, Z, Y): # gender imbalance
    filter_y_val = lambda y_val: Y == y_val # filter on True Y predictions
    Z_prop = dict()
    for y_val in np.unique(Y):
        Z_prop[y_val] = np.mean(Z[filter_y_val(y_val)] == z_val)
    return Z_prop


def TPR_RMS(X, Y, Z, z1_id, z2_id, clfs):
    list_TPR_gaps = [TPR_Gaps(clf.predict(X), Z, Y, z1_id, z2_id) for clf in clfs]
    RMS_dict = lambda D : np.sqrt(np.mean([D[y_val]**2 for y_val in np.unique(Y)]))
    list_RMS = [RMS_dict(D) for D in list_TPR_gaps]

    return np.mean(list_RMS), np.std(list_RMS)

def TPR_corr_coef(X, Y, Z, z1_id, z2_id, clfs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning) # ignore warnings for correlation coefficient when std is 0
        list_TPR_gaps = [TPR_Gaps(clf.predict(X), Z, Y, z1_id, z2_id) for clf in clfs]
        Z_props = Z_proportions(z1_id, Z, Y)
        Z_props_list = [Z_props[y_val] for y_val in np.unique(Y)]
        Corr_coef = lambda D : np.corrcoef(Z_props_list, [D[y_val] for y_val in np.unique(Y)])[0,1]
        list_corr_coefs = [Corr_coef(D) for D in list_TPR_gaps] 
        return np.mean(list_corr_coefs), np.std(list_corr_coefs)
