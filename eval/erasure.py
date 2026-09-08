from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from statistics import mean, stdev
import warnings
from random import shuffle
import numpy as np
import math
from collections import Counter
from scipy.stats import entropy


def linear_probing(x, y, x_test, y_test, nprobing=1, max_iter=20, ignore_warnings=True, random_state=None):
    scores_train, scores_test = [], []
    for i in range(nprobing):
        seed = random_state + i if random_state is not None else None
        with warnings.catch_warnings(action="ignore" if ignore_warnings else "all"):
            clf = SGDClassifier(max_iter=max_iter, early_stopping=True, random_state=seed).fit(x, y)
        scores_train.append(clf.score(x, y))
        scores_test.append(clf.score(x_test, y_test))
    return (mean(scores_train), mean(scores_test), stdev(scores_train), stdev(scores_test)) if nprobing > 1 else (scores_train[0], scores_test[0], 0.0, 0.0)

def nonlinear_probing(x, y, x_test, y_test, nprobing=1, max_iter=20, ignore_warnings=True, random_state=None):
    scores_train, scores_test = [], []
    for i in range(nprobing):
        seed = random_state + i if random_state is not None else None
        with warnings.catch_warnings(action="ignore" if ignore_warnings else "all"):
            clf = MLPClassifier(max_iter=max_iter, early_stopping=True, random_state=seed).fit(x, y)
        scores_train.append(clf.score(x, y))
        scores_test.append(clf.score(x_test, y_test))
    return (mean(scores_train), mean(scores_test), stdev(scores_train), stdev(scores_test)) if nprobing > 1 else (scores_train[0], scores_test[0], 0.0, 0.0)


def mdl(x, y, max_iter=1000, ignore_warnings=True):
    if len(x) < 1001:
        raise ValueError("MDL evaluation requires at least 1000 samples.")
    nsamples = len(x)
    indices = np.random.permutation(nsamples)
    ratios = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5, 1]
    block_indices = [indices[:int(r * nsamples)] for r in ratios]
    nlabels = len(np.unique(y))
    # compute the number of bits required for the first transmission
    score = len(block_indices[0]) * math.log(nlabels, 2)

    for i, indices in enumerate(block_indices[:-1]):
        x_train, y_train = x[indices], y[indices]
        
        with warnings.catch_warnings(action="ignore" if ignore_warnings else "always"):
            clf = MLPClassifier(max_iter=max_iter)
            clf.fit(x_train, y_train)

        next_indices = block_indices[i + 1]
        x_test, y_test = x[next_indices], y[next_indices]

        y_pred = clf.predict_proba(x_test)

        for y_gold, y_pred in zip(y_test, y_pred):
            try:
                score -= math.log(y_pred[y_gold], 2)
            except:
                pass
    return (score / 1024)  # output the MDL in Kbits
        


def privacy(x, y, max_iter=500, random_state=0):
    vals = np.array(list(Counter(y).values()))
    p_y = vals/sum(vals)
    hy = entropy(p_y, base=2) # entropy of Z

    # def get_mi(samples, A, HA):
    # clf = MLPClassifier(random_state=random_state, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes)
    clf = MLPClassifier(random_state=random_state, max_iter=max_iter)
    # clf = MLPClassifier(random_state=0, max_iter=max_iter, hidden_layer_sizes=(5,))
    clf.fit(x, y)
    print("Privacy score:", clf.score(x, y))
    
    hyx = np.mean([entropy(y_pred, base=2) for y_pred in clf.predict_proba(x)])
    Izx = hy - hyx
    return max(Izx, 0)


