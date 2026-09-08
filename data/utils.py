import numpy as np
import pickle
from sklearn.model_selection import train_test_split


def load_data(dataset_name, base_path=None):
    if dataset_name == 'GloVe':
        load_func=load_GloVe
    if dataset_name == 'biasbios':
        load_func=load_biasbios 
    if dataset_name == 'biasbios-v2t':
        load_func=load_biasbios_v2t
    if dataset_name == 'biasbios-v2t-cf': # counterfactual samples of biasbios-v2t
        load_func=load_biasbios_v2t_cf
    if dataset_name == 'deepmoji':
        load_func=load_deepmoji
    if dataset_name == 'jigsaw':
        load_func=load_jigsaw
    x, z_raw, y_raw, x_val, z_val_raw, y_val_raw, x_test, z_test_raw, y_test_raw = load_func(base_path=base_path)
    # (label encoding) Mapping unique categorical values to incremental integers
    # z_values: sorted list of unique concept labels
    # y_values: sorted list of unique downstream task labels
    z_values, z = np.unique(z_raw, return_inverse=True) if z_raw is not None else (None, None)
    y_values, y = np.unique(y_raw, return_inverse=True) if y_raw is not None else (None, None)
    z_val = np.searchsorted(z_values, z_val_raw) if z_val_raw is not None else None
    y_val = np.searchsorted(y_values, y_val_raw) if y_val_raw is not None else None
    z_test = np.searchsorted(z_values, z_test_raw) if z_test_raw is not None else None
    y_test = np.searchsorted(y_values, y_test_raw) if y_test_raw is not None else None

    return x, z, y, x_val, z_val, y_val, x_test, z_test, y_test, z_values, y_values



# GLOVE DATASET
#########################################################################


def load_dump(file_path):
    """
    Load data from a .pkl file.

    Args:
        file_path (str): The path to the .pkl file.

    Returns:
        The data from the .pkl file.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def form_dataset(male_words, fem_words, neut_words):
    x, y = [], []

    for w, v in male_words.items():
        x.append(v)
        y.append("m")
    for w, v in fem_words.items():
        x.append(v)
        y.append("f")
    for w, v in neut_words.items():
        x.append(v)
        y.append("n")
    return np.array(x), np.array(y)



def load_GloVe(base_path=None):
    if base_path is None:
        data_path="data/GloVe"
    else:
        data_path=str(base_path) + "/data/GloVe"

    male_words = load_dump(data_path + '/male_words.pkl')
    fem_words = load_dump(data_path + '/fem_words.pkl')
    neut_words = load_dump(data_path + '/neut_words.pkl')

    x, z = form_dataset(male_words, fem_words, neut_words)
    
    x_train_dev, x_test, z_train_dev, z_test = train_test_split(
        x, z, test_size=0.3, random_state=0)
    x_train, x_dev, z_train, z_dev = train_test_split(
        x_train_dev, z_train_dev, test_size=0.3, random_state=0)
    # no downstream task labels for this dataset, so we return None for y
    return x_train, z_train, np.empty(0), x_dev, z_dev, np.empty(0), x_test, z_test, np.empty(0)


def load_ws3353(base_path=None):
    if base_path is None:
        data_path="data/GloVe"
    else:
        data_path=str(base_path) + "/data/GloVe"
    with open(data_path + '/glove-top-150k.pickle', "rb") as f:
        data = pickle.load(f)
    glove_150k = dict(zip(data['words'], data['vecs']))

    x1, x2, y_sim = [], [], []
    with open(data_path + '/evaluation/ws353simrel/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt', "r") as f:
        for line in f:
            if line[0] != '#':
                word_1, word_2, similarity_score = line.strip().split('\t')
                try:
                    v1, v2 = glove_150k[word_1], glove_150k[word_2]
                    x1.append(v1)
                    x2.append(v2)
                    y_sim.append(float(similarity_score))
                except:
                    None
    return np.array(x1), np.array(x2), np.array(y_sim)
    




# BIAS IN BIOS  DATASET
#########################################################################


def load_biasbios(base_path=None):
    '''
    Load the Bias in Bios dataset (Bert embeddings).

    Returns:
        x, z_raw, y_raw: Training data and labels.
        x_val, z_val_raw, y_val_raw: Validation data and labels.
        x_test, z_test_raw, y_test_raw: Test data and labels.
    '''
    if base_path is None:
        data_path="data/biasbios/bert"
    else:
        data_path=str(base_path) + "/data/biasbios/bert"
    # Load the original train, validation, test split
    x, z_raw, y_raw = np.load(data_path + '/train_cls.npy', allow_pickle=True), np.load(data_path + '/train_z.npy', allow_pickle=True), np.load(data_path + '/train_y.npy', allow_pickle=True)
    x_val, z_val_raw, y_val_raw =  np.load(data_path + '/dev_cls.npy', allow_pickle=True), np.load(data_path + '/dev_z.npy', allow_pickle=True), np.load(data_path + '/dev_y.npy', allow_pickle=True)
    x_test, z_test_raw, y_test_raw = np.load(data_path + '/test_cls.npy', allow_pickle=True), np.load(data_path + '/test_z.npy', allow_pickle=True), np.load(data_path + '/test_y.npy', allow_pickle=True)
    return x, z_raw, y_raw, x_val, z_val_raw, y_val_raw, x_test, z_test_raw, y_test_raw

def load_biasbios_texts(base_path=None):
    '''
    Load the original texts for the Bias in Bios dataset.

    Returns:
        train_data: Training data containing original texts.
        dev_data: Validation data containing original texts.
        test_data: Test data containing original texts.
    '''
    if base_path is None:
        data_path="data/biasbios"
    else:
        data_path=str(base_path) + "/data/biasbios"

    # load the train.pickle, dev.pickle, test.pickle files containing the original texts
    with open(data_path + '/train.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open(data_path + '/dev.pickle', 'rb') as f:
        dev_data = pickle.load(f)
    with open(data_path + '/test.pickle', 'rb') as f:
        test_data = pickle.load(f) 
    return train_data, dev_data, test_data

def load_biasbios_texts_cf(base_path=None):
    '''
    Load the counterfactually augmented Bias in Bios dataset.

    Returns:
        train_data: Training data containing original and counterfactual texts.
        dev_data: Validation data containing original and counterfactual texts.
        test_data: Test data containing original and counterfactual texts.
    '''
    if base_path is None:
        data_path="data/biasbios"
    else:
        data_path=str(base_path) + "/data/biasbios"

    # load the train_cf_augmented.pickle, dev_cf_augmented.pickle, test_cf_augmented.pickle files containing the counterfactual texts
    with open(data_path + '/train_cf_augmented.pickle', 'rb') as f:
        train_data = pickle.load(f)
    with open(data_path + '/dev_cf_augmented.pickle', 'rb') as f:
        dev_data = pickle.load(f)
    with open(data_path + '/test_cf_augmented.pickle', 'rb') as f:
        test_data = pickle.load(f) 
    return train_data, dev_data, test_data


def load_biasbios_v2t(base_path=None):
    '''
    Load the biasbios dataset (T5 generated embeddings).

    Returns:
            x, z_raw, y_raw: Training data and labels.
            x_val, z_val_raw, y_val_raw: Validation data and labels.
            x_test, z_test_raw, y_test_raw: Test data and labels.
    '''
    if base_path is None:
        data_path="data/biasbios/T5"
    else:
        data_path=str(base_path) + "/data/biasbios/T5"

    x, z_raw, y_raw = np.load(data_path + '/train_meanpool.npy', allow_pickle=True), np.load(data_path + '/train_z.npy', allow_pickle=True), np.load(data_path + '/train_y.npy', allow_pickle=True)
    x_val, z_val_raw, y_val_raw =  np.load(data_path + '/dev_meanpool.npy', allow_pickle=True), np.load(data_path + '/dev_z.npy', allow_pickle=True), np.load(data_path + '/dev_y.npy', allow_pickle=True)
    x_test, z_test_raw, y_test_raw = np.load(data_path + '/test_meanpool.npy', allow_pickle=True), np.load(data_path + '/test_z.npy', allow_pickle=True), np.load(data_path + '/test_y.npy', allow_pickle=True)
    return x, z_raw, y_raw, x_val, z_val_raw, y_val_raw, x_test, z_test_raw, y_test_raw
    

def load_biasbios_v2t_cf(base_path=None):
    if base_path is None:
        data_path="data/biasbios/T5"
    else:
        data_path=str(base_path) + "/data/biasbios/T5"

    x, z_raw, y_raw = np.load(data_path + '/train_cf_meanpool.npy', allow_pickle=True), np.load(data_path + '/train_cf_z.npy', allow_pickle=True), np.load(data_path + '/train_y.npy', allow_pickle=True)
    x_val, z_val_raw, y_val_raw =  np.load(data_path + '/dev_cf_meanpool.npy', allow_pickle=True), np.load(data_path + '/dev_cf_z.npy', allow_pickle=True), np.load(data_path + '/dev_y.npy', allow_pickle=True)
    x_test, z_test_raw, y_test_raw = np.load(data_path + '/test_cf_meanpool.npy', allow_pickle=True), np.load(data_path + '/test_cf_z.npy', allow_pickle=True), np.load(data_path + '/test_y.npy', allow_pickle=True)

    return x, z_raw, y_raw, x_val, z_val_raw, y_val_raw, x_test, z_test_raw, y_test_raw



# DIAL  DATASET
#########################################################################


def load_deepmoji(base_path=None):
    '''
    Load the DIAL dataset (deepmoji embeddings).

    Returns:
            x, z_raw, y_raw: Training data and labels.
            x_val, z_val_raw, y_val_raw: Validation data and labels.
            x_test, z_test_raw, y_test_raw: Test data and labels.
    '''
    if base_path is None:
        data_path="data/deepmoji"
    else:
        data_path=str(base_path) + "/data/deepmoji"


    nspc_train = 40000
    split = "train"
    x_neg_aa = np.load(data_path + "/" + split + '/neg_pos.npy') # sentiment : negative, language: AAE
    x_neg_wh = np.load(data_path + "/" + split + '/neg_neg.npy') # sentiment : negative, language: SAE
    x_pos_aa = np.load(data_path + "/" + split + '/pos_pos.npy') # sentiment : positive, language: AAE
    x_pos_wh = np.load(data_path + "/" + split + '/pos_neg.npy') # sentiment : positive, language: SAE

    x = np.concatenate((x_neg_aa[:nspc_train], x_neg_wh[:nspc_train], x_pos_aa[:nspc_train], x_pos_wh[:nspc_train]), axis = 0)
    z = np.array(nspc_train*[1] + nspc_train*[0] + nspc_train*[1] + nspc_train*[0]) # 0: SAE (Standard American English) - 1: AAE (African-American English)
    y = np.array(nspc_train*[0] + nspc_train*[0] + nspc_train*[1] + nspc_train*[1]) # 0: "sad", "1": "happy"

    nspc_val = 2000
    split = "dev"
    x_neg_aa = np.load(data_path + "/" + split + '/neg_pos.npy') # sentiment : negative, language: AAE
    x_neg_wh = np.load(data_path + "/" + split + '/neg_neg.npy') # sentiment : negative, language: SAE
    x_pos_aa = np.load(data_path + "/" + split + '/pos_pos.npy') # sentiment : positive, language: AAE
    x_pos_wh = np.load(data_path + "/" + split + '/pos_neg.npy') # sentiment : positive, language: SAE

    x_val = np.concatenate((x_neg_aa[:nspc_val], x_neg_wh[:nspc_val], x_pos_aa[:nspc_val], x_pos_wh[:nspc_val]), axis = 0)
    z_val = np.array(nspc_val*[1] + nspc_val*[0] + nspc_val*[1] + nspc_val*[0]) # 0: SAE (Standard American English) - 1: AAE (African-American English)
    y_val = np.array(nspc_val*[0] + nspc_val*[0] + nspc_val*[1] + nspc_val*[1]) # 0: "sad", "1": "happy"

    nspc_test = 1999 # not as in KRaM
    split = "test"
    x_neg_aa = np.load(data_path + "/" + split + '/neg_pos.npy') # sentiment : negative, language: AAE
    x_neg_wh = np.load(data_path + "/" + split + '/neg_neg.npy') # sentiment : negative, language: SAE
    x_pos_aa = np.load(data_path + "/" + split + '/pos_pos.npy') # sentiment : positive, language: AAE
    x_pos_wh = np.load(data_path + "/" + split + '/pos_neg.npy') # sentiment : positive, language: SAE

    x_test = np.concatenate((x_neg_aa[:nspc_test], x_neg_wh[:nspc_test], x_pos_aa[:nspc_test], x_pos_wh[:nspc_test]), axis = 0)
    z_test = np.array(nspc_test*[1] + nspc_test*[0] + nspc_test*[1] + nspc_test*[0]) # 0: SAE (Standard American English) - 1: AAE (African-American English)
    y_test = np.array(nspc_test*[0] + nspc_test*[0] + nspc_test*[1] + nspc_test*[1]) # 0: "sad", "1": "happy"

    # shuffle the observations in the sets
    np.random.seed(0)
    indices = np.arange(4*nspc_train)
    np.random.shuffle(indices)
    x, z, y = x[indices], z[indices], y[indices]

    indices = np.arange(4*nspc_val)
    np.random.shuffle(indices)
    x_val, z_val, y_val = x_val[indices], z_val[indices], y_val[indices]

    indices = np.arange(4*nspc_test)
    np.random.shuffle(indices)
    x_test, z_test, y_test = x_test[indices], z_test[indices], y_test[indices]

    return x, z, y, x_val, z_val, y_val, x_test, z_test, y_test 



# JIGSAW  DATASET
#########################################################################

def load_jigsaw(base_path=None):
    '''
    Load the Jigsaw dataset (GPT-4 generated embeddings).

    Returns:
            x, z_raw, y_raw: Training data and labels.
            x_val, z_val_raw, y_val_raw: Validation data and labels.
            x_test, z_test_raw, y_test_raw: Test data and labels.
    '''
    if base_path is None:
        data_path="data/jigsaw"
    else:
        data_path=str(base_path) + "/data/jigsaw"

    with open(data_path + "/jigsaw_train.pkl", "rb") as file:
        df_train = pickle.load(file)
    with open(data_path + "/jigsaw_test.pkl", "rb") as file:
        df_test = pickle.load(file)
    label_set = [
        'buddhist', 
        'christian', 
        'hindu', 
        'jewish', 
        'muslim', 
        'other_religion'
    ]
    religions = ["christian", "muslim", "jewish", "hindu", "buddhist"]
    df_train = df_train[df_train[label_set].idxmax(axis=1).isin(religions)]
    df_test = df_test[df_test[label_set].idxmax(axis=1).isin(religions)]
    z_raw = df_train[religions].idxmax(axis=1).tolist()
    z_test_raw = df_test[religions].idxmax(axis=1).tolist()
    # get the embeddings from the "embedded" column and transform them into numpy arrays
    x = np.array(df_train["embedded"].tolist()).astype(np.float32)
    x_test = np.array(df_test["embedded"].tolist()).astype(np.float32)
    # get the toxicity labels
    y_raw = (df_train["toxicity"] >= 0.5).astype(int).values
    y_test_raw = (df_test["toxicity"] >= 0.5).astype(int).values
    y_raw = ["toxic" if label == 1 else "non-toxic" for label in y_raw]
    y_test_raw = ["toxic" if label == 1 else "non-toxic" for label in y_test_raw]
    

    return x, z_raw, y_raw, None, None, None, x_test, z_test_raw, y_test_raw


if __name__ == "__main__":
    from collections import Counter

    dataset = "biasbios"

    if dataset == "glove": x, z_raw, y_raw, x_val, z_val, y_val, x_test, z_test_raw, y_test_raw = load_GloVe() 
    if dataset == "biasbios": x, z_raw, y_raw, x_val, z_val, y_val, x_test, z_test_raw, y_test_raw = load_biasbios()
    if dataset == "biasbios_v2t": x, z_raw, y_raw, x_val, z_val, y_val, x_test, z_test_raw, y_test_raw = load_biasbios_v2t()
    if dataset == "biasbios_v2t_cf": x, z_raw, y_raw, x_val, z_val, y_val, x_test, z_test_raw, y_test_raw = load_biasbios_v2t_cf()
    if dataset == "deepmoji": x, z_raw, y_raw, x_val, z_val, y_val, x_test, z_test_raw, y_test_raw = load_deepmoji()
    if dataset == "jigsaw": x, z_raw, y_raw, x_val, z_val, y_val, x_test, z_test_raw, y_test_raw = load_jigsaw()
    print("Number of samples in the training set:", len(x))
    print("Number of samples in the validation set:", len(x_val) if x_val is not None else 0)
    print("Number of samples in the test set:", len(x_test))
    print("-------------")
    print("Number of samples in the training with each concept class:", Counter(z_raw))
    print("Number of samples in the test set with each concept class:", Counter(z_test_raw))
    print("-------------")
    print("Number of samples in the training set with each downstream label:", Counter(y_raw))
    print("Number of samples in the test set with each downstream label:", Counter(y_test_raw))

    # display some samples
    if dataset == "biasbios": 
        # train_data, dev_data, test_data = load_biasbios_texts()
        _, _, test_cf_data = load_biasbios_texts_cf()
        for i in range(5):
            print("-------------")
            print("Original:", test_cf_data[i]["hard_text_untokenized"])
            print("Counterfactual:", test_cf_data[i]["text_gender_reversed"])
  
            
