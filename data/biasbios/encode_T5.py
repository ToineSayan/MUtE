import tqdm
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import List
from transformers import AutoModel, AutoTokenizer
warnings.filterwarnings(action='ignore')


IS_FIRST = True
MAX_SEQUENCE_LENGTH = 64
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512

############################################################

def mean_pool(
        hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs

def get_gtr_embeddings(text_list, encoder, tokenizer):
    samples_len = [len(s) for s in tokenizer(text_list)['input_ids']]

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=MAX_SEQUENCE_LENGTH,
                       truncation=True,
                       padding="max_length", ).to(DEVICE)

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = mean_pool(hidden_state, inputs['attention_mask']).cpu().detach().numpy()

    return embeddings, samples_len

encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to(DEVICE)
encoder.eval()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

############################################################

def read_data_file(input_file):
    """
    read the data file with a pickle format
    :param input_file: input path, string
    :return: the file's content
    """
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    return data

############################################################


base_path = "."
out_dir = "./T5"


# extract and save the concept labels and target labels for each split
for split in ['train', 'dev', 'test']:
    data = read_data_file(f"{split}.pickle")
    data_df = pd.DataFrame(data)

    z = data_df['g'].to_numpy()
    y = data_df['p'].to_numpy()
    np.save(f'{out_dir}/{split}_z.npy', z)
    np.save(f'{out_dir}/{split}_y.npy', y)

    encodings = []
    # encode the train set
    for i in tqdm.tqdm(range(0, len(data_df), BATCH_SIZE)):
        sents_batch = data_df.loc[i:i+BATCH_SIZE-1, 'hard_text_untokenized'].tolist() # previously hard_text
        embeddings, samples_len = get_gtr_embeddings(sents_batch, encoder, tokenizer)
        encodings.append(embeddings)
    x = np.concatenate(encodings, axis=0)
    x = x[:len(data_df)] # Nan are added to the last batch, let's remove them
    np.save(f'{out_dir}/{split}_meanpool.npy', x)
    

for split in ['train', 'dev', 'test']:
    data = read_data_file(f"{split}_cf_augmented.pickle")
    data_df = pd.DataFrame(data)

    z = data_df['g_cf'].to_numpy()
    cf_success = data_df['cf_success'].to_numpy()
    z[cf_success == False] = 'nan' # let's assign a special value to samples for which counterfactual augmentation failed
    np.save(f'{out_dir}/{split}_cf_z.npy', z)

    encodings = []
    # encode the train set
    for i in tqdm.tqdm(range(0, len(data_df), BATCH_SIZE)):
        sents_batch = data_df.loc[i:i+BATCH_SIZE-1, 'text_gender_reversed'].tolist()
        embeddings, samples_len = get_gtr_embeddings(sents_batch, encoder, tokenizer)
        encodings.append(embeddings)
    x = np.concatenate(encodings, axis=0)
    x = x[:len(data_df)] # Nan are added to the last batch, let's remove them
    np.save(f'{out_dir}/{split}_cf_meanpool.npy', x)
