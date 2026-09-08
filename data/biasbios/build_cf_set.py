import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Ensure the required NLTK tokenizer models are available
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
nltk.download('punkt')

def custom_tokenize(text: str) -> list:
    """
    Tokenizes text using NLTK's word_tokenize, but enforces a specific 
    custom constraint: abbreviations like "Mr." are forcibly split into 
    the title and the period (e.g., ["Mr", "."]).

    Args:
        text (str): The input text string to tokenize.

    Returns:
        list: A list of string tokens with the applied constraints.
    """
    # Standard NLTK tokenization
    raw_tokens = word_tokenize(text)
    
    processed_tokens = []
    for token in raw_tokens:
        # Enforce the constraint: if the token is exactly "Mr.", split it.
        if token == "Mr.":
            processed_tokens.extend(["Mr", "."])
        elif token == "Ms.":
            processed_tokens.extend(["Ms", "."])
        elif token == "Mrs.":
            processed_tokens.extend(["Mrs", "."])
        elif token == "M.":
            processed_tokens.extend(["Mr", "."])
        elif token == "_.":
            processed_tokens.extend(["_", "."])
        # elif token == "Inc.":
        #     processed_tokens.extend(["Inc", "."])
        # elif token == "van":
        #     processed_tokens.extend([])
        else:
            processed_tokens.append(token)
            
    return processed_tokens

def reverse_gender(t_orig: str, t_nogender: str, label: str) -> str:
    """
    Replaces the blank indicator '_' in a gender-neutralized text with the 
    opposite gender pronoun from the original text.

    Args:
        t_orig (str): The original text containing the original gendered words.
        t_nogender (str): The text where gendered words have been replaced by '_'.
        label (str): The label indicating the original gender ("f" for female, "m" for male).

    Returns:
        str: A new string where the '_' are replaced by the opposite gender words.
        
    Raises:
        ValueError: If the lengths of the tokenized strings do not match, or if an invalid label is provided.
    """
    
    # Define the gender dictionaries mapping original words to their opposites
    m_dic = {"he": "she", "him": "her", "his": "hers", "mr": "ms", "m": "ms", "himself": "herself"}
    f_dic = {"she": "he", "her": "him", "hers": "his", "ms": "mr", "mrs": "mr", "herself": "himself"}
    
    # Select the appropriate dictionary based on the provided label
    if label == "f":
        swap_dic = f_dic
    elif label == "m":
        swap_dic = m_dic
    else:
        raise ValueError("Invalid label. Please use 'f' or 'm'.")

    # Tokenize the input texts into lists of words/punctuation
    tokens_orig = custom_tokenize(t_orig)
    tokens_nogender = custom_tokenize(t_nogender)
    
    # Ensure the tokenized arrays align perfectly
    if len(tokens_orig) != len(tokens_nogender):
        raise ValueError("Token alignment failed: t_orig and t_nogender have different token counts.")
        
    # Convert standard Python lists to numpy arrays for array operations
    arr_orig = np.array(tokens_orig)
    arr_nogender = np.array(tokens_nogender)
    
    # Find the numpy indices where the nogender array has the blank character "_"
    # np.where returns a tuple, so we grab the first element [0] to get the index array
    blank_indices = np.where(arr_nogender == "_")[0]
    
    # Iterate over the identified indices to perform the swap
    for idx in blank_indices:
        # Extract the original word that corresponds to the blank
        orig_word = arr_orig[idx]
        
        # Preserve capitalization state (e.g., "She" vs "she")
        is_capitalized = orig_word[0].isupper() if orig_word else False
        orig_word_lower = orig_word.lower()
        
        # Look up the opposite word. If the word isn't in the dict, fallback to the original word.
        opp_word = swap_dic.get(orig_word_lower, orig_word_lower)
        
        # Re-apply capitalization if the original word was capitalized
        if is_capitalized:
            opp_word = opp_word.capitalize()
            
        # Update the target numpy array with the new opposite gender word
        arr_nogender[idx] = opp_word
        
    # Detokenize the numpy array back into a properly spaced string
    # TreebankWordDetokenizer handles punctuation spacing better than a simple ' '.join()
    detokenizer = TreebankWordDetokenizer()
    reversed_text = detokenizer.detokenize(arr_nogender)
    
    return reversed_text


if __name__ == "__main__":
    import pickle

    data_path = "."

    for split in ['train', 'dev', 'test']:
        with open(f"{data_path}/{split}.pickle", 'rb') as f:
            data = pickle.load(f) 

        samples_cf_augmented = []
        cnt = 0
        for sample in data:
            sample_text = sample['hard_text_untokenized']
            sample_nogender = sample['text_without_gender']
            sample_label = sample['g']
            reversed_gender = 'm' if sample_label == 'f' else 'f'
            sample_augmented = sample
            sample_augmented["g_cf"] = reversed_gender
            try:
                reversed_text = reverse_gender(sample_text, sample_nogender, sample_label)
                sample_augmented["text_gender_reversed"] = reversed_text
                sample_augmented["cf_success"] = True
                cnt +=1
            except ValueError as e:
                sample_augmented["text_gender_reversed"] = ""  # Assign empty string if reversal fails
                sample_augmented["cf_success"] = False
            samples_cf_augmented.append(sample_augmented)
                # print(f"Error processing sample: {e}")
        print(f"Successfully processed {cnt} samples out of {len(data)} ({cnt/len(data)*100:.2f}%) for counterfactual augmentation.")

        # Save the augmented train samples to a pickle file
        with open(f"{data_path}/{split}_cf_augmented.pickle", "wb") as f:
            pickle.dump(samples_cf_augmented, f)
            
        # Load data and transform them into pandas dataframes
        with open(f"{data_path}/{split}_cf_augmented.pickle", "rb") as f:
            bios_set = pickle.load(f)
        bios_set_df = pd.DataFrame(bios_set)
        print(bios_set_df.head())
