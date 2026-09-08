# Instructions to download the datasets and compute the embeddings

## GloVe

For the WS-353 calculation, download the top 150k GloVe embeddings from the [RLACE (Ravfogel et al., 2022) repository](https://nlp.biu.ac.il/~ravfogs/rlace/glove/glove-top-50k.pickle).
Name it `glove-top-150k.pickle`.

### Directory structure

```bash
└── GloVe
    ├── evaluation/ws353simrel/wordsim353_sim_rel
    │   ├── wordsim_relatedness_goldstandard.txt
    │   ├── wordsim_similarity_goldstandard.txt
    │   ├── wordsim353_agreed.txt
    │   ├── wordsim353_annotator1.txt
    │   └── wordsim353_annotator2.txt
    ├── fem_words.pkl
    ├── glove-top-150k.pickle
    ├── male_words.pkl
    └── neut_words.pkl   
```

## Bias in Bios

### Download 

(Version used in the article "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection, Ravfogel et al., 2020")

```bash
mkdir -p data/biasbios
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle -P data/biasbios/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle -P data/biasbios/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle -P data/biasbios/
```

### Typical entries:

```json
{
	g:	f
	p:	professor
	text:	Janine Langan is an Associate Professor at the University of Toronto. She founded and teaches the Christianity and Culture program. Janine has written and lectured extensively on art, the family, the media, and the problems of Catholic education.
	start:	69
	hard_text:	She founded and teaches the Christianity and Culture program . Janine has written and lectured extensively on art , the family , the media , and the problems of Catholic education .
	hard_text_untokenized:	She founded and teaches the Christianity and Culture program. Janine has written and lectured extensively on art, the family, the media, and the problems of Catholic education.
	text_without_gender:	_ founded and teaches the Christianity and Culture program. _ has written and lectured extensively on art, the family, the media, and the problems of Catholic education.
}

{
	g:	f
	p:	psychologist
	text:	Gillian Burrell is a psychoanalytic psychotherapist in private practice in Sydney. She is a past secretary of the NSW Institute of Psychoanalyic Psychotherapy. Before that she worked with Relationships Australia as a family and relationship therapist.
	start:	82
	hard_text:	She is a past secretary of the NSW Institute of Psychoanalyic Psychotherapy . Before that she worked with Relationships Australia as a family and relationship therapist .
	hard_text_untokenized:	She is a past secretary of the NSW Institute of Psychoanalyic Psychotherapy. Before that she worked with Relationships Australia as a family and relationship therapist.
	text_without_gender:	_ is a past secretary of the NSW Institute of Psychoanalyic Psychotherapy. Before that _ worked with Relationships Australia as a family and relationship therapist.
}
```

###  Key statistics

- Number of observations (train/validation/test): 255,710 / 39,369 / 98,344
- Target concept: gender - 2 labels (f /m)
- Downstream task : occupation prediction - 28 labels (accountant, architect, attorney, chiropractor, comedian, composer, dentist, dietitian, dj, filmmaker, interior_designer, journalist, model, nurse, painter, paralegal, pastor, personal_trainer, photographer, physician, poet, professor, psychologist, rapper, software_engineer, surgeon, teacher, yoga_teacher)


### Bert embeddings

To compute Bert embeddings, go to directory `./biasbios/` and run

```python
python encode_bert.py
```
This script generates the files `train_cls.npy`, `train_z.npy`, `train_y.npy`, `dev_cls.npy`, `dev_z.npy`, `dev_y.npy`, `test_cls.npy`, `test_z.npy`, `test_y.npy` inside the directory `./biasbios/bert/`

### T5 embeddings (for Embedding to Text)

To compute T5 embeddings, go to directory `./biasbios/` and run the two following scripts to: 

1. generate the counterfactual texts of Bias in Bios
```python
python build_cf_set.py
```

This script generates the files `train_cf_augmented.pickle`, `dev_cf_augmented.pickle`, `dev_cf_augmented.pickle` inside the current directory.

2. embed the original texts and the counterfactual texts

```python
python encode_T5.py
```
This script generates the files `train_meanpool.npy`, `train_z.npy`, `train_y.npy`, `train_cf_meanpool.npy`, `train_cf_z.npy`, `dev_meanpool.npy`, `dev_z.npy`, `dev_y.npy`, `dev_cf_meanpool.npy`, `dev_cf_z.npy`, `test_meanpool.npy`, `test_z.npy`, `test_y.npy`, `test_cf_meanpool.npy`, `test_cf_z.npy`, inside the directory `./biasbios/T5/`

### Directory structure

```bash
└── biasbios
    ├── bert
    │   ├── train_cls.npy
    │   ├── train_z.npy
    │   ├── train_y.npy
    │   ├── dev_cls.npy
    │   ├── dev_z.npy
    │   ├── dev_y.npy
    │   ├── test_cls.npy
    │   ├── test_z.npy
    │   └── test_y.npy
    ├── T5
    │   ├── train_meanpool.npy
    │   ├── train_z.npy
    │   ├── train_y.npy
    │   ├── train_cf_meanpool.npy
    │   ├── train_cf_z.npy
    │   ├── dev_meanpool.npy
    │   ├── dev_z.npy
    │   ├── dev_y.npy
    │   ├── dev_cf_meanpool.npy
    │   ├── dev_cf_z.npy
    │   ├── test_meanpool.npy
    │   ├── test_z.npy
    │   ├── test_y.npy 
    │   ├── test_cf_meanpool.npy
    │   └── test_cf_z.npy
    ├── build_cf_set.py 
    ├── encode_bert.py
    ├── encode_T5.py
    ├── train.pickle
    ├── train_cf_augmented.pickle
    ├── dev.pickle
    ├── dev_cf_augmented.pickle
    ├── test.pickle
    └── test_cf_augmented.pickle
```


## DIAL

### Download 

Follow the instructions from the [KRaM (Chowdhury et al., 2023) repository](https://github.com/brcsomnath/KRaM/blob/master/data/README.md)


### Directory structure

```bash
└── deepmoji
    ├── train
    │   ├── neg_neg.npy
    │   ├── neg_pos.npy
    │   ├── pos_neg.npy
    │   └── pos_pos.npy
    ├── dev
    │   ├── neg_neg.npy
    │   ├── neg_pos.npy
    │   ├── pos_neg.npy
    │   └── pos_pos.npy
    └── test
        ├── neg_neg.npy
        ├── neg_pos.npy
        ├── pos_neg.npy
        └── pos_pos.npy
```

## Jigsaw

Download [Jigsaw Unintended Bias in Toxicity all_data](https://www.kaggle.com/datasets/frankmollard/jigsaw-unintended-bias-in-toxicity-all-data) and put the `all_data.csv` file in the `./jigsaw/` directory.

Put your opeanai key into the `./jigsaw/` directory.

Run the notebook `./jigsaw/data_analysis.ipynb` to embed the data and generate the train / test files (`jigsaw_train.pkl` and `jigsaw_test.pkl`)

### Directory structure

```bash
└── jigsaw
    ├── all_data.csv
    ├── data_analysis.ipynb
    ├── jigsaw_train.pkl
    ├── jigsaw_test.pkl
    └── openai.key
```