## Obtaining Better Static Word Embeddings Using Contextual Embedding Models
This folder contains the code used in the paper "Obtaining Better Static Word Embeddings Using Contextual Embedding Models"

## Software Requirements
+ Python3.6+
+ PyTorch1.4.0

### Libraries for Python

Use pip to install the following libraries -
+ transformers -
+ tqdm
+ ftfy
+ spacy
+ nltk

## Hardware Requirements
+ GPU


## Files used

+ learn_from_bert_ver2.py - used to obtain BERT2STATIC<sub>sent</sub>
+ learn_from_bert_ver2_paragraph.py - used to obtain BERT2STATIC<sub>para</sub>
+ learn_from_gpt2_ver2.py - used to obtain GPT22STATIC<sub>sent</sub>
+ learn_from_gpt2_ver2_paragraph.py - used to obtain GPT22STATIC<sub>para</sub>
+ learn_from_roberta_ver2.py - used to obtain RoBERTa2STATIC<sub>sent</sub>
+ learn_from_roberta_ver2_paragraph.py - used to obtain RoBERTa2STATIC<sub>para</sub>
+ make_vocab_dataset.py - to create a pickled dataset from a preprocessed one

## Usage -

### Preprocessing -
Use make_vocab_dataset.py to create a pickled dataset from a preprocessed one. Example -
python make_vocab_dataset.py --dataset_location wiki_text --min_count 10 --max_vocab_size 750000 --location_save_vocab_dataset processed_data/

Then the processed data will be saved in the folder processed_data/ with a vocabulary constructed for at most 750000 words which occur at least 10 times in the dataset.

Make sure that you use a preprocessed file with 1 paragraph per line for X2STATIC<sub>para</sub> embeddings and a preprocessed file with 1 sentence per line for X2STATIC<sub>sent</sub> embeddings with make_vocab_dataset.py

### Training models -
Suppose, you want to obtain BERT2STATIC<sub>para</sub> embeddings from BERT-12 and the preprocessed data is in the folder processed_data/, use the following code (with the same hyperparameters as the paper)
```
python learn_from_bert_ver2_paragraph.py --gpu_id 0 --num_epochs 1 --lr 0.001 --algo SparseAdam --t 5e-6 --word_emb_size 768 --location_dataset  processed_data/ --model_folder model/ --num_negatives 10 --pretrained_bert_model bert-base-uncased
```

Then the vectors will be stored in model/vectors_final.txt . You can also download the preprocessed data used in the paper from [here](https://zenodo.org/record/5055755). 

For other models, use the relevant python file and relevant word_emb_size (768 for models with 12 layers and 1024 for models with 24 layers) and select the correct pretrained_bert_model/ pretrained_RoBERTa_model/pretrained_gpt2_model as shown in the list below -

+ BERT-12 - bert-base-uncased
+ RoBERTa-12 -roberta-base
+ GPT2-12 - gpt2

+ BERT-24 - bert-large-uncased
+ RoBERTa-24 - roberta-large
+ GPT2-24 - gpt2-medium

### ASE Models -
For ASE models, use the code provided by Bommasani et al. here - https://github.com/AnonymousICLR2020Submission/BERT-Wears-GloVes
