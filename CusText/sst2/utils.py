import os
import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from tqdm import tqdm, trange
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import json
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import warnings
from args import *
from transformers import BertTokenizer,BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

import datetime
from logger import get_logger

warnings.filterwarnings('ignore')

parser = get_parser()
args = parser.parse_args()

"""
加载数据
"""
def load_data(dataset=None):

    print(f'__loading__{args.dataset}__')
    train_df = pd.read_csv(f"datasets/{args.dataset}/train.tsv",'\t')
    dev_df = pd.read_csv(f"datasets/{args.dataset}/dev.tsv",'\t')
    test_df = pd.read_csv(f"datasets/{args.dataset}/test.tsv",'\t')
    return train_df,dev_df,test_df

"""
构造Dataset类
"""
class Bert_dataset(Dataset):
    def __init__(self,df):
        self.df=df
        self.tokenizer = BertTokenizer.from_pretrained(f"{args.model_type}",do_lower_case=True)

    def __getitem__(self,index):
        # get the sentence from the dataframe
        sentence = self.df.loc[index,'sentence']

        encoded_dict = self.tokenizer.encode_plus(
            sentence,              # sentence to encode
            add_special_tokens = True,         # Add '[CLS]' and '[SEP]'
            max_length = args.max_len,
            pad_to_max_length= True,
            truncation='longest_first',
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        # These are torch tensors already
        input_ids = encoded_dict['input_ids'][0]
        attention_mask = encoded_dict['attention_mask'][0]
        token_type_ids = encoded_dict['token_type_ids'][0]

        #Convert the target to a torch tensor
        target = torch.tensor(self.df.loc[index,'label'])

        sample = (input_ids,attention_mask,token_type_ids,target)
        return sample

    def __len__(self):
        return len(self.df)



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False



def get_customized_mapping(eps,top_k):



    df_train = pd.read_csv(f"datasets/{args.dataset}/train.tsv",'\t')
    df_dev = pd.read_csv(f"datasets/{args.dataset}/dev.tsv",'\t')
    train_corpus = " ".join(df_train.sentence)
    dev_corpus = " ".join(df_dev.sentence)
    corpus = train_corpus + " " + dev_corpus
    word_freq = [x[0] for x in Counter(corpus.split()).most_common()]

    if args.embedding_type == "glove_840B-300d":
        file = open(f'./embeddings/{args.embedding_type}.txt','r')
        js = file.read()
        word_embeddings_glove= json.loads(js)
        file.close()

        embeddings = []
        idx2word = []
        word2idx = {}

        for i,(k,v) in enumerate(word_embeddings_glove.items()):
            idx2word.append(k)
            word2idx[k] = i
            embeddings.append(v)

        embeddings = np.asarray(embeddings)
        idx2word = np.asarray(idx2word)
    else:
        embedding_path = f"./embeddings/{args.embedding_type}.txt"
        embeddings = []
        idx2word = []
        word2idx = {}
        with open(embedding_path,'r') as file:
            for i,line in enumerate(file):
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
                idx2word.append(line.strip().split()[0])
                word2idx[line.strip().split()[0]] = i
        embeddings = np.array(embeddings)
        idx2word = np.asarray(idx2word)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        print(embeddings.T.shape)

    if args.embedding_type == "glove_840B-300d":
        word_hash = defaultdict(str)
        sim_word_dict = defaultdict(list)
        p_dict = defaultdict(list)
        for i in trange(len(word_freq)):
            word = word_freq[i]
            if word in word2idx:
                if word not in word_hash:
                    index_list = euclidean_distances(embeddings[word2idx[word]].reshape(1,-1),embeddings)[0].argsort()[:top_k]
                    word_list = [idx2word[x] for x in index_list]
                    embedding_list = np.array([embeddings[x] for x in index_list])
                        
                    if args.mapping_strategy == "aggressive":
                        sim_dist_list = euclidean_distances(embeddings[word2idx[word]].reshape(1,-1), embedding_list)[0]
                        min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                        min_dist = min(sim_dist_list)
                        new_sim_dist_list = [-(x-min_dist)/min_max_dist for x in sim_dist_list]
                        tmp = [np.exp(eps*x/2) for x in new_sim_dist_list]
                        norm = sum(tmp)
                        p = [x/norm for x in tmp]
                        p_dict[word] = p
                        sim_word_dict[word] =  word_list
                    else:
                        for x in word_list:
                            if x not in word_hash:
                                word_hash[x] = word
                                sim_dist_list = euclidean_distances(embeddings[word2idx[x]].reshape(1,-1), embedding_list)[0]
                                min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                                min_dist = min(sim_dist_list)
                                new_sim_dist_list = [-(x-min_dist)/min_max_dist for x in sim_dist_list]
                                tmp = [np.exp(eps*x/2) for x in new_sim_dist_list]
                                norm = sum(tmp)
                                p = [x/norm for x in tmp]
                                p_dict[x] = p
                                sim_word_dict[x] =  word_list
                        if args.mapping_strategy == "conservative":
                            inf_embedding = [1e9] * 300
                            for i in index_list:
                                embeddings[i,:] = inf_embedding
    else:
        word_hash = defaultdict(str)
        sim_word_dict = defaultdict(list)
        p_dict = defaultdict(list)
        for i in trange(len(word_freq)):
            word = word_freq[i]
            if word in word2idx:
                if word not in word_hash:
                    index_list = np.dot(embeddings[word2idx[word]], embeddings.T).argsort()[::-1][:top_k]
                    word_list = [idx2word[x] for x in index_list]
                    embedding_list = np.array([embeddings[x] for x in index_list])
                        
                    if args.mapping_strategy == "aggressive":
                        sim_dist_list = np.dot(embeddings[word2idx[x]], embedding_list.T)
                        min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                        min_dist = min(sim_dist_list)
                        new_sim_dist_list = [(x-min_dist)/min_max_dist for x in sim_dist_list]
                        tmp = [np.exp(eps*x/2) for x in new_sim_dist_list]
                        norm = sum(tmp)
                        p = [x/norm for x in tmp]
                        p_dict[word] = p
                        sim_word_dict[word] =  word_list
                    else:
                        for x in word_list:
                            if x not in word_hash:
                                word_hash[x] = word
                                sim_dist_list = np.dot(embeddings[word2idx[x]], embedding_list.T)
                                min_max_dist = max(sim_dist_list) - min(sim_dist_list)
                                min_dist = min(sim_dist_list)
                                new_sim_dist_list = [(x-min_dist)/min_max_dist for x in sim_dist_list]
                                tmp = [np.exp(eps*x/2) for x in new_sim_dist_list]
                                norm = sum(tmp)
                                p = [x/norm for x in tmp]
                                p_dict[x] = p
                                sim_word_dict[x] =  word_list
                        if args.mapping_strategy == "conservative":
                            inf_embedding = [0] * 300
                            for i in index_list:
                                embeddings[i,:] = inf_embedding

    try:
        with open(f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt", 'w') as json_file:
            json_file.write(json.dumps(p_dict, ensure_ascii=False, indent=4))
    except IOError:
        pass
    else:
        pass
    finally:
        pass

    try:
        with open(f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt", 'w') as json_file:
            json_file.write(json.dumps(sim_word_dict, ensure_ascii=False, indent=4))
    except IOError:
        pass
    else:
        pass
    finally:
        pass



    return sim_word_dict,p_dict



def generate_new_sents_s1(df,sim_word_dict,p_dict,save_stop_words,type="train"):

    punct = list(string.punctuation)

    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))
    
    cnt = 0 
    raw_cnt = 0 
    stop_cnt = 0 
    dataset = df.sentence
    new_dataset = []

    for i in trange(len(dataset)):
        record = dataset[i].split()
        new_record = []
        for word in record:
            if (save_stop_words and word in stop_words) or (word not in sim_word_dict):
                if word in stop_words:
                    stop_cnt += 1  
                    raw_cnt += 1   
                if is_number(word):
                    try:
                        word = str(round(float(word))+np.random.randint(1000))
                    except:
                        pass                   
                new_record.append(word)
            else:
                p = p_dict[word]
                new_word = np.random.choice(sim_word_dict[word],1,p=p)[0]
                new_record.append(new_word)
                if new_word == word:
                    raw_cnt += 1 

            cnt += 1 
        new_dataset.append(" ".join(new_record))

    df.sentence = new_dataset

    if not os.path.exists(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}"):
        os.mkdir(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}")
    if type == "train":
        df.to_csv(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/train.tsv","\t",index=0)
    else:
        df.to_csv(f"./privatized_dataset/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}_{args.privatization_strategy}_save_stop_words_{args.save_stop_words}/test.tsv","\t",index=0)

    return df
