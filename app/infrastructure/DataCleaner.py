import re
import os
import random
import math
import bert
from bert.tokenization import bert_tokenization
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from joblib import dump, load

import app.config as cf




class CleanData :
    """
    cleans text data from raw file
    """

    def __init__(self, path='', cols=[], cols_to_keep=[]) :
        self.path = path
        self.cols = cols
        self.cols_to_keep = cols_to_keep



    def get_df_from_path(self) :
        name, extension = os.path.splitext(self.path)
        if extension == '.csv':
            df = pd.read_csv(self.path, names=self.cols)
        elif extension == '.parquet':
            df = pd.read_parquet(self.path, names=self.cols)
        else:
            raise FileExistsError('Extension must be parquet or csv.')

        return df[self.cols_to_keep]



    def clean_tweet(self, tweet):
        tweet = BeautifulSoup(tweet, "lxml").get_text()
        tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
        tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)
        tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
        tweet = re.sub(r" +", ' ', tweet)
        return tweet



    def get_cleaned_df(self):
        df = self.get_df_from_path()
        df['text'] = df['text'].apply(lambda tweet : self.clean_tweet(tweet))
        return df



    def get_data_clean(self) :
        data_clean = self.get_cleaned_df()['text'].values
        return data_clean



    def get_data_labels(self) :
        data_labels = self.get_cleaned_df()['sentiment'].values
        print('get_data_labels')
        return data_labels



    def get_tokenizer(self):
        FullTokenizer = bert_tokenization.FullTokenizer
        bert_layer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
            trainable=False
        )
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        tokenizer = FullTokenizer(vocab_file, do_lower_case)
        print('get_tokenizer')
        return tokenizer



    def get_encode_sentence(self):
        tokenizer = self.get_tokenizer()
        def encode_sentence(sent):
            return ["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"]
        print('get_encode_sentence')
        return encode_sentence



    def get_data_inputs(self) :
        data_clean = self.get_data_clean()
        encode_sentence = self.get_encode_sentence()
        print('get_data_inputs')
        return [encode_sentence(sentence) for sentence in data_clean]



    def get_ids(self, tokens):
        tokenizer = self.get_tokenizer()
        return tokenizer.convert_tokens_to_ids(tokens)



    def get_mask(self, tokens):
        return np.char.not_equal(tokens, "[PAD]").astype(int)



    def get_segments(self, tokens):
        seg_ids = []
        current_seg_id = 0
        for tok in tokens:
            seg_ids.append(current_seg_id)
            if tok == "[SEP]":
                current_seg_id = 1-current_seg_id # convierte los 1 en 0 y vice versa
        return seg_ids



    def get_sorted_all(self) :
        data_labels = self.get_data_labels()
        data_inputs = self.get_data_inputs()
        data_with_len = [[sent, data_labels[i], len(sent)] for i, sent in enumerate(data_inputs)]
        random.shuffle(data_with_len)
        data_with_len.sort(key=lambda x: x[2])
        sorted_all = [
            ([get_ids(sent_lab[0]), get_mask(sent_lab[0]), get_segments(sent_lab[0])], sent_lab[1])
            for sent_lab in data_with_len if sent_lab[2] > 7
        ]
        print('get_sorted_all')
        return sorted_all



    def get_all_dataset(self) :
        sorted_all = self.get_sorted_all()
        all_dataset = tf.data.Dataset.from_generator(lambda: sorted_all, output_types=(tf.int32, tf.int32))
        return all_dataset



    def get_all_batched(self) :
        all_dataset = self.get_all_dataset()
        all_batched = all_dataset.padded_batch(cf.BATCH_SIZE, padded_shapes=((3, None), ()), padding_values=(0, 0))
        return all_batched


    def get_train_test(self) :
        all_batched = self.get_all_batched()
        NB_BATCHES = math.ceil(len(sorted_all) / cf.BATCH_SIZE)
        NB_BATCHES_TEST = NB_BATCHES // 10
        all_batched.shuffle(NB_BATCHES)
        test_dataset = all_batched.take(NB_BATCHES_TEST)
        train_dataset = all_batched.skip(NB_BATCHES_TEST)
        return test_dataset, train_dataset






if __name__ == "__main__":

    cd = CleanData(
        path=cf.INPUTS_FILE, 
        cols=cf.COLS,
        cols_to_keep=cf.COLS_TO_KEEP
    )

    sorted_all = cd.get_sorted_all()

    print(sorted_all)


