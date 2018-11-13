import numpy as np
import pandas as pd
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split


class Model():
    def __init__(self, dataset):
        self.datapath = os.path.join(os.getcwd(), dataset)
        self.df = self.load_mbti_dataset(self.datapath)

    def metrics(self, df):
        num_of_samples = len(df.index)

        num_words = [len(s.split()) for s in df["posts"]]
        words_per_sample = np.average(num_words)

        classes = df["type"].unique
        samples_per_class = df["type"].value_counts()

        return num_of_samples, words_per_sample, samples_per_class, classes

    def load_mbti_dataset(self, data_path, seed=123):
        # Load and shuffle the data
        data = pd.read_csv(data_path)
        random.seed(seed)
        data.reindex(np.random.permutation(data.index))
        return data

    def split_data(self):
        # 80:20 split for training and testing
        train, test = train_test_split(self.df, test_size=0.2)
        train_texts = train["posts"]
        train_labels = train["type"]
        test_texts = test["posts"]
        return train_texts, np.array(train_labels), test_texts

    def vectorize(self):
        train_texts, train_labels, test_texts = self.split_data()

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents='unicode',
            decode_error='replace',
            analyzer='word',    # create word tokens
            min_df=2)   # gets rid of tokens with appearing in less than 2 docs

        # Learns vocabulary and idf from texts and return term-document matrix
        x_train = vectorizer.fit_transform(train_texts)
        x_test = vectorizer.transform(test_texts)

        # Use f_classif to calculate feature importance and get the 20,000 most important features
        selector = SelectKBest(f_classif, k=20)
        selector.fit(x_train, train_labels)

        x_train = selector.transform(x_train).astype('float32')
        print(x_train.shape)
        x_test = selector.transform(x_test).astype('float32')

if __name__ == "__main__":  
    a = Model("mbti.csv")
    a.vectorize()
