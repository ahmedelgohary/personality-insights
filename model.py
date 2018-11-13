import numpy as np
import pandas as pd
import os, random


def metrics(df):
    num_of_samples = len(df.index)
    
    num_words = [len(s.split()) for s in df["posts"]]
    words_per_sample = np.average(num_words)

    classes = df["type"].unique
    samples_per_class = df["type"].value_counts()

    return num_of_samples, words_per_sample, samples_per_class, classes

def load_mbti_dataset(data_path, seed=123):
    # Load data
    data = pd.read_csv(data_path)
    random.seed(seed)
    data.reindex(np.random.permutation(data.index))
    metrics(data)
    num_validation = int(0.8 * len(data))
    training, validation = data[:num_validation], data[num_validation:]
    # print(training.head())


    # train_la?bels, validation_labels = np.array(training.iloc[:, 0]), np.array(validation.iloc[:, 0])

    
    return training


if __name__ == "__main__":
    load_mbti_dataset(os.path.join(os.getcwd(), "mbti.csv"))