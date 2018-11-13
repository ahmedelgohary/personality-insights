import numpy as np
import pandas as pd
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, callbacks, losses


class Model():
    def __init__(self, dataset):
        self.datapath = os.path.join(os.getcwd(), dataset)
        self.df = self.load_mbti_dataset(self.datapath)

    def metrics(self, df):
        '''
        Used to calculate key metrics which were used to determine the type of model to be useds
        '''
        num_of_samples = len(df.index)

        num_words = [len(s.split()) for s in df["posts"]]
        words_per_sample = np.median(num_words)

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
        test_labels = test["type"]
        return train_texts, np.array(train_labels), test_texts, np.array(test_labels)

    def vectorize(self):
        train_texts, train_labels, test_texts, test_labels = self.split_data()

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            strip_accents='unicode',
            decode_error='replace',
            analyzer='word',    # create word tokens
            min_df=2)   # gets rid of tokens appearing in less than 2 docs

        # Learns vocabulary and idf from texts and return term-document matrix
        x_train = vectorizer.fit_transform(train_texts)
        x_test = vectorizer.transform(test_texts)

        # Use f_classif to calculate feature importance and get the 20,000 most important features
        selector = SelectKBest(f_classif, k=20000)
        selector.fit(x_train, train_labels)

        x_train = selector.transform(x_train).astype('float32')
        x_test = selector.transform(x_test).astype('float32')

        return x_train, x_test, train_labels, test_labels

    def mlp_model(self, input_shape):
        # An instance of a multi-layer perceptron model using Softmax as the activation function
        # Has a Dropout rate of 20% and 64 as our units
        model = Sequential([
            # Our input layer with our shape tuple
            Dropout(rate=0.2, input_shape=input_shape),

            # Use 2 Dense layers
            Dense(units=64, activation='relu'),
            Dropout(rate=0.2),
            Dense(units=64, activation='relu'),
            Dropout(rate=0.2),

            # Output layer, using softmax as our activation function for multi-class classification
            Dense(units=16, activation='softmax')])

        return model

    def train_model(self):
        ''' 
        Train our model using:
        Accuracy as the metric
        Categorical Cross Entropy as our loss function
        Adam as our optimizer
        '''
        x_train, x_test, train_labels, test_labels = self.vectorize()
        model = self.mlp_model(x_train.shape)

        optimizer = optimizers.Adam(lr=1e-3)    # learning rate of 1e-3
        model.compile(optimizer=optimizer,
                      loss=losses.categorical_crossentropy, metrics=['acc'])

        # call back to stop early when losing validation, i.e. stop if loss doesn't decrease in two consecutive tries
        callback = [callbacks.EarlyStopping(monitor='val_loss', patience=2)]

        # Train and validate model.
        history = model.fit(
            x_train,
            train_labels,
            epochs=1000,
            callbacks=callback,
            validation_data=(x_test, test_labels),
            verbose=2,  # Logs once per epoch.
            batch_size=512)

        # Print the results
        history = history.history
        print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

        # Save the model
        model.save('mlp_model.h5')
        return history['val_acc'][-1], history['val_loss'][-1]


if __name__ == "__main__":
    a = Model("mbti.csv")
    a.train_model()
