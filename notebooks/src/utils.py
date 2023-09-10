from src.text import classes_table
import tensorflow as tf
import pandas as pd
import gzip
import json
import math
import numpy as np
from src.text import standardize_method

def load_data_generator(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)


def load_dataset(path):
    return pd.DataFrame([row for row in load_data_generator("../datasets/awzm_products.jsonl.gz")])


def split_data(dataset):

    X = dataset["title"].to_numpy()
    Y = dataset["main_cat"].to_numpy()
    N = X.shape[0]

    train_split=0.90
    val_split=(1-train_split)/2

    assert (train_split+val_split*2) == 1.0

    n_train = math.ceil(N*train_split)
    n_val = math.ceil(N*val_split)

    np.random.RandomState(42).shuffle(X)
    np.random.RandomState(42).shuffle(Y)

    X_train = X[:n_train]
    Y_train = Y[:n_train]

    X_val = X[n_train:n_train+n_val]
    Y_val = Y[n_train:n_train+n_val]

    X_test = X[n_train+n_val:]
    Y_test = Y[n_train+n_val:]

    print(f"train split: {train_split} | val split: {val_split} | test split: {val_split}")
    print(f"Train shape: {(X_train.shape[0], Y_train.shape[0])}, Val shape: {(X_val.shape[0], Y_val.shape[0])}, Test count: {(X_test.shape[0], Y_test.shape[0])}")

    assert X_train.shape[0]==Y_train.shape[0]
    assert X_val.shape[0]==Y_val.shape[0]
    assert X_test.shape[0]==Y_test.shape[0]

    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


class DataLoader():

    def __init__(self, vocab_size, classes, batch_size):
        self.vocab_size = vocab_size
        self.classes = classes
        self.batch_size = batch_size
        self.text_vectorizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size,
                                                        standardize=standardize_method,
                                                        split="whitespace",
                                                        output_mode="tf-idf")

    def build_datasets(self, path):
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(load_dataset(path))

        self.text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(self.batch_size))

        train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                            .shuffle(self.batch_size)
                            .batch(self.batch_size)
                            .map(lambda text, label: (self.text_vectorizer(text), tf.one_hot(classes_table.lookup(label), self.classes)),
                                                     num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE))

        val_dataset = (tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(self.batch_size)
                            .map(lambda text, label: (self.text_vectorizer(text), tf.one_hot(classes_table.lookup(label), self.classes)),
                                                     num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)).cache()

        test_dataset = (tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(self.batch_size)
                            .map(lambda text, label: (self.text_vectorizer(text), tf.one_hot(classes_table.lookup(label), self.classes)),
                                                     num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)).cache()

        return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}