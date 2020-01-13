
import pickle
import sys

import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
)

from domain import get_clustering_data_level0


def featurize(texts, ngram_range=(1, 2), feature_type="tfidf", binary=False):
    if feature_type == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    else:
        vectorizer = CountVectorizer(ngram_range=ngram_range, binary=binary)

    X = vectorizer.fit_transform(texts).toarray()
    return X

def train(
    texts,
    output_folder,
    model_name,
    ngram_range=(1, 2),
    feature_type="tfidf",
    binary=False,
    min_sample=5,
):
    X = featurize(
        texts=texts,
        ngram_range=ngram_range,
        binary=binary,
        feature_type=feature_type,
    )

    clustering = OPTICS(min_samples=min_sample)
    print ("Training ...")
    clustering.fit(X)
    print ("Done!")

    pickle_file = f"{output_folder}/{model_name}.pkl"
    with open(pickle_file, "wb") as f:
        pickle.dump(clustering, f)
    print (f"Write file {pickle_file}")

def train_level0():
    texts = get_clustering_data_level0(
        excel_file="data/sach_mem/human_res_concat.xlsx"
    )
    train(
        texts=texts,
        output_folder="data/sach_mem",
        model_name="optics_clustering_level0",
        ngram_range=(1, 2),
        feature_type="tfidf",
        binary=False,
        min_sample=5,
    )

def train_level1():
    texts = get_clustering_data_level0(
        excel_file="data/sach_mem/predict_level0/group_to_response.xlsx"
    )

    train(
        texts=texts,
        output_folder="data/sach_mem",
        model_name="optics_clustering_level1",
        ngram_range=(1, 2),
        feature_type="tfidf",
        binary=False,
        min_sample=3,
    )

if __name__ == "__main__":
    args = sys.argv

    if args[1] == "level0":
        train_level0()
    elif args[1] == "level1":
        train_level1()
