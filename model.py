import pickle
import sys

import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
from pandas import DataFrame

from domain import (
    MAX_CONTEXT_LENGTH,
    MAX_RESPONSE_LENGTH,
)


class Model:

    def __init__(self):
        pass

    def train(self, **kwargs):
        raise NotImplementedError(
            "Not implement train function"
        )

    def predict(self, **kwargs):
        raise NotImplementedError(
            "Not implement predict function"
        )

    def save(self, **kwargs):
        raise NotImplementedError(
            "Not implement save function"
        )

    def load(self, **kwargs):
        raise NotImplementedError(
            "Not implement load function"
        )


class SklearnModel(Model):

    def __init__(self, name):
        self.name = name

    def train(self, data):
        texts = data[0],
        y = data[1]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2)
        )
        # We simply concat the context and the response to
        # a paragraph
        X = vectorizer.fit_transform(texts)

        # build the model
        lg = LogisticRegression()
        parameters = {
            "C": [0.1, 1, 10, 100],
            "max_iter": [1000],
        }
        model = GridSearchCV(lg, parameters)
        model.fit(X, y)
        print ("resutls: ", model.cv_results_)

        self.vectorizer = vectorizer
        self.model = model

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X), self.model.predict_proba(X)[:, 1]

    def save(self, data_dir):
        with open(f"{data_dir}/{self.name}.vec", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(f"{data_dir}/{self.name}.cls", "wb") as f:
            pickle.dump(self.model, f)

    def load(self, data_dir):
        with open(f"{data_dir}/{self.name}.vec", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(f"{data_dir}/{self.name}.cls", "rb") as f:
            self.model = pickle.load(f)

def get_rnn_layer(units, reg=1e-7):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=units,
            recurrent_initializer="glorot_uniform",
            return_sequences=True,
            return_state=True,
            kernel_regularizer=tf.keras.regularizers.l2(reg),
            recurrent_regularizer=tf.keras.regularizers.l2(reg),
            activity_regularizer=tf.keras.regularizers.l2(reg),
        ),
        merge_mode="concat"
    )


class DSTC7FirstPlace(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        embedd_dim=200,
        units=200, # use for both context and response
        local_units=200,
        composition_units=200,
        final_hidden_units=512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedd_dim = embedd_dim

        # input_length = None because we use this embedding for both context and response
        # which are different length
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedd_dim, mask_zero=True, input_length=None)

        # return (a1...aT, cT) for regular rnn, if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        self.c_rnn = get_rnn_layer(units)
        self.r_rnn = get_rnn_layer(units)

        # to transform context_units quals to response_units
        # to calculate the E matrix attention weights
        # need *2 because we use bidirectional
        # self.atten_dense = tf.keras.layers.Dense(2*response_units, activation="linear")

        # reduce the dimension after local matching
        self.local_dense_c = tf.keras.layers.Dense(local_units, activation="relu")
        self.local_dense_r = tf.keras.layers.Dense(local_units, activation="relu")

        # return (a1...aT, cT) for regular rnn, if LSTM (a1...aT, aT, cT)
        # if Bidirectional (a1...aT, a1T, a2T, c1T, c2T)
        # read local matching vectors and learn to discrimate critical local matching vectors
        # for the overall utterance-level relationship
        self.c_rnn_composition = get_rnn_layer(composition_units)
        self.r_rnn_composition = get_rnn_layer(composition_units)

        # pooling operators
        self.global_max_pool = tf.keras.layers.GlobalMaxPool1D(data_format="channels_last")
        self.global_average_pool = tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last")

        # final mlp layers
        self.final_hidden_dense = tf.keras.layers.Dense(final_hidden_units, activation="tanh")
        self.final_dense = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=True):
        context = inputs["context"]
        response = inputs["response"]

        # tf.print("context[0]: ", context[0])
        # tf.print("response[0]: ", response[0])
        # tf.print("training: ", training)

        # (batch, time, embedd_dim)
        embedd_context = self.embedding(context, training=training)
        embedd_response = self.embedding(response, training=training)

        # (batch, time)
        context_mask = tf.dtypes.cast(embedd_context._keras_mask, tf.bool)
        response_mask = tf.dtypes.cast(embedd_response._keras_mask, tf.bool)

        # (batch, time, units)
        context_hiddens = self.c_rnn(embedd_context, mask=context_mask, training=training)[0]
        response_hiddens = self.r_rnn(embedd_response, mask=response_mask, training=training)[0]

        # mask the outputs, so make all padding vectors to zero vectors
        context_hiddens = context_hiddens * tf.cast(tf.expand_dims(context_mask, axis=2), tf.float32)
        response_hiddens = response_hiddens * tf.cast(tf.expand_dims(response_mask, axis=2), tf.float32)

        # local matching, calculate the E mmatrix
        # (batch, m, units) x (batch, n, units)T -> (batch, m, n)
        E = tf.matmul(context_hiddens, response_hiddens, transpose_b=True)

        # tf.print("E[0]: ", E[0])
        # normalize attention weights
        # here we check if the weights are padding weights
        eE = tf.keras.backend.switch(tf.math.equal(tf.math.exp(E), 1), tf.zeros_like(E), tf.math.exp(E))
        # tf.print("eE[0]: ", eE[0])

        # calculate normalized weights
        # (batch, m, n)
        alpha = eE / tf.reduce_sum(eE, axis=2, keepdims=True)
        # tf.print("alpha[0]: ", alpha[0])
        # tf.print("alpha[0][-1] sum: ", tf.math.reduce_sum(alpha[0][-1]))
        # make sure not getting nan when dividing by zero (some cases sum of weights in a column is zero)
        alpha = tf.keras.backend.switch(tf.math.is_nan(alpha), tf.zeros_like(alpha), alpha)
        # tf.print("alpha[0] no nan: ", alpha[0])

        beta  = eE / tf.reduce_sum(eE, axis=1, keepdims=True)
        # make sure not getting nan when dividing by zero (some cases sum of weights in a column is zero)
        beta  = tf.keras.backend.switch(tf.math.is_nan(beta), tf.zeros_like(beta), beta)

        # tf.print("beta[0]: ", beta[0])
        # tf.print("beta[0] sum: ", tf.math.reduce_sum(beta[0][:, -1]))

        # calculate dual vectors to capture contextual meaning
        context_dual = tf.matmul(alpha, response_hiddens)
        response_dual = tf.matmul(beta, context_hiddens, transpose_a=True)

        # tf.print("context_dual shape: ", context_dual.shape)
        # tf.print("response_dual shape: ", response_dual.shape)

        # state and dual vectors
        context_local = tf.concat([
            context_hiddens,
            context_dual,
            context_hiddens - context_dual,
            context_hiddens * context_dual,
        ],  axis=2)
        response_local = tf.concat([
            response_hiddens,
            response_dual,
            response_hiddens - response_dual,
            response_hiddens * response_dual,
        ],  axis=2)

        # reduce the local vectors dimension
        context_local = self.local_dense_c(context_local)
        response_local = self.local_dense_r(response_local)

        # tf.print("context_local shape: ", context_local.shape)
        # tf.print("response_local shape: ", response_local.shape)
        # tf.print("context_local[0]: ", context_local[0])

        # matching composition
        # read local matching vectors and learn to discrimate critical local matching vectors
        # for the overall utterance-level relationship
        context_composition = self.c_rnn_composition(context_local, training=training)[0]
        response_composition = self.r_rnn_composition(response_local, training=training)[0]

        # tf.print("context_composition shape: ", context_composition.shape)
        # tf.print("response_composition shape: ", response_composition.shape)

        # pooling
        # (batch, composition_units)
        context_max = self.global_max_pool(context_composition)
        context_average = self.global_average_pool(context_composition)
        response_max = self.global_max_pool(response_composition)
        response_average = self.global_average_pool(response_composition)

        # concat
        final_features = tf.concat([
            context_max, context_average,
            response_max, response_average,
        ], axis=1)
        # tf.print("final_features shapre: ", final_features.shape)

        # final mlp layers
        final_features = self.final_hidden_dense(final_features)
        probs = self.final_dense(final_features)

        # logits = tf.math.reduce_sum(logits, axis=1)

        # tf.print("logits shape: ", logits.shape)
        # tf.print("logits[0]: ", logits[0])

        return probs
