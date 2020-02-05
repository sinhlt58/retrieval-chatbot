import argparse
import os

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
import sentencepiece as sp

from model import SklearnModel, DSTC7FirstPlace
from utils import read_json_data, write_json_data
from domain import (
    MAX_CONTEXT_LENGTH,
    MAX_RESPONSE_LENGTH,
    padd_ids,
)


def get_data_by_type(data_dir, model_type, training=True):
    batch_size = 64

    if model_type == "baseline":
        df = pd.read_excel(f"{data_dir}/train.xlsx", encoding="utf-8")
        # we simply concat context and response
        texts = (df["context"] + " <sep> " + df["response"]).values
        return (texts, df["label"].values)
    elif model_type in ["dstc7"]:
        name_to_features = {
            "context": tf.io.FixedLenFeature([MAX_CONTEXT_LENGTH], tf.int64),
            "response": tf.io.FixedLenFeature([MAX_RESPONSE_LENGTH], tf.int64),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        def _decode_record(example_proto):
            """Decodes a record to a TensorFlow example."""
            example_dict = tf.io.parse_single_example(example_proto, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            features_dict = {}
            for name in list(example_dict.keys()):
                t = example_dict[name]
                if t.dtype == tf.int64:
                    t = tf.cast(t, tf.int32)
                example_dict[name] = t
                if name != "label":
                    features_dict[name] = t

            label = example_dict["label"]

            return (features_dict, label)

        record_path = f"{data_dir}/train.tfrecord"
        filenames = [record_path]
        raw_dataset = tf.data.TFRecordDataset(filenames)
        parsed_dataset = raw_dataset.map(_decode_record)

        if training:
            return parsed_dataset.shuffle(1000, reshuffle_each_iteration=True) \
                                 .batch(batch_size, drop_remainder=False)
        else:
            return parsed_dataset.batch(batch_size, drop_remainder=False)

def _get_model_framework(model_type):
    if model_type in ["baseline"]:
        return "sklearn"
    if model_type in ["dstc7"]:
        return "tf"
    return None

def _get_model(data_dir, framework_type, model_type):
    if framework_type == "sklearn":
        return SklearnModel(name="baseline")
    elif framework_type == "tf":
        if model_type == "dstc7":
            # load the tokenizer to get the vocab size
            tokenizer = sp.SentencePieceProcessor()
            tokenizer.load(f"{data_dir}/spm.model")
            vocab_size = len(tokenizer)

            model = DSTC7FirstPlace(vocab_size)

            opt = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            # metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

            model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

            return model

def _predict_tf_model(data, model, threshold):
    probs = model.predict(data).sum(axis=1)
    y_pred = (probs >= threshold).astype(int)
    return y_pred, probs

def _load_model_to_predict(data_dir, model_type):
    framework_type = _get_model_framework(model_type)
    model = _get_model(data_dir, framework_type, model_type)

    if framework_type == "sklearn":
        model.load(data_dir)
    elif framework_type == "tf":
        # expect_partial for skipping the optimizer params
        model.load_weights(f"{data_dir}/cp.ckpt").expect_partial()

    return model

def _predict_model(model, model_type, data, threshold):
    framework_type = _get_model_framework(model_type)
    if framework_type == "sklearn":
        y_pred, probs = model.predict(data)
    elif framework_type == "tf":
        y_pred, probs = _predict_tf_model(data, model, threshold)

    return y_pred, probs

def evaluate(root_dir, data_dir, model_type, threshold=0.8):
    # output json
    out_json = {}
    out_json_file = f"{data_dir}/summary_{model_type}.json"

    # TODO: Create test set later
    # for now just use the train set
    data = get_data_by_type(data_dir, model_type, training=False)

    # load model
    model = _load_model_to_predict(data_dir, model_type)

    # get data
    framework_type = _get_model_framework(model_type)
    if framework_type == "sklearn":
        X, y = data[0], data[1]
    elif framework_type == "tf":
        # get the true y
        y = get_data_by_type(data_dir, "baseline")[1]
        X = data

    # normal metrics
    y_pred, probs = _predict_model(model, model_type, X, threshold)
    cls_report_json = classification_report(y, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y, probs)
    print ("roc_auc_score: ", roc_auc)
    out_json["classification_report"] = cls_report_json
    out_json["roc_auc_score"] = roc_auc

    ########################### calculate recall@k and MRR #######################
    df = pd.read_excel(f"{data_dir}/train.xlsx", encoding="utf-8")
    df = df.loc[df["label"] == True, :].reset_index()

    # load the tokenizer and responses
    tokenizer, responses, group_to_idx = _load_tokenizer_and_responses(
        root_dir, data_dir, model_type
    )

    # (num true queries, num responses)
    # df = df.iloc[:4, :].reset_index()
    probs_df = []
    for _, row in df.iterrows():
        X = create_chat_x(row["context"], responses, model_type, tokenizer)
        _, probs = _predict_model(model, model_type, X, threshold=0.0)
        probs_df.append(probs)

    df["probs"] = probs_df
    df["probs_sorted_index"] = df["probs"].apply(lambda probs: (-probs).argsort())

    # calculate mrr
    def mrr_row(row):
        found_i = row["probs_sorted_index"].tolist().index(group_to_idx[row["group_id"]])
        rank = found_i + 1
        return 1 / rank

    df["rr"] = df.apply(mrr_row, axis=1)
    mrr = df["rr"].values.mean()
    print (f"MRR: {mrr}")
    out_json["MRR"] = mrr

    # calculate recall@k(s)
    # for different @k(s)
    ks = [1, 2, 3, 4, 5]
    thresholds = [0.8 + i/100 for i in range(0, 10)]
    thresholds += [0.9 + i/100 for i in range(0, 10)]
    metrics_df = pd.DataFrame({
        "threshold": thresholds,
    })
    out_json["best_threshold_for_recall_at_k"] = {}
    for k in ks:
        mean_recalls = []
        for threshold in thresholds:
            def recall_at_k(row):
                num_greater_than_threshold = (row["probs"] >= threshold).sum()
                num_get = k if k <= num_greater_than_threshold else num_greater_than_threshold
                # check if the true response is in the top num_get responses
                return int(group_to_idx[row["group_id"]] in row["probs_sorted_index"][:num_get])

            # apply for every row
            col = f"recall_{threshold}_{k}"
            df[col] = df.apply(recall_at_k, axis=1)
            mean_recalls.append(df[col].values.mean())
            del df[col]

        metrics_df[f"recall@{k}"] = mean_recalls
        out_json["best_threshold_for_recall_at_k"][k] = \
            metrics_df["threshold"].values[metrics_df[f"recall@{k}"].values.argmax()]

    metrics_file = f"{data_dir}/metrics_{model_type}.xlsx"
    metrics_df.to_excel(metrics_file, index=None)
    print (f"Write file {metrics_file}")

    write_json_data(out_json_file, out_json)
    print (f"Write file {out_json_file}")

    # generate debug file
    debug_file = f"{data_dir}/debug.json"
    debug_json = []
    if os.path.exists(debug_file):
        debug_json = read_json_data(debug_file)
    else:
        debug_json = df.shape[0]*[{}]

    for idx, row in df.iterrows():
        debug_obj = debug_json[idx]
        debug_obj["context"] = row["context"]
        debug_obj["response"] = row["response"]
        debug_obj["is_in_recall_at_3"] = False
        debug_obj["is_in_recall_at_5"] = False

        debug_obj[model_type] = []
        for j, p_i in enumerate(row["probs_sorted_index"][:5]):
            debug_obj[model_type].append({
                "score": str(row["probs"][p_i]),
                "response": responses[p_i],
            })

            if responses[p_i] == row["response"]:
                if j < 3:
                    debug_obj["is_in_recall_at_3"] = True
                debug_obj["is_in_recall_at_5"] = True

    write_json_data(debug_file, debug_json)
    print (f"Write file {debug_file}")

def train(data_dir, model_type):
    data = get_data_by_type(data_dir, model_type)

    framework_type = _get_model_framework(model_type)
    model = _get_model(data_dir, framework_type, model_type)

    if framework_type == "sklearn":
        model.train(data)
        model.save(data_dir)
    elif framework_type == "tf":
        # create checkpoint callback
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{data_dir}/cp.ckpt",
            save_weights_only=True,
            save_best_only=True,
            verbose=1,
        )
        model.fit(data, validation_data=data, epochs=40, callbacks=[cp_callback])

def create_chat_x(context, responses, model_type, tokenizer):
    framework_type = _get_model_framework(model_type)
    if framework_type == "sklearn":
        return [context + " <sep> " + r for r in responses]
    elif framework_type == "tf":
        X = {
            "context": [],
            "response": [],
        }
        context_ids = tokenizer.EncodeAsIds(context)
        context_ids = padd_ids(context_ids, MAX_CONTEXT_LENGTH)
        for response in responses:
            response_ids = tokenizer.EncodeAsIds(response)
            response_ids = padd_ids(response_ids, MAX_RESPONSE_LENGTH)
            X["context"].append(context_ids)
            X["response"].append(response_ids)

        X = tf.data.Dataset.from_tensor_slices(X)
        X = X.batch(57, drop_remainder=False)
        return X

def _load_tokenizer_and_responses(root_dir, data_dir, model_type):
    # load the tokenizer to get the vocab size
    framework_type = _get_model_framework(model_type)
    tokenizer = None
    if framework_type == "tf":
        tokenizer = sp.SentencePieceProcessor()
        tokenizer.load(f"{data_dir}/spm.model")

    group_to_response = read_json_data(f"{root_dir}/group_to_response.json")
    responses = []
    group_to_idx = {}
    i = 0
    for g, r in group_to_response.items():
        responses.append(r)
        group_to_idx[g] = i
        i += 1

    return tokenizer, responses, group_to_idx

def chat(args):
    model_type = args.model_type

    max_turn = input("Enter max_turn model: ")
    print (f"You choose model {model_type}, max_turn {max_turn}")

    max_turn_data_dir = f"{args.data_dir}/train/max_turn_{max_turn}"

    # load model
    model = _load_model_to_predict(max_turn_data_dir, model_type)

    # load the tokenizer and responses
    tokenizer, responses, _ = _load_tokenizer_and_responses(
        args.data_dir, max_turn_data_dir, model_type
    )

    while True:
        text = input("You: ")
        if text == "exit":
            break

        X = create_chat_x(text, responses, model_type, tokenizer)
        y_pred, probs = _predict_model(model, model_type, X, threshold=0.8)

        indices = (-probs).argsort()
        for i in indices[:5]:
            print (f"Bot: {responses[i]} ({probs[i]})")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Wether to run training.",
    )

    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Wether to run evaluating."
    )

    parser.add_argument(
        "--do_chat",
        action="store_true",
        help="Do chat to test the bot."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="baseline",
        required=True,
        help="Choose the model: baseline, dstc7",
    )

    parser.add_argument(
        "--min_max_turn",
        type=int,
        required=False,
        default=1,
        help="Min max_turn data."
    )

    parser.add_argument(
        "--max_max_turn",
        type=int,
        required=False,
        default=2,
        help="Max max_turn data."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="data/sach_mem",
        help="The data folder."
    )

    args = parser.parse_args()

    if args.do_chat:
        chat(args)

    for max_turn in range(args.min_max_turn, args.max_max_turn + 1):
        data_dir = f"{args.data_dir}/train/max_turn_{max_turn}"

        if args.do_train:
            train(data_dir, args.model_type)

        if args.do_eval:
            evaluate(args.data_dir, data_dir, args.model_type)
