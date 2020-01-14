import argparse

import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)

from model import SklearnModel

MODELS_MAP ={
    "baseline": SklearnModel(name="baseline"),
}

def get_data_by_type(data_dir, model_type):

    if model_type == "baseline":
        df = pd.read_excel(f"{data_dir}/train.xlsx", encoding="utf-8")
        # we simply concat context and response
        texts = (df["context"] + " <sep> " + df["response"]).values
        return texts, df["label"].values
    elif model_type in ["dstc7"]:
        pass

def evaluate(data_dir, model_type):
    # TODO: Create test set later
    # for now just use the train set
    texts, y = get_data_by_type(data_dir, model_type)

    model = MODELS_MAP[model_type]

    model.load(data_dir)

    y_pred, probs = model.predict(texts)

    print (classification_report(y, y_pred))
    print ("roc_auc_core: ", roc_auc_score(y, probs[:,1]))

def train(data_dir, model_type):
    texts, y = get_data_by_type(data_dir, model_type)

    model = MODELS_MAP[model_type]

    model.train(texts, y)
    model.save(data_dir)

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
        "--data_folder",
        type=str,
        required=False,
        default="data/sach_mem/train",
        help="The data folder."
    )

    args = parser.parse_args()

    for max_turn in range(args.min_max_turn, args.max_max_turn + 1):
        data_dir = f"{args.data_folder}/max_turn_{max_turn}"

        if args.do_train:
            train(data_dir, args.model_type)

        if args.do_eval:
            evaluate(data_dir, args.model_type)
