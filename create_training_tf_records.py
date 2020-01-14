import collections

import tensorflow as tf
import sentencepiece as sp
import pandas as pd


def _int64_feature(list_value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_value))

def serialize_example(context, response, label):
    feature = collections.OrderedDict()
    feature["context"] = _int64_feature(context)
    feature["response"] = _int64_feature(response)
    feature["label"] = _int64_feature(label)

    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    return tf_example.SerializeToString()

def create_records_for_max_turn(data_folder):
    train_file = f"{data_folder}/train.xlsx"
    tokenizer_model_file = f"{data_folder}/spm.model"

    # load train data to dataframe, sentencepiece model
    df = pd.read_excel(train_file, encoding="utf-8")
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.load(tokenizer_model_file)

    # create tf Dataset
    def gen():
        for idx, row in df.iterrows():
            yield serialize_example(
                tokenizer.EncodeAsIds(row["context"]),
                tokenizer.EncodeAsIds(row["response"]),
                [int(row["label"])],
            )

    # write the dataset
    serialized_dataset = tf.data.Dataset.from_generator(
        gen, output_types=tf.string, output_shapes=()
    )
    out_file = f"{data_folder}/train.tfrecord"
    writer = tf.data.experimental.TFRecordWriter(out_file)
    writer.write(serialized_dataset)
    print (f"Write file {out_file}")


def create_records_for_max_turns(input_folder, max_turns):
    for max_turn in max_turns:
        create_records_for_max_turn(
            data_folder=f"{input_folder}/max_turn_{max_turn}"
        )

if __name__ == "__main__":
    create_records_for_max_turns(
        input_folder="data/sach_mem/train",
        max_turns=[1, 2],
    )
