import glob
import os

import pandas as pd
import numpy as np

from utils import write_json_data


def get_to_user_speaker(text, bot_texts):
    for bot_text in bot_texts:
        if bot_text in text:
            return "bot"
    return "human"


def create_conversation(df, bot_texts):
    conversation_id = None # user the user id
    messages = []

    for idx, row in df.iterrows():
        if conversation_id is None:
            conversation_id = row["user_id"].split(":")[-1]

        text = row["raw_text"]
        from_type = row["from"]

        if from_type == "USER":
            speaker = "user"
        else:
            speaker = get_to_user_speaker(text, bot_texts)

        message_dict = {
            "speaker": speaker,
            "timestamp": row["timestamp"],
            "text": text,
        }
        messages.append(message_dict)

    return conversation_id, messages


def logcsvs2json(input_folder, output_folder):
    csv_paths = glob.glob("{}/*.csv".format(input_folder))
    csv_paths.sort(key=os.path.getmtime)

    with open("{}/auto_responses.txt".format(output_folder), "r", encoding="utf-8") as f:
        bot_texts = f.read().split('\n')[:-1]

    conversations_dict = {}
    for csv_path in csv_paths:
        if os.name == "nt":  # on Windows
            split_char = "\\"
        else:  # on Lunix
            split_char = "/"
        file_name = csv_path.split(split_char)[-1].replace(".csv", "").strip()

        csv_df = pd.read_csv(csv_path, encoding="utf-8")
        print (csv_df.shape)
        csv_df_list = np.split(csv_df, csv_df[csv_df.isnull().all(1)].index)

        for df in csv_df_list:
            df = df.dropna(how='all')
            conv_id, messages = create_conversation(df, bot_texts)

            if conv_id not in conversations_dict:
                conversations_dict[conv_id] = []

            conversations_dict[conv_id].extend(messages)
            # sort by the timestamp for the case the user id is in multiple csv log files
            conversations_dict[conv_id].sort(key=lambda m: m["timestamp"])

    # write the conversations file
    conv_path = "{}/conversations.json".format(output_folder)
    write_json_data(conv_path, conversations_dict)
    print ("Write file {}".format(conv_path))
    print ("Number of conversation is {}".format(len(conversations_dict)))


if __name__ == "__main__":
    logcsvs2json("../rasabot/reports/data/sach_mem/labeled_data", "data/sach_mem")
