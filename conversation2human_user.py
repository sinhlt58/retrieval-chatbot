import json
import pandas as pd

from utils import read_json_data, write_json_data
from domain import *

def pre_process_human_res_concat(output_folder, texts):
    return domain_pre_process_human_res_concat(output_folder, texts)

def pre_process_user_response(text):
    return domain_pre_process_user_response(text)

def conversation2human_user(input_folder, output_folder):
    conv_path = f"{input_folder}/conversations.json"
    conversations_dict = read_json_data(conv_path)

    # for human_res_concat for clustering
    conv_ids = []
    human_res_concat_ids = []
    human_res_concat_array = []
    human_res_concats = []
    human_res_concat_start_timestamps = []
    human_res_concat_end_timestamps = []
    human_res_concat_timestamps = []

    # for train_conversations.json
    train_conversations_dict = {}

    # create human_res_concat from conversations.json
    for conv_id, messages in conversations_dict.items():
        curr_human_res_concat_array = []
        curr_human_res_concat_timestamps = []

        # for train_conversations.json
        train_conversation = []
        curr_sub_train_conversation_timestamps = [] # user or human concat
        curr_sub_train_texts = [] # for user only, human we use template later

        # remove speak bot
        no_bot_messages = list(filter(lambda m: m["speaker"] != "bot", messages))

        n = len(no_bot_messages)
        for idx, message in enumerate(no_bot_messages):
            speaker = message["speaker"]

            if speaker == "human":
                curr_human_res_concat_array.append(message["text"])
                curr_human_res_concat_timestamps.append(message["timestamp"])

            if (idx == n - 1 or speaker != "human") and\
                curr_human_res_concat_array:

                conv_ids.append(conv_id)
                start = curr_human_res_concat_timestamps[0]
                end = curr_human_res_concat_timestamps[-1]
                human_res_concat_id = f'{conv_id}_{start}_{end}'
                curr_human_res_concat = " __eou__ ".join(curr_human_res_concat_array).strip()

                human_res_concat_ids.append(human_res_concat_id)
                human_res_concat_array.append(curr_human_res_concat_array)
                human_res_concats.append(curr_human_res_concat)
                human_res_concat_start_timestamps.append(start)
                human_res_concat_end_timestamps.append(end)
                human_res_concat_timestamps.append(curr_human_res_concat_timestamps)

                curr_human_res_concat_array = []
                curr_human_res_concat_timestamps = []

            # for train_conversations.json
            curr_sub_train_conversation_timestamps.append(
                message["timestamp"]
            )
            curr_sub_train_texts.append(message["text"])

            if (idx == n - 1 or ( idx < n - 1 and speaker != no_bot_messages[idx + 1]["speaker"])) and\
                curr_sub_train_conversation_timestamps:
                start = curr_sub_train_conversation_timestamps[0]
                end = curr_sub_train_conversation_timestamps[-1]
                res_concat_id = f"{conv_id}_{start}_{end}"

                if speaker == "human":
                    text = "later use cluster + template"
                else:
                    text = " __eou__ ".join(curr_sub_train_texts).strip()\
                    # pre-process user response
                    text = pre_process_user_response(text)

                train_conversation.append({
                    "speaker": speaker,
                    "timestamp_id": res_concat_id,
                    "text": text,
                })

                curr_sub_train_conversation_timestamps = []
                curr_sub_train_texts = []

        if conv_id not in train_conversations_dict:
            train_conversations_dict[conv_id] = {}
        train_conversations_dict[conv_id]["messages"] = train_conversation

    # write train_conversations.json
    train_conversations_file = f"{output_folder}/train_conversations.json"
    write_json_data(train_conversations_file, train_conversations_dict)
    print (f"Write file {train_conversations_file}")

    # create human_res_concat DataFrame
    df = pd.DataFrame({
        "user_id": conv_ids,
        "id": human_res_concat_ids,
        "human_res_concat_array": human_res_concat_array,
        "human_res_concat": human_res_concats,
        "start_timestamp": human_res_concat_start_timestamps,
        "end_timestamp": human_res_concat_end_timestamps,
        "timestamps": human_res_concat_timestamps,
    })

    # pre-process human_res_concat
    df["human_res_concat_pre"] = pre_process_human_res_concat(
        output_folder, df["human_res_concat"].values
    )

    # write file human_res_concat.xlsx
    df = df.sort_values(by="human_res_concat_pre")
    human_res_concat_path = f"{output_folder}/human_res_concat.xlsx"
    df.to_excel(human_res_concat_path, encoding="utf-8", index=False)
    print (f"Write file {human_res_concat_path}")

if __name__ == "__main__":
    conversation2human_user("data/sach_mem", "data/sach_mem")
