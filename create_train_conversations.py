import random

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import (
    TfidfVectorizer
)
from sklearn.metrics.pairwise import (
    euclidean_distances
)

from utils import read_json_data, write_json_data, create_folder
from domain import(
    find_excluded_groups,
    is_valid_user_text,
)


def create_sim_matrix(group_to_response, excluded_groups):
    """
        Create group to response matrix for k-nn
        for negative negative sampling

        :return: sim matrix, look up group index
    """
    # create lookup index for group id
    group2index = {}
    idx = 0
    group_responses = []
    for group, response in group_to_response.items():
        if group in excluded_groups:
            continue

        group2index[group] = idx
        group_responses.append(response)
        idx += 1
    index2group = {v: k for k, v in group2index.items()}

    # featurizer
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(group_responses).toarray()

    # calculate sim matrix
    sim_matrix = euclidean_distances(X, X)

    # get indices of sorted sim matrix
    # here we sort with descending order
    sim_matrix = np.argsort(-sim_matrix, axis=1)

    return sim_matrix, group2index, index2group

def generate_negative_samples(
    sim_matrix,
    group,
    group2index,
    index2group,
    k=2,
    gen_range=0.8,
):
    idx = group2index[group]
    n = len(sim_matrix[0])
    row = sim_matrix[idx]

    # here we cut off (1 - gen_range) percentage
    # samples that are too similar to the group
    n_sample_range = int(gen_range*n)
    row = row[:n_sample_range]

    # random k times
    negative_groups = []
    for i in range(k):
        sample_idx = random.choice(row)
        negative_groups.append(index2group[sample_idx])

    return negative_groups

def find_sub_train_conversation(
    train_conversations_file,
    timestime_id_to_group_file,
    group_to_response_file,
    output_folder,
    max_turns=[1, 2],
    k=2,
    gen_range=0.8,
):
    train_conversations_dict = read_json_data(train_conversations_file)
    timestime_id_to_group = read_json_data(timestime_id_to_group_file)
    group_to_response = read_json_data(group_to_response_file)

    # find groups to exclude like NOT_TEXT, -1, ...
    excluded_groups = find_excluded_groups(group_to_response)
    print (f"Exclude groups: {excluded_groups}")

    # max_turn counts
    max_turns_counts = [0]*len(max_turns)
    num_invalid_user_texts = 0
    max_conversation_len = 0

    # prepare sim matrix for negative sampling later
    sim_matrix, group2index, index2group = create_sim_matrix(
        group_to_response,
        excluded_groups,
    )

    # training data dataframes
    max_turn_dfs = {}

    # update human response to template response in train_conversations.json
    # generate sub conversation for training examples
    for user_id, data in train_conversations_dict.items():
        messages = data["messages"]

        if len(messages) > max_conversation_len:
            max_conversation_len = len(messages)

        # empty the max turn data
        for max_turn in max_turns:
            max_turn_key = f"max_turn_{max_turn}"
            data[max_turn_key] = []
            if max_turn not in max_turn_dfs:
                max_turn_dfs[max_turn] = []

        # update human response to template response
        for i, message in enumerate(messages):
            if message["speaker"] == "human":
                group_in = timestime_id_to_group[message["timestamp_id"]]
                response_template = f'__response_{group_in}__'
                message["text"] = response_template
                group_text = group_to_response[group_in]
                message["group_text"] = group_text
                message["group_id"] = group_in

                # we skip human responses that are in excluded groups
                # or the first message is from human
                if group_in in excluded_groups or\
                    i == 0:
                    continue

                # generate sub conversations for training
                # also do negative sampling
                for max_turn in max_turns:
                    max_turn_key = f"max_turn_{max_turn}"

                    # we generate the context for the response
                    # depends on the max_turn
                    context_texts = []
                    max_num_context_texts = 1 + (max_turn - 1)*2
                    min_j = i - max_num_context_texts
                    min_j = 0 if min_j < 0 else min_j
                    for j in reversed(range(min_j, i)):
                        speaker = messages[j]["speaker"]
                        if speaker == "user" and is_valid_user_text(messages[j]["text"]):
                            context_texts.append(messages[j]["text"])
                        elif speaker == "human" and\
                            messages[j]["group_id"] not in excluded_groups:
                            context_texts.append(messages[j]["group_text"])
                        else:
                            # NOTE: here we make sure the context always start with the user
                            # if we break at this message and the previous is from human
                            # then we remove the latest text in the context_texts
                            if speaker == "user": # which means the previous text is from human
                                context_texts = context_texts[:-1]
                            break

                    # generate context, response pair training samples
                    if context_texts:
                        context = " __eot__ ".join(reversed(context_texts))
                        sample_groups = [group_in]
                        sample_labels = [True]

                        if not is_valid_user_text(context):
                            num_invalid_user_texts += 1

                        # generate k negative samples
                        negative_groups = generate_negative_samples(
                            sim_matrix=sim_matrix,
                            group=group_in,
                            group2index=group2index,
                            index2group=index2group,
                            k=k,
                            gen_range=gen_range,
                        )
                        # add negative samples
                        sample_groups += negative_groups
                        sample_labels += k*[False]

                        for g, l in zip(sample_groups, sample_labels):
                            sample_dict = {
                                "context": context,
                                "response": group_to_response[g],
                                "label": l,
                                "group_id": g,
                            }
                            data[max_turn_key].append(sample_dict)
                            max_turn_dfs[max_turn].append(sample_dict)
                        max_turns_counts[max_turn - 1] += 1

    for i, c in enumerate(max_turns_counts):
        print (f"Max turn {i + 1} has {c} samples")
    print ("num_invalid_user_texts: ", num_invalid_user_texts)
    print ("max_conversation_len: ", max_conversation_len)

    write_json_data(train_conversations_file, train_conversations_dict)
    print (f"Write update file {train_conversations_file}")

    # Write max turn training dataframes, .txt for sentencepice
    create_folder(f"{output_folder}/train")
    for max_turn, data in max_turn_dfs.items():
        out_turn_folder = f"{output_folder}/train/max_turn_{max_turn}"
        create_folder(out_turn_folder)

        # write df training data
        out_file_df = f"{out_turn_folder}/train.xlsx"
        df = pd.DataFrame.from_records(data)
        df.to_excel(out_file_df, encoding="utf-8", index=False)
        print (f"Write file {out_file_df}")

        # write .txt for sentencepiece
        texts = df["context"].tolist() + df["response"].tolist()
        out_file_texts = f"{out_turn_folder}/texts.txt"
        with open(out_file_texts, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        print (f"Write file {out_file_texts}")


def create_final_group_to_response(
    id_to_group_level0_file, # timestamp id to group level0
    group_to_response_level0_file,
    group_to_group_file,
    group_to_response_level1_file,
    output_folder
):
    """
    Combine group_to_response level0 and group_to_response level1
    and excluded groups
    group level0 -> group level1 -> group response
    """
    id_to_group = read_json_data(id_to_group_level0_file)
    group_to_response_level0 = read_json_data(group_to_response_level0_file)
    group_to_group = read_json_data(group_to_group_file)
    group_to_response_level1 = read_json_data(group_to_response_level1_file)

    out_id_to_group_file = f"{output_folder}/id_to_group.json"
    out_group_to_response_file = f"{output_folder}/group_to_response.json"

    final_group_to_response = {}

    def get_new_group_to_response_pair(group0):
        group1 = group_to_group[group0]
        if group1 == "-1": # keep the group level 0
            return f"0_{group0}", group_to_response_level0[group0]
        else: # use the response from group level 1
            return f"1_{group1}", group_to_response_level1[group1]

    for group0, res0 in group_to_response_level0.items():
        new_group, new_response = get_new_group_to_response_pair(
            group0
        )
        final_group_to_response[new_group] = new_response

    write_json_data(out_group_to_response_file, final_group_to_response)
    print (f"Write file {out_group_to_response_file}")

    # create new timestampe to group
    new_id_to_group = {}
    for timestamp_id, group0 in id_to_group.items():
        new_group, _ = get_new_group_to_response_pair(
            group0
        )
        new_id_to_group[timestamp_id] = new_group

    write_json_data(out_id_to_group_file, new_id_to_group)
    print (f"Write file {out_id_to_group_file}")

if __name__ == "__main__":
    create_final_group_to_response(
        id_to_group_level0_file="data/sach_mem/predict_level0/id_to_group.json",
        group_to_response_level0_file="data/sach_mem/predict_level0/group_to_response.json",
        group_to_group_file="data/sach_mem/predict_level1/group_to_group.json",
        group_to_response_level1_file="data/sach_mem/predict_level1/group_to_response.json",
        output_folder="data/sach_mem",
    )

    find_sub_train_conversation(
        train_conversations_file="data/sach_mem/train_conversations.json",
        timestime_id_to_group_file="data/sach_mem/id_to_group.json",
        group_to_response_file="data/sach_mem/group_to_response.json",
        output_folder="data/sach_mem",
        max_turns=[1, 2],
        k=2,
        gen_range=0.8,
    )
