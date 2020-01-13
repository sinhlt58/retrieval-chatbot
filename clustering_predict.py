import sys
import pandas as pd
import pickle

from utils import write_json_data, create_folder
from domain import (
    plot_to_file,
    pick_group_response,
)


def get_predicted_labels(model_file):
    with open(model_file, "rb") as f:
        clustering = pickle.load(f)
    print (f"labels: {clustering.labels_}")
    return clustering.labels_

def create_group_to_response_files(
    group_excel_file,
    group_to_response_json_file,
    group_to_response_excel_file,
    previous_group_to_group_json_file=None,
):
    df = pd.read_excel(group_excel_file, encoding="utf-8")

    responses_dict_map = {}
    unique_groups = df["cluster_label"].unique()
    out_df_dict = {
        "id": [], # for next level clustering
        "previous_cluster_label": [],
        "human_res_concat_pre": [], # for next level clustering
    }
    for i, group in enumerate(unique_groups):
        group_res = pick_group_response(df[df["cluster_label"] == group].copy())
        responses_dict_map[f"{group}"] = group_res
        out_df_dict["id"].append(i)
        out_df_dict["previous_cluster_label"].append(group)
        out_df_dict["human_res_concat_pre"].append(group_res)

    write_json_data(group_to_response_json_file, responses_dict_map)
    print (f"Write file {group_to_response_json_file}")

    out_df = pd.DataFrame(out_df_dict)
    out_df.to_excel(group_to_response_excel_file, encoding="utf-8", index=False)
    print (f"Write file {group_to_response_excel_file}")

    if previous_group_to_group_json_file:
        group_to_group_dict = {}
        for i, row in df.iterrows():
            group_to_group_dict[f"{row['previous_cluster_label']}"] = str(row["cluster_label"])
        write_json_data(previous_group_to_group_json_file, group_to_group_dict, sort_keys=True)
        print (f"Write file {previous_group_to_group_json_file}")

def predict(
    model_file,
    in_excel_file,
    out_group_excel_file,
    out_id_to_group_json_file,
    out_group_to_response_json_file,
    out_group_to_response_excel_file,
    out_bar_graph_name,
    output_folder,
    out_previous_group_to_group_json_file=None,
):

    # Create output_folder if not existed
    create_folder(output_folder)

    # Read in_excel_file file.
    # Make sure the file contains human_res_concat_pre, id columns
    df = pd.read_excel(in_excel_file, encoding="utf-8")

    # Get predicted labels
    predicted_labels = get_predicted_labels(model_file)

    # 1. Write ouput the group excel file
    # Which adds the cluster_label column
    df["cluster_label"] = predicted_labels
    df = df.sort_values(by=["cluster_label", "human_res_concat_pre"])
    df.to_excel(out_group_excel_file, encoding="utf-8", index=False)
    print (f"Write file {out_group_excel_file}")

    # 2. Write id to group file which maps each id row to a group
    # For level 0 group, ids are human timestamp ids
    id_to_group_dict = {}
    for i, row in df.iterrows():
        id_to_group_dict[row["id"]] = str(row["cluster_label"])
    write_json_data(out_id_to_group_json_file, id_to_group_dict)
    print (f"Write file {out_id_to_group_json_file}")

    # 3, 4. Write group to response files
    # For each group, we pick a response that represents the entire group using heristic
    create_group_to_response_files(
        group_excel_file=out_group_excel_file,
        group_to_response_json_file=out_group_to_response_json_file,
        group_to_response_excel_file=out_group_to_response_excel_file,
        previous_group_to_group_json_file=out_previous_group_to_group_json_file,
    )

    # 5. Write the bar graph images result, that counts the number of datapoint
    # for each group
    label_count_df = df.groupby(["cluster_label"]).size().reset_index(name="count")
    label_count_df = label_count_df.sort_values(by="count", ascending=False)
    plot_to_file(label_count_df, "cluster_label", 90, (80, 6), output_folder, out_bar_graph_name)

def predict_level0():
    predict(
        model_file="data/sach_mem/optics_clustering_level0.pkl",
        in_excel_file="data/sach_mem/human_res_concat.xlsx",
        out_group_excel_file="data/sach_mem/predict_level0/group.xlsx",
        out_id_to_group_json_file="data/sach_mem/predict_level0/id_to_group.json",
        out_group_to_response_json_file="data/sach_mem/predict_level0/group_to_response.json",
        out_group_to_response_excel_file="data/sach_mem/predict_level0/group_to_response.xlsx",
        out_bar_graph_name="optics",
        output_folder="data/sach_mem/predict_level0",
        out_previous_group_to_group_json_file=None,
    )

def predict_level1():
    predict(
        model_file="data/sach_mem/optics_clustering_level1.pkl",
        in_excel_file="data/sach_mem/predict_level0/group_to_response.xlsx",
        out_group_excel_file="data/sach_mem/predict_level1/group.xlsx",
        out_id_to_group_json_file="data/sach_mem/predict_level1/id_to_group.json",
        out_group_to_response_json_file="data/sach_mem/predict_level1/group_to_response.json",
        out_group_to_response_excel_file="data/sach_mem/predict_level1/group_to_response.xlsx",
        out_bar_graph_name="optics",
        output_folder="data/sach_mem/predict_level1",
        out_previous_group_to_group_json_file="data/sach_mem/predict_level1/group_to_group.json"
    )

if __name__ == "__main__":
    args = sys.argv

    if args[1] == "level0":
        predict_level0()
    elif args[1] == "level1":
        predict_level1()
