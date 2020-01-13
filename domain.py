import re
import string
import json

import pandas as pd
from urlextract import URLExtract
from nltk.tokenize import WhitespaceTokenizer
import demoji

from utils import read_json_data, write_json_data


demoji.download_codes()

# pre-process and create conversations.json stuff
class Entity:

    def __init__(self, type, text, start, end):
        self.type = type
        self.text = text
        self.start = start
        self.end = end


url_extractor = URLExtract()
email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
phone_patterns = [
    r'[0-9]{2,}\.[0-9]{2,}\.[0-9]{2,}\.[0-9]{2,}',
    r'\([0-9]{2,}\)[0-9]{2,}\.[0-9]{2,}\.[0-9]{2,}',
    r'[0-9]{2,}\.[0-9]{2,}\.[0-9]{2,}',
    r'[0-9]{2,} [0-9]{2,} [0-9]{2,}',
    r'[0-9]{2,}\.[0-9]{2,}',
]

whitespace_tokenizer = WhitespaceTokenizer()
chars_to_remove = string.punctuation.replace("_", "")
translator = str.maketrans(chars_to_remove, "".join([" "]*len(chars_to_remove)))

def build_entities(text, subs, entity_type):
    entities = []
    for sub in subs:
        start = text.find(sub)
        if start != -1:
            entity = Entity(
                type=entity_type,
                text=sub,
                start=start,
                end=start + len(sub),
            )
            entities.append(entity)
    return entities

def find_urls(text):
    urls = url_extractor.find_urls(text)
    entities = build_entities(text, urls, "url")

    return entities

def find_emails(text):
    match = re.findall(email_pattern, text)
    entities = build_entities(text, match, "email")

    return entities

def find_phones(text):
    match = []
    for phone_pattern in phone_patterns:
        m = re.findall(phone_pattern, text)
        if m:
            match = m
            break
    entities = build_entities(text, match, "phone")

    return entities

def replace_entities(text, entities_dict, entities):
    for e in entities:
        entity_index = entities_dict[e.type]["entity2index"][e.text]
        text = text.replace(e.text, entity_index)
    return text

def update_entities_dict(entity_dict, entity_value, entity_type):
    if "entity2index" not in entity_dict:
        entity_dict["entity2index"] = {}
        entity_dict["index2entity"] = {}
        entity_dict["count"] = {}

    if entity_value not in entity_dict["entity2index"]:
        new_index = f'__{entity_type}{len(entity_dict["entity2index"])}__'
        entity_dict["entity2index"][entity_value] = new_index
        entity_dict["index2entity"][new_index] = entity_value
        entity_dict["count"][new_index] = 0

    index = entity_dict["entity2index"][entity_value]
    entity_dict["count"][index] += 1

def add_entities_dict(entity_dict, entities, entity_type):
    for entity in entities:
        update_entities_dict(entity_dict, entity.text, entity_type)

def domain_pre_process_human_res_concat(output_folder, texts):
    pre_texts = []

    entities_dict = {
        "url": {},
        "email": {},
        "phone": {},
    }
    for text in texts:
        # find url
        url_entities = find_urls(text)
        add_entities_dict(entities_dict["url"], url_entities, "url")

        # find email
        email_entities = find_emails(text)
        add_entities_dict(entities_dict["email"], email_entities, "email")

        # find phone
        phone_entities = find_phones(text)
        add_entities_dict(entities_dict["phone"], phone_entities, "phone")

        # replace entity value with entity index
        entities = url_entities + email_entities + phone_entities
        pre_text = replace_entities(text, entities_dict, entities)

        # remove punctuation
        pre_text = " ".join(pre_text.split())
        pre_text = pre_text.translate(translator)
        pre_text = " ".join(pre_text.split())
        pre_text = pre_text.lower()

        if not pre_text:
            pre_text = "EMPTY"

        pre_texts.append(pre_text)

    # write entities.json
    entities_json_path = f"{output_folder}/entities.json"
    with open(entities_json_path, "w", encoding="utf-8") as f:
        json.dump(entities_dict, f, ensure_ascii=False, indent=4)
    print (f"Write file {entities_json_path}")

    # write entities.xlsx
    df_dict = {
        "type": [],
        "index": [],
        "entity": [],
        "count": [],
    }
    for entity_type, entity_dict in entities_dict.items():
        for index, entity in entity_dict["index2entity"].items():
            df_dict["type"].append(entity_type)
            df_dict["index"].append(index)
            df_dict["entity"].append(entity)
            df_dict["count"].append(entity_dict["count"][index])
    df = pd.DataFrame(df_dict).sort_values(by=["type", "entity", "count"])
    entities_xlsx_path = f"{output_folder}/entities.xlsx"
    df.to_excel(entities_xlsx_path, encoding="utf-8", index=False)
    print (f"Write file {entities_xlsx_path}")

    return pre_texts

def domain_pre_process_user_response(text):
    # remove emojis
    text = demoji.replace(text, " ")

    # remove tab, break line, spaces
    text = " ".join(text.split())

    # find emails and replace them with __email__ tokens
    email_entities = find_emails(text)
    for e in email_entities:
        text = text.replace(e.text, "__email__")

    # lower case
    text = text.lower()

    # remove punctuation
    text = text.translate(translator)
    text = " ".join(text.split())

    return text

# create train conversation stuff
def pick_group_response(df):
    df["res_length"] = df["human_res_concat_pre"].apply(lambda r: len(r))
    df = df.sort_values(by=["res_length"])

    count_df = df.groupby(["human_res_concat_pre"]).size().reset_index(name="count")
    count_df = count_df.sort_values(by=["count"], ascending=False)

    if count_df["count"][0] > 1:
        return count_df["human_res_concat_pre"].values[0]

    return df["human_res_concat_pre"].values[0]

def _does_contain(res_text, special_contains):
    for s in special_contains:
        if s in res_text:
            return True
    return False

def find_excluded_groups(group_to_response):
    r = []

    special_texts = ["not_text", "fall_back"]
    special_contains = ["→ đăng nhập → thêm sách → nhập mã số", "vô văn hóa"]

    for group_id, res_text in group_to_response.items():
        if res_text in special_texts or\
           group_id == "0_-1" or\
           _does_contain(res_text, special_contains): # this case is specific to the book code case
            r.append(group_id)
    return r

def is_valid_user_text(text):
    if "not_text" in text:
        return False
    return True

# clustering stuff
def get_clustering_data_level0(excel_file):
    df = pd.read_excel(excel_file, encoding="utf-8")
    return df["human_res_concat_pre"]

# write cluster bar graph count
def plot_to_file(df, x_key, rot, figsize, out_folder, file_name):
    out_path = f"{out_folder}/{file_name}.png"
    ax = df.plot.bar(x=x_key, rot=rot, figsize=figsize)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()))
    # count for each column legend
    columns = list(df.columns)
    columns.remove(x_key)
    legends = []
    for c in columns:
        s = df[c].values.sum()
        legend = "{} - {}".format(s, c)
        legends.append(legend)
    ax.legend(legends)

    fig = ax.get_figure()
    fig.savefig(out_path)
    print (f"Write file {out_path}")
