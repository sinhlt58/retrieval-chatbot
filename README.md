# Retrieval chatbot

## What is this probject?

This probject is an implementation of the [Sequential Attention-based Network for Noetic End-to-End Response Selection](http://workshop.colips.org/dstc7/papers/07.pdf) paper. We implemented the **Cross attention-based method** model (Figure 1 (b) in the paper) since we only need that model.

## Should I use this porject to develop a real-world chatbot to interact with real custumers on a small closed domain dataset?

Short answer. NO! It is better just using rule-based systems for real-world closed domain chatbots!

## Why this project?

The purpose of this project is attempting to use state of the art machine learning models to solve real-word problems. In the research perspective, it is a small step on the path to achieving truly AI chatbot.

## What dataset is used?

The dataset contains real-world conversations between users and customer support staff on Facebook **sachmem** page. After processing, clustering, generating; we end up with 4300 examples (1/3 true, 2/3 false). Each example contains a pair of context and response, the label for an example is true when the response is correct for the corresponding context. We create two datasets with max turn 1 and 2.

## Where are the train data files?

[Train dataset max turn 1](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_1/train.xlsx)

[Train dataset max turn 2](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_2/train.xlsx)

## How do I make sure the code is correct?

We try to overfit the train dataset to make sure the model is capable of capturing the patterns in the data. It is a reasonable way for checking bugs in the code.

## What are the results on the train dataset?

Currently, we use the train dataset for evaluating the model (just make sure the code is correct and testing out our workflow) since the dataset is quite small. We don't do cross-validation because it is quite slow. We will create the test set in future work.

Below are the resuls on the train dataset:
**baseline**: Logistic Regression model
**dstc7**: The cross-attention model

### For max turn 1 model

[baseline summary](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_1/summary_baseline.json)

[baseline recall at diffrent k](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_1/metrics_baseline.xlsx)

[dstc7 summary](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_1/summary_dstc7.json)

[dstc7 recall at diffrent k](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_1/metrics_dstc7.xlsx)

### For max turn 2 model

[baseline summary](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_2/summary_baseline.json)

[baseline recall at diffrent k](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_2/metrics_baseline.xlsx)

[dstc7 summary](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_2/summary_dstc7.json)

[dstc7 recall at diffrent k](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_2/metrics_dstc7.xlsx)

The most important metrics for this problem are:

**MRR** (Mean reciprocal rank): https://en.wikipedia.org/wiki/Mean_reciprocal_rank

**Recall@k** (Recall at k): https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54

## How do I debug predicted responses by the models for each context in the train data?

[max turn 1 debug](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_1/debug.json)

[max turn 2 debug](https://github.com/sinhlt58/retrieval-chatbot/blob/master/data/sach_mem/train/max_turn_2/debug.json)

## How do I run the project to reproduce the results?

### Train

``python run.py --do_train --model_type=baseline|dstc7``

### Eval

``python run.py --do_eval --model_type=baseline|dstc7``

### Test

``python run.py --do_chat --model_type=baseline|dstc7``

### References

http://workshop.colips.org/dstc7/papers/07.pdf

