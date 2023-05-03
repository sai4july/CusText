# Introduction
This repo contains source code for "A Customized Text Sanitization Mechanism with Differential Privacy" (accepted to ACL Findings 2023 )

# How to get MedSTS dataset?
The MedSTS dataset is from the paper: MedSTS: A Resource for Clinical Semantic Textual Similarity https://arxiv.org/pdf/1808.09397.pdf \
This is a medical dataset and cannot be directly downloaded. You can contact the first author of the paper for the access to the dataset.

# How to run CusText?

 python main.py \
  --dataset sst2 \
  --epsilon 1.0 \
  --top_k 20 \
  --embedding_type ct_vectors 

# How to run CusText+?

 python main.py \
  --dataset sst2 \
  --epsilon 1.0 \
  --top_k 20 \
  --embedding_type ct_vectors \
  --save_stop_words True
