# Introduction
This is an anonymous_GitHub_repository for ACL Rolling Review.

# How to run CusText

 python main.py \
  --dataset "sst2" \
  --epsilon 1.0 \
  --top_k 20 \
  --embedding_type ct_vectors \

# How to run CusText+ 

 python main.py \
  --dataset "sst2" \
  --epsilon 1.0 \
  --top_k 20 \
  --embedding_type ct_vectors \
  --save_stop_words True
