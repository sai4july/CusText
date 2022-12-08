import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # ---training params---
    parser.add_argument("--log_path", type=str, default="./log")
    parser.add_argument("--dataset", type=str, default="qnli")
    parser.add_argument("--save_path", type=str, default="./trained_model") 
    parser.add_argument("--model_type", type=str, default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_labels", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len",type=int,default=128)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--num_workers",type=int,default=os.cpu_count())
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--log_steps",type=int,default=500) 
    parser.add_argument("--eval_steps",type=int,default=500) 

    # ---CusText params---
    parser.add_argument("--eps", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--embedding_type", type=str, default="ct_vectors")
    parser.add_argument("--mapping_strategy", type=str, default="conservative")
    parser.add_argument("--privatization_strategy", type=str, default="s1")
    parser.add_argument("--save_stop_words", type=bool, default=False)
    return parser