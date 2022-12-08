import datetime
from utils import *
import torch
import torch.nn as nn
from logger import get_logger
from training import Trainer
from transformers import AdamW,get_linear_schedule_with_warmup,BertModel,AutoConfig
from args import *

parser = get_parser()
args = parser.parse_args()

logger = get_logger(log_file=f"{args.embedding_type}_{args.mapping_strategy}_{args.privatization_strategy}_eps_{args.eps}_top_{args.top_k}_save_{args.save_stop_words}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.txt")
logger.info(f"{args.dataset}, args: {args}")

if __name__ == "__main__":
        parser = get_parser()
        args = parser.parse_args()

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        train_data,dev_data,test_data = load_data(args.dataset)

        if os.path.exists(f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt") and os.path.exists(f"./sim_word_dict/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt"):
                with open(f"./p_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt", 'r') as dic:
                        p_dict = json.load(dic)
                
                with open(f"./sim_word_dict/{args.embedding_type}/{args.mapping_strategy}/eps_{args.eps}_top_{args.top_k}.txt", 'r') as dic:
                        sim_word_dict = json.load(dic)
        else:
                sim_word_dict,p_dict = get_customized_mapping(eps = args.eps, top_k = args.top_k)
                
        if args.privatization_strategy == "s1":
                train_data = generate_new_sents_s1(df = train_data ,sim_word_dict = sim_word_dict ,p_dict = p_dict ,save_stop_words = args.save_stop_words)
                test_data = generate_new_sents_s1(df = test_data ,sim_word_dict = sim_word_dict ,p_dict = p_dict ,save_stop_words = args.save_stop_words,type="test")

        train_dataset = Bert_dataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        dev_dataset = Bert_dataset(dev_data)
        dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = Bert_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        logger.info(f"train_data:{len(train_data)},dev_data:{len(dev_data)},test_data:{len(test_data)}")

        model = BertForSequenceClassification.from_pretrained(
        args.model_type,
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False)

        optimizer = AdamW(model.parameters(),lr=args.lr,eps=1e-8)  
        
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=len(train_loader)*args.epochs)
        trainer = Trainer(
                model,
                scheduler,
                optimizer,
                args.epochs,
                args.log_steps,
                args.eval_steps,
                args.use_cuda,
                logger
                )

        trainer.train(train_loader, test_loader)

        # evaluate test dataset #
        acc = trainer.predict(test_loader)
        logger.info(f"test acc = {acc:.4f}.")