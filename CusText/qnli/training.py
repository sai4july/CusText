import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils import *


class Trainer:
    def __init__(
        self,
        model,
        scheduler,
        optimizer,
        n_epochs,
        log_steps,
        eval_steps,
        use_cuda=True,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.logger = logger

        self.dev_scores = []

        self.train_epoch_losses = [] 
        self.train_step_losses = []  
        self.dev_losses = []
        
        self.best_score = 0

        if self.device == "cuda":
            self.model.cuda()

    def train(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """
        self.model.train()
        if val_loader is not None:
            init_val_acc = self.evaluate(val_loader)
            self.logger.info(f"Init_val_acc: {init_val_acc:5f}")
            self.best_score = init_val_acc
            
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        num_training_steps = self.n_epochs * len(train_loader)

        global_step = 0
        
        self.logger.info(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()


        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            tr_loss = 0
            tr_examples = 0
            for batch_data in tqdm(train_loader):
                batch_data = tuple(data.to(self.device) for data in batch_data) 
                inputs_ids, inputs_masks,token_type_ids,inputs_labels = batch_data
                self.optimizer.zero_grad()
                outputs = self.model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks, labels=inputs_labels)
                loss = outputs['loss']
                tr_loss += loss.item()
                tr_examples += inputs_ids.size(0)
                self.train_step_losses.append((global_step,tr_loss / tr_examples))
                if self.log_steps and global_step  % self.log_steps == 0:
                    self.logger.info(f"[Train] epoch:{epoch+1}/{self.n_epochs}, step: {global_step}/{num_training_steps}, step_loss:{loss.item():.4f}")
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                global_step += 1
    
                if val_loader is not None and self.eval_steps > 0 and global_step != 0 and \
                    (global_step % self.eval_steps == 0 or global_step == (num_training_steps - 1)):
                    val_acc= self.evaluate(val_loader)
                    self.logger.info(f"[Evaluate] epoch:{epoch+1}/{self.n_epochs}, step: {global_step}/{num_training_steps}, val_acc:{val_acc:.5f}")

                    self.model.train()

                    if val_acc > self.best_score:
                        self.save_model()
                        self.logger.info(f"[Evaluate] best accuracy performance has been updated: {self.best_score:.5f} -> {val_acc:.5f}")
                        self.best_score = val_acc

            epoch_time = time.time() - epoch_start
            s = (
                f"[Epoch {epoch + 1}] "
                f"train_epoch_loss = {tr_loss / tr_examples:.4f}, "
            )

            if val_loader is not None:
                val_acc= self.evaluate(val_loader)
                s += (
                    f" ---- val_acc = {val_acc:.4f}, "
                )
            s += f" [{epoch_time:.1f}s]"
            self.logger.info(s)

        train_time = int(time.time() - train_start)
        self.logger.info(f"-- Training done in {train_time}s.")


    def evaluate(self, data_loader):
        self.model.eval()
        y_list = []
        y_hat_list = []

        for batch in tqdm(data_loader):
            batch = tuple(data.to(self.device) for data in batch)
            inputs_ids, inputs_masks,token_type_ids,inputs_labels = batch
            with torch.no_grad():
                preds = self.model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks) # 模型预测
            y_list.extend(inputs_labels.detach().cpu().numpy())
            y_hat_list.extend(preds['logits'].detach().cpu().numpy())

        y_list = np.array(y_list)
        y_hat_list = np.array(y_hat_list)
        preds = np.argmax(y_hat_list, axis=1).flatten() # shape = (1, :)
        labels = y_list.flatten()
        acc = np.sum(preds==labels) / len(y_list)
        return acc
    
    def predict(self,data_loader):
        self.model.load_state_dict(torch.load(f'{args.save_path}/{args.embedding_type}_{args.mapping_strategy}_{args.privatization_strategy}_eps_{args.eps}_top_{args.top_k}_save_{args.save_stop_words}_model.pkl'))
        self.model.eval()
        y_list = []
        y_hat_list = []
        for batch in tqdm(data_loader):
            batch = tuple(data.to(self.device) for data in batch)
            inputs_ids, inputs_masks,token_type_ids,inputs_labels = batch
            with torch.no_grad():
                preds = self.model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks) 
            y_list.extend(inputs_labels.detach().cpu().numpy())
            y_hat_list.extend(preds['logits'].detach().cpu().numpy())

        y_list = np.array(y_list)
        y_hat_list = np.array(y_hat_list)
        preds = np.argmax(y_hat_list, axis=1).flatten() # shape = (1, :)
        labels = y_list.flatten()
        acc = np.sum(preds==labels) / len(y_list)
        return acc
        
    def save_model(self):
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        torch.save(self.model.state_dict(),f'{args.save_path}/{args.embedding_type}_{args.mapping_strategy}_{args.privatization_strategy}_eps_{args.eps}_top_{args.top_k}_save_{args.save_stop_words}_model.pkl')


