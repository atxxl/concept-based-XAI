import os
import rich
import torch
import wandb 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm     import tqdm
from torch    import nn
from torch.optim  import AdamW, lr_scheduler
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau


class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    # Preprocess text (username and link placeholders)
    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)    
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.preprocess(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True,
        )
        
        return {
          'text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'labels': torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model, device, test_loader):
    model.eval()
    total, correct = 0, 0
    total_val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            total_val_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_val_loss = total_val_loss / len(test_loader)
    # f1 = f1_score(all_labels, all_preds, average='macro') # balanced
    
    return accuracy, avg_val_loss


def __main__():    
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Set the environment variable for w&b
    os.environ["WANDB_DIR"] = "/work3/s216410/"
    
    # Load the BERTweet model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print(f"RoBERTa_sentiment on {device} w. balanced dataset.")

    # Adjust the configuration to have 5 classes instead of the original number
    config  = AutoConfig.from_pretrained(model_name, num_labels=5) # Adjust num_labels to 5
    # roberta = AutoModelForSequenceClassification.from_pretrained(model_name, 
    #                                                              config=config,
    #                                                              ignore_mismatched_sizes=True)
    # roberta.to(device)
    # rich.print(f"RoBERTa loaded on [red]{device}.")
    
    # Load the datasets
    df_train = pd.read_csv("/zhome/81/3/170259/data/twitter-2016train-CE.tsv", 
                           sep='\t', header=None, quoting=3)
    df_test  = pd.read_csv("/zhome/81/3/170259/data/twitter-2016test-CE.tsv", 
                           sep='\t', header=None, quoting=3)
    df_tweet = pd.concat([df_train, df_test], ignore_index=True)
    df_tweet['converted_score'] = df_tweet[2] + 2
    
    # Resample each class to have the same number of the smallest class
    class_counts = df_tweet['converted_score'].value_counts()
    smallest_class_size = class_counts.min()
    final_sample = pd.DataFrame() 

    for class_value in class_counts.index:
        df_class = df_tweet[df_tweet['converted_score'] == class_value]
        df_class_sampled = df_class.sample(n=smallest_class_size, 
                                           random_state=42, replace=False)
        final_sample = pd.concat([final_sample, df_class_sampled], axis=0)

    # Fix the final sample for training and testing
    final_sample.reset_index(drop=True, inplace=True)
    df_train, df_test = train_test_split(final_sample, test_size=0.2, 
                                         random_state=42, 
                                         stratify=final_sample['converted_score'])

    # # Save the sampled dataset
    # final_sample.to_csv('balanced_goemo.csv', index=False)

    # Create datasets
    dataset = TwitterDataset(
        texts=df_train[3].values,
        labels=df_train['converted_score'].values,
        tokenizer=tokenizer,
        max_len=512     #768 for the spareCBM
    )
    
    test_dataset = TwitterDataset(
        texts=df_test[3].values,
        labels=df_test['converted_score'].values,
        tokenizer=tokenizer,
        max_len=512
    )

    # # Grid search parameters

    # best_accuracy = 0
    # best_params = {}
    # param_combo = {}
    
    # epochs = 20
    # for lr in learning_rates:
    #     for batch_size in batch_sizes:
    #         run = wandb.init(project="RoBERTa_sentiment_balanced_1", # change the w&b log here
    #                          reinit=True,
    #                          config={"epochs": epochs,
    #                                 "learning_rate": lr,
    #                                 "batch_size": batch_size,
    #             })
            
    #         model = AutoModelForSequenceClassification.from_pretrained(model_name, 
    #                                                                     config=config,
    #                                                                     ignore_mismatched_sizes=True)
    #         train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #         test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #         optimizer = AdamW(model.parameters(), lr=lr)
    #         # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.3)
    #         total_loss = 0
    #         model.train()
    #         model.to(device)
            
    #         accuracies = []
    #         best_in_loop = 0
    #         rich.print(f"lr: [green]{lr} [black]batch size: [blue]{batch_size}")
            
    #         for epoch in range(epochs):
    #             loop = tqdm(train_loader, leave=True)
    #             for batch in loop:
    #                 input_ids = batch['input_ids'].to(device)
    #                 attention_mask = batch['attention_mask'].to(device)
    #                 labels = batch['labels'].to(device)
                    
    #                 optimizer.zero_grad()
    #                 outputs = model(input_ids, attention_mask)
    #                 loss = nn.CrossEntropyLoss()(outputs.logits , labels)
    #                 loss.backward()
    #                 optimizer.step()
    #                 total_loss += loss.item()

    #                 # Update tqdm description
    #                 loop.set_description(f'Epoch {epoch+1}/{epochs}')
    #                 loop.set_postfix(loss=loss.item())
    #                 # scheduler.step()
                    
    #                 # Evaluate model performance
    #                 avg_loss = total_loss / len(train_loader)
    #                 accuracy, _ = evaluate_model(model, device, test_loader)
    #                 accuracies.append(accuracy)
    #                 if accuracy > best_accuracy:
    #                     best_accuracy = accuracy
    #                     best_in_loop = accuracy
    #                     best_params = {'learning_rate': lr, 'batch_size': batch_size}
    #                 elif accuracy > best_in_loop:
    #                     best_in_loop = accuracy
    #                 run.log({"accuracy": accuracy, "train-loss": loss, "avg_loss": avg_loss})
                    
    #         print(f'Finish with Total Loss: {total_loss / len(train_loader)}, Best Accuarcy: {best_in_loop}')

    #         param_key = f"LR: {lr}, Batch: {batch_size}"
    #         param_combo[param_key] = accuracies
            
    #         run.config.update({
    #             "best_learning_rate": best_params["learning_rate"],
    #             "best_batch_size": best_params["batch_size"],
    #             "best_accuracy": best_accuracy
    #             })
    #         run.finish()

    # print(f"Best Params: {best_params}, Best Accuracy: {best_accuracy}")

    # enable when not using grid search 
    # best_lr = best_params['learning_rate']
    # best_batch_size = best_params['batch_size']
    best_lr = 5e-05
    best_batch_size = 64
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                               config=config,
                                                               ignore_mismatched_sizes=True)
    train_loader = DataLoader(dataset, batch_size=best_batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=best_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
    model.to(device)
    rich.print(f"RoBERTa loaded on [red]{device}.")
    
    # Training loop
    epochs = 20
    model.train()
    best_accuracy = 0
    rich.print(f"lr: [green]{best_lr} [black]batch size: [blue]{best_batch_size}")
    
    run = wandb.init(project="RoBERTa_sentiment_balanced_best", # change the w&b log here
                     reinit=False,
                     config={"epochs": epochs,
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "batch_size": best_batch_size,
                    })
       
    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        total_loss = 0
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Update tqdm description
            loop.set_description(f'Epoch {epoch+1}/{epochs}')
            loop.set_postfix(loss=loss.item(),lr=optimizer.param_groups[0]['lr'])
            # scheduler.step()
            
            # Evaluate model performance
            avg_loss = total_loss / len(train_loader)
            accuracy, val_loss = evaluate_model(model, device, test_loader)
            scheduler.step(val_loss)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            run.log({"accuracy": accuracy, "train-loss": loss.item(), "avg_loss": avg_loss,
                      "lr": optimizer.param_groups[0]['lr'], "epoch": epoch})
            
    print(f'Finish with Avg. Loss: {avg_loss}, Best Accuarcy: {best_accuracy}')

    # model saving
    path = "/zhome/81/3/170259/models/"
    os.makedirs(path, exist_ok=True)
    name = "retrained_RoBERTa_sentiment_balanced_acc_" + str(round(accuracy, 3)) + ".pth"
    full_model_path = os.path.join(path, name)
    torch.save(model, full_model_path)
    print(f"Model saved to: {full_model_path}")
    
if __name__ == "__main__":
    __main__()