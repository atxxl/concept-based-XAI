import os, rich
import torch
import numpy  as np
import pandas as pd
import random as rnd

from tqdm     import tqdm
from torch    import nn
from torch.optim  import AdamW
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

class BertweetClassifier(nn.Module):
    def __init__(self, bertweet_model, num_labels):
        super(BertweetClassifier, self).__init__()
        self.bertweet = bertweet_model
        torch.manual_seed(seed=42)
        self.classifier = nn.Linear(self.bertweet.config.hidden_size, 
                                    num_labels)
        
    def forward(self, input_ids, attention_mask=None, return_logits=True):
        outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        if return_logits: # Use for training with cross-entropy loss
            return logits  
        else:  # +1 to adjust class range to 1-5
            predictions = torch.argmax(logits, dim=1) + 1 
            return predictions

    
class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
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
        
        
def evaluate_model(model, device, val_loader):
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            preds = model(input_ids, attention_mask, return_logits=False)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average='macro') # balanced
    # f1 = f1_score(all_labels, all_preds, average='weight') # unbalance

    return accuracy, f1


def __main__():    
    # Set the seed for reproducibility
    # seed = 42
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # rnd.seed(seed)

    # # If using CUDA
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    # Load the BERTweet model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bertweet  = AutoModel.from_pretrained("vinai/bertweet-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    bertweet.to(device)
    rich.print(f"BERTweet loaded on [red]{device}.")       
    
    # Load the datasets
    df_train = pd.read_csv("/zhome/81/3/170259/data/twitter-2016train-CE.tsv", 
                           sep='\t', header=None, quoting=3)
    df_test  = pd.read_csv("/zhome/81/3/170259/data/twitter-2016test-CE.tsv", 
                           sep='\t', header=None, quoting=3)
    df_tweet = pd.concat([df_train, df_test], ignore_index=True)
    df_tweet['converted_score'] = df_tweet[2] + 3
    # df_train['converted_score'] = df_train[2] + 3
    # df_test['converted_score']  = df_test[2] + 3
    
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
                                         random_state=42, stratify=final_sample['converted_score'])

    # # Save the sampled dataset
    # final_sample.to_csv('balanced_goemo.csv', index=False)

    # Create datasets
    dataset = TwitterDataset(
        texts=df_train[3].values,
        labels=df_train['converted_score'].values,
        tokenizer=tokenizer,
        max_len=130 #768 for the spareCBM
    )
    
    test_dataset = TwitterDataset(
        texts=df_test[3].values,
        labels=df_test['converted_score'].values,
        tokenizer=tokenizer,
        max_len=130
    )

    # Grid search parameters
    learning_rates = [5e-2, 1e-2, 5e-3, 1e-3, 5e-5, 1e-5]
    batch_sizes = [8, 16, 24, 32, 64]
    best_accuracy = 0
    best_params = {}
    
    # # Adjusting class weights when using the origin train dataset(unbalanced)
    # class_counts = df_train['converted_score'].value_counts()
    # total_samples = sum(class_counts)
    # # print(class_counts, total_samples)
    # # num_classes = len(class_counts)
    # weights = [total_samples / class_counts[label] for label in class_counts.index]

    # class_weights = torch.tensor(weights, device=device, dtype=torch.float)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    epochs = 10
    for lr in learning_rates:
        for batch_size in batch_sizes:
            model = BertweetClassifier(bertweet, num_labels=5)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            optimizer = AdamW(model.parameters(), lr=lr)
            total_loss = 0
            model.to(device)
            model.train()
            
            rich.print(f"lr: [green]{lr} [black]batch size: [blue]{batch_size}")
            
            for epoch in range(epochs):
                loop = tqdm(train_loader, leave=True)
                for batch in loop:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device) - 1
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    # loss = criterion(outputs, labels) # for unbalanced data
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    # Update tqdm description
                    loop.set_description(f'Epoch {epoch+1}/{epochs}')
                    loop.set_postfix(loss=loss.item())

            # Evaluate model performance
            accuracy, _ = evaluate_model(model, device, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                
            print(f'Finish with Total Loss: {total_loss / len(train_loader)}, Accuarcy: {accuracy}')

    print(f"Best Params: {best_params}, Best Accuracy: {best_accuracy}")
    
    
    # # enable when not using grid search 
    # best_lr = best_params['learning_rate']
    # best_batch_size = best_params['batch_size']
    # model  = BertweetClassifier(bertweet, num_labels=5)
    # train_loader = DataLoader(dataset, batch_size=best_batch_size, shuffle=True)
    # test_loader  = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=True)
    # optimizer = AdamW(model.parameters(), lr=best_lr)
    # model.to(device)
    
    # # Training loop
    # epochs = 20
    # model.train()
    # problematic_batch_indices = []
    
    # # early stopping
    # patience  = 3
    # min_delta = 0.01
    # best_score = None
    # epochs_no_improve = 0
    # early_stop = False

    # for epoch in range(epochs):
    #     loop = tqdm(train_loader, leave=True)
    #     all_preds, all_labels = [], []
    #     if early_stop:
    #         rich.print(f"[red]Stopped Early last time.")
    #         model.load_state_dict(torch.load('best_model.pth'))
    #         break
        
    #     for batch_idx, batch in enumerate(loop):
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device) - 1  # Ensure labels are 0-indexed for cross-entropy

    #         try:
    #             outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    #             loss = nn.functional.cross_entropy(outputs, labels)
    #             # loss = criterion(outputs, labels) # for unbalanced data 

    #             # Calculate predictions
    #             preds = model(input_ids, attention_mask, return_logits=False)
    #             all_preds.extend(preds.cpu().numpy())
    #             all_labels.extend(labels.cpu().numpy())

    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             # scheduler.step()
                
    #             loop.set_description(f'Epoch {epoch+1}/{epochs}')
    #             # max_input_id = input_ids.max().item()
    #             # min_input_id = input_ids.min().item()
    #             # loop.set_postfix(loss=loss.item(), max_input_id=max_input_id, min_input_id=min_input_id)
    #             loop.set_postfix(loss=loss.item())

    #         except Exception as e:
    #             print(f"\nError encountered at batch index {batch_idx}. Error details: {str(e)}")
    #             # Decode the input IDs back to text and print them
    #             decoded_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    #             print("Problematic input texts:", decoded_texts)
    #             # Append the index of the problematic batch to the list
    #             problematic_batch_indices.append(batch_idx)
        
    #     # Calculate and print accuracy and F1 score at the end of the epoch
    #     accuracy, f1 = evaluate_model(model, device, test_loader)
    #     if best_score is None:
    #         best_score = accuracy
    #     elif accuracy < best_score - min_delta:
    #         best_score = accuracy
    #         epochs_no_improve = 0
    #         # Save the best model if necessary
    #         torch.save(model, 'best_model.pth')
    #     else:
    #         epochs_no_improve += 1
        
    #     print(f'\nEpoch {epoch+1} completed. Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        
    #     if epochs_no_improve >= patience:
    #         rich.print(f"[red]Early stopping triggered.")
    #         early_stop = True
    #         break

    # # After training, print out all problematic batch indices
    # # print("Indices of problematic batches:", problematic_batch_indices)

    # torch.save(model, "retrained_BERTweet_balanced_acc_"+str(round(accuracy, 2))+".pth")
    
if __name__ == "__main__":
    __main__()