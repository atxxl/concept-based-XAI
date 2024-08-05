import os, rich
import torch
import wandb 
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm     import tqdm
from torch    import nn
from torch.optim  import AdamW
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

class RoBERTaClassifier(nn.Module):
    def __init__(self, roberta_model, num_labels):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = roberta_model
        for param in self.roberta.parameters(): # freeze the parameters
            param.requires_grad = False
        torch.manual_seed(42)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, return_logits=True):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Get the last hidden state
        cls_representation = last_hidden_state[:, 0, :]
        logits = self.classifier(cls_representation)
        if return_logits:
            return logits
        else: # Adjust the range to 1-5 for the class labels
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
        
        
def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    total, correct = 0, 0
    total_val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device) - 1

            outputs = model(input_ids, attention_mask, return_logits=True)
            loss = criterion(outputs, labels) # for unbalanced data
            total_val_loss += loss.item()
            preds = model(input_ids, attention_mask, return_logits=False)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy() + 1 )

            # Calculate accuracy
            correct += (preds == labels + 1).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_val_loss = total_val_loss / len(test_loader)
    # f1 = f1_score(all_labels, all_preds, average='micro') # unbalanced
    
    return accuracy, avg_val_loss


def __main__():
    # Set the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set the environment variable for w&b
    os.environ["WANDB_DIR"] = "/work3/s216410/"
    
    # Load the BERTweet model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roberta = RobertaModel.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", use_fast=False)
    roberta.to(device)
    rich.print(f"RoBERTa loaded on [red]{device}.")       
    
    # Load the datasets
    df_train = pd.read_csv("/zhome/81/3/170259/data/twitter-2016train-CE.tsv", 
                           sep='\t', header=None, quoting=3)
    df_test  = pd.read_csv("/zhome/81/3/170259/data/twitter-2016test-CE.tsv", 
                           sep='\t', header=None, quoting=3)
    df_train['converted_score'] = df_train[2] + 3 
    df_test['converted_score']  = df_test[2] + 3
    
    # downsize and resample the test set
    _, X_test, _, y_test = train_test_split(df_test.drop('converted_score', axis=1),
                                            df_test['converted_score'], 
                                            test_size=.1,
                                            random_state=seed, 
                                            stratify=df_test['converted_score'])

    # Combine the features and labels back into a single DataFrame
    df_test = pd.concat([X_test, y_test], axis=1)
    
    # Adjusting class weights when using the origin train dataset(unbalanced)
    class_counts = df_train['converted_score'].value_counts()
    total_samples = sum(class_counts)
    weights = [total_samples / class_counts[label] for label in class_counts.index]

    class_weights = torch.tensor(weights, device=device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
    learning_rates = [5e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    batch_sizes = [16, 32, 64]
    best_accuracy = 0
    best_params = {}
    param_combo = {}
    
    epochs = 10
    for lr in learning_rates:
        for batch_size in batch_sizes:
            run = wandb.init(project="RoBERTa_base_unbalanced", # change the w&b log here
                             reinit=True,
                             config={"epochs": epochs,
                                    "learning_rate": lr,
                                    "batch_size": batch_size,
            })
            model = RoBERTaClassifier(roberta, num_labels=5)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            optimizer = AdamW(model.parameters(), lr=lr)
            total_loss = 0
            model.to(device)
            model.train()
            
            accuracies = []
            best_in_loop = 0
            rich.print(f"lr: [green]{lr} [black]batch size: [blue]{batch_size}")
            
            for epoch in range(epochs):
                loop = tqdm(train_loader, leave=True)
                for batch in loop:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device) - 1
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels) # for unbalanced data
                    # loss = nn.CrossEntropyLoss()(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    # Update tqdm description
                    loop.set_description(f'Epoch {epoch+1}/{epochs}')
                    loop.set_postfix(loss=loss.item())

                # Evaluate model performance
                avg_loss = total_loss / len(train_loader)
                accuracy, _ = evaluate_model(model, device, test_loader, criterion)
                accuracies.append(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_in_loop = accuracy
                    best_params = {'learning_rate': lr, 'batch_size': batch_size}
                elif accuracy > best_in_loop:
                    best_in_loop = accuracy
                run.log({"accuracy": accuracy, "train-loss": loss, "avg_loss": avg_loss})
                
            print(f'Finish with Total Loss: {total_loss / len(train_loader)}, Accuarcy: {accuracy}')
            
            param_key = f"LR: {lr}, Batch: {batch_size}"
            param_combo[param_key] = accuracies
            
            run.config.update({
                "best_learning_rate": best_params["learning_rate"],
                "best_batch_size": best_params["batch_size"],
                "best_accuracy": best_accuracy
                })
            run.finish()

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