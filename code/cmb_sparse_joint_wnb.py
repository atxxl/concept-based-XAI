import os 
import rich
import torch
import wandb
import numpy  as np
import pandas as pd
import argparse
import transformers
import torch.nn.functional as F

from tqdm import tqdm
from datasets import load_dataset
from gensim.models import FastText
from torch.optim.lr_scheduler import StepLR
from transformers import RobertaTokenizer, RobertaModel,BertModel, BertTokenizer,GPT2Model, GPT2Tokenizer, DistilBertModel, DistilBertTokenizer, OPTModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from cbm_template_models import MLP, FC
from cbm_models import ModelXtoC_function, ModelCtoY_function,ModelXtoCtoY_function
from sparse_model import insert_sparse_mask, set_required_gradient, set_sparse_index, setup_seed, SubnetLinear_Mask
from obert import EmpiricalBlockFisherInverse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
# Set the environment variable for w&b
os.environ["WANDB_DIR"] = "/work3/s216410/"

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--VERBOSE', action='store_true', help='Enable VERBOSE logs.')
parser.add_argument('--model_name', default='bert-base-uncased',
                     help='Backbone model.')
parser.add_argument('--FINETUNE_EPOCH', type=int, default=30,
                    help='Number of epochs to train the model.')
parser.add_argument('--START_EPOCH', type=int, default=2,
                    help='Number of epochs to start pruning.')
parser.add_argument('--END_EPOCH', type=int, default=15,
                    help='Number of epochs to end pruning.')
parser.add_argument('--CLF_EPOCH', type=int, default=20,
                    help='Maximum Number of epochs to train the clf head.')
parser.add_argument('--INIT_SPARSITY', type=float, default=0.2,
                    help='Rate for the initial target sparsity.')
parser.add_argument('--FINAL_SPARSITY', type=float, default=0.7,
                    help='Rate for the final target sparsity.')
parser.add_argument('--M_GRAD', type=int, default=20,
                    help='Values of M_GRAD.')
parser.add_argument('--BLOCK_SIZE', type=int, default=20,
                    help='Value of BLOCK_SIZE.')
parser.add_argument('--LAMBD', type=float, default=1e-7,
                    help='Values of LAMBD.')
parser.add_argument('--LAMBD_XtoC', type=float, default=5.,
                    help='Rate for the final target sparsity.')
parser.add_argument('--model_path', default='/zhome/81/3/170259/SparseCBM/run_cebab/models/',
                     help='Path for output pre-trained model.')

args = parser.parse_args()  

model_path = args.model_path


# Enable concept or not
mode = 'joint'

# Define the paths to the dataset and pretrained model
# model_name = "microsoft/deberta-base"
model_name = args.model_name # 'bert-base-uncased' / 'roberta-base' / 'gpt2' / 'lstm' / 'distilbert-base-uncased'
setup_seed(args.seed)

# Define the maximum sequence length and batch size
max_len = 512
batch_size = 3  # changed
lambda_XtoC = args.LAMBD_XtoC  # lambda > 0
is_aux_logits = False
num_labels = 5  #label的个数
num_each_concept_classes = 3  #每个concept有几个类


# Load the tokenizer and pretrained model
if model_name == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
elif model_name == 'bert-base-uncased':
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
elif model_name == 'distilbert-base-uncased':
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)
elif model_name == 'gpt2':
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
elif model_name == 'facebook/opt-125m':
    model = OPTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif model_name == 'facebook/opt-350m':
    model = OPTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif model_name == 'facebook/opt-1.3b':
    model = OPTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
elif model_name == 'lstm':
    fasttext_model = FastText.load_fasttext_format('./fasttext/cc.en.300.bin')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    class BiLSTMWithDotAttention(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            embeddings = fasttext_model.wv.vectors
            self.embedding.weight = torch.nn.Parameter(torch.tensor(embeddings))
            self.embedding.weight.requires_grad = False
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers = 1, bidirectional=True, batch_first=True)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim*2, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2)
        )

        def forward(self, input_ids, attention_mask):
            input_lengths = attention_mask.sum(dim=1)
            embedded = self.embedding(input_ids)
            output, _ = self.lstm(embedded)
            weights = F.softmax(torch.bmm(output, output.transpose(1, 2)), dim=2)
            attention = torch.bmm(weights, output)
            logits = self.classifier(attention.mean(1))
            return logits

    model = BiLSTMWithDotAttention(len(tokenizer.vocab), 300, 128)

data_type = "goemo" # "pure_cebab"/"aug_cebab"/"aug_yelp"/"aug_cebab_yelp"
# Load data
if data_type == "goemo":
    data = pd.read_csv('/zhome/81/3/170259/data/goemo_resampled_100.csv')
    num_concept_labels = 28
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    test_df, val_df = train_test_split(temp_df, test_size=0.5, random_state=42) 


class MyDataset(Dataset):
    # Split = train/dev/test
    def __init__(self, dataframe, skip_class = "no majority"):
        self.data = dataframe
        self.labels = self.data["predicted_sentiment"].values
        self.text = self.data["text"].values
        self.map_dict = {"Negative":0, "Positive":1, "unknown":2, "":2,"no majority":2}
        
        # the 28 concepts
        self.concepts = ['admiration', 'amusement', 'anger', 'annoyance', 
                    'approval', 'caring', 'confusion', 'curiosity', 'desire', 
                    'disappointment', 'disapproval', 'disgust', 'embarrassment', 
                    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
                    'nervousness', 'optimism', 'pride', 'realization', 'relief',
                    'remorse', 'sadness', 'surprise', 'neutral']
        for c in self.concepts:
            setattr(self, f"{c}_aspect", self.data[c].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.labels[idx] - 1
        
        # gold labels
        for c in self.concepts:
            setattr(self, f"{c}_concept", self.data[c].map(self.map_dict).values)

        concept_labels = []
        tensor_dict = {}
        
        for concept in self.concepts:
            concept_value = getattr(self, f"{concept}_concept")[idx]
            concept_labels.append(concept_value)
            tensor_dict[f"{concept}_concept"] = torch.tensor(concept_value, dtype=torch.long)
        
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
                "concept_labels": torch.tensor(concept_labels, dtype=torch.long),
                **tensor_dict
            }


# Load the data
train_dataset = MyDataset(train_df)
test_dataset = MyDataset(test_df)
val_dataset = MyDataset(val_df)

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


#Set ModelXtoCtoY_layer
    # concept_classes 每个concept有几类；    label_classes  label的个数；  n_attributes concept的个数； n_class_attr 每个concept有几类；
if model_name == 'lstm':
    ModelXtoCtoY_layer = ModelXtoCtoY_function(concept_classes = num_each_concept_classes, label_classes = num_labels, n_attributes = num_concept_labels, bottleneck = True, expand_dim = 0, n_class_attr=num_each_concept_classes, use_relu=False, use_sigmoid=False,Lstm=True,aux_logits=is_aux_logits)
else:
    ModelXtoCtoY_layer = ModelXtoCtoY_function(concept_classes = num_each_concept_classes, label_classes = num_labels, n_attributes = num_concept_labels, bottleneck = True, expand_dim = 0, n_class_attr=num_each_concept_classes, use_relu=False, use_sigmoid=False,aux_logits=is_aux_logits)

# Set up the optimizer and loss function
# optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-5)
optimizer = torch.optim.Adam(list(model.parameters()) + list(ModelXtoCtoY_layer.parameters()), lr=1e-5)

if model_name == 'lstm':
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rich.print(f"Device: [red]{device}")


# Modify model with sparse mask
ModelXtoCtoY_layer.to(device)
model = insert_sparse_mask(model, num_concept_labels, None, pruning_type='obert')
model.to(device)

# make sure that the concepts are here
concepts = ['admiration', 'amusement', 'anger', 'annoyance', 
            'approval', 'caring', 'confusion', 'curiosity', 'desire', 
            'disappointment', 'disapproval', 'disgust', 'embarrassment', 
            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
            'nervousness', 'optimism', 'pride', 'realization', 'relief',
            'remorse', 'sadness', 'surprise', 'neutral']

FINETUNE_EPOCH=args.FINETUNE_EPOCH
START_EPOCH=args.START_EPOCH
END_EPOCH=args.END_EPOCH
INIT_SPARSITY=args.INIT_SPARSITY
FINAL_SPARSITY=args.FINAL_SPARSITY

M_GRAD=args.M_GRAD
BLOCK_SIZE=args.BLOCK_SIZE
LAMBD=args.LAMBD
EPS=torch.finfo(torch.float32).eps

run_XtoCtoY = wandb.init(project="CMB_sparse_joint", # change the w&b log here
                         reinit=True,
                         config={"finetune_epoch": FINETUNE_EPOCH,
                                 "start_epoch": START_EPOCH,
                                 "end_epoch": END_EPOCH,
                                 "init_sparsity": INIT_SPARSITY,
                                 "final_sparsity": FINAL_SPARSITY,
                                 "m_grad": M_GRAD,
                                 "block_size": BLOCK_SIZE,
                                 "lambd": LAMBD,
                                 })

#step 1.1  XtoCtoY finetune with fixed mask
print("train XtoCtoY! Fix Sparse Mask")
for epoch in range(FINETUNE_EPOCH):
    predicted_concepts_train = []
    predicted_concepts_train_label = []
    ModelXtoCtoY_layer.train()
    model.train()
    
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)
        concept_tensors = {}
        for concept in concepts:
            concept_key = f"{concept}_concept"  # Construct the key name as stored in the batch
            if concept_key in batch:
                concept_tensors[concept] = batch[concept_key].to(device)

        concept_labels=batch["concept_labels"].to(device)
        concept_labels = torch.t(concept_labels)
        concept_labels = concept_labels.contiguous().view(-1) 

        optimizer.zero_grad()

        XtoC_outputs = []
        XtoY_outputs = []
        for concept_idx in range(num_concept_labels):
            set_sparse_index(model, concept_idx)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if model_name == 'lstm':
                pooled_output = outputs
            else:
                pooled_output = outputs.last_hidden_state.mean(1)  

            outputs  = ModelXtoCtoY_layer(pooled_output) 

            XtoC_outputs.append(outputs[concept_idx+1]) 
            XtoY_outputs.extend(outputs[0:1])

        # XtoC_loss
        XtoC_logits = torch.nn.Sigmoid()(torch.cat(XtoC_outputs, dim=0)) # 32*4 00000000111111112222222233333333
        XtoC_loss = loss_fn(XtoC_logits, concept_labels)
        # XtoY_loss
        Y_batch = XtoY_outputs[0]
        XtoY_loss = loss_fn(Y_batch, label)
        loss = XtoC_loss*lambda_XtoC+XtoY_loss
        loss.backward()
        optimizer.step()
        run_XtoCtoY.log({"XtoC_loss": XtoC_loss.item(), "XtoY_loss": XtoY_loss.item(), 
                         "loss":loss, "epoch": epoch,})

    model.eval()
    ModelXtoCtoY_layer.eval()
    val_accuracy = 0.
    concept_val_accuracy = 0.
    test_accuracy = 0.
    concept_test_accuracy = 0.
    best_acc_score = 0
    predict_labels = np.array([])
    true_labels = np.array([])
    concept_predict_labels = np.array([])
    concept_true_labels = np.array([])
    predict_concepts = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_tensors = {}
            for concept in concepts:
                concept_key = f"{concept}_concept"  # Construct the key name as stored in the batch
                if concept_key in batch:
                    concept_tensors[concept] = batch[concept_key].to(device)     
            concept_labels=batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels)
            concept_labels = concept_labels.contiguous().view(-1)


            XtoC_outputs = []
            XtoY_outputs = []
            for concept_idx in range(num_concept_labels):
                set_sparse_index(model, concept_idx)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if model_name == 'lstm':
                    pooled_output = outputs
                else:
                    pooled_output = outputs.last_hidden_state.mean(1)  

                outputs  = ModelXtoCtoY_layer(pooled_output)  
                XtoC_outputs.append(outputs[concept_idx+1]) 
                XtoY_outputs.extend(outputs [0:1])

                    
            predictions = torch.argmax(XtoY_outputs[0], axis=1)
            val_accuracy += torch.sum(predictions == label).item()
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
            
            #concept accuracy
            XtoC_logits = torch.cat(XtoC_outputs, dim=0)
            concept_predictions = torch.argmax(XtoC_logits, axis=1)
            concept_val_accuracy += torch.sum(concept_predictions == concept_labels).item()
            concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())
            concept_predictions = concept_predictions.reshape(-1,num_concept_labels)  # reshape 二维向量[batch_size*num_concept_labels]
        
        val_accuracy /= len(val_dataset)
        num_labels = len(np.unique(true_labels))

        concept_val_accuracy /= len(val_dataset)
        concept_num_true_labels = len(np.unique(concept_true_labels))
        
        macro_f1_scores = []
        for label in range(num_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

        concept_macro_f1_scores = []
        for concept_label in range(concept_num_true_labels):
            concept_label_pred = np.array(concept_predict_labels) == concept_label
            concept_label_true = np.array(concept_true_labels) == concept_label
            concept_macro_f1_scores.append(f1_score(concept_label_true, concept_label_pred, average='macro'))
            concept_mean_macro_f1_score = np.mean(concept_macro_f1_scores)

    print(f"Epoch {epoch + 1}: Val concept Acc = {concept_val_accuracy*100/num_concept_labels} Val concept Macro F1 = {concept_mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Val Acc = {val_accuracy*100} Val Macro F1 = {mean_macro_f1_score*100}")
    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        os.makedirs(model_path, exist_ok=True)
        torch.save(model, model_path+model_name+"_"+str(FINAL_SPARSITY)+"_joint.pth")
        torch.save(ModelXtoCtoY_layer, model_path+model_name+"_"+str(FINAL_SPARSITY)+"_ModelXtoCtoY_layer_joint.pth")
        
    run_XtoCtoY.log({"val_accuracy": val_accuracy, "mean_macro_f1_score": mean_macro_f1_score, 
                     "concept_val_accuracy": concept_val_accuracy, "concept_mean_macro_f1_score": concept_mean_macro_f1_score, 
                     "epoch": epoch,})
    run_XtoCtoY.finish()

    #step 1.2  Update Mask
    if epoch >= START_EPOCH and epoch <= END_EPOCH:
        target_sparsity = (FINAL_SPARSITY - INIT_SPARSITY) * (epoch - START_EPOCH) / (END_EPOCH - START_EPOCH) + INIT_SPARSITY
        print('Start Pruning for Target-Sparsity {}, density'.format(target_sparsity, 1 - target_sparsity))

        for concept_idx in range(num_concept_labels):
            print('Concept: {}'.format(concept_idx+1))
            set_sparse_index(model, concept_idx)
            grad_steps = 0

            finnvs_dict = {}
            for name, module in model.named_modules():
                if isinstance(module, SubnetLinear_Mask):
                    finnvs_dict[name] = EmpiricalBlockFisherInverse(
                        num_grads = M_GRAD,
                        fisher_block_size = BLOCK_SIZE,
                        num_weights = module.mask_list[concept_idx].numel(),
                        damp = LAMBD,
                        device = device,
                    )

            for batch in tqdm(train_loader, desc="Training", unit="batch"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                label = batch["label"].to(device)
                concept_tensors = {}
                for concept in concepts:
                    concept_key = f"{concept}_concept"
                    if concept_key in batch:
                        concept_tensors[concept] = batch[concept_key].to(device)              
                concept_labels=batch["concept_labels"].to(device)
                concept_labels = torch.t(concept_labels)
                concept_labels = concept_labels.contiguous().view(-1) 

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state.mean(1)
                outputs  = ModelXtoCtoY_layer(pooled_output)
                XtoC_output = outputs[concept_idx+1]
                XtoC_logits = torch.nn.Sigmoid()(XtoC_output)
                XtoY_output = outputs[0:1]

                BS=XtoC_logits.shape[0]

                XtoC_loss = loss_fn(XtoC_logits, concept_labels[concept_idx*BS: concept_idx*BS+BS])
                XtoY_loss = loss_fn(XtoY_output[0], label)

                loss = XtoC_loss*lambda_XtoC+XtoY_loss
                loss.backward()

                for name, module in model.named_modules():
                    if isinstance(module, SubnetLinear_Mask):
                        if not name in finnvs_dict: continue
                        if module.fc.weight.grad == None:
                            del finnvs_dict[name]
                            continue
                        finnvs_dict[name].add_grad(module.fc.weight.grad.reshape(-1))
                grad_steps += 1
                
                if grad_steps >= M_GRAD:
                    break


            # Calculate Scores and Update Mask
            for name, module in model.named_modules():
                if isinstance(module, SubnetLinear_Mask):

                    if not name in finnvs_dict: continue
                    scores = (
                        (module.fc.weight.data.reshape(-1) ** 2).to(device)
                        / (2.0 * finnvs_dict[name].diag() + EPS)
                    ).reshape(module.fc.weight.shape)
                    d = module.mask_list[concept_idx].numel()
                    kth_score = torch.kthvalue(scores.reshape(-1), round(target_sparsity * d))[0]
                    module.mask_list[concept_idx].data = (scores > kth_score).to(module.mask_list[concept_idx].dtype)
                    print('Remaining weight for {} = {:.4f}'.format(name, module.mask_list[concept_idx].gt(0).float().mean()))

####################### test
num_epochs = 1
print("Test!")
model = torch.load(model_path+model_name+"_"+str(FINAL_SPARSITY)+"_joint.pth")
ModelXtoCtoY_layer = torch.load(model_path+model_name+"_"+str(FINAL_SPARSITY)+"_ModelXtoCtoY_layer_joint.pth") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)
            concept_tensors = {}
            for concept in concepts:
                concept_key = f"{concept}_concept"
                if concept_key in batch:
                    concept_tensors[concept] = batch[concept_key].to(device)  
            concept_labels=batch["concept_labels"].to(device)
            concept_labels = torch.t(concept_labels)
            concept_labels = concept_labels.contiguous().view(-1)

            XtoC_outputs = []
            XtoY_outputs = []
            for concept_idx in range(num_concept_labels):
                set_sparse_index(model, concept_idx)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if model_name == 'lstm':
                    pooled_output = outputs
                else:
                    pooled_output = outputs.last_hidden_state.mean(1)  

                outputs  = ModelXtoCtoY_layer(pooled_output)  
                XtoC_outputs.append(outputs[concept_idx+1]) 
                XtoY_outputs.extend(outputs [0:1])

            predictions = torch.argmax(XtoY_outputs[0], axis=1)
            test_accuracy += torch.sum(predictions == label).item()
            predict_labels = np.append(predict_labels, predictions.cpu().numpy())
            true_labels = np.append(true_labels, label.cpu().numpy())
            #concept accuracy
            XtoC_logits = torch.cat(XtoC_outputs, dim=0)
            concept_predictions = torch.argmax(XtoC_logits, axis=1)
            concept_test_accuracy += torch.sum(concept_predictions == concept_labels).item()
            concept_predict_labels = np.append(concept_predict_labels, concept_predictions.cpu().numpy())
            concept_true_labels = np.append(concept_true_labels, concept_labels.cpu().numpy())
            concept_predictions = concept_predictions.reshape(-1,num_concept_labels)  # reshape 二维向量[batch_size*num_concept_labels]
        
        test_accuracy /= len(test_dataset)
        num_labels = len(np.unique(true_labels))

        concept_test_accuracy /= len(test_dataset)
        concept_num_true_labels = len(np.unique(concept_true_labels))
        
        macro_f1_scores = []
        for label in range(num_labels):
            label_pred = np.array(predict_labels) == label
            label_true = np.array(true_labels) == label
            macro_f1_scores.append(f1_score(label_true, label_pred, average='macro'))
            mean_macro_f1_score = np.mean(macro_f1_scores)

        concept_macro_f1_scores = []
        for concept_label in range(concept_num_true_labels):
            concept_label_pred = np.array(concept_predict_labels) == concept_label
            concept_label_true = np.array(concept_true_labels) == concept_label
            concept_macro_f1_scores.append(f1_score(concept_label_true, concept_label_pred, average='macro'))
            concept_mean_macro_f1_score = np.mean(concept_macro_f1_scores)

    print(f"Epoch {epoch + 1}: Test concept Acc = {concept_test_accuracy*100/num_concept_labels} Test concept Macro F1 = {concept_mean_macro_f1_score*100}")
    print(f"Epoch {epoch + 1}: Test Acc = {test_accuracy*100} Test Macro F1 = {mean_macro_f1_score*100}")