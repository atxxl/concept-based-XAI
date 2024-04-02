import rich
import torch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from cbm_template_models import ModelXtoC
from sklearn.model_selection import train_test_split
from cbm_joint_sparse_obert import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rich.print(f"Device: [red]{DEVICE}")

model = torch.load('/zhome/81/3/170259/SparseCBM/run_cebab/s4e_model/distilbert-base-uncased_0.7_ModelXtoCtoY_layer_joint.pth')
model.to(device)
model.eval()

data = pd.read_csv('/zhome/81/3/170259/data/goemo_100.csv')
dataset = MyDataset(data)

batch_size = 8

# Define the dataloaders
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


def extract_concept_embeddings(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_concept_embeddings = []  # Store all concept embeddings
    for batch in dataloader:
        inputs = batch['input_ids'].to(device)
        outputs = model(inputs)  # Forward pass
        # Get the concept embeddings, assuming they are the second output onwards
        concept_embeddings = torch.cat(outputs[1:], dim=1).detach().cpu().numpy()
        all_concept_embeddings.append(concept_embeddings)
    return np.concatenate(all_concept_embeddings, axis=0)

# Function to perform PCA and plot
def plot_pca(concept_embeddings):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(concept_embeddings)
    plt.figure(figsize=(10, 10))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title('PCA of Concept Embeddings')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

# Extract embeddings
concept_embeddings = extract_concept_embeddings(model, train_loader, device)

# Plot PCA
plot_pca(concept_embeddings)
