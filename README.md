# Thesis Project: Exploring the Concept Bottleneck Models for Sentiment Analysis: A case study with GoEmotion Dataset

This repository contains the code for reproducing the results from the project.

## Abstract
This thesis explores the impact of different sparsity settings on the performance of Concept Bottleneck Models (CBMs) in sentiment analysis tasks, specifically applied to the GoEmotions dataset. The study aims to determine if varying sparsity levels can maintain high accuracy while improving the interpretability of model predictions. Initially, a reproduction of Sparse CBMs using the CEBaB dataset was performed to establish a baseline. The GoEmotions dataset, containing 28 emotion concepts with significant imbalances, was then employed to evaluate the models under various sparsity configurations. 

Through a series of experiments, including augmentation techniques and hyperparameter tuning, the research demonstrates that fine-tuned sparsity levels can enhance model performance, even when having misclassification in concept prediction. The evaluation of the model utilized both confusion matrices and McNemar’s test to assess the significance of performance differences.

The findings suggest that with careful adjustment of sparsity, CBMs can achieve a balance between interpretability and accuracy. Future work should address the imbalances in the dataset, improve sentiment score reliability and explore advanced techniques for better generalization. This research contributes to the field of interpretable AI by offering insights into how sparsity can be leveraged to enhance the accuracy and transparency of sentiment analysis models.

## Experiments

The code is tested on Tesla A100 PCIE 80GB GPUs. An example for running the experiments is as follows:

```shell
bash ~/code/job_script.sh
```

## Project structure

The directory structure of the project looks like this:

```txt

├── code                        <- The code for sparse cmb.
│
├── data                        <- The twitter dataset.
│
├── sentiment models            <- The code for reproducing the sentiment model.
│
├── visulizations_in_notebook   <- The code for reproducing the figures.
│
└──  environment.yaml            <- The file for reproducing the environment.
```

