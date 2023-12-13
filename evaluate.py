from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import json
import torch
import numpy as np

from train import CustomModelTrainer

texts_df = pd.read_csv('BankStatements/output.csv')
labels_df = pd.read_csv('BankStatements/labels.csv')

texts_df = texts_df[:len(labels_df)]

labels_df['category'] = labels_df['category_group'] + \
    '/' + labels_df['category_subgroup']
with open('categories.json', 'r') as f:
    classes = json.load(f)

classes_list = [f'{k}/{x}' for k, v in classes.items() for x in v]
# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts_df, labels_df, test_size=0.2, random_state=99)

trainer = CustomModelTrainer(
    train_texts, train_labels, val_texts, val_labels, classes_list)
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
trainer.model.load_state_dict(
    torch.load('model2.pt', map_location=device))
average_f1_score, preds, gt = trainer.validate(return_preds=True)

print(average_f1_score)
all_labels = list(range(len(classes_list)))

print(classification_report(
    gt, preds,
    target_names=sorted(classes_list),
    labels=all_labels,
    zero_division=np.nan))
