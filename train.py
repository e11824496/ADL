from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
import json

from model import Model

# Assuming you have a list of texts and corresponding labels
texts_df = pd.read_csv('BankStatements/output.csv')
labels_df = pd.read_csv('BankStatements/labels.csv')

texts_df = texts_df[:len(labels_df)]

labels_df['category'] = labels_df['category_group'] + \
    '/' + labels_df['category_subgroup']
with open('categories.json', 'r') as f:
    classes = json.load(f)

classes_list = [f'{k}/{x}' for k, v in classes.items() for x in v]
num_classes = len(classes_list)
print(f'Num of Classes: {num_classes}')


class CustomDataset(Dataset):
    def __init__(self, texts_df, labels_df):
        self.descriptions = texts_df['Description'].tolist()
        self.times = texts_df['Time'].tolist()
        self.amounts = texts_df['Amount'].tolist()

        self.labels = labels_df['category'].tolist()

        le = preprocessing.LabelEncoder()
        le = le.fit(classes_list)
        self.labels = le.fit_transform(self.labels)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        sample = {
            'description': self.descriptions[idx],
            'time': self.times[idx],
            'amount': self.amounts[idx],
            'label': self.labels[idx]
        }
        return sample


# Split the data into training and validation sets
# train_texts, val_texts, train_labels, val_labels = train_test_split(
#    texts_df, labels_df, test_size=0.2, random_state=42)

train_texts = val_texts = texts_df
train_labels = val_labels = labels_df

train_dataset = CustomDataset(train_texts, train_labels)
val_dataset = CustomDataset(val_texts, val_labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=num_classes)
model.to(device)


# Create DataLoader for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Set up optimizer and scheduler
optimizer = AdamW([
    {'params': model.embedding.parameters(), 'lr': 5e-5},
    {'params': model.fc1.parameters()},
    {'params': model.fc2.parameters()},
], lr=5e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Training loop
epochs = 30


for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs['loss']
        print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Validation
    model.eval()
    val_preds = []
    val_gt = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            outputs = model(**batch)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_gt.extend(batch['label'].cpu().numpy())

    accuracy = accuracy_score(val_gt, val_preds)
    print(f"Epoch {epoch + 1} - Validation Accuracy: {accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'model.pt')
