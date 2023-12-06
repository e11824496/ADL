import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
import json

from model2 import Model


class CustomDataset(Dataset):
    def __init__(self, texts_df, labels_df, classes_list):
        self.descriptions = texts_df['Description'].tolist()
        self.times = texts_df['Time'].tolist()
        self.amounts = texts_df['Amount'].tolist()

        self.labels = labels_df['category'].tolist()

        le = preprocessing.LabelEncoder()
        le = le.fit(classes_list)
        self.labels = le.transform(self.labels)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        amount = torch.tensor(self.amounts[idx], dtype=torch.float32)
        sample = {
            'description': self.descriptions[idx],
            'time': self.times[idx],
            'amount': amount,
            'label': self.labels[idx]
        }
        return sample


class CustomModelTrainer:
    def __init__(self, train_texts, train_labels,
                 val_texts, val_labels, classes_list):
        self.train_dataset = CustomDataset(
            train_texts, train_labels, classes_list)
        self.val_dataset = CustomDataset(val_texts, val_labels, classes_list)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=8, shuffle=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=8, shuffle=False)

        self.train_amount_mean = train_texts['Amount'].mean()
        self.train_amount_std = train_texts['Amount'].std()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(num_classes=len(classes_list),
                           mean_amount=self.train_amount_mean,
                           std_amount=self.train_amount_std)
        self.model.to(self.device)

        self.optimizer = AdamW([
            {'params': self.model.embedding.parameters(), 'lr': 5e-5},
            {'params': self.model.fc1.parameters()},
            {'params': self.model.fc2.parameters()},
        ], lr=5e-3)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.9)

    def train(self, epochs=30):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
            self.scheduler.step()

            # Validation
            self.model.eval()
            val_preds = []
            val_gt = []
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Validation"):
                    outputs = self.model(**batch)
                    logits = outputs['logits']
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_gt.extend(batch['label'].cpu().numpy())

            accuracy = accuracy_score(val_gt, val_preds)
            print(
                f"Epoch {epoch + 1} - Validation Accuracy: {accuracy:.4f} -" +
                f" Total Loss: {total_loss:.4f}")

            # Save the trained model
            torch.save(self.model.state_dict(), 'model2.pt')


if __name__ == '__main__':
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
        texts_df, labels_df, test_size=0.2, random_state=42)
    trainer = CustomModelTrainer(
        train_texts, train_labels, val_texts, val_labels, classes_list)
    trainer.train()
