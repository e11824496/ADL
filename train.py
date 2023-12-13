import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
import json

from model2 import Model


class CustomDataset(Dataset):
    """
    A custom dataset class for handling text data with corresponding labels.

    Args:
        texts_df (pandas.DataFrame): DataFrame containing the text descriptions.
        labels_df (pandas.DataFrame): DataFrame containing the labels.
        classes_list (list): List of all possible classes.

    Attributes:
        descriptions (list): List of text descriptions.
        times (list): List of time values.
        amounts (list): List of amount values.
        labels (list): List of label values.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the sample at the given index.

    """

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
                 val_texts, val_labels, classes_list,
                 save_path='model2.pt'):
        """
        Initializes the training process.

        Args:
            train_texts (list): List of training texts.
            train_labels (list): List of training labels.
            val_texts (list): List of validation texts.
            val_labels (list): List of validation labels.
            classes_list (list): List of classes.
            save_path (str, optional): Path to save the trained model. Defaults to 'model2.pt'.
        """
        self.save_path = save_path
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

    def validate(self, return_preds=False):
        """
        Performs validation on the trained model.

        Args:
            return_preds (bool, optional): Whether to return the predictions. Defaults to False.

        Returns:
            float or tuple: Average F1 score if return_preds is False, otherwise a tuple containing average F1 score, validation predictions, and validation ground truth.
        """
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

        average_f1_score = f1_score(val_gt, val_preds, average='macro')
        if return_preds:
            return average_f1_score, val_preds, val_gt
        return average_f1_score

    def train(self, epochs=30):
        """
        Trains the model for the specified number of epochs.

        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to 30.
        """
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

            average_f1_score = self.validate()
            print(
                f"Epoch {epoch + 1} - " +
                f"Validation average_f1_score: {average_f1_score:.4f} - " +
                f"Total Loss: {total_loss:.4f}")

            # Save the trained model
            torch.save(self.model.state_dict(), self.save_path)


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
        texts_df, labels_df, test_size=0.2, random_state=99)
    trainer = CustomModelTrainer(
        train_texts, train_labels, val_texts, val_labels, classes_list)
    trainer.train()
