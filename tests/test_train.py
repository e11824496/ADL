import unittest
import pandas as pd
from sklearn.model_selection import train_test_split
from train import CustomDataset, CustomModelTrainer


class TestCustomDataset(unittest.TestCase):
    def setUp(self):
        # Create a small dataset for testing
        data = {
            'Description': ['desc1', 'desc2', 'desc3'],
            'Time': [1, 2, 3],
            'Amount': [10.0, 20.0, 30.0],
            'category': ['A', 'B', 'A']
        }
        labels_df = pd.DataFrame(data)
        texts_df = pd.DataFrame(data)
        classes_list = ['A', 'B']
        self.dataset = CustomDataset(texts_df, labels_df, classes_list)

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertTrue('description' in sample)
        self.assertTrue('time' in sample)
        self.assertTrue('amount' in sample)
        self.assertTrue('label' in sample)


class TestCustomModelTrainer(unittest.TestCase):
    def setUp(self):
        # Create a small dataset for testing
        data = {
            'Description': ['desc1', 'desc2', 'desc3'],
            'Time': [1, 2, 3],
            'Amount': [10.0, 20.0, 30.0],
            'category': ['A', 'B', 'A']
        }
        labels_df = pd.DataFrame(data)
        texts_df = pd.DataFrame(data)
        classes_list = ['A', 'B']
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts_df, labels_df, test_size=0.2, random_state=42)
        self.trainer = CustomModelTrainer(
            train_texts, train_labels, val_texts, val_labels, classes_list,
            save_path='tests/test_model.pt')

    def test_train(self):
        # Test that training does not raise any errors
        self.trainer.train(epochs=1)


if __name__ == '__main__':
    unittest.main()
