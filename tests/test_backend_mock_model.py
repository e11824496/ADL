import unittest
from fastapi.testclient import TestClient
from backend import app, get_model, ModelWrapper
import torch


class MockModel:
    def __init__(self, num_classes):
        pass

    def __call__(self, **input_data):
        return {'logits': torch.Tensor([[0.1, 0.9]])}


class MockLabelEncoder:
    def fit(self, classes_list):
        pass

    def inverse_transform(self, preds):
        return ['category']


class TestBackendMockModel(unittest.TestCase):

    def test_classify(self):
        app.dependency_overrides[get_model] = lambda: ModelWrapper(
            MockModel(2), MockLabelEncoder())
        client = TestClient(app)

        # Define a sample input data
        input_data = {
            'amount': 100.0,
            'description': 'Sample description',
            'time': '2023-01-01T12:00:00'
        }

        response = client.post("/classify", json=input_data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), 'category')


if __name__ == '__main__':
    unittest.main()
