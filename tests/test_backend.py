import unittest
from fastapi.testclient import TestClient


import os
# skip tests if model.pt not present
if not os.path.isfile('model.pt'):
    raise unittest.SkipTest("model.pt not present")
from backend import app


class TestClassifyEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        pass

    def test_classify_valid_input(self):
        valid_input = {
            "amount": 100.0,
            "description": "Eurospar Dankt 4141 Wien 1050",
            "time": "2023-01-01T12:00:00"
        }
        response = self.client.post("/classify", json=valid_input)
        self.assertEqual(response.status_code, 200)
        # Replace with the expected class string
        self.assertEqual(response.json(), "Food/Groceries")

    def test_classify_invalid_input(self):
        invalid_input = {
            "amount": 100.0,
            "time": "2023-01-01T12:00:00"
        }
        response = self.client.post("/classify", json=invalid_input)
        # Unprocessable Entity due to validation error
        self.assertEqual(response.status_code, 422)

    # Add more test methods as needed


if __name__ == "__main__":
    unittest.main()
