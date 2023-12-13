import functools
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import torch
from sklearn import preprocessing

from model import Model

app = FastAPI()


class BankStatementData(BaseModel):
    amount: float
    description: str
    time: str


class ModelWrapper:
    def __init__(self, model: Model,
                 label_encoder: preprocessing.LabelEncoder):
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, input_data: BankStatementData):
        """
        Predicts the class label for the given input data.

        Parameters:
        input_data (BankStatementData): The input data to be predicted.

        Returns:
        str: The predicted class label.
        """
        input_data = input_data.model_dump()
        outputs = self.model(**input_data)
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        class_str = self.label_encoder.inverse_transform(preds)[0]
        return class_str


@functools.lru_cache()
def get_model():
    """
    Load and return a trained model along with its label encoder.

    Returns:
        ModelWrapper: A wrapper object containing
        the trained model and label encoder.
    """
    with open('categories.json', 'r') as f:
        classes = json.load(f)

    classes_list = [f'{k}/{x}' for k, v in classes.items() for x in v]
    num_classes = len(classes_list)
    model = Model(num_classes)

    le = preprocessing.LabelEncoder()
    le = le.fit(classes_list)

    model.load_state_dict(torch.load('model.pt'))
    print('Model loaded')

    return ModelWrapper(model, le)


@app.post("/classify")
async def classify(input_data: BankStatementData,
                   model: ModelWrapper = Depends(get_model)):
    try:
        result = model.predict(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
