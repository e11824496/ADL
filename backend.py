from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import torch


from model import Model


app = FastAPI()

with open('categories.json', 'r') as f:
    classes = json.load(f)

classes_list = [f'{k}/{x}' for k, v in classes.items() for x in v]
num_classes = len(classes_list)
model = Model(num_classes)

model.load_state_dict(torch.load('model.pt'))


class BankStatementData(BaseModel):
    amount: float
    description: str
    time: str


@app.post("/classify")
async def classify(input_data: BankStatementData):
    input = input_data.model_dump()
    outputs = model(**input)
    logits = outputs['logits']
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    class_str = classes_list[preds[0]]
    return class_str

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
