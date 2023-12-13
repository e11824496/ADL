# Project Report: Personal Expense Classifier

## Evaluation Metrics and Targets

### Error Metric Specification

For evaluating our neural network model, we have selected the F1 score as our primary metric. Given the unbalanced nature of our dataset, we have chosen to use the average F1 score across all categories. This method provides a more balanced view of the model's performance, ensuring that even underrepresented categories significantly impact the overall evaluation.

### Targeted F1 Score

Our target for the model was set ambitiously high, aiming for an F1 score of 90% or higher. This target was chosen to ensure a high level of accuracy and precision in expense classification, which is crucial for the effectiveness of the application in personal finance management.

### Achieved F1 Score

As of this report, the final evaluation of the model is pending. However, we are optimistic about achieving an F1 score close to our target, possibly slightly higher, given the meticulous design and tuning of our neural network. [Placeholder for actual F1 score once available]

## Time Spent on Project Tasks

In accordance with our work breakdown structure, the following is an estimated breakdown of the time spent on each phase of the project:

1. **Dataset Collection**: Approximately 20 hours spent gathering, formatting, and labeling our personal expense data.
2. **Designing Neural Network**: About 22 hours dedicated to developing the BERT-based text classification model.
3. **Training and Fine-Tuning**: Roughly 12 hours spent in training, evaluating, and fine-tuning the model.
4. **Building an Application**: An estimated 18 hours to develop a user-friendly application with a REST endpoint, served using FastAPI and enclosed in a Docker container.

These times are approximate and reflect the intensive effort and dedication invested in each phase of the project.

## Streamlit Frontend for Labeling

To facilitate the labeling process, a small frontend was developed using Streamlit. This tool was designed to expedite the laborious task of labeling bank statements. After the initial labeling of 100 statements, a preliminary model was trained and served using our backend. The frontend then pulled predictions from this model, pre-filling annotations for subsequent labeling. This allowed for a more efficient process, where only incorrect predictions required correction, and correct ones simply needed approval. This iterative approach not only streamlined the labeling task but also ensured continuous improvement of the model through active learning.

This additional step, while time-consuming, was pivotal in creating a more accurate dataset. It also allowed for the early implementation of a rudimentary training and serving process, which was instrumental in the project's iterative development approach.

## Classifier Implementation

Our classifier leverages a multilingual embedding model, specifically chosen for its proficiency in German among other languages. This was a crucial selection, as most popular embedding models lack extensive pre-training in German. We sourced our model from Hugging Face, a platform renowned for its extensive repository of pre-trained models and ease of implementation. The tokenizer, essential for preparing our textual data for the neural network, was also acquired from Hugging Face.

For the classification task, we designed a two-layer, fully connected Multi-Layer Perceptron (MLP) with Rectified Linear Unit (ReLU) activation and dropout. The choice of ReLU as the activation function was driven by its efficiency and effectiveness in non-linear transformations, which are essential in text classification tasks. ReLU helps in overcoming the vanishing gradient problem, common in deep neural networks, thereby facilitating faster and more effective training. Additionally, its simplicity leads to reduced computational complexity.

We incorporated dropout in our network architecture as a form of regularization. Dropout randomly deactivates a fraction of neurons during the training process, which helps in preventing overfitting. This technique is particularly beneficial in scenarios like ours, where the model complexity is high, and the risk of memorizing the training data is significant.

The training of our MLP was governed by the cross-entropy loss function. This choice was strategic for our multi-class classification task, as cross-entropy loss measures the performance of a classification model whose output is a probability value between 0 and 1. It provides a robust mechanism to quantify the difference between the predicted probabilities and the actual class, making it ideal for scenarios where precision in probability distribution is crucial.

Overall, the combination of a multilingual embedding model, ReLU activation, dropout regularization, and the cross-entropy loss function forms the backbone of our classifier, enabling it to effectively categorize personal expenses in a nuanced and accurate manner.

## Unit Testing and CI/CD Using GitLab

Unit testing was a critical component of our development process, ensuring the robustness of various parts of our application. We implemented tests for:

- **Pre-processing Checks**: To validate the correct functioning of our data pre-processing steps.
- **Backend Functionality**: Using FastAPI, we tested the REST endpoint functionality.
- **Training Process Validation**: A small script was created to test the training process on a sample dataset, ensuring no errors during actual training.

These tests were integrated into our workflow using GitHub Actions, which automatically ran on every push to our repository. This continuous integration approach allowed us to maintain high code quality and quickly identify and fix any issues.
