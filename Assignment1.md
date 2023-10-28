# Assignment 1 Report

## Introduction

Managing personal finances is essential, but it can be challenging when automated systems categorize expenses in ways that don't resonate with our understanding of our spending habits. This assignment is driven by the frustration with these limitations and the desire to create a more tailored solution.

## Goal

Our primary goal is to build a classifier for personal expenses. By training a deep neural network model, we aim to categorize expenses based on our definitions, allowing us to get better insights into our spending patterns and track expenses effectively. We also want to spot any deviations between monthly spending for better financial planning.

## Project Summary

We've chosen the "Bring Your Own Data" approach, which involves collecting our personal expense data. Here's a summary of our approach:

1. Dataset Collection: We'll start by gathering our expense data, formatting it, and labeling it with predefined categories. This phase is expected to take around 20 hours.

2. Designing Neural Network: The core of our project involves designing a neural network model for text classification. We plan to use BERT-based embeddings with a classification head and develop a robust training process. This stage is estimated to require about 20 hours.

3. Training and Fine-Tuning: Once the model is set up, we'll train it, evaluate its performance, and make adjustments as needed. We anticipate this phase to take roughly 10 hours.

4. Building an Application: To make our classification results accessible, we'll develop an application, possibly as a REST endpoint or a web interface, enclosed in a Docker container. This task is estimated to take about 20 hours.

5. Writing the Final Report: A crucial part of the project is documenting the entire process. We plan to allocate approximately 5 hours to create a detailed report summarizing our methodology and findings.

6. Preparing the Presentation: Lastly, we'll spend around 2 hours preparing a presentation to effectively share our project's results and insights.

## Challenges

This project comes with its set of challenges. The primary one is the limited variety of expenses in our personal history, with most being groceries. This lack of diversity might make it tricky to gauge the model's performance on new and varied inputs. Additionally, evaluating the model might be complex as we aim for it to classify even exotic statements correctly and not generalize everything as groceries.

## Conclusion

This assignment addresses a real-world problem with significant implications for personal finance and budgeting. Our aim to customize expense categorization using deep learning and BERT-based embeddings has the potential to provide better insights into our financial habits. Despite the challenges, we believe this project can change how we manage our expenses, leading to a deeper understanding of our financial well-being.
