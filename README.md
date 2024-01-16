# ADL

The primary goal of this assignment is to develop a classifier capable of categorizing personal expenses accurately. By building and training a deep neural network model, we aim to categorize expenses into predefined categories based on our own definitions, not dictated by an automated system.
To learn more about the objective, timeframe and anticipated challenges, see the [Assignment 1 Report](./Assignment1.md).

For a description about the Labling tools, the model and the results, see the [Assignment 2 Report](./Assignment2.md).

A simple frontend to explore the categorized expenses is described in my [Assignment 3 Report](./Assignment3.md).

A quick summary of the project, the challenges and main takeaways is available in the [Report](./Report.md).

And here is a quick video displaying the project: [Youtube](https://youtu.be/TTNTPjDMj8I?feature=shared).

## Run the Application

### Setup and Launch

The application is conveniently available via Docker Compose, which simplifies setup and ensures consistent environments across different systems. The images are provided using Docker Hub. To get started, run the following command in your terminal:

```bash
docker compose up
```

This command fetches the necessary Docker images and runs the containers, setting up both the frontend and backend services.

### Accessing the Frontend

After running the Docker Compose command, the frontend of the application will be accessible at:

<http://localhost:8501>

Navigate to this URL in your web browser to interact with the application's user interface.

### Testing the Application

For testing purposes, sample data files are provided:

* Bank Statements: test/test_data_export_sample.csv
* PayPal Export: test/test_data/paypal_sample.csv

Follow these steps to test the application:

* Upload Data: Use the provided sample files to upload bank statements and PayPal data.
* Merge and Label: Click on the Merge and Label button in the application. This will combine the datasets and apply labels using the included model.
* First-Time Setup: During the initial run, the backend needs to fetch some model data (e.g., Tokenizer). This process might take some time, so please be patient as the application prepares and displays the results.

### Notes

The application is designed to handle typical data formats used by my bank and PayPal systems. Ensure your data files are in the correct format for optimal performance.

## Features

### Dataset Generation

The existing code is designed to train a model for classifying bank statements. A common challenge in this process arises from the fact that many personal expenses are conducted through PayPal, and the associated bank statements often record these transactions simply as "PayPal," without revealing the specific item or service purchased. This lack of detailed information can significantly hinder accurate classification.

To address this issue, our preprocessing routine integrates PayPal transaction details with the corresponding entries in bank statements. This integration is crucial as it enriches the bank statement data with more descriptive information about each PayPal transaction. By doing so, it provides a clearer insight into the nature of each expense, thereby enhancing the model's ability to classify transactions accurately.

For more details on how this preprocessing is implemented, please refer to the script [datasetGeneration.py](./datasetGeneration.py) in our repository.

### Labeling Data

To facilitate efficient labeling of custom data, we provide a Streamlit-based application in our [label frontend](./label_frontend/label_frontend.py). This application is designed for easy classification of bank statements that have been preprocessed using our dataset generator.

The frontend is configured to interact with a model exposed through a REST API, allowing it to retrieve initial predictions. These predictions can then be quickly reviewed, approved, or corrected by users directly within the frontend interface. This setup streamlines the data labeling process, making it more user-friendly and efficient.

### Training Model

For model training, the [train.py](./train.py) script offers a straightforward training loop. This script leverages a multilingual embedding model to train on the labeled data. It simplifies the process of training the classification model, ensuring that even users with limited technical expertise can effectively train and refine the model based on their specific data sets.

### REST-API Server

To make model predictions accessible to end-users and the label frontend, we have developed a REST-API server, detailed [here](./backend.py). This server is built using FastAPI, offering a basic yet efficient solution for deploying the model. It's designed for quick deployment, allowing for easy integration with various applications, including the label frontend, for real-time predictions.

### Frontend

There is a basic stream-lit frotend to enable a quick exploration of the predictions and the resulting spending-habit. Future work could extend this work, to enable extensive filtering and visualizations.

## Run the code

For preprocessing, there is currently a file-structure in place, that handels the bank and paypal statments.

* `BankStatements/` folder:
  * `export.csv`: Bank statement CSV file.
  * `paypal.csv`: PayPal transactions CSV file.
  * `output.csv`: Output file after processing.

Install dependencies

```bash
pip install -r requirements.txt
```

### Preprocessing

1. Place your bank statement file as `BankStatements/export.csv` and your PayPal file as `BankStatements/paypal.csv`. Ensure the format of these files matches the required structure (see Data Format section below). Due to privacy reasons, I'm unable to provide you with my training data. A minimal working subset is available for testing under  [tests/test_data](./tests/test_data/).

2. Run the script:

   ```python
   python dataset_generation.py
   ```

3. The processed data will be saved as `BankStatements/output.csv`.

#### Bank Statement CSV (`export.csv`)

* Format: Semi-colon separated values (`;`).
* Columns: 'Date', 'Description', 'Date2', 'Amount', 'Currency', 'Datetime'.
* Example:

  ```txt
  01-01-2023;Purchase at Store;02-01-2023;123.45;USD;01.01.2023 10:30:00:000
  ```

* This output format is the default if you use Raiffeisenbank csv export.

#### PayPal CSV (`paypal.csv`)

* Columns: 'Date', 'Time', 'Name', 'Transaction ID', 'Item Title', 'Gross', 'Status', 'Reference Txn ID', 'Bank Reference ID'.
* Example:

  ```txt
  01/01/2023;10:30:00;John Doe;TXN1234;Item A;1,234.56;Completed;REF123;BANK123
  ```

* This output format is the default if you use Paypal activity report export.

### Training

1. Preprocess your data
2. Create labels for your training data
3. Run the training procedure:

    ```bash
    python train.py
    ```

### Serving

Once you have trained a model, you can serve it using our backend.

```sh
python backend.py
```

### Label-Frontend

You currently need a model-deployed as a REST-API to fully utilize the Label frontend.

```sh
cd ./label_frontend/
streamlit run label_frontend.py
```
