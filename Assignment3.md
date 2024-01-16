# Assignment 3 Report

## Introduction

This report provides an update on the ongoing development of the personal expense classifier project. My efforts have been focused on enhancing the application's frontend. The primary objective was to integrate basic visualization and filtering capabilities, demonstrating how the model can be incorporated into an application for personal finance management. While the current version is a proof of concept with limited features, it lays the groundwork for a more comprehensive and advanced personal finance management tool.

## New Features

### Frontend Development for Visualization and Filtering

#### Overview

The new frontend feature represents a significant step towards a more interactive and user-friendly application. It allows users to upload bank statements and PayPal data directly from their sources. Once uploaded, the data is preprocessed and labeled using the classifier model, which is hosted in a separate Docker container and accessible via a REST API.

#### Key Functionalities

1. **Data Upload and Processing**: Users can directly upload their bank statements and PayPal transaction data. The application then preprocesses and labels these data using the deployed model.

2. **Basic Filtering and Visualization**: The frontend includes rudimentary filtering options that allow users to view where and how much money was spent in each category. This feature enhances the user experience by providing a clearer overview of spending habits.

3. **Pie Chart Visualizations**: The application offers pie chart visualizations, making it easier for users to understand their expense distribution across different categories.

### Docker Containers for Easy Deployment

The application has been containerized using Docker, ensuring ease of deployment and consistency across different environments. There are two separate Docker containers:

1. **Frontend Container**: Hosts the newly developed frontend features, including visualization and filtering functionalities.

2. **Backend Container**: Contains the classifier model and handles data processing tasks. It communicates with the frontend via a REST API.

Running the Application

The application can be easily deployed using the following command:

```bash
docker compose up
```

This command fetches and runs the Docker containers for both the frontend and backend, streamlining the deployment process.

## Conclusion

The addition of the frontend with visualization and filtering capabilities marks a significant milestone in our project. It not only enhances the user experience but also demonstrates the practical application of our classifier model in personal finance management. As a proof of concept, it sets the stage for future development, where we aim to introduce more advanced features to transform it into a comprehensive personal finance management tool.
