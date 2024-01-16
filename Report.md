# Report

This project aimed to develop a classifier using a deep neural network model to accurately categorize personal expenses. The primary motivation was to enhance personal finance management by providing more accurate and personalized categorization than automated systems.

The problem addressed in this project is the inaccurate categorization of personal expenses by automated systems. This issue is significant as it affects the ability to manage personal finances effectively.

## Why Is It a Problem?

The issue of inaccurate categorization of personal expenses by automated systems is a significant problem in personal finance management for several reasons. First, it hinders the ability to gain an accurate understanding of spending habits. Automated systems often apply generic categorizations that may not align with an individual's perception of their expenses. This misalignment can lead to misconceptions about where and how one's money is being spent.

Moreover, inaccurate categorization complicates budgeting and financial planning. When expenses are not categorized correctly, it becomes challenging to track spending against a budget, identify areas for cost-saving, or understand spending trends over time. This lack of clarity can impede effective financial decision-making and long-term financial health.

## Solution

My solution to this problem is a sophisticated neural network-based classifier specifically designed for personal expense categorization. The core of my solution is the integration of BERT-based embeddings for text classification. This advanced deep learning technique allows the model to understand the context and nuances of expense descriptions, leading to more accurate and personalized categorization.

Deep learning emerged as a key solution for this project due to its ability to handle large datasets and complex patterns. The use of a multilingual embedding model allowed for nuanced understanding and categorization of textual data in various languages, particularly German. Deep learning's capability to learn from and adapt to diverse data types made it an ideal choice for accurately classifying a wide range of personal expenses.

Additionally, the solution includes a custom labeling application that facilitates efficient data annotation. This tool allows users to quickly review and correct the model's predictions, thereby enhancing the training process and ultimately the accuracy of the classifier.

Furthermore, the development of a user-friendly visualization frontend complements the classifier by enabling users to easily interact with and understand their categorized expenses. This frontend includes basic functionalities such as filtering and pie chart visualizations, but there is potential for further development to offer more advanced analytical tools.

## Why Is It a Solution?

Deep learning, particularly BERT-based embeddings, offers a robust approach for text classification. It captures contextual nuances in expense descriptions, leading to more accurate categorizations.

## Project Implementation

    Dataset Collection: 8 hours
    Neural Network Design: 16 hours
    Training and Fine-Tuning: 5 hours
    Labeling-Application Development: 27 hours
    Frontend-Application Development: 12 hours
    Testing and CI/CD: 12 hours
    Documentation: 12 hours

The total time spent is reasonable, with unexpected additional time in Testing and CI/CD due to a bug in VSCode. This was higher than the initial estimate, reflecting the complexity and challenges faced in developing a machine learning-based application. Overall I think my estimates were somewhat reasonable.

## Main Takeaways and Insights

The project journey offered valuable insights, particularly in the realms of data handling and model training. One of the primary challenges we faced was the underrepresentation of certain categories in our dataset. This imbalance posed a significant hurdle in training the neural network model, as it struggled to accurately classify rare or unique expense categories. The experience underscored the importance of having a diverse and well-balanced dataset for training machine learning models to ensure broad and accurate categorization capabilities.

A problem which occured during training was the use of Paypal as a payment-menthod. This service charged my bank account but did not provide any information on the actual type of expense. Therefore, the integration of bank and PayPal data proved crucial in enriching the dataset, providing more context for accurate classification.

A crucial breakthrough in our project was the development and implementation of a Label-frontend. This tool was instrumental in enhancing the efficiency of our data labeling process. By fetching predictions from an early version of our model, the Label-frontend allowed for quick review and correction of data labels. This setup significantly expedited the process of moving through the most common expenses. It's important to note that while this approach greatly sped up the labeling process, it did not directly address the issue of underrepresented data. However, it enabled us to label more data and therefor increase the amount of labeld data available.

Another significant aspect of our learning experience was the realization of the trade-offs between model complexity and project constraints. The project's limited timeframe necessitated prioritizing certain aspects over others. In hindsight, exploring smaller models for performance comparison could have been insightful. However, due to time constraints, we were unable to delve deeply into this area.

## Reflections and Future Work

Reflecting on the project, there are several areas where future work could enhance the system's capabilities and user experience. While the current implementation of the classifier and its associated frontends has been successful, there is always room for improvement and expansion.

One of the key areas for future development lies in the visualization frontend we created to display categorized expenses. This tool currently offers basic functionalities, such as minor filtering options and pie chart visualizations. However, there is significant potential to extend its capabilities to provide more in-depth insights into personal financial data. Enhancements could include advanced filtering mechanisms, allowing users to view expenses over different time periods, categories, or other criteria. Additionally, integrating more sophisticated visualization techniques, such as trend lines, bar graphs, and scatter plots, could offer users a more comprehensive and interactive experience in analyzing their spending patterns.

Moreover, incorporating predictive analytics into the visualization frontend could be an exciting direction for future work. This could involve using the categorized data to forecast future spending trends, identify potential areas for budget optimization, and even provide personalized financial advice based on past spending behaviors. Such features would not only enhance the practical utility of the application but also provide users with actionable insights for better financial planning.

In terms of the neural network model, ongoing training and refinement with a more diverse set of data could further improve its accuracy, especially in underrepresented categories. Exploring different neural network architectures and comparing their performances could also yield beneficial insights, potentially leading to a more efficient and effective classification system.

## Conclusion

This project demonstrates the potential of deep learning in personal finance management. The neural network-based classifier significantly improved the accuracy of expense categorization, contributing to better financial planning and budgeting. The insights gained, especially regarding data handling and the labeling process, are valuable for future projects in similar domains.
