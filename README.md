# credit-risk-classification 
 
## Overview of the Analysis

The purpose of this analysis is to assess the credit risk of loans using a machine learning model, specifically logistic regression. By predicting whether loans are likely to be healthy or high-risk (default), we aim to support financial institutions in making informed lending decisions. This analysis helps identify high-risk loans early, minimizing potential losses. The model's performance is evaluated on key metrics like accuracy, precision, recall, and the F1-score to ensure it can effectively distinguish between healthy and high-risk loans, even in the presence of imbalanced data.


- The logistic regression model does a great job predicting both healthy (0) and high-risk (1) loans.
- For healthy loans, it achieves perfect precision (1.00) and nearly perfect recall (0.99), resulting in an F1-score of 1.00.
- For high-risk loans, it performs well with a precision of 0.84, recall of 0.94, and an F1-score of 0.89. Overall, the model is 99% accurate.
- This is impressive given the imbalance in the data, with many more healthy loans than high-risk ones.

- The model's high precision, recall, and F1-scores—both overall and for each class—demonstrate its effectiveness in handling this imbalance and accurately predicting both loan types.




# Import the modules 
   
 import numpy as np
 import pandas as pd 
 from sklearn.model_selection import train_test_split 
 from sklearn.linear_model import LogisticRegression 
 from sklearn.metrics import confusion_matrix, classification_report   
 import warnings




The instructions for this Challenge are divided into the following subsections:

  Create a Logistic Regression Model with the Original Data
  Write a Credit Risk Analysis Report
  Split the Data into Training and Testing Sets


Open the starter code notebook and use it to complete the following steps:

    # Read salary data
lending_data_df = pd.read_csv('Resources/lending_data.csv')  

    # Review the Dataframe
lending_data_df.head()   

![image](https://github.com/user-attachments/assets/7db56917-1f28-439b-9f46-d0b1faff7071)   





## Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns. 

    # Separate the data into labels and features
    # Separate the y variable, the labels
y = lending_data_df['loan_status']
     
    # Separate the X variable, the features
x = lending_data_df.drop(columns='loan_status') 

     # Review the y variable Series
y.head()
![image](https://github.com/user-attachments/assets/f2640220-1515-457d-8082-9c6d32c7d5fb)

    # Review the X variable DataFrame
x.head()


![image](https://github.com/user-attachments/assets/e3dcd841-5967-4f64-8464-a4770754816e)


# Split the data using train_test_split


    # Import the train_test_learn module
from sklearn.model_selection import train_test_split

    # Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

   
    
# Create a Logistic Regression Model with the Original Data

Use your knowledge of logistic regression to complete the following steps:

    # Import the LogisticRegression module from SKLearn
from sklearn.linear_model import LogisticRegression

    # Instantiate the Logistic Regression model
    # Assign a random_state parameter of 1 to the model
linear_classifier = LogisticRegression(random_state=1)

    # Fit the model using training data
linear_classifier.fit(X_train,y_train)

    
![image](https://github.com/user-attachments/assets/c7dbb884-70c2-4567-aca3-a104810edb94)

    
 Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.
  
    # Make a prediction using the testing data  
predictions = linear_classifier.predict(X_test) 
predictions   


Evaluate the model’s performance by doing the following:

    # Generate a confusion matrix for the model
confusion_matrix_model = confusion_matrix(y_test, predictions)
confusion_matrix_model_df = pd.DataFrame(
    confusion_matrix_model,
    index=['Actual Healthy (0)', 'Actual High-Risk (1)'],
    columns=['Predicted Healthy (0)', 'Predicted High-Risk (1)']
)

    # display the findings 
confusion_matrix_model_df

        
![image](https://github.com/user-attachments/assets/91b9a049-72fc-41bd-a7dc-32f9291895eb)

     
    # Print the classification report for the model
print(classification_report(y_test, predictions))

![image](https://github.com/user-attachments/assets/4e1eaaee-f2ac-4132-8e3c-253c649d606d)



    

## To close out the project:

Write a Credit Risk Analysis Report

Write a brief report that includes a summary and analysis of the performance of the machine learning models that you used in this homework. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, ensuring that it contains the following:

    An overview of the analysis: Explain the purpose of this analysis.

    The results: Using a bulleted list, describe the accuracy score, the precision score, and recall score of the machine learning model.

    A summary: Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning.

 ## Results 
 - Accurarcy: 99%
 - Precision: 84%
 - Recall:    94%

   ## Summary
   
   The logistic regression model performs exceptionally well, with 99% overall accuracy in predicting both healthy and high-risk loans, despite the imbalance in the dataset. The model achieves perfect precision (1.00) and nearly perfect recall (0.99) for healthy loans, yielding an F1-score of 1.00. For high-risk loans, the precision (0.84), recall (0.94), and F1-score (0.89) are strong, indicating the model is effective at identifying high-risk cases without too many false positives.

### Recommendations for Future Analysis:

   - Addressing Data Imbalance: While the model handles imbalance well, exploring techniques like SMOTE (Synthetic Minority Oversampling Technique) or under-sampling the majority class could further improve high-risk loan predictions.
   - Threshold Tuning: Adjusting the decision threshold may improve the precision-recall trade-off, especially for high-risk loans, depending on business priorities (e.g., minimizing false positives vs. maximizing recall).
   - Feature Importance: Analyzing the model’s coefficients or using feature importance methods could reveal key drivers for predicting loan outcomes, guiding more effective risk assessment.
   - Model Comparison: Comparing the logistic regression model with more complex models like random forests or gradient boosting could identify potential performance gains, especially in capturing high-risk loans more accurately.
   - Cross-Validation: Conducting cross-validation on different subsets of the data can provide insights into model robustness and help refine it further.
   - Explainability: Implementing SHAP or LIME to explain individual predictions could help provide actionable insights, especially for high-risk loans, aiding in decision-making.
