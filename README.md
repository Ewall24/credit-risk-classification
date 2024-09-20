# credit-risk-classification 

Instructions

The instructions for this Challenge are divided into the following subsections:

    # Import the modules
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report  


    Split the Data into Training and Testing Sets

    Create a Logistic Regression Model with the Original Data

    Write a Credit Risk Analysis Report

Split the Data into Training and Testing Sets

Open the starter code notebook and use it to complete the following steps:

    # Read salary data
lending_data_df = pd.read_csv('Resources/lending_data.csv')  

    # Review the Dataframe
lending_data_df.head()   

![image](https://github.com/user-attachments/assets/7db56917-1f28-439b-9f46-d0b1faff7071)   





  Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns. 

    # Import the train_test_learn module
from sklearn.model_selection import train_test_split

    # Split the data using train_test_split
    # Assign a random_state of 1 to the function
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

   
    
  Create a Logistic Regression Model with the Original Data

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


    Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?


Write a Credit Risk Analysis Report

Write a brief report that includes a summary and analysis of the performance of the machine learning models that you used in this homework. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, ensuring that it contains the following:

    An overview of the analysis: Explain the purpose of this analysis.

    The results: Using a bulleted list, describe the accuracy score, the precision score, and recall score of the machine learning model.

    A summary: Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning.
