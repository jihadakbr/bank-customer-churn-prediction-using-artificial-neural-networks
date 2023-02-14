
# Bank Customer Churn Prediction Using Artificial Neural Networks (ANN)

Bank customer churn prediction using neural networks is a valuable business project because it helps banks identify customers who are at risk of leaving and take proactive measures to retain them. By analyzing customer data and predicting churn, banks can gain insights into why customers leave and what factors influence their decision to stay or go. This information can be used to improve customer experience, tailor marketing and promotional activities, and develop strategies to increase customer loyalty. Ultimately, reducing churn can have a significant positive impact on a bank's bottom line, as it is typically more cost-effective to retain existing customers than to acquire new ones.
## Dataset Source

[Bank Customer Churn Modelling (Kaggle)](https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling)
## Content of Dataset

1. RowNumber: the index of the row in the data
2. CustomerId: unique customer identifier
3. Surname: surname
4. CreditScore: credit rating
5. Geography: country of residence
6. Gender: gender
7. Age: age
8. Tenure: how many years a person has been a client of the bank
9. Balance: account balance
10. NumOfProducts: the number of bank products used by the client
11. HasCrCard: availability of a credit card
12. IsActiveMember: client activity
13. EstimatedSalary: estimated salary
14. Exited: the fact of the client's departure (target variable)

* Exited = 1
* Not Exited = 0
## Tools

Jupyter notebook:
* Data exploring and cleansing
* Data visualization
* Looking for the best accuracy deep learning model using artificial neural networks (ANN)

Spyder:
* Create python functions for data visualization
* Create a web application using flask and deploy it to replit.com
## Results

Link to the web application:
* [https://bank-customer-churn-prediction.jihadakbr.repl.co/](https://bank-customer-churn-prediction.jihadakbr.repl.co/)
* or [https://replit.com/@jihadakbr/Bank-Customer-Churn-Prediction?v=1](https://replit.com/@jihadakbr/Bank-Customer-Churn-Prediction?v=1)
* Note that this web application is only active when I log in to replit.com

![Accuracy and Lost of Bank Customer Churn Prediction](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg6W0I9sTrDEEO2U3z0eM7t7Pvq8gYmFajKEk8fE6Mmc94R44aS9M4IumG66A-BiiYiCPvsBNZYXXg6-hCQ5Wk7udQ11A_M4CQL09hMHf8Fb8tQxmOrHQlddOxjj5kSltjP7skVm_gqw7BUMRiaIr4YMwdjbcy_ZuDvonarrDJYZ1KeA6-HMZ1GMu54/s1600/bank-customer-churn-prediction-accuracy-lost.png)

![Bank Customer Churn Prediction](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEghtWFO5GUz9J1O0j00yKYv6I9QxSxjoXPRveZ6wpyak_sP5sZP0s1WYZQnyj_Zfsl88rUo5zmWL27ZCkVd-U6xanWhdk7_um7yboAZxJQ2-9zVdazkOW8AHucBEXGet0ZnwMbDrIy53IyypA1UCFKu8p-mB8VsGTcLb2BU66PwvWeL3wQEkC7Pq87y/s1600/bank-customer-churn-prediction.png)

![Bank Customer Churn Prediction 2](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEi8ZQqv1Qazh6-k_nag9voJsj4dy6bL2yfhK_ctdHybq9PcyLypk4iowGneEaSxFdidT-2tYHjdKnCu2DCs5-soOFWTHCxJUKbSp8k3s3cu_3wZGULU3JLxYa5le4POquZEWFgoYJ3CSCDXAYx9Rj4uJptP-nHcmwFQlSU0XhbkTmd3s3b-O4nJHoPQ/s1600/bank-customer-churn-prediction-2.png)
## Conclusions

* The datasets contain no missing values and duplicates. Moreover, several outliers have been filtered
* The neural network model achieved an accuracy of 87.0% in correctly classifying the testing data for the bank customer churn prediction project
* This accuracy is achieved by these configurations: 1 input with 11 neurons, 1 hidden layer and dropout with 6 neurons, and 1 output with 1 neuron
* A successful web application has been deployed in replit.com
