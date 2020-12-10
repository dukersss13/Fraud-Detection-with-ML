# Fraud Detection with ML
 Detecting Frauds using Random Forest Classifier & Neural Network
# Abstract
The following model is an approach to precisely and accurately detect fradulent transactions using a simulated financial dataset. Two different variations of the dataset (will be elaborated in the Final Report) were trained on a Random Forest classifier & a Neural Network. Although the data was synthetically generated, it is assumed that the samples resemble actual operations of real financial activities.
# Data
The main dataset was obtained from Kaggle. It contains over 6 million samples and has 11 columns. Due to the confidentiality of real financial data, the data was simulated by PaySim simulator and does not contain actual transactions. More details about the dataset can be found in the Final Report.

Source: https://www.kaggle.com/ntnu-testimon/paysim1

# Description
Due to the imbalance nature of the dataset, the discrepancy of the target variable could make it very difficult for any classification model to consistently predict at a high level. In addition, it is also important to note that the overall accuracy of any algorithm when trained on an imbalance dataset should be further analyzed and not taken based upon face value. In other words, if all the model is required to do is correctly label the 99.99% of non-fraudulent transactions but fail to pinpoint frauds, then it defeats the sole purpose of having this classification algorithm. 

To overcome this potential issue, we applied SMOTE to the minority sample size in order to achieve a more balanced dataset. Subsequently, we run the two variations of the training data (without SMOTE & with SMOTE) through a Neural Network and Random Forest classifier and compare the results. Please refer to the Final Report for further details and analysis.
