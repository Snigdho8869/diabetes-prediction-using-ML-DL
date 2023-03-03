# diabetes-prediction-using-ML-DL
This repository contains machine learning and deep learning models for predicting the onset of diabetes in patients. The models used include Logistic Regression, SVM, Random Forest, Ensemble Learning, Adaptive Boosting, and Deep Neural Networks. The performance of the models is compared, and the best performing model is identified.
This project aims to predict the onset of diabetes in patients based on various features such as age, BMI, blood pressure, etc. I have used several Machine Learning and Deep Learning models to predict whether a patient is diabetic or not.

The following machine learning algorithms are used in this project:

Logistic Regression

Support Vector Machines (SVM)

Random Forest (RF)

GradientBoosting

Ensemble Learning (using voting classifier)

AdaBoost

In addition, a deep neural network (DNN) is also used for diabetes prediction.

Requirements:
The following libraries are required to run the project:

numpy

pandas

scikit-learn

tensorflow

keras

You can install the required libraries using the following command:

Copy code

pip install numpy pandas scikit-learn tensorflow keras

Dataset

The dataset used in this project contains 8 features and 1 target variable, which is the onset of diabetes. The features are as follows:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

Models:

Logistic Regression:

Logistic regression is a simple and interpretable linear classifier. In this project, logistic regression is used to classify whether a person has diabetes or not based on the 8 features in the dataset.

Support Vector Machines:

Support vector machines (SVM) are a popular classification algorithm used in many applications. In this project, SVM is used with a linear kernel to classify whether a person has diabetes or not.

Random Forest:

Random Forest is a powerful machine learning algorithm that is used for both classification and regression tasks. In this project, a random forest classifier is used to predict the onset of diabetes.

GradientBoosting:

Gradient boosting is a machine learning algorithm used for regression and classification problems. It involves combining multiple weak predictive models, typically decision trees, into a strong predictive model.

Ensemble Learning:

Ensemble learning is a machine learning technique where multiple models are trained to solve the same problem and their predictions are combined to produce a final prediction. In this project, two ensemble learning techniques are used:

Voting Classifier: 

A voting classifier is an ensemble learning technique where multiple models are trained and their predictions are combined using a majority vote.

AdaBoost:

AdaBoost is a boosting algorithm that is used to improve the accuracy of a weak classifier. In this project, AdaBoost is used with a decision tree classifier as the base estimator to classify whether a person has diabetes or not.

Deep Neural Network:

A deep neural network (DNN) is a type of artificial neural network that is capable of learning complex patterns from data. In this project, a DNN is used to predict the onset of diabetes.

Results:

The results of the different models are compared based on their accuracy score, precision, recall, and F1-score. The best performing model is selected based on the evaluation metrics. And the best performing models are RF,DNN.

Conclusion:

This project demonstrates the use of machine learning and deep learning models to predict the onset of diabetes. The different models are evaluated and compared based on their accuracy score, precision, recall, and F1-score. The best performing model can be used to predict the onset of diabetes in patients and can help healthcare professionals take preventive measures.
