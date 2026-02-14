Heart Disease Classification System
a. Problem Statement
The goal of this project is to implement and evaluate six different Machine Learning classification models to predict the presence of heart disease. In a medical context, the objective is to maximize the model's ability to correctly identify patients with heart disease (high Recall) to ensure timely intervention.

b. Dataset Description
Source: UCI Heart Disease Dataset (Cleveland)

Features: 16 (including id, age, sex, cp, thalach, oldpeak, etc.)

Instances: 920

Target Variable: num (Mapped to binary 0 for Healthy and 1 for Heart Disease)

Preprocessing: Handled missing values via median/mode imputation, performed feature scaling using StandardScaler, and categorical encoding via OneHotEncoder.

c. Models Used and Evaluation Metrics
I implemented six models using a Scikit-learn Pipeline to ensure robust preprocessing for each. 

Model Name,Accuracy,AUC,Precision,Recall,F1,MCC
Logistic Regression,0.842,0.913,0.877,0.853,0.865,0.676
Decision Tree,0.793,0.801,0.874,0.761,0.814,0.591
kNN,0.875,0.906,0.884,0.908,0.896,0.740
Naive Bayes,0.804,0.902,0.876,0.780,0.825,0.610
Random Forest,0.880,0.940,0.914,0.881,0.897,0.755
XGBoost,0.842,0.933,0.892,0.835,0.863,0.680

Observations about Model Performance

ML Model Name,Observation about model performance
Logistic Regression,"Performed well as a baseline, indicating that clinical features like age and blood pressure have a strong linear relationship with the target."
Decision Tree,"Showed the lowest Accuracy and MCC. It likely struggled with the small dataset size, leading to some overfitting compared to ensemble methods."
kNN,Achieved the highest Recall (0.908). This is critical in medical diagnosis as it minimizes False Negatives (missing a sick patient).
Naive Bayes,"Solid AUC (0.902), but lower Recall. It serves as a fast and efficient classifier despite its feature independence assumption."
Random Forest,Best overall model. Highest Accuracy (0.880) and AUC (0.940). It effectively reduced variance and handled non-linear relationships in the medical data.
XGBoost,Very robust performance with the second-highest AUC (0.933). It showed strong class separation capabilities.

Conclusion
After evaluating six models, the Random Forest classifier proved to be the most robust for this dataset. However, for clinical screening where missing a case is dangerous, kNN's high recall makes it a secondary model of interest.