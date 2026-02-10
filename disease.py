
Lifestyle Disease (Diabetes) Prediction Using Machine Learning
This project implements an end-to-end machine learning pipeline to predict the risk of lifestyle disease (Diabetes) using health indicators from the BRFSS 2015 dataset.
The system performs data loading, exploratory data analysis,
preprocessing, model training, evaluation, cross-validation, visualization, and prediction, with model persistence for future use.
 Project Objective
The main goal of this project is to predict whether a person is at high risk of diabetes based on lifestyle and health-related features such as BMI, 
blood pressure, cholesterol, physical activity, mental health, and age.
Two machine learning models are trained and compared:
Random Forest Classifier
XGBoost Classifier
 Dataset

Source: BRFSS 2015 (Behavioral Risk Factor Surveillance System)
Target Variable: Diabetes_binary
0 → Healthy
1 → Diabetes
Features Used (18 total):
HighBP, HighChol, BMI, Smoker, Stroke
HeartDiseaseorAttack, PhysActivity, Fruits, Veggies
HvyAlcoholConsump, AnyHealthcare, NoDocbcCost
GenHlth, MentHlth, PhysHlth, DiffWalk
Sex, Age
1.Workflow Overview
1.Data Loading
Reads the dataset using Pandas
Displays dataset shape for verification
2.Exploratory Data Analysis (EDA)
Summary statistics of all features
Target class distribution visualization
Correlation heatmap to understand feature relationships
3. Data Preprocessing
Target column renamed to disease
Feature selection based on domain relevance
StandardScaler used for feature normalization
Dataset split using stratified train-test split
️4. Model Training
Random Forest Classifier
XGBoost Classifier
Models trained with fixed random state for reproducibility
5️.Model Evaluation
Accuracy score
Precision, Recall, F1-Score
Classification report
ROC-AUC score (for probabilistic models)
6. Cross-Validation
5-Fold cross-validation using accuracy as the metric
Helps measure model stability and generalization
️7. Visualization
Confusion matrix (saved as image)
Feature importance plot (XGBoost)
️8. Prediction System
Supports:
Sample prediction
User-input based prediction
Outputs:
Risk label (High / Low)
Probability score (if available)
️9.Model Persistence
Trained XGBoost model saved using Joblib
Scaler saved separately for consistent future predictions
Key Highlights
End-to-end ML pipeline in a single executable script
Comparison between ensemble learning models
Proper feature scaling and stratified sampling
Reusable prediction function for real-world deployment
Visual outputs saved automatically
Clean, modular, and production-ready code structure
Tech Stack
Programming Language: Python
Libraries & Frameworks:
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
XGBoost
Joblib
