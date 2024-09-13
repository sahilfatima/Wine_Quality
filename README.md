# 🍷 Wine Quality Prediction Using SVM

Welcome to the **Wine Quality Prediction** project! This repository contains the code for training and evaluating a Support Vector Machine (SVM) model to predict the quality of wine based on various chemical properties. By analyzing wine features such as acidity, sugar, and alcohol content, we aim to classify wines into different quality levels. Let's dive into the details! 🎉

## 🌟 Objective

The main goal of this project is to predict wine quality (on a scale of 3-8) using a dataset of various chemical properties of wine samples. We employ machine learning techniques like **Support Vector Machines (SVM)** and fine-tune the model using **Grid Search** to achieve the best performance.

---

## 📂 Dataset Overview

The dataset used for this project contains 1143 instances of wine samples with 13 columns representing different chemical features and the quality score of the wine. The key features are:

- **Fixed Acidity**: Determines the sour taste in wine.
- **Volatile Acidity**: Impacts the vinegary taste.
- **Citric Acid**: Adds freshness and protects from spoilage.
- **Residual Sugar**: Sweetness of the wine.
- **Chlorides**: Salt content in wine.
- **Free Sulfur Dioxide**: Protects wine from oxidation and bacterial spoilage.
- **Total Sulfur Dioxide**: Combined SO2 levels in wine.
- **Density**: Linked to alcohol and sugar content.
- **pH**: Acidity or alkalinity of the wine.
- **Sulphates**: Contributes to wine preservation.
- **Alcohol**: Alcohol content of the wine.
- **Quality**: The quality score of wine (our target label).
  
---

## ⚙️ Project Workflow

The following steps outline the workflow of the project:

### 1. **Library Imports**
We start by importing essential libraries for data manipulation, visualization, and modeling:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
```

### 2. **Loading and Exploring the Dataset**
We load the wine quality dataset, explore its structure using `data.head()` and `data.info()` to understand the various features and ensure there are no missing values.

### 3. **Data Visualization** 📊
We generate visualizations to understand the relationships between various features and the target label:
- **Correlation Heatmap**: Visualizes correlations between different features.
- **Boxplot**: Compares alcohol content across different quality levels.
- **Scatter Plot**: Visualizes the relationship between fixed acidity and residual sugar, colored by quality.

### 4. **Data Preprocessing**
- The **target variable** is the `quality` column, while all other features are used as predictors.
- We remove the `Id` column (irrelevant for modeling) and standardize the data to prepare it for SVM training.

### 5. **Splitting Data for Training and Testing**
We split the dataset into **training** and **testing** sets (80-20 split):
```python
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 6. **Training the SVM Model** 🧠
We use the **SVC** model with the Radial Basis Function (RBF) kernel:
```python
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)
```

### 7. **Initial Model Performance**
The initial SVM model achieves an accuracy of **64%**, and the classification report shows the precision, recall, and F1-scores for different quality levels. There are some challenges with predicting rare quality levels (like 4 or 8), leading to low scores for those categories.

### 8. **Fine-Tuning the SVM Model with Grid Search** 🔧
We perform **Grid Search** to find the best hyperparameters (C, gamma, and kernel) for the SVM model:
```python
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5, verbose=1, n_jobs=-1)
svm_grid_search.fit(X_train_scaled, y_train)
```

The best parameters found are:
- **C**: 1
- **Gamma**: 0.1
- **Kernel**: 'rbf'

After tuning, the model achieves an improved accuracy of **65%**. While the accuracy isn't drastically higher, the fine-tuning helps better generalize the model.

---

## 📊 Results and Evaluation

- **Accuracy** before tuning: 64%
- **Accuracy** after tuning: 65%
- The model struggles with predicting some quality levels like **4** and **8** due to data imbalance, but performs well for more common quality levels like **5** and **6**.

Here’s a summary of the performance after tuning:
```plaintext
SVM Classification Report after Grid Search:
              precision    recall  f1-score   support
           4       0.00      0.00      0.00         9
           5       0.67      0.77      0.71       143
           6       0.63      0.67      0.65       146
           7       0.73      0.39      0.51        41
           8       0.00      0.00      0.00         4
    accuracy                           0.65       343
```

The main takeaway is that the SVM model can reasonably predict wine quality for the most common categories but struggles with rare ones due to imbalanced data. 

---

## 💡 Key Learnings

1. **Support Vector Machines (SVM)** are powerful classifiers, especially for smaller datasets, as they create decision boundaries to maximize margin between classes.
2. **Data Standardization** is essential when using SVMs as the model is sensitive to the scale of features.
3. **Grid Search** is a helpful tool for tuning hyperparameters and improving model performance.
4. **Class Imbalance** can affect the model’s ability to generalize for minority classes. Advanced techniques like **oversampling** or **class weighting** may help improve this.

---



## 📈 Future Improvements

1. **Handle Class Imbalance**: Techniques like SMOTE (Synthetic Minority Oversampling Technique) can be applied to handle class imbalance.
2. **Explore More Models**: Try other algorithms like Random Forest, Gradient Boosting, or XGBoost to compare performance.
3. **Feature Engineering**: Create new features or use feature selection techniques to improve model accuracy.

---

## 🎯 Conclusion

This project demonstrates the potential of machine learning in predicting wine quality based on chemical features. With further improvements and exploration, we can enhance model performance and gain more insights into what makes a wine great! 🍇🍷

---

Feel free to explore, experiment, and contribute to this project! Let's make wine prediction fun and insightful. 😊🍾

--- 

#### Made with ❤️ by [Sahil Fatima]
