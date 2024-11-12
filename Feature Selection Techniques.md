#𝗣𝘆𝘁𝗵𝗼𝗻 𝗖𝗼𝗱𝗲 𝗜𝗺𝗽𝗹𝗲𝗺𝗲𝗻𝘁𝗮𝘁𝗶𝗼𝗻 for Feature Selection

## Python Libraries

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, mutual_info_regression, VarianceThreshold, RFE

from sklearn.ensemble import RandomForestClassifier


#1. 𝗙𝗲𝗮𝘁𝘂𝗿𝗲 𝗥𝗲𝗹𝗮𝘁𝗶𝗼𝗻𝘀𝗵𝗶𝗽𝘀

## Load your dataset

data = pd.read_csv('dataset.csv')

correlation_matrix = data.corr()

# Set a threshold to identify highly correlated features

high_corr_features = [col for col in correlation_matrix.columns if any(correlation_matrix[col] > 0.8)]


#2. 𝗖𝗵𝗶-𝘀𝗾𝘂𝗮𝗿𝗲 𝗧𝗲𝘀𝘁 / 𝗔𝗡𝗢𝗩𝗔 𝗧𝗲𝘀𝘁

# Chi-square for categorical target, e.g., Loan_Approval

X, y = data.drop(columns=['Loan_Approval']), data['Loan_Approval']

y_encoded = LabelEncoder().fit_transform(y)

# Perform Chi-square test

chi2_scores, _ = chi2(X, y_encoded)

# For ANOVA with continuous targets

f_scores, _ = f_classif(X, y_encoded)


#3. 𝗠𝘂𝘁𝘂𝗮𝗹 𝗜𝗻𝗳𝗼𝗿𝗺𝗮𝘁𝗶𝗼𝗻

# For classification

mutual_info = mutual_info_classif(X, y_encoded)

# For regression

# mutual_info = mutual_info_regression(X, y)


#4. 𝗩𝗮𝗿𝗶𝗮𝗻𝗰𝗲 𝗧𝗵𝗿𝗲𝘀𝗵𝗼𝗹𝗱

# Set a threshold for variance (e.g., 0.01)

selector = VarianceThreshold(threshold=0.01)

X_selected = selector.fit_transform(X)


#5. 𝗥𝗲𝗰𝘂𝗿𝘀𝗶𝘃𝗲 𝗙𝗲𝗮𝘁𝘂𝗿𝗲 𝗘𝗹𝗶𝗺𝗶𝗻𝗮𝘁𝗶𝗼𝗻 (𝗥𝗙𝗘)

# Initialize model and RFE

model = RandomForestClassifier()

rfe = RFE(estimator=model, n_features_to_select=5)

X_selected = rfe.fit_transform(X, y_encoded)

