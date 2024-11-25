# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:33:10 2024

@author: User
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from catboost import CatBoostClassifier

os.chdir('C://Users//User//Desktop//人工智慧')


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# class_transform = {-1:0, 0:1, 1:2}

def create_lagged_features(df, days=30):
    features = []
    targets = []
    
    for i in range(days-1, len(df)):
        feature_row = []
        target_1d = df.iloc[i]['1_trend']
        # target_5d = df.iloc[i]['5_trend']
        #target_10d = df.iloc[i]['10_trend']
        # Concat features from day 1 to day 30.
        for j in range(days-1, -1, -1):
            feature_row.extend([
                df.iloc[i - j]['close'],
                # df.iloc[i - j]['open'],
                # df.iloc[i - j]['high'],
                # df.iloc[i - j]['low'],
                # df.iloc[i - j]['volume']
            ])
        features.append(feature_row)
        targets.append([target_1d])
    
    return np.array(features), np.array(targets)

# Create feature
features, targets = create_lagged_features(train_df, days=30)


# Use for Standardization
scaler = StandardScaler()

# Split train set and validation set
X_train, y_train = features[:round(len(features) * 0.8)], targets[:round(len(targets) * 0.8)]
X_val, y_val = features[round(len(features) * 0.8):], targets[round(len(targets) * 0.8):] 

# Standardization
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = CatBoostClassifier(n_estimators=500, max_depth=10, learning_rate=0.01)
# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_val)


# Generate Classification Result
print(f"Report for 1_trend:")
print(classification_report(y_val[:], y_pred[:], zero_division=True))
print('---------------------------------------------------')
# A warning is UndefinedMetricWarning, this means some class have 0 predictions by model.
# You can set zero_division to avoid warning, but this may lead to higher F1 score.

output_train = model.predict(X_train)
output_val = model.predict(X_val)

train_f1 = f1_score(y_train.flatten(),output_train.flatten(), average='macro')
val_f1 = f1_score(y_val.flatten(), output_val.flatten(), average='macro')

print(f"Training F1: {train_f1}")
print(f"Valid F1: {val_f1}")



# print the order of output classes
class_map ={-1:"decline", 0:"unchanged", 1:"rise"}
class_order = [class_map[model.classes_[0]], class_map[model.classes_[1]], class_map[model.classes_[2]]]

# The code here draws 1 label(1_trend) chart
prob_1_trend = model.predict_proba(X_val)

# Draw Prediction Probability Distribution 
plt.figure(figsize=(10, 6))

for i in range(prob_1_trend.shape[1]):
    plt.hist(prob_1_trend[:, i], bins=20, alpha=0.5, label=class_order[i])

plt.title('Prediction Probability Distribution', fontsize=16)
plt.xlabel('Probability', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend()
plt.show()
# The code here draws 1 label(1_trend) chart
# Draw confusion_matrix
cm = confusion_matrix(y_val[:], y_pred[:])
# 繪製混淆矩陣
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['decline', 'unchanged', 'rise'], yticklabels=['decline', 'unchanged', 'rise'])
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.show()


######################################################3
def create_test_features(df, days=30):
    
    features = []
    for i in range(days-1, len(df)+1, days):
        feature_row = []

        for j in range(days-1, -1, -1):
            feature_row.extend([
                df.iloc[i - j]['close'],
                # df.iloc[i - j]['open'],
                # df.iloc[i - j]['high'],
                # df.iloc[i - j]['low'],
                # df.iloc[i - j]['volume']
            ])

        features.append(feature_row)
    
    return np.array(features)

test_features = create_test_features(test_df, days=30)



standard_features = scaler.transform(test_features)
predictions = model.predict(test_features)








