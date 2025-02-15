# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:12:42 2025

@author: MUSTAFA
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv('Ads.csv')
print(df.head())
print(df.isna().sum())
print(df.info())
print(df.describe())

# Display unique values for categorical columns
categorical_cols = ['Platform', 'Content_Type', 'Target_Age', 'Target_Gender', 'Region']
for col in categorical_cols:
    print(f"\n{col} distribution:")
    print(df[col].value_counts())

#============================Visualizations======================================
def one_variable_charts(charttype, x, color='Red'):
    if charttype == 'histplot':
        plt.figure(figsize=(10,8))
        sns.histplot(df, x=x, color=color, kde=True)
        plt.title(f'{x} distribution')
        plt.xlabel(f'{x}')
        plt.ylabel('Frequency')
        plt.show()
    elif charttype == 'countplot':
        plt.figure(figsize=(10,8))
        sns.countplot(df, x=x, color=color)
        plt.title(f'{x} frequency trend')
        plt.show()
    elif charttype == 'kdeplot':
        plt.figure(figsize=(10,8))
        sns.kdeplot(df, x=x, color=color, fill=True, alpha=0.5, linewidth=2)
        plt.title(f'{x} distribution')
        plt.show()
    else:
        print('Error in one of the values!')
        
def categorical_success_chart(x):
    plt.figure(figsize=(10,8))
    sns.barplot(x=x, y='Success', data=df, palette='viridis')
    plt.title(f"Success Rate by {x}")
    plt.show()

# Plotting different charts
one_variable_charts('histplot', 'Clicks', 'Red')
one_variable_charts('histplot', 'Duration', 'Blue')
one_variable_charts('histplot', 'Budget', 'Green')
one_variable_charts('histplot', 'CPC', 'Orange')
one_variable_charts('kdeplot', 'CPC', 'Orange')
one_variable_charts('histplot', 'CTR', 'Pink')
one_variable_charts('kdeplot', 'CTR', 'Pink')
one_variable_charts('histplot', 'Conversion_Rate', 'Purple')
one_variable_charts('kdeplot', 'Conversion_Rate', 'Purple')

# Success charts for categorical columns
categorical_cols = ['Platform', 'Content_Type', 'Target_Age', 'Target_Gender', 'Region']
for col in categorical_cols:
    categorical_success_chart(col)

# Count plots for categorical columns
for col in categorical_cols:
    one_variable_charts('countplot', col, color='grey')

#============================Feature Engineering======================================
df.drop(columns=["Campaign_ID"], inplace=True)  # Drop unique identifier column that we won't use

# Creating new features
df["Daily_Budget"] = df["Budget"] / df["Duration"]
df["Click_to_Conversion_Rate"] = df["Conversions"] / df["Clicks"]
df["Click_to_Conversion_Rate"].fillna(0, inplace=True)  # Replace NaN values with 0
df["Region_Age_Group"] = df["Region"] + "_" + df["Target_Age"]

# Convert categorical variables into numerical ones
df = pd.get_dummies(df, columns=["Platform", "Content_Type", "Target_Age", "Target_Gender", "Region"], drop_first=True)

# Convert boolean columns to integers (0 or 1)
df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)

# Scale numerical variables
scaler = StandardScaler()
num_cols = ["Budget", "Duration", "Clicks", "Conversions", "CTR", "CPC", "Conversion_Rate", "Daily_Budget", "Click_to_Conversion_Rate"]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Split data into train and test sets
x = df.drop(['Region_Age_Group', 'Success'], axis=1)
y = df['Success']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Building the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train, epochs=40, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# ROC-AUC
y_prob = model.predict(x_test)  # Model's predicted probabilities
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(12,12))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC - {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 0.1])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

print(f"ROC-AUC Score: {roc_auc:.4f}")





        

        
        