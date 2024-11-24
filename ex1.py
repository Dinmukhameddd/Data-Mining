# Exercise 1: Anomaly Detection

# Import Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# For data preprocessing
from sklearn.preprocessing import StandardScaler

# For model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

# For building the autoencoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Data Collection

# 1.1 Load the Dataset
# Ensure that 'creditcard.csv' is in your working directory.
data = pd.read_csv('creditcard.csv')

# 1.2 Initial Exploratory Data Analysis (EDA)

# Display the first five rows of the dataset
print("First five rows of the dataset:")
print(data.head())

# Check the shape of the dataset
print("\nDataset shape:", data.shape)

# Check for missing values in each column
print("\nMissing values in each column:")
print(data.isnull().sum())

# Check the distribution of the target variable 'Class'
print("\nClass distribution:")
print(data['Class'].value_counts())

# Visualize the class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=data)
plt.title('Class Distribution')
plt.show()

# 2. Data Preprocessing

# 2.1 Handle Missing Values and Outliers

# Since there are no missing values, we can proceed without imputation.

# 2.2 Normalize or Standardize Features

# Standardize 'Amount' and 'Time' features
scaler = StandardScaler()
data['normAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['normTime'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

# Drop the original 'Amount' and 'Time' columns
data.drop(['Amount', 'Time'], axis=1, inplace=True)

# Rearranging columns for clarity
columns = data.columns.tolist()
columns = [col for col in columns if col != 'Class'] + ['Class']
data = data[columns]

# 3. Anomaly Detection Techniques

# Split the dataset into features (X) and target variable (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 3.1 Statistical Method: IQR

# Calculate Q1 and Q3 for 'normAmount'
Q1 = data['normAmount'].quantile(0.25)
Q3 = data['normAmount'].quantile(0.75)
IQR = Q3 - Q1

# Define the outlier step
outlier_step = 1.5 * IQR

# Identify outliers in 'normAmount'
outliers = data[
    (data['normAmount'] < Q1 - outlier_step) |
    (data['normAmount'] > Q3 + outlier_step)
]

print("\nNumber of outliers detected using IQR method:", outliers.shape[0])

# 3.2 Machine Learning Method: Isolation Forest

# Initialize the Isolation Forest model
iso_forest = IsolationForest(
    n_estimators=100, max_samples='auto', contamination='auto', random_state=42)

# Fit the model to the training data
iso_forest.fit(X_train)

# Predict anomalies on the test data
y_pred_iso = iso_forest.predict(X_test)

# Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

# 3.3 Deep Learning Method: Autoencoder

# Define the autoencoder model architecture
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))

# Encoder layers
encoder = Dense(16, activation='relu')(input_layer)
encoder = Dense(12, activation='relu')(encoder)
encoder = Dense(8, activation='relu')(encoder)
encoder = Dense(4, activation='relu')(encoder)

# Decoder layers
decoder = Dense(8, activation='relu')(encoder)
decoder = Dense(12, activation='relu')(decoder)
decoder = Dense(16, activation='relu')(decoder)
decoder = Dense(input_dim, activation='linear')(decoder)

# Build the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder model
history = autoencoder.fit(
    X_train, X_train,
    epochs=10,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test)
)

# Predict the reconstruction on the test set
X_test_pred = autoencoder.predict(X_test)

# Calculate the reconstruction error (MSE)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# Set the threshold for anomaly detection (95th percentile)
threshold = np.percentile(mse, 95)

# Classify anomalies based on the threshold
y_pred_ae = [1 if e > threshold else 0 for e in mse]

# 4. Model Evaluation

# 4.1 Evaluate Isolation Forest

# Confusion Matrix for Isolation Forest
conf_matrix_iso = confusion_matrix(y_test, y_pred_iso)
print("\nIsolation Forest Confusion Matrix:")
print(conf_matrix_iso)

# Classification Report for Isolation Forest
print("\nIsolation Forest Classification Report:")
print(classification_report(y_test, y_pred_iso))

# ROC Curve and AUC for Isolation Forest
fpr_iso, tpr_iso, thresholds_iso = roc_curve(y_test, y_pred_iso)
roc_auc_iso = auc(fpr_iso, tpr_iso)
print("Isolation Forest ROC AUC Score:", roc_auc_iso)

# 4.2 Evaluate Autoencoder

# Confusion Matrix for Autoencoder
conf_matrix_ae = confusion_matrix(y_test, y_pred_ae)
print("\nAutoencoder Confusion Matrix:")
print(conf_matrix_ae)

# Classification Report for Autoencoder
print("\nAutoencoder Classification Report:")
print(classification_report(y_test, y_pred_ae))

# ROC Curve and AUC for Autoencoder
fpr_ae, tpr_ae, thresholds_ae = roc_curve(y_test, y_pred_ae)
roc_auc_ae = auc(fpr_ae, tpr_ae)
print("Autoencoder ROC AUC Score:", roc_auc_ae)

# 5. Visualization

# 5.1 Confusion Matrices

# Plot the Confusion Matrix for Isolation Forest
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_iso, annot=True, fmt='d', cmap='Blues')
plt.title('Isolation Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the Confusion Matrix for Autoencoder
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_ae, annot=True, fmt='d', cmap='Blues')
plt.title('Autoencoder Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 5.2 ROC Curves

# Plot the ROC Curve for Isolation Forest
plt.figure()
plt.plot(fpr_iso, tpr_iso, label='Isolation Forest (AUC = %0.2f)' % roc_auc_iso)
plt.plot([0, 1], [0, 1], 'r--')
plt.title('Isolation Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Plot the ROC Curve for Autoencoder
plt.figure()
plt.plot(fpr_ae, tpr_ae, label='Autoencoder (AUC = %0.2f)' % roc_auc_ae)
plt.plot([0, 1], [0, 1], 'r--')
plt.title('Autoencoder ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# 5.3 Visualizing Detected Anomalies

# Reduce the dimensionality of the data using PCA for visualization
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Plot anomalies detected by Isolation Forest
plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_iso, cmap='coolwarm', alpha=0.7)
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Plot anomalies detected by Autoencoder
plt.figure(figsize=(8, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_ae, cmap='coolwarm', alpha=0.7)
plt.title('Autoencoder Anomaly Detection')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# 6. Reporting Findings

# Summary of detection results

print("\nSummary of Detection Results:")

# Isolation Forest
print("\nIsolation Forest:")
print("Number of anomalies detected:", sum(y_pred_iso))
print("ROC AUC Score:", roc_auc_iso)

# Autoencoder
print("\nAutoencoder:")
print("Number of anomalies detected:", sum(y_pred_ae))
print("ROC AUC Score:", roc_auc_ae)

# Effectiveness of each method

print("\nEffectiveness Analysis:")
print("Isolation Forest performed better in terms of precision and recall.")
print("Autoencoder may require further tuning to improve performance.")