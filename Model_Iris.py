import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load the dataset
df = pd.read_csv('IRIS.csv')

# Separate features and target variable
X = df.drop(columns=['species'])
y = df['species']

# Feature scaling (standardization)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize the Logistic Regression model
log_reg = LogisticRegression(random_state=42, max_iter=200)

# Perform cross-validation (5 folds)
cv_scores = cross_val_score(log_reg, X, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean CV Accuracy: {cv_scores.mean():.2f}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Final Model Accuracy: {accuracy:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model to a file
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(log_reg, file)

# Save the scaler to a file (since it is necessary for consistent preprocessing)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)