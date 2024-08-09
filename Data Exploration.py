import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('IRIS.csv')
print(df.head(5))

# Display statistical data
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Data Visualization

# Histograms
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal_length'].dropna(), bins=10, kde=False)
plt.title('Sepal Length Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['petal_length'].dropna(), bins=10, kde=False)
plt.title('Petal Length Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['sepal_width'].dropna(), bins=10, kde=False)
plt.title('Sepal Width Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['petal_width'].dropna(), bins=10, kde=False)
plt.title('Petal Width Distribution')
plt.show()

# Scatter plots
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', palette='bright')
plt.title('Sepal Length vs Sepal Width')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species', palette='bright')
plt.title('Petal Length vs Petal Width')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', palette='bright')
plt.title('Sepal Length vs Petal Length')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='sepal_width', y='petal_width', hue='species', palette='bright')
plt.title('Sepal Width vs Petal Width')
plt.show()

# Display correlation matrix excluding the target variable
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=df, palette='bright')
plt.title('Boxplot of Sepal Length by Species')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_width', data=df, palette='bright')
plt.title('Boxplot of Sepal Width by Species')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal_length', data=df, palette='bright')
plt.title('Boxplot of Petal Length by Species')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal_width', data=df, palette='bright')
plt.title('Boxplot of Petal Width by Species')
plt.show()

