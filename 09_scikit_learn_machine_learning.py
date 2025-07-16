# 06 - Machine Learning with Scikit-learn
# Scikit-learn is the go-to library for machine learning in Python

"""
SCIKIT-LEARN
============

Scikit-learn provides:
- Simple and efficient tools for data mining and analysis
- Classification, regression, clustering algorithms
- Model selection and evaluation tools
- Data preprocessing and feature engineering
- Dimensionality reduction techniques
- Ensemble methods and model pipelines

Key Machine Learning Workflow:
1. Data Collection and Exploration
2. Data Preprocessing
3. Feature Engineering
4. Model Selection
5. Training and Validation
6. Model Evaluation
7. Hyperparameter Tuning
8. Final Model Deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, r2_score, silhouette_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print(f"Scikit-learn version: {sklearn.__version__}")

# =============================================================================
# 1. DATA LOADING AND EXPLORATION
# =============================================================================

print("="*60)
print("1. DATA LOADING AND EXPLORATION")
print("="*60)

# Load built-in datasets
iris = datasets.load_iris()
boston = datasets.load_boston()
wine = datasets.load_wine()

print("Available built-in datasets:")
print("- Iris: Classification (flower species)")
print("- Boston Housing: Regression (house prices)")
print("- Wine: Classification (wine quality)")

# Create DataFrame for easier handling
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(f"\nIris dataset shape: {iris_df.shape}")
print("First few rows:")
print(iris_df.head())

print(f"\nTarget distribution:")
print(iris_df['species'].value_counts())

# Basic statistics
print(f"\nBasic statistics:")
print(iris_df.describe())

# Visualize the data
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iris Dataset Exploration', fontsize=16, fontweight='bold')

# Pairplot-style visualization
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', 
                hue='species', ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length vs Width')

sns.scatterplot(data=iris_df, x='petal length (cm)', y='petal width (cm)', 
                hue='species', ax=axes[0, 1])
axes[0, 1].set_title('Petal Length vs Width')

# Distribution plots
sns.histplot(data=iris_df, x='sepal length (cm)', hue='species', ax=axes[1, 0])
axes[1, 0].set_title('Sepal Length Distribution')

sns.boxplot(data=iris_df, x='species', y='petal length (cm)', ax=axes[1, 1])
axes[1, 1].set_title('Petal Length by Species')

plt.tight_layout()
plt.show()

print("✅ Data exploration completed")

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================

print("\n" + "="*60)
print("2. DATA PREPROCESSING")
print("="*60)

# Create a dataset with missing values and mixed types
np.random.seed(42)
n_samples = 1000

# Generate synthetic customer data
customer_data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
    'experience': np.random.randint(0, 40, n_samples),
    'city': np.random.choice(['New York', 'LA', 'Chicago', 'Houston', 'Phoenix'], n_samples),
    'purchased': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
})

# Introduce missing values
missing_indices = np.random.choice(n_samples, size=50, replace=False)
customer_data.loc[missing_indices, 'income'] = np.nan

missing_indices_2 = np.random.choice(n_samples, size=30, replace=False)
customer_data.loc[missing_indices_2, 'education'] = np.nan

print("Dataset with missing values:")
print(customer_data.info())
print(f"\nMissing values per column:")
print(customer_data.isnull().sum())

# Handle missing values
print(f"\nHandling missing values...")

# For numerical columns: fill with median
customer_data['income'].fillna(customer_data['income'].median(), inplace=True)

# For categorical columns: fill with mode
customer_data['education'].fillna(customer_data['education'].mode()[0], inplace=True)

print("After handling missing values:")
print(customer_data.isnull().sum())

# Encode categorical variables
print(f"\nEncoding categorical variables...")

# Label encoding for ordinal data
education_order = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
customer_data['education_encoded'] = customer_data['education'].map(education_order)

# One-hot encoding for nominal data
city_encoded = pd.get_dummies(customer_data['city'], prefix='city')
customer_data = pd.concat([customer_data, city_encoded], axis=1)

print("Encoded features:")
print(customer_data[['education', 'education_encoded']].head())
print(f"\nOne-hot encoded cities: {city_encoded.columns.tolist()}")

# Feature scaling
print(f"\nFeature scaling...")
scaler = StandardScaler()

# Select numerical features for scaling
numerical_features = ['age', 'income', 'experience']
customer_data[numerical_features] = scaler.fit_transform(customer_data[numerical_features])

print("Scaled features (first 5 rows):")
print(customer_data[numerical_features].head())

print("✅ Data preprocessing completed")

# =============================================================================
# 3. CLASSIFICATION ALGORITHMS
# =============================================================================

print("\n" + "="*60)
print("3. CLASSIFICATION ALGORITHMS")
print("="*60)

# Prepare data for classification
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Initialize classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate classifiers
results = {}

for name, classifier in classifiers.items():
    # Train the model
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(classifier, X, y, cv=5)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }
    
    print(f"{name}:")
    print(f"  Test Accuracy: {accuracy:.3f}")
    print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print()

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy comparison
models = list(results.keys())
accuracies = [results[model]['accuracy'] for model in models]
cv_means = [results[model]['cv_mean'] for model in models]

x_pos = np.arange(len(models))
ax1.bar(x_pos, accuracies, alpha=0.7, label='Test Accuracy')
ax1.bar(x_pos, cv_means, alpha=0.7, label='CV Mean')
ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy')
ax1.set_title('Model Comparison')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Confusion matrix for best model
best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_predictions = results[best_model]['predictions']

cm = confusion_matrix(y_test, best_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title(f'Confusion Matrix - {best_model}')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.show()

print(f"✅ Best performing model: {best_model}")
print(f"Classification report for {best_model}:")
print(classification_report(y_test, best_predictions, target_names=iris.target_names))

# =============================================================================
# 4. REGRESSION ALGORITHMS
# =============================================================================

print("\n" + "="*60)
print("4. REGRESSION ALGORITHMS")
print("="*60)

# Load Boston housing dataset
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['price'] = boston.target

print("Boston Housing Dataset:")
print(boston_df.head())
print(f"Dataset shape: {boston_df.shape}")

# Prepare data
X_boston = boston.data
y_boston = boston.target

# Split the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_boston, y_boston, test_size=0.3, random_state=42)

# Initialize regressors
regressors = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

# Train and evaluate regressors
reg_results = {}

for name, regressor in regressors.items():
    # Train the model
    regressor.fit(X_train_reg, y_train_reg)
    
    # Make predictions
    y_pred_reg = regressor.predict(X_test_reg)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    # Cross-validation score
    cv_scores = cross_val_score(regressor, X_boston, y_boston, cv=5, scoring='r2')
    
    reg_results[name] = {
        'mse': mse,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred_reg
    }
    
    print(f"{name}:")
    print(f"  MSE: {mse:.3f}")
    print(f"  R² Score: {r2:.3f}")
    print(f"  CV R² Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print()

# Visualize regression results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Regression Model Comparison', fontsize=16, fontweight='bold')

# R² comparison
models_reg = list(reg_results.keys())
r2_scores = [reg_results[model]['r2'] for model in models_reg]
cv_r2_means = [reg_results[model]['cv_mean'] for model in models_reg]

x_pos = np.arange(len(models_reg))
axes[0, 0].bar(x_pos, r2_scores, alpha=0.7, label='Test R²')
axes[0, 0].bar(x_pos, cv_r2_means, alpha=0.7, label='CV Mean R²')
axes[0, 0].set_xlabel('Models')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_title('R² Score Comparison')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(models_reg, rotation=45)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MSE comparison
mse_scores = [reg_results[model]['mse'] for model in models_reg]
axes[0, 1].bar(models_reg, mse_scores, alpha=0.7, color='orange')
axes[0, 1].set_xlabel('Models')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Mean Squared Error Comparison')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Actual vs Predicted for best model
best_reg_model = max(reg_results.keys(), key=lambda x: reg_results[x]['r2'])
best_reg_predictions = reg_results[best_reg_model]['predictions']

axes[1, 0].scatter(y_test_reg, best_reg_predictions, alpha=0.7)
axes[1, 0].plot([y_test_reg.min(), y_test_reg.max()], 
                [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual Values')
axes[1, 0].set_ylabel('Predicted Values')
axes[1, 0].set_title(f'Actual vs Predicted - {best_reg_model}')
axes[1, 0].grid(True, alpha=0.3)

# Residuals plot
residuals = y_test_reg - best_reg_predictions
axes[1, 1].scatter(best_reg_predictions, residuals, alpha=0.7)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title(f'Residuals Plot - {best_reg_model}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"✅ Best performing regression model: {best_reg_model}")

# =============================================================================
# 5. CLUSTERING ALGORITHMS
# =============================================================================

print("\n" + "="*60)
print("5. CLUSTERING ALGORITHMS")
print("="*60)

# Use Iris dataset for clustering (without labels)
X_cluster = iris.data
true_labels = iris.target

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_cluster)

# Calculate silhouette score
silhouette_avg = silhouette_score(X_cluster, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# Find optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    cluster_labels_temp = kmeans_temp.fit_predict(X_cluster)
    inertias.append(kmeans_temp.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster, cluster_labels_temp))

# Visualize clustering results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Clustering Analysis', fontsize=16, fontweight='bold')

# Elbow method
axes[0, 0].plot(k_range, inertias, marker='o')
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Inertia')
axes[0, 0].set_title('Elbow Method')
axes[0, 0].grid(True, alpha=0.3)

# Silhouette scores
axes[0, 1].plot(k_range, silhouette_scores, marker='o', color='orange')
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score vs k')
axes[0, 1].grid(True, alpha=0.3)

# Clustering visualization (using first two features)
axes[1, 0].scatter(X_cluster[:, 0], X_cluster[:, 1], c=cluster_labels, cmap='viridis')
axes[1, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   c='red', marker='x', s=200, linewidths=3)
axes[1, 0].set_xlabel('Sepal Length')
axes[1, 0].set_ylabel('Sepal Width')
axes[1, 0].set_title('K-Means Clustering Results')

# True labels for comparison
axes[1, 1].scatter(X_cluster[:, 0], X_cluster[:, 1], c=true_labels, cmap='viridis')
axes[1, 1].set_xlabel('Sepal Length')
axes[1, 1].set_ylabel('Sepal Width')
axes[1, 1].set_title('True Species Labels')

plt.tight_layout()
plt.show()

print("✅ Clustering analysis completed")

# =============================================================================
# 6. DIMENSIONALITY REDUCTION
# =============================================================================

print("\n" + "="*60)
print("6. DIMENSIONALITY REDUCTION")
print("="*60)

# Apply PCA to Iris dataset
pca = PCA()
X_pca = pca.fit_transform(X_cluster)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("Explained variance ratio by component:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"  PC{i+1}: {ratio:.3f}")

print(f"\nCumulative explained variance:")
for i, ratio in enumerate(cumulative_variance_ratio):
    print(f"  First {i+1} components: {ratio:.3f}")

# Visualize PCA results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Principal Component Analysis', fontsize=16, fontweight='bold')

# Explained variance
axes[0, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Explained Variance Ratio')
axes[0, 0].set_title('Explained Variance by Component')
axes[0, 0].grid(True, alpha=0.3)

# Cumulative explained variance
axes[0, 1].plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
axes[0, 1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Cumulative Explained Variance')
axes[0, 1].set_title('Cumulative Explained Variance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 2D PCA visualization
scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='viridis')
axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
axes[1, 0].set_title('PCA: First Two Components')
plt.colorbar(scatter, ax=axes[1, 0])

# Feature importance in PC1 and PC2
feature_names = iris.feature_names
pc1_importance = np.abs(pca.components_[0])
pc2_importance = np.abs(pca.components_[1])

x_pos = np.arange(len(feature_names))
width = 0.35
axes[1, 1].bar(x_pos - width/2, pc1_importance, width, label='PC1', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, pc2_importance, width, label='PC2', alpha=0.7)
axes[1, 1].set_xlabel('Features')
axes[1, 1].set_ylabel('Absolute Loading')
axes[1, 1].set_title('Feature Importance in PC1 and PC2')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels([name.split()[0] for name in feature_names], rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("✅ Dimensionality reduction analysis completed")

# =============================================================================
# 7. MODEL PIPELINES AND HYPERPARAMETER TUNING
# =============================================================================

print("\n" + "="*60)
print("7. MODEL PIPELINES AND HYPERPARAMETER TUNING")
print("="*60)

# Create a pipeline for classification
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid for grid search
param_grid = {
    'pca__n_components': [2, 3, 4],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7, None]
}

# Perform grid search
print("Performing grid search...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)

print(f"Test accuracy with best model: {best_accuracy:.3f}")

# Feature importance from the best model
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    # Get PCA components used
    n_components = best_model.named_steps['pca'].n_components_
    feature_importance = best_model.named_steps['classifier'].feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_components), feature_importance)
    plt.xlabel('Principal Components')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance in Best Model')
    plt.xticks(range(n_components), [f'PC{i+1}' for i in range(n_components)])
    plt.grid(True, alpha=0.3)
    plt.show()

print("✅ Pipeline and hyperparameter tuning completed")

# =============================================================================
# 8. PRACTICAL EXAMPLE: COMPLETE ML PROJECT
# =============================================================================

print("\n" + "="*60)
print("8. PRACTICAL EXAMPLE: CUSTOMER CHURN PREDICTION")
print("="*60)

# Create synthetic customer churn dataset
np.random.seed(42)
n_customers = 2000

# Generate realistic customer data
churn_data = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'age': np.random.randint(18, 80, n_customers),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'monthly_charges': np.random.normal(65, 20, n_customers),
    'total_charges': np.random.normal(2000, 1000, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
    'tech_support': np.random.choice(['Yes', 'No'], n_customers),
    'online_security': np.random.choice(['Yes', 'No'], n_customers),
    'num_services': np.random.randint(1, 8, n_customers)
})

# Create realistic churn based on features
churn_probability = (
    0.1 +  # Base probability
    0.3 * (churn_data['contract_type'] == 'Month-to-month') +
    0.2 * (churn_data['tenure_months'] < 12) +
    0.15 * (churn_data['monthly_charges'] > 80) +
    0.1 * (churn_data['tech_support'] == 'No') +
    0.1 * (churn_data['payment_method'] == 'Electronic check') -
    0.1 * (churn_data['num_services'] > 4)
)

churn_data['churn'] = np.random.binomial(1, np.clip(churn_probability, 0, 1), n_customers)

print("Customer Churn Dataset:")
print(churn_data.head())
print(f"Dataset shape: {churn_data.shape}")
print(f"Churn rate: {churn_data['churn'].mean():.1%}")

# Exploratory Data Analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Churn Analysis', fontsize=16, fontweight='bold')

# Churn by contract type
churn_by_contract = churn_data.groupby('contract_type')['churn'].mean()
axes[0, 0].bar(churn_by_contract.index, churn_by_contract.values)
axes[0, 0].set_title('Churn Rate by Contract Type')
axes[0, 0].set_ylabel('Churn Rate')
axes[0, 0].tick_params(axis='x', rotation=45)

# Churn by tenure
axes[0, 1].hist([churn_data[churn_data['churn']==0]['tenure_months'],
                 churn_data[churn_data['churn']==1]['tenure_months']], 
                bins=20, alpha=0.7, label=['No Churn', 'Churn'])
axes[0, 1].set_title('Tenure Distribution by Churn')
axes[0, 1].set_xlabel('Tenure (months)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()

# Monthly charges vs churn
sns.boxplot(data=churn_data, x='churn', y='monthly_charges', ax=axes[0, 2])
axes[0, 2].set_title('Monthly Charges by Churn')
axes[0, 2].set_xticklabels(['No Churn', 'Churn'])

# Churn by internet service
churn_by_internet = churn_data.groupby('internet_service')['churn'].mean()
axes[1, 0].bar(churn_by_internet.index, churn_by_internet.values)
axes[1, 0].set_title('Churn Rate by Internet Service')
axes[1, 0].set_ylabel('Churn Rate')

# Age distribution
axes[1, 1].hist([churn_data[churn_data['churn']==0]['age'],
                 churn_data[churn_data['churn']==1]['age']], 
                bins=20, alpha=0.7, label=['No Churn', 'Churn'])
axes[1, 1].set_title('Age Distribution by Churn')
axes[1, 1].set_xlabel('Age')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()

# Correlation heatmap
numeric_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'num_services', 'churn']
correlation_matrix = churn_data[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
axes[1, 2].set_title('Correlation Matrix')

plt.tight_layout()
plt.show()

# Data preprocessing for ML
print("\nPreparing data for machine learning...")

# Encode categorical variables
le_contract = LabelEncoder()
le_payment = LabelEncoder()
le_internet = LabelEncoder()

churn_data['contract_encoded'] = le_contract.fit_transform(churn_data['contract_type'])
churn_data['payment_encoded'] = le_payment.fit_transform(churn_data['payment_method'])
churn_data['internet_encoded'] = le_internet.fit_transform(churn_data['internet_service'])
churn_data['tech_support_encoded'] = (churn_data['tech_support'] == 'Yes').astype(int)
churn_data['online_security_encoded'] = (churn_data['online_security'] == 'Yes').astype(int)

# Select features for modeling
feature_columns = ['age', 'tenure_months', 'monthly_charges', 'total_charges', 'num_services',
                  'contract_encoded', 'payment_encoded', 'internet_encoded', 
                  'tech_support_encoded', 'online_security_encoded']

X_churn = churn_data[feature_columns]
y_churn = churn_data['churn']

# Split the data
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.3, random_state=42, stratify=y_churn)

# Scale the features
scaler_churn = StandardScaler()
X_train_churn_scaled = scaler_churn.fit_transform(X_train_churn)
X_test_churn_scaled = scaler_churn.transform(X_test_churn)

# Train multiple models
churn_models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

churn_results = {}

for name, model in churn_models.items():
    # Train model
    if name == 'SVM':
        model.fit(X_train_churn_scaled, y_train_churn)
        y_pred_churn = model.predict(X_test_churn_scaled)
        y_pred_proba = model.predict_proba(X_test_churn_scaled)[:, 1]
    else:
        model.fit(X_train_churn, y_train_churn)
        y_pred_churn = model.predict(X_test_churn)
        y_pred_proba = model.predict_proba(X_test_churn)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_churn, y_pred_churn)
    
    churn_results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred_churn,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} Accuracy: {accuracy:.3f}")

# Model comparison and final results
best_churn_model = max(churn_results.keys(), key=lambda x: churn_results[x]['accuracy'])
print(f"\nBest model: {best_churn_model}")
print(f"Best accuracy: {churn_results[best_churn_model]['accuracy']:.3f}")

# Feature importance for Random Forest
if 'Random Forest' in churn_models:
    rf_model = churn_models['Random Forest']
    feature_importance = rf_model.feature_importances_
    
    plt.figure(figsize=(12, 6))
    sorted_idx = np.argsort(feature_importance)[::-1]
    plt.bar(range(len(feature_importance)), feature_importance[sorted_idx])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.xticks(range(len(feature_importance)), 
               [feature_columns[i] for i in sorted_idx], rotation=45)
    plt.tight_layout()
    plt.show()

print("✅ Complete ML project example completed")

print("\n" + "="*70)
print("SCIKIT-LEARN SUMMARY")
print("="*70)
print("✅ Comprehensive machine learning library")
print("✅ Classification, regression, and clustering algorithms")
print("✅ Data preprocessing and feature engineering tools")
print("✅ Model selection and evaluation metrics")
print("✅ Pipeline creation for streamlined workflows")
print("✅ Hyperparameter tuning with GridSearchCV")
print("✅ Dimensionality reduction techniques")
print("✅ Easy-to-use API with consistent interface")
print("\nNext: Move to file 07 - Multi-core Parallelization!")
print("="*70)