#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Data preprocessing:
Load the data from all three files.
Combine the three datasets into a single unified dataset.
Remove the columns start_time, axle, cluster, tsne_1, and tsne_2 from the dataset.
Replace all normal events with 0 and all other events with 1.

Data transformation:
Normalize the dataset.
'''
# Data preprocessing
# Read the data from all all three files into a pandas dataframe
df1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1.csv')
df2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1.csv')
df3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0.csv')

pd.set_option('display.width', 73)
pd.set_option('display.max_columns', None)
np.set_printoptions(linewidth=73)

print("File 1 shape:", df1.shape)
print("File 2 shape:",df2.shape)
print("File 3 shape:",df3.shape)

# Combine the three dataframes into one single unified dataset
df = pd.concat([df1, df2, df3], ignore_index=True)

# Display the shape of the combined dataframe
print("\nCombined dataset shape:", df.shape)

# Remove the specified columns
# The columns 'start_time', 'axle', 'cluster', 'tsne_1', and 'tsne_2' are dropped from the dataframe
df = df.drop(columns=['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2'])

# Display the shape of the dataframe after removing the columns
print("\nShape of the dataframe after removing specified columns:", df.shape)

# Show original event distribution
print("\nOriginal event types:")
print(df['event'].value_counts(), "\n")

# Display the first few rows of the dataframe
print("First few rows of the dataframe:\n", df.head()) 

# Replace 'normal' with 0 and all other events with 1
df['event'] = np.where(df['event'] == 'normal', 0, 1)

# Display the unique values in the 'event' column after replacement
print("\nUnique values in the event column after replacement:", df['event'].unique())

# Display the event column after replacement, shows the first rows
print("\nEvent column after replacement: \n", df['event'].head())

#%%
# Data transformation
# Normalize the dataset to scale the feature values so they are on a similar range
from sklearn.preprocessing import MinMaxScaler

# Identify numerical columns to normalize, excluding 'event' column
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('event')

# Normalize the numerical columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the first rows of the normalized dataframe
print("First few rows of the normalized dataframe:\n", df.head())

#%%
'''
Dataset splitting:
Split the data into training and testing sets in an 80/20 ratio.

Cross-Validation:
Perform k-fold cross-validation (e.g., 5-fold) on the training set to evaluate model stability.

Comparison task:
Compare between the 80/20 train-test split and k-fold cross-validation using SVM (Support Vector Machine).  
Train an SVM model using both methods and evaluate its performance. 
Discuss the differences in accuracy, consistency of results, and generalization ability.

'''
# Dataset splitting
# Split the data into features and target variable

# The target variable is the column 'event' and all other columns are features
# The features are stored in a new dataframe x, which contains all columns except 'event'
# Drop the 'event' column from features and keep it as target variable
x = df.drop(columns=['event'])

# The target variable 'event' is extracted from the dataframe as new dataframe y
# This dataframe contains the labels for classification where 'normal' is 0 and all other events are 1
y = df['event']

# Display the class distribution in the target variable
# This shows how many instances belong to each class in the target variable
print("Class distribution in y:", y.value_counts(), "\n")

# Display the shape of the dataframe x that contains the features
print("Shape of features x:", x.shape, "\n")

# Display the shape of the dataframe y
print("Shape of the target variable y:", y.shape, "\n") 

# Display the first few rows of the features
print("First few rows of features (x):\n", x.head(), "\n")

# Display the first few rows of the target variable
print("First few rows of target variable (y):\n", y.head(), "\n")

# Display the unique values in the target variable
print("Unique values in target variable (y):", y.unique(), "\n")

# Display the number of unique values in each column of the features
print("Number of unique values in each column of features (x):\n", x.nunique(), "\n")

# Display the summary statistics of the features
print("Summary statistics of features (x):\n", x.describe())

#%%
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80/20 ratio)
# The stratify parameter ensures that the class distribution is preserved in both training and testing sets
# The random_state parameter ensures reproducibility of the split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Display the class distribution in the training sets
print("Class distribution in y_train:", y_train.value_counts())

# Display the class distribution in the testing sets
print("\nClass distribution in y_test:", y_test.value_counts())

# Display the summary statistics of the training set features
print("\nSummarize features in x_train:\n", x_train.describe())

# Display the first few rows of the training and testing sets
print("\nFirst few rows of x_train:\n", x_train.head())
print("\nFirst few rows of x_test:\n", x_test.head())

# Display the first few rows of the training and testing sets
print("\nFirst few rows of y_train:\n", y_train.head())
print("\nFirst few rows of y_test:\n", y_test.head())

# Display the shapes of the training and testing sets
print("\nShape of x_train:", x_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_train:", y_train.shape)  
print("Shape of y_test:", y_test.shape)

#%%
# Perform k-fold cross-validation (e.g., 5-fold) on the training set
from sklearn.svm import SVC

# Create an SVM model, Faster on large datasets, Assumes data is linearly separable
# The kernel parameter specifies the type of kernel to be used in the SVM model
# 'linear' kernel is used for linear classification, 'rbf' kernel is used for non-linear classification

# For linear classification
model = SVC(kernel='linear', random_state=42)

# The fit method trains the model on the training data
model.fit(x_train, y_train)

# Predict and evaluate
# The predict method is used to make predictions on the test data
# The accuracy_score function calculates the accuracy of the model on the test data
# The classification_report function provides a detailed report of the model's performance
from sklearn.metrics import accuracy_score, classification_report

# Predict the labels for the test set
y_pred = model.predict(x_test)

# Calculate the accuracy of the model on the test set
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy of the model on the test set
print("Accuracy of the model on the test set:", accuracy)

# Display the classification report for the model on the test set
print("\nClassification report for the model on the test set:\n", classification_report(y_test, y_pred))

# Display the confusion matrix for the model on the test set
from sklearn.metrics import confusion_matrix

# The confusion_matrix function computes the confusion matrix to evaluate the accuracy of a classification
# The confusion matrix is a table that is often used to describe the performance of a classification model  
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("Confusion matrix:\n", cm)

import seaborn as sns
import matplotlib.pyplot as plt

# Display the confusion matrix using seaborn heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#%%
from sklearn.model_selection import cross_val_score, cross_validate

# Perform 5-fold cross-validation
# The cross_val_score function performs k-fold cross-validation on the model
# It returns the accuracy scores for each fold
cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')

# Display the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Calculate the mean and standard deviation of the cross-validation scores
mean_score = np.mean(cv_scores)
std_score = np.std(cv_scores)

# Display the mean cross-validation score, 
# indicating the average accuracy across all folds
print("Mean cross-validation score:", mean_score)

# Display the standard deviation of the cross-validation scores, 
# indicating the stability of the model
# This shows how much the accuracy varies across different folds
print("Standard deviation of cross-validation scores:", std_score)

# Define the scoring metrics to be used in cross-validation 
# to evaluate the model's performance
scoring = ['accuracy', 'precision', 'recall', 'f1']

# The cross_validate function performs cross-validation and 
# returns a dictionary containing the scores for each metric
# It also returns the fit time and score time for each fold
cv_results = cross_validate(model, x_train, y_train, cv=5, scoring=scoring, return_train_score=True)

# Display results
# Print the cross-validation results including test accuracy, 
# mean accuracy, train accuracy, fit times, and score times
# This provides a comprehensive overview of the model's performance across different folds
print("\nCross-validation results:")
print("Test Accuracy:", cv_results['test_accuracy'])
print("Mean Accuracy:", np.mean(cv_results['test_accuracy']))
print("Train Accuracy:", cv_results['train_accuracy'])
print("Fit Times:", cv_results['fit_time'])
print("Score Times:", cv_results['score_time'])

#%%
'''
Implement feature selection algorithms to identify and retain the most relevant features, 
improving model performance by reducing noise and dimensionality.
Research and understand various feature selection techniques, such as:
Filter methods (e.g., Pearson correlation, chi-square test).
Wrapper methods (e.g., recursive feature elimination).
Embedded methods (e.g., LASSO, feature importance in tree-based models).

Implement at least four feature selection algorithms in this project, 
applying them to the dataset.
'''
# Feature selection using Pearson correlation
# Calculate the Pearson correlation coefficient between features and the target variable
# The corr() method computes pairwise correlation of columns, excluding NA/null values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print("Correlation matrix:\n", correlation_matrix)

# Select features with high correlation to the target variable 'event'
# The abs() function is used to get the absolute value of the correlation coefficients
# The sort_values() method sorts the correlation coefficients in descending order  
# Pearson correlation (Filter)
high_correlation_features = correlation_matrix['event'].abs().sort_values(ascending=False)

# Display features with high correlation to the target variable
# This shows the features that have a high correlation with the target variable 'event'
# This helps in identifying the most relevant features for the classification task
print("\nFeatures ranked by absolute correlation with 'event':\n", high_correlation_features)

# Select features with correlation above a certain threshold (e.g., 0.1)
threshold = 0.1
selected_features_corr = high_correlation_features[high_correlation_features > threshold].index.tolist()
# Remove the target variable 'event' from the selected features
selected_features_corr.remove('event')

print(f"\nSelected features (correlation > {threshold}):\n",selected_features_corr)
print(f"Number of selected features: {len(selected_features_corr)}")

# Visualize correlation with target
plt.figure(figsize=(10, 8))
correlation_with_target = high_correlation_features.drop('event')
plt.barh(range(len(correlation_with_target)), correlation_with_target.values)
plt.yticks(range(len(correlation_with_target)), correlation_with_target.index)
plt.xlabel('Absolute Correlation with Target')
plt.title('Feature Correlation with Event Detection')
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.legend()
plt.tight_layout()
plt.show()

#%%
# Feature selection using Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Create an SVM model
# This model will be used for feature selection
# The SVC model is used as the estimator for RFE
# The kernel parameter specifies the type of kernel to be used in the SVM model
# 'linear' kernel is used for linear classification

# Create an RFE model with SVM as the estimator
model_rfe = SVC(kernel='linear', random_state=42)

# Test different numbers of features
feature_counts = [3, 5, 8, 10, 12]
rfe_results = {}

for n_features in feature_counts:
    # Create an RFE model and select different top fetaure
    rfe = RFE(estimator=model_rfe, n_features_to_select=n_features)
    # Fit RFE on the training data
    rfe.fit(x_train, y_train)
    
    # Evaluate performance with cross-validation
    x_train_rfe = rfe.transform(x_train)
    cv_scores = cross_val_score(model_rfe, x_train_rfe, y_train, cv=5, scoring='accuracy')
    
    rfe_results[n_features] = {
        'features': x_train.columns[rfe.support_].tolist(),
        'ranking': rfe.ranking_,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\nRFE with {n_features} features:")
    print(f"Selected features: {rfe_results[n_features]['features']}")
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Find optimal number of features
optimal_n = max(rfe_results.keys(), key=lambda k: rfe_results[k]['cv_mean'])
print(f"\nOptimal number of features: {optimal_n}")
print(f"Best features: {rfe_results[optimal_n]['features']}")

# Visualize performance vs number of features
plt.figure(figsize=(10, 6))
n_features_list = list(rfe_results.keys())
cv_means = [rfe_results[n]['cv_mean'] for n in n_features_list]
cv_stds = [rfe_results[n]['cv_std'] for n in n_features_list]

plt.errorbar(n_features_list, cv_means, yerr=cv_stds, marker='o', capsize=5)
plt.xlabel('Number of Selected Features')
plt.ylabel('Cross-Validation Accuracy')
plt.title('RFE Performance vs Feature Count')
plt.grid(True)
plt.show()


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Use Logistic Regression with L1 penalty for classification
# This is the proper way to do LASSO for binary classification
print("Using Logistic Regression with L1 penalty (LASSO for classification)")

# Define a range of regularization strengths (C is the inverse of regularization)
# Higher C means weaker regularization
# Smaller C means stronger regularization = more sparsity
C_values = np.logspace(-2, 2, 20)  # from 0.01 to 100
lasso_results = {}

# Evaluate performance across different C values
for C in C_values:
    # LogisticRegression with L1 penalty
    lasso_logreg = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42, max_iter=10000)
    lasso_logreg.fit( x_train, y_train)
    
    # Count non-zero (selected) features
    n_features = np.sum(lasso_logreg.coef_[0] != 0)
    
    # Perform 5-fold cross-validation using accuracy
    cv_scores = cross_val_score(lasso_logreg, x_train, y_train, cv=5, scoring='accuracy')
    
    # Save results
    lasso_results[C] = {
        'n_features': n_features,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'coefficients': pd.Series(lasso_logreg.coef_[0], index=x_train.columns)
    }

# Identify the best C (highest cross-validation accuracy)
best_C = max(lasso_results, key=lambda c: lasso_results[c]['cv_mean'])

# Train the best model on the full training set
best_model = LogisticRegression(penalty='l1', solver='liblinear', C=best_C, random_state=42, max_iter=10000)
best_model.fit(x_train, y_train)

# Extract important coefficients
best_coefs = pd.Series(best_model.coef_[0], index=x_train.columns)
selected_features_lasso = best_coefs[best_coefs != 0].index.tolist()

# Print summary
print(f"\nOptimal C: {best_C}")
print(f"Selected features ({len(selected_features_lasso)}): {selected_features_lasso}")
print("\nFeature coefficients (sorted by absolute value):")
print(best_coefs[best_coefs != 0].sort_values(key=abs, ascending=False))

# Visualization of CV accuracy and number of features
cv_means = [lasso_results[C]['cv_mean'] for C in C_values]
n_features_list = [lasso_results[C]['n_features'] for C in C_values]

plt.figure(figsize=(12, 8))

# Plot 1: CV Accuracy vs C
plt.subplot(2, 1, 1)
plt.semilogx(C_values, cv_means, marker='o', color='blue')
plt.axvline(x=best_C, linestyle='--', color='red', label=f"Best C = {best_C:.4f}")
plt.title('Cross-Validation Accuracy vs Regularization Strength (C)')
plt.xlabel('C (1 / Regularization strength)')
plt.ylabel('Mean CV Accuracy')
plt.grid(True)
plt.legend()

# Plot 2: Number of selected features vs C
plt.subplot(2, 1, 2)
plt.semilogx(C_values, n_features_list, marker='s', color='green')
plt.axvline(x=best_C, linestyle='--', color='red')
plt.title('Number of Selected Features vs Regularization Strength (C)')
plt.xlabel('C (1 / Regularization strength)')
plt.ylabel('Number of Non-Zero Features')
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
# Feature selection using feature importance from a tree-based model (e.g., Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Create a Random Forest model
# This model will be used for feature selection
# Random Forest is an ensemble method that uses multiple decision trees 
# to improve classification accuracy
# The n_estimators parameter specifies the number of trees in the forest
# The random_state parameter ensures reproducibility of the results
# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

# Fit the Random Forest model
# The fit method trains the model on the training data
# The model learns the relationships between features and the target variable
# The fit method is called on the Random Forest model with the training data
rf.fit(x_train, y_train)

# The feature_importances_ attribute provides the importance of each feature
# The importance is calculated based on how much each feature contributes to the model's accuracy
# The feature importances are stored in a pandas Series
# The index of the Series is the feature names
# The values are the importance scores
# Get the feature importances
feature_importances_rf = pd.Series(rf.feature_importances_, index=x_train.columns)

# Display the feature importances
# This shows the importance of each feature in the Random Forest model
print("\nFeature importances from Random Forest:\n", feature_importances_rf)

# Sort the feature importances in descending order
feature_importances_rf = feature_importances_rf.sort_values(ascending=False)

# Display the sorted feature importances
print("\nSorted feature importances:\n", feature_importances_rf)

# Calculate permutation importance for more robust estimates
perm_importance = permutation_importance(rf, x_train, y_train, n_repeats=10, random_state=42)
perm_importance_df = pd.DataFrame({
    'feature': x_train.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

# Select features using different thresholds
# The threshold can be adjusted based on the dataset and model performance
# Features with importance below the threshold are considered less relevant
# The threshold can be adjusted based on the dataset and model performance
thresholds = [0.01, 0.02, 0.05]
rf_results = {}

for threshold in thresholds:
    # Select features with importance above the threshold
    # The selected features are those that have an importance score above the threshold
    # This helps in reducing the dimensionality of the dataset by selecting only the most relevant features
    # The selected features are stored in a list
    selected_features = feature_importances_rf[feature_importances_rf > threshold].index.tolist()
    
    # Create a new dataframe with selected features
    # The new dataframe contains only the features that have been selected based on 
    # their importance scores
    # This helps in reducing the dimensionality of the dataset by selecting only 
    # the most relevant features
    # Evaluate performance with selected features
    x_train_selected = x_train[selected_features]
    cv_scores = cross_val_score(rf, x_train_selected, y_train, cv=5, scoring='accuracy')
    
    rf_results[threshold] = {
        'features': selected_features,
        'n_features': len(selected_features),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\nThreshold {threshold}: {len(selected_features)} features")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Visualize feature importances
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.barh(range(len(feature_importances_rf)), feature_importances_rf.values)
plt.yticks(range(len(feature_importances_rf)), feature_importances_rf.index)
plt.xlabel('Feature Importance (Gini)')
plt.title('Random Forest Feature Importance')

plt.subplot(2, 1, 2)
plt.errorbar(range(len(perm_importance_df)), perm_importance_df['importance_mean'], 
             yerr=perm_importance_df['importance_std'], fmt='o', capsize=3)
plt.xticks(range(len(perm_importance_df)), perm_importance_df['feature'], rotation=45)
plt.ylabel('Permutation Importance')
plt.title('Permutation-Based Feature Importance')
plt.tight_layout()
plt.show()

print(f"\nTop 5 features by Random Forest importance:")
print(feature_importances_rf.head())

#%%
import textwrap
from sklearn.model_selection import cross_val_score

# Evaluate model using Pearson-selected features
x_train_corr = x_train[selected_features_corr]
model_corr = SVC(kernel='linear', random_state=42)
cv_scores_corr = cross_val_score(model_corr, x_train_corr, y_train, cv=5, scoring='accuracy')
pearson_cv_mean = cv_scores_corr.mean()

# Compare all methods
comparison_data = {
    'Method': ['Pearson Correlation', 'RFE (5 features)', 'LASSO', 'Random Forest (0.01 threshold)'],
    'Features_Selected': [
        len(selected_features_corr),
        len(rfe_results[5]['features']),
        len(selected_features_lasso),
        len(rf_results[0.01]['features'])
    ],
    'Top_Features': [
        selected_features_corr[:5],
        rfe_results[5]['features'],
        selected_features_lasso,
        rf_results[0.01]['features'][:5]
    ],
    'CV_Accuracy_Mean': [
        pearson_cv_mean,
        rfe_results[5]['cv_mean'],
        lasso_results[best_C]['cv_mean'],
        rf_results[0.01]['cv_mean']
    ]
}

print("Feature Selection Methods Comparison:\n")

for method, n_features, top_feats, cv_score in zip(
    comparison_data['Method'],
    comparison_data['Features_Selected'],
    comparison_data['Top_Features'],
    comparison_data['CV_Accuracy_Mean']
):
    print(f"▶ Method: {method}")
    print(f"Number of Selected Features: {n_features}")
    print(f"CV Accuracy Mean: {cv_score:.4f}")
    
    # Format and wrap top features
    formatted_features = ", ".join(sorted(top_feats))
    wrapped_features = textwrap.fill(formatted_features, width=73)
    print(f"Top Features:\n{wrapped_features}\n")

# %%
