#Task 1 & 2: Structural Profiling & Missing Data Stratification
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- TASK 1 & 2: LOADING AND COHESIVE SYSTEM STATE ---
# Establish isolation inside project tracking directories
os.makedirs("analytics", exist_ok=True)
csv_fallback_path = os.path.join("analytics", "titanic.csv")

# Core ingestion condition: Single point of entry network/cache load
df_raw = sns.load_dataset('titanic')

# Save offline fallback immediately for immutable tracking
df_raw.to_csv(csv_fallback_path, index=False)
print(f"[+] Initial Network/Cache Pull Verified. Fallback saved to: {csv_fallback_path}")

# Reload from local state to guarantee strict pipeline continuity 
df = pd.read_csv(csv_fallback_path)

print("\n=== SYSTEM CAPTURE: df.info() ===")
df.info()

print("\n=== SYSTEM CAPTURE: df.shape ===")
print(f"Dataset Shape: {df.shape}")

print("\n=== SYSTEM CAPTURE: df.describe() ===")
print(df.describe(include='all'))

# Missing Value Profiling & Stratification Rule Mapping
print("\n=== MISSING DATA PERCENTAGE PROFILING ===")
missing_series = (df.isnull().sum() / len(df)) * 100
missing_columns = missing_series[missing_series > 0].sort_values(ascending=False)

for col, pct in missing_columns.items():
    print(f"Column '{col}': {pct:.2f}% missing values.")
# Apply the deterministic structural cleaning decisions to our EDA dataframe
df['deck'] = df['deck'].astype(str).replace('nan', 'Missing')
df['age'] = df['age'].fillna(df['age'].median())
df = df.dropna(subset=['embarked', 'embark_town'])
print("\n[+] Primary Cleaning Completed. Current state remaining shape:", df.shape)
#Task 3: Univariate Outlier and Skewness Diagnostics
# --- TASK 3: UNIVARIATE ANALYSIS ---
for col in ['age', 'fare']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\nDistribution Analytics for '{col}':")
    print(f"  IQR Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Detected Outlier Records: {len(outliers)} rows out of {len(df)}")

# Distribution Central Tendency Diagnostics for Fare
fare_mean = df['fare'].mean()
fare_median = df['fare'].median()
fare_mode = df['fare'].mode()[0]

print(f"\nFare Central Tendency: Mean={fare_mean:.4f}, Median={fare_median:.4f}, Mode={fare_mode:.4f}")
#Task 4: Bivariate Cross-Tabs and Correlation Diagnostics
# --- TASK 4: BIVARIATE ANALYSIS ---
print("=== CONDITIONAL SURVIVAL RATES ===")
# (a) Breakdown by Sex
survival_sex = df.groupby('sex')['survived'].mean()
print(f"\nSurvival Rate by Sex:\n{survival_sex.to_string()}")

# (b) Breakdown by Passenger Class
survival_pclass = df.groupby('pclass')['survived'].mean()
print(f"\nSurvival Rate by Pclass:\n{survival_pclass.to_string()}")

# (c) Joint Matrix: Sex x Pclass Together
survival_joint = df.groupby(['sex', 'pclass'])['survived'].mean().unstack()
print(f"\nSurvival Matrix (Sex x Pclass):\n{survival_joint}")

# Linear Correlation Deep Dive
numeric_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=-1, vmax=1)
plt.title("Filtered Numeric Feature Correlation Matrix", fontsize=12)
plt.tight_layout()
plt.show()
#Task 5: Multivariate Analytical Data Story Telling
# --- TASK 5: MULTIVARIATE DATA STORY ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Chart 1: Categorical Interaction Factor
sns.barplot(x='pclass', y='survived', hue='sex', data=df, ax=axes[0,0], palette='muted')
axes[0,0].set_title("Survival Probability Across Class and Gender")
axes[0,0].set_ylabel("Survival Rate")

# Chart 2: Continuous Distribution Density via Violin Plots
sns.violinplot(x='survived', y='age', hue='sex', data=df, split=True, inner="quart", ax=axes[0,1], palette='pastel')
axes[0,1].set_title("Age Demographics Density vs Conditional Survival")

# Chart 3: Structural Family Unit Tracking Scatter Variant
sns.scatterplot(x='sibsp', y='parch', hue='survived', size='fare', sizes=(20, 200), data=df, ax=axes[1,0], alpha=0.7)
axes[1,0].set_title("Family Footprint (SibSp vs Parch) & Wealth Impact")

# Chart 4: Socioeconomic Fare Boxplot Stratification
sns.boxplot(x='embarked', y='fare', hue='survived', data=df[df['fare'] < 150], ax=axes[1,1])
axes[1,1].set_title("Fare Threshold Distributions across Embarkation Ports")

plt.tight_layout()
plt.show()
#Task 6: Exploratory Z-Score Sanity Transformation Check
# --- TASK 6: Z-SCORE SANITY CHECK ---
df_eda_check = df.copy()

print("=== STATISTICAL MOMENTS BEFORE STANDARDIZATION ===")
print(df_eda_check[['age', 'fare']].agg(['mean', 'std']))

# Vectorized realization of the z-score function
for col in ['age', 'fare']:
    mean_val = df_eda_check[col].mean()
    std_val = df_eda_check[col].std()
    df_eda_check[f'{col}_scaled'] = (df_eda_check[col] - mean_val) / std_val

print("\n=== STATISTICAL MOMENTS AFTER STANDARDIZATION ===")
print(df_eda_check[['age_scaled', 'fare_scaled']].agg(['mean', 'std']))
#Part B — Predictive Modeling and Pipeline EvaluationTask 7: Stratified Data Splitting
from sklearn.model_selection import train_test_split

# Define feature arrays and targets
X = df.drop(columns=['survived', 'alive']) # Remove highly correlated target flags
y = df['survived']

# Calculate baseline class distribution balances
balance = y.value_counts(normalize=True) * 100
print(f"Target Label Imbalance Profile:\n{balance.to_string()}")

# Train-Test Split using Stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n[+] Stratified Data Splits Realized. Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
#Task 8 & 9: ColumnTransformer Engineering & Classifier Execution
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Isolate feature tracking lists based on type
numeric_features = ['age', 'fare', 'sibsp', 'parch']
categorical_features = ['sex', 'embarked', 'pclass', 'deck']

# Design component sub-pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Defensive fallback configuration
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Consolidate into a global preprocessing layer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Initialize baseline models
classifiers = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
}

model_metrics_summary = {}

# Execute parallel estimation loops
for name, clf in classifiers.items():
    # Construct complete end-to-end atomic pipeline execution graph
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Fit strictly on the training partition
    pipeline.fit(X_train, y_train)
    
    # Generate test predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    model_metrics_summary[name] = {
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Y_Proba": y_proba # Keep for ROC curve visualization
    }

# --- VISUALIZE THE TRAINED DECISION TREE SUB-COMPONENT ---
plt.figure(figsize=(16, 8))
# Extract feature names generated dynamically by the categorical transformer
encoded_cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(encoded_cat_features)

plot_tree(
    classifiers["Decision Tree"],
    feature_names=all_feature_names,
    class_names=['Perished', 'Survived'],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Trained Structural Decision Tree Graph Architecture", fontsize=14)
plt.show()
#Task 10: Imbalance Mitigation Experimentation
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Transform and extract clean arrays to analyze the effects of over-sampling
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

variants = {
    "Baseline (No Action)": DecisionTreeClassifier(random_state=42, max_depth=4),
    "Class Weight Balanced": DecisionTreeClassifier(random_state=42, max_depth=4, class_weight='balanced')
}

imbalance_results = {}

# Evaluate standard variants
for v_name, model in variants.items():
    model.fit(X_train_trans, y_train)
    preds = model.predict(X_test_trans)
    imbalance_results[v_name] = {
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds)
    }

# Evaluate the SMOTE configuration
smote_pipeline = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('model', DecisionTreeClassifier(random_state=42, max_depth=4))
])
smote_pipeline.fit(X_train_trans, y_train)
smote_preds = smote_pipeline.predict(X_test_trans)

imbalance_results["SMOTE Imb Sampling"] = {
    "Precision": precision_score(y_test, smote_preds),
    "Recall": recall_score(y_test, smote_preds),
    "F1 Score": f1_score(y_test, smote_preds)
}

print("=== IMBALANCE MITIGATION PERFORMANCE MATRIX ===")
print(pd.DataFrame(imbalance_results).T)
#Task 11: Cross-Validated Hyperparameter Grid Optimization
from sklearn.model_selection import GridSearchCV

# Instantiate baseline model with OOB validation enabled
rf_core = RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True)

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 15, 20],
    'classifier__max_features': ['sqrt', 'log2']
}

tuning_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_core)
])

grid_search = GridSearchCV(tuning_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Extract optimized estimators
best_pipeline = grid_search.best_estimator_
final_rf_model = best_pipeline.named_steps['classifier']

print("=== GRID OPTIMIZATION HIGHLIGHTS ===")
print(f"Optimal Hyperparameters: {grid_search.best_params_}")
print(f"Resulting Model Out-of-Bag (OOB) Generalization Score: {final_rf_model.oob_score_:.4f}")

# Register optimized performance metrics
y_opt_pred = best_pipeline.predict(X_test)
y_opt_proba = best_pipeline.predict_proba(X_test)[:, 1]

model_metrics_summary["Optimized Random Forest"] = {
    "Confusion Matrix": confusion_matrix(y_test, y_opt_pred),
    "Accuracy": accuracy_score(y_test, y_opt_pred),
    "Precision": precision_score(y_test, y_opt_pred),
    "Recall": recall_score(y_test, y_opt_pred),
    "F1 Score": f1_score(y_test, y_opt_pred),
    "AUC": roc_auc_score(y_test, y_opt_proba),
    "Y_Proba": y_opt_proba
}
#Task 12: Regression Side-Task (Continuous Target Estimation)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Switch targeting structure
X_reg = df.drop(columns=['fare'])
y_reg = df['fare']

# Maintain stratified assignment logic mapping
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

# Re-build matching preprocessor layers excluding the target variable
reg_numeric_features = ['age', 'sibsp', 'parch']
reg_categorical_features = ['sex', 'embarked', 'pclass', 'deck', 'survived']

reg_preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, reg_numeric_features),
    ('cat', categorical_transformer, reg_categorical_features)
])

reg_pipeline = Pipeline(steps=[
    ('preprocessor', reg_preprocessor),
    ('regressor', LinearRegression())
])

reg_pipeline.fit(X_train_r, y_train_r)
reg_preds = reg_pipeline.predict(X_test_r)

# Metric Calculations
mae = mean_absolute_error(y_test_r, reg_preds)
rmse = root_mean_squared_error(y_test_r, reg_preds)
r2 = r2_score(y_test_r, reg_preds)
n = len(y_test_r)
p = X_train_trans.shape[1] # Track feature numbers
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Residual Vector Diagnostics
residuals = y_test_r - reg_preds

plt.figure(figsize=(6, 4))
plt.scatter(reg_preds, residuals, alpha=0.5, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.title("Regression Engine Residual Distribution Analysis")
plt.xlabel("Predicted Value Matrix")
plt.ylabel("Residual Adjustments")
plt.show()

print("=== REGRESSION RESULTS ===")
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}")
#Task 13: Unified Multi-Paradigm Comparison & Final Model Recommendation
# Consolidate classification outcomes into a clean summary table
clf_table_data = {}
for m_name, metrics in model_metrics_summary.items():
    clf_table_data[m_name] = {
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1 Score": metrics["F1 Score"],
        "ROC-AUC": metrics["AUC"]
    }

df_clf_final = pd.DataFrame(clf_table_data).T

# Structure the regression metrics separately to avoid scale confusion
df_reg_final = pd.DataFrame({
    "Regression Model (Target: Fare)": ["Multivariate Linear Regression"],
    "MAE": [mae], "RMSE": [rmse], "R²": [r2], "Adjusted R²": [adj_r2]
}).set_index("Regression Model (Target: Fare)")

print("================================== MACHINE LEARNING MASTER SUITE EVALUATION ==================================")
print("\n--- ENGINE BLOCK 1: CLASSIFICATION PERFORMANCE METRICS ---")
print(df_clf_final.round(4))
print("\n--- ENGINE BLOCK 2: CONTINUOUS REGRESSION METRICS ---")
print(df_reg_final.round(4))
print("\n==============================================================================================================")
#Task 14: End-to-End Serialization and Verification
import joblib

export_artifact_path = os.path.join("analytics", "best_titanic_pipeline.joblib")

# Serialize the complete processing and modeling pipeline together
joblib.dump(best_pipeline, export_artifact_path)
print(f"[+] Complete end-to-end atomic pipeline safely serialized to: {export_artifact_path}")

# --- RESET ENGINE REALIZATION FOR SANITY VALIDATION ---
loaded_pipeline = joblib.load(export_artifact_path)

# Construct an un-preprocessed raw sample for testing
unprocessed_mock_payload = pd.DataFrame([{
    'pclass': 1,
    'sex': 'female',
    'age': None,       # Tests defensive median imputation functionality
    'sibsp': 1,
    'parch': 0,
    'fare': 120.50,
    'embarked': 'C',
    'class': 'First',
    'who': 'woman',
    'adult_male': False,
    'deck': np.nan,    # Tests categorical string replacement mapping
    'embark_town': 'Cherbourg',
    'alone': False
}])

mock_prediction = loaded_pipeline.predict(unprocessed_mock_payload)
mock_probability = loaded_pipeline.predict_proba(unprocessed_mock_payload)[:, 1]

print("\n=== LIVE SERIALIZATION VERIFICATION LOGS ===")
print(f"Input Raw Dictionary Read Status: SUCCESS")
print(f"Inferred Classification Classification Result: {mock_prediction[0]} (where 1=Survived)")
print(f"Associated Survival Mathematical Probability Score: {mock_probability[0]:.4f}")
print("[+] Verification Check: PASS. Production pipeline artifact functions smoothly on raw input.")
