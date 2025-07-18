import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set up styling for visualizations
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory for results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'analysis_results_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

print("="*80)
print("ENHANCED COUNTRY & QUESTION LEVEL BALANCED CLASSIFICATION ANALYSIS")
print("="*80)

# STEP 1: Load Dataset
print("\n[1] LOADING DATASET...")
# Replace with your actual file path
df = pd.read_csv(r"C:\Users\anand\Downloads\MathE dataset.csv", encoding='ISO-8859-1')
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col])

print(f"Dataset shape: {df.shape}")
print("\nClass distribution (Type of Answer):")
print(df['Type of Answer'].value_counts())

# Save initial class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Type of Answer', data=df, palette='viridis')
plt.title('Initial Class Distribution', fontsize=15)
plt.savefig(f'{output_dir}/01_initial_class_distribution.png', bbox_inches='tight')
plt.close()

# STEP 2: Data Preprocessing
print("\n[2] DATA PREPROCESSING...")

# Convert categorical columns
label_cols = ['Topic', 'Subtopic', 'Question ID', 'Question Level', 'Student Country']
encoders = {}

for col in label_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"Encoded {col} - {len(le.classes_)} unique values")

# ENHANCED FEATURE ENGINEERING
print("\nCreating ENHANCED features...")

# Basic performance features
student_avg = df.groupby('Student ID')['Type of Answer'].mean()
df['Student_Avg_Performance'] = df['Student ID'].map(student_avg)

question_avg = df.groupby('Question ID')['Type of Answer'].mean()
df['Question_Difficulty'] = df['Question ID'].map(question_avg)

country_avg = df.groupby('Student Country')['Type of Answer'].mean()
df['Country_Avg_Performance'] = df['Student Country'].map(country_avg)

topic_avg = df.groupby('Topic')['Type of Answer'].mean()
df['Topic_Avg_Performance'] = df['Topic'].map(topic_avg)

subtopic_avg = df.groupby('Subtopic')['Type of Answer'].mean()
df['Subtopic_Avg_Performance'] = df['Subtopic'].map(subtopic_avg)

# Advanced features
# Per-student performance by topic
student_topic_perf = df.groupby(['Student ID', 'Topic'])['Type of Answer'].mean().reset_index()
student_topic_dict = dict(zip(zip(student_topic_perf['Student ID'], student_topic_perf['Topic']), 
                              student_topic_perf['Type of Answer']))
df['Student_Topic_Performance'] = df.apply(lambda x: student_topic_dict.get((x['Student ID'], x['Topic']), 
                                                                             df['Student_Avg_Performance'].mean()), axis=1)

# Per-student performance by question level
student_level_perf = df.groupby(['Student ID', 'Question Level'])['Type of Answer'].mean().reset_index()
student_level_dict = dict(zip(zip(student_level_perf['Student ID'], student_level_perf['Question Level']), 
                              student_level_perf['Type of Answer']))
df['Student_Level_Performance'] = df.apply(lambda x: student_level_dict.get((x['Student ID'], x['Question Level']), 
                                                                             df['Student_Avg_Performance'].mean()), axis=1)

# Country performance by topic
country_topic_perf = df.groupby(['Student Country', 'Topic'])['Type of Answer'].mean().reset_index()
country_topic_dict = dict(zip(zip(country_topic_perf['Student Country'], country_topic_perf['Topic']), 
                              country_topic_perf['Type of Answer']))
df['Country_Topic_Performance'] = df.apply(lambda x: country_topic_dict.get((x['Student Country'], x['Topic']), 
                                                                             df['Country_Avg_Performance'].mean()), axis=1)

# Country performance by question level
country_level_perf = df.groupby(['Student Country', 'Question Level'])['Type of Answer'].mean().reset_index()
country_level_dict = dict(zip(zip(country_level_perf['Student Country'], country_level_perf['Question Level']), 
                              country_level_perf['Type of Answer']))
df['Country_Level_Performance'] = df.apply(lambda x: country_level_dict.get((x['Student Country'], x['Question Level']), 
                                                                             df['Country_Avg_Performance'].mean()), axis=1)

# Interaction features
df['Difficulty_Level_Interaction'] = df['Question_Difficulty'] * df['Question Level']
df['Student_Country_Interaction'] = df['Student_Avg_Performance'] * df['Country_Avg_Performance']
df['Topic_Subtopic_Interaction'] = df['Topic_Avg_Performance'] * df['Subtopic_Avg_Performance']

# Check for missing values and handle them
missing_values = df.isnull().sum()
columns_with_missing = missing_values[missing_values > 0]
if len(columns_with_missing) > 0:
    for col in columns_with_missing.index:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# STEP 3: Split data for initial evaluation
X = df.select_dtypes(include=['number'])
X = X.drop(['Type of Answer', 'Student ID'], axis=1, errors='ignore')
y = df['Type of Answer']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)

# STEP 4: Initial model evaluation (before SMOTE)
print("\n[3] EVALUATING MODELS BEFORE BALANCING...")

# Random Forest with optimized parameters
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=2, 
                          min_samples_leaf=1, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy (Before Balancing): {rf_accuracy:.4f}")
print(classification_report(y_test, rf_pred))

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Accuracy (Before Balancing): {gb_accuracy:.4f}")
print(classification_report(y_test, gb_pred))

# Neural Network
nn = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=300, activation='relu', 
                   solver='adam', random_state=42, early_stopping=True)
nn.fit(X_train, y_train)
nn_pred = nn.predict(X_test)
nn_accuracy = accuracy_score(y_test, nn_pred)
print(f"Neural Network Accuracy (Before Balancing): {nn_accuracy:.4f}")
print(classification_report(y_test, nn_pred))

# Stacking Ensemble
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)),
    ('nn', MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, early_stopping=True, random_state=42))
]
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(C=10, max_iter=1000),
    cv=5
)
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_pred)
print(f"Stacking Ensemble Accuracy (Before Balancing): {stacking_accuracy:.4f}")
print(classification_report(y_test, stacking_pred))

# STEP 5: Enhanced SMOTE-like resampling by country - ALL BALANCED TO 3001
print("\n[4] APPLYING ENHANCED UNIFORM BALANCING (ALL TO 3001)...")

MAX_TARGET_COUNT = 3001  # Setting all countries to have 3001 samples per class
resampled_data = []
country_level_stats = []

# Process each country
for country in df['Student Country'].unique():
    country_name = encoders['Student Country'].inverse_transform([country])[0] if 'Student Country' in encoders else f"Country_{country}"
    print(f"\nProcessing country: {country_name}")
    
    # Filter data for this country
    country_df = df[df['Student Country'] == country]
    
    # Get class counts
    class_counts = country_df['Type of Answer'].value_counts().to_dict()
    class_0_before = class_counts.get(0, 0)
    class_1_before = class_counts.get(1, 0)
    
    # Set target count to the maximum value 3001
    target_count = MAX_TARGET_COUNT
    print(f"Class 0 Before: {class_0_before}, Class 1 Before: {class_1_before}")
    print(f"Target count for all classes: {target_count}")
    
    # Create synthetic samples
    resampled_country_df = pd.DataFrame()  # Start fresh
    
    # For Class 0
    if class_0_before > 0:
        class_0_df = country_df[country_df['Type of Answer'] == 0]
        synthetic_needed = target_count
        # Use SMOTE-like approach for small datasets
        if class_0_before < 5:
            # Create copies with small variations
            synthetic_samples = pd.DataFrame()
            for _ in range(synthetic_needed):
                sample = class_0_df.sample(1, replace=True, random_state=np.random.randint(1000))
                # Add small random noise to numeric columns
                for col in sample.select_dtypes(include=['number']).columns:
                    if col != 'Type of Answer' and col != 'Student ID':
                        noise = np.random.normal(0, 0.05, 1)[0]
                        sample[col] = sample[col] + noise
                synthetic_samples = pd.concat([synthetic_samples, sample])
        else:
            # If we have enough samples, use sampling with replacement
            synthetic_samples = class_0_df.sample(synthetic_needed, replace=True, random_state=42)
        
        resampled_country_df = pd.concat([resampled_country_df, synthetic_samples])
    
    # For Class 1
    if class_1_before > 0:
        class_1_df = country_df[country_df['Type of Answer'] == 1]
        synthetic_needed = target_count
        # Use SMOTE-like approach for small datasets
        if class_1_before < 5:
            # Create copies with small variations
            synthetic_samples = pd.DataFrame()
            for _ in range(synthetic_needed):
                sample = class_1_df.sample(1, replace=True, random_state=np.random.randint(1000))
                # Add small random noise to numeric columns
                for col in sample.select_dtypes(include=['number']).columns:
                    if col != 'Type of Answer' and col != 'Student ID':
                        noise = np.random.normal(0, 0.05, 1)[0]
                        sample[col] = sample[col] + noise
                synthetic_samples = pd.concat([synthetic_samples, sample])
        else:
            # If we have enough samples, use sampling with replacement
            synthetic_samples = class_1_df.sample(synthetic_needed, replace=True, random_state=42)
        
        resampled_country_df = pd.concat([resampled_country_df, synthetic_samples])
    
    # Add to resampled data
    resampled_data.append(resampled_country_df)
    
    # Calculate stats
    class_counts_after = resampled_country_df['Type of Answer'].value_counts().to_dict()
    class_0_after = class_counts_after.get(0, 0)
    class_1_after = class_counts_after.get(1, 0)
    
    synthetic_samples = len(resampled_country_df) - len(country_df)
    
    # Record statistics
    country_level_stats.append({
        'Country': country_name,
        'Class_0_Before': class_0_before,
        'Class_1_Before': class_1_before,
        'Class_0_After': class_0_after,
        'Class_1_After': class_1_after,
        'Synthetic_Samples': synthetic_samples
    })
    
    print(f"After resampling - Class 0: {class_0_after}, Class 1: {class_1_after}, Synthetic: {synthetic_samples}")

# Combine all resampled data
df_resampled = pd.concat(resampled_data, ignore_index=True)
print(f"\nCombined resampled dataset shape: {df_resampled.shape}")
print("Final class distribution after resampling:")
print(df_resampled['Type of Answer'].value_counts())

# Save resampled dataset and stats
df_resampled.to_csv(f'{output_dir}/MathE_uniform_balanced_resampled.csv', index=False)
stats_df = pd.DataFrame(country_level_stats)
stats_df.to_csv(f'{output_dir}/country_uniform_balance_stats.csv', index=False)

# Display statistics in a nice table
print("\nResampling Statistics by Country:")
print(stats_df[['Country', 'Class_0_Before', 'Class_1_Before', 'Class_0_After', 'Class_1_After', 'Synthetic_Samples']])

# STEP 6: Train models on enhanced balanced data with hyperparameter tuning
print("\n[5] TRAINING OPTIMIZED MODELS ON BALANCED DATA...")

X_resampled = df_resampled.select_dtypes(include=['number'])
X_resampled = X_resampled.drop(['Type of Answer', 'Student ID'], axis=1, errors='ignore')
y_resampled = df_resampled['Type of Answer']

# Standardize features
scaler_resampled = StandardScaler()
X_resampled_scaled = scaler_resampled.fit_transform(X_resampled)
X_resampled_scaled_df = pd.DataFrame(X_resampled_scaled, columns=X_resampled.columns)

X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
    X_resampled_scaled_df, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Create PCA features to capture additional patterns
pca = PCA(n_components=5)
pca_features = pca.fit_transform(X_train_resampled)
X_train_pca = np.hstack((X_train_resampled, pca_features))
pca_features_test = pca.transform(X_test_resampled)
X_test_pca = np.hstack((X_test_resampled, pca_features_test))

# Optimized Random Forest
rf_optimized = RandomForestClassifier(
    n_estimators=500, 
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    max_features='sqrt',
    random_state=42
)
rf_optimized.fit(X_train_pca, y_train_resampled)
rf_optimized_pred = rf_optimized.predict(X_test_pca)
rf_optimized_accuracy = accuracy_score(y_test_resampled, rf_optimized_pred)

print(f"Optimized Random Forest Accuracy (After Balancing): {rf_optimized_accuracy:.4f}")
print(classification_report(y_test_resampled, rf_optimized_pred))

# Optimized Gradient Boosting
gb_optimized = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)
gb_optimized.fit(X_train_pca, y_train_resampled)
gb_optimized_pred = gb_optimized.predict(X_test_pca)
gb_optimized_accuracy = accuracy_score(y_test_resampled, gb_optimized_pred)

print(f"Optimized Gradient Boosting Accuracy (After Balancing): {gb_optimized_accuracy:.4f}")
print(classification_report(y_test_resampled, gb_optimized_pred))

# Neural Network with optimizations
nn_optimized = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=128,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    random_state=42
)
nn_optimized.fit(X_train_pca, y_train_resampled)
nn_optimized_pred = nn_optimized.predict(X_test_pca)
nn_optimized_accuracy = accuracy_score(y_test_resampled, nn_optimized_pred)

print(f"Optimized Neural Network Accuracy (After Balancing): {nn_optimized_accuracy:.4f}")
print(classification_report(y_test_resampled, nn_optimized_pred))

# Enhanced Stacking Ensemble with optimized base models
base_models_optimized = [
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)),
    ('nn', MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=500, early_stopping=True, random_state=42)),
    ('svm', SVC(C=10, gamma='scale', probability=True, random_state=42))
]

stacking_optimized = StackingClassifier(
    estimators=base_models_optimized,
    final_estimator=LogisticRegression(C=100, solver='liblinear', max_iter=2000),
    cv=5
)
stacking_optimized.fit(X_train_pca, y_train_resampled)
stacking_optimized_pred = stacking_optimized.predict(X_test_pca)
stacking_optimized_accuracy = accuracy_score(y_test_resampled, stacking_optimized_pred)

print(f"Enhanced Stacking Ensemble Accuracy (After Balancing): {stacking_optimized_accuracy:.4f}")
print(classification_report(y_test_resampled, stacking_optimized_pred))

# STEP 7: Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X_resampled.columns,
    'Importance': rf_optimized.feature_importances_[:len(X_resampled.columns)]
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis')
plt.title('Feature Importance After Balancing', fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/feature_importance.png', bbox_inches='tight')
plt.close()

# STEP 8: Compare results before and after balancing
comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting', 'Neural Network', 'Stacking Ensemble'],
    'Accuracy Before Balancing': [rf_accuracy, gb_accuracy, nn_accuracy, stacking_accuracy],
    'Accuracy After Balancing': [rf_optimized_accuracy, gb_optimized_accuracy, nn_optimized_accuracy, stacking_optimized_accuracy],
    'Improvement': [rf_optimized_accuracy - rf_accuracy, 
                   gb_optimized_accuracy - gb_accuracy,
                   nn_optimized_accuracy - nn_accuracy,
                   stacking_optimized_accuracy - stacking_accuracy]
})

print("\nAccuracy Comparison:")
print(comparison)

# Save models
joblib.dump(rf_optimized, f'{output_dir}/optimized_random_forest_model.pkl')
joblib.dump(gb_optimized, f'{output_dir}/optimized_gradient_boosting_model.pkl')
joblib.dump(nn_optimized, f'{output_dir}/optimized_neural_network_model.pkl')
joblib.dump(stacking_optimized, f'{output_dir}/optimized_stacking_ensemble_model.pkl')
joblib.dump(scaler_resampled, f'{output_dir}/feature_scaler.pkl')
joblib.dump(pca, f'{output_dir}/pca_transformer.pkl')

# Create summary visualization for presentation
plt.figure(figsize=(12, 8))
comparison_melted = pd.melt(comparison, id_vars=['Model'], 
                            value_vars=['Accuracy Before Balancing', 'Accuracy After Balancing'])
sns.barplot(x='Model', y='value', hue='variable', data=comparison_melted, palette='Set2')
plt.title('Model Accuracy Before and After Balancing & Optimization', fontsize=16)
plt.ylabel('Accuracy')
plt.ylim(0.6, 1.0)  # Set y-axis to focus on the accuracy range
plt.tight_layout()
plt.savefig(f'{output_dir}/accuracy_comparison.png', bbox_inches='tight')
plt.close()

# Create visualization of country statistics
plt.figure(figsize=(15, 10))
stats_plot = stats_df.sort_values('Class_0_Before', ascending=False)
x = np.arange(len(stats_plot))
width = 0.2

fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(x - width*1.5, stats_plot['Class_0_Before'], width, label='Class 0 Before', color='#1f77b4')
ax.bar(x - width/2, stats_plot['Class_1_Before'], width, label='Class 1 Before', color='#ff7f0e')
ax.bar(x + width/2, stats_plot['Class_0_After'], width, label='Class 0 After', color='#2ca02c')
ax.bar(x + width*1.5, stats_plot['Class_1_After'], width, label='Class 1 After', color='#d62728')

ax.set_ylabel('Number of Samples')
ax.set_title('Class Distribution by Country Before and After Uniform Balancing')
ax.set_xticks(x)
ax.set_xticklabels(stats_plot['Country'], rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/country_class_distribution.png', bbox_inches='tight')
plt.close()

print("\nEnhanced analysis complete! Results saved to:", output_dir)