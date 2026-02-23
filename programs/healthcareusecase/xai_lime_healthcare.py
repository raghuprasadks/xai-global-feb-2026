"""
XAI LIME Library - Healthcare Application
==========================================

This program demonstrates the working of the LIME (Local Interpretable Model-agnostic Explanations)
library using a healthcare dataset. It covers:

1. Loading a healthcare dataset (Breast Cancer Classification)
2. Training a machine learning model
3. Using LIME to explain individual predictions
4. Visualizing feature importance for model interpretability
5. Understanding which features contribute to predictions

Dataset: Breast Cancer Wisconsin (Diagnostic)
- Target: Malignant (1) vs Benign (0)
- Features: 30 computed features from cell nuclei measurements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

print("=" * 80)
print("LIME (Local Interpretable Model-agnostic Explanations) - Healthcare Demo")
print("=" * 80)

# ============================================================================
# STEP 2: LOAD HEALTHCARE DATASET
# ============================================================================

print("\n[STEP 1] Loading Breast Cancer Healthcare Dataset...")
print("-" * 80)

# Load the breast cancer dataset
cancer_data = load_breast_cancer()
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = pd.Series(cancer_data.target, name='diagnosis')

print(f"Dataset loaded successfully!")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: 0 = Benign (Negative), 1 = Malignant (Positive)")
print(f"\nTarget distribution:")
print(f"  - Benign (0): {(y == 0).sum()} samples ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"  - Malignant (1): {(y == 1).sum()} samples ({(y == 1).sum() / len(y) * 100:.1f}%)")

print(f"\nDataset shape: X = {X.shape}, y = {y.shape}")
print(f"\nFirst few features:\n{X.head()}")

# ============================================================================
# STEP 3: PREPARE DATA - TRAIN-TEST SPLIT
# ============================================================================

print("\n[STEP 2] Preparing Data - Train-Test Split...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"\nTraining set class distribution:")
print(f"  - Benign (0): {(y_train == 0).sum()}")
print(f"  - Malignant (1): {(y_train == 1).sum()}")

# ============================================================================
# STEP 4: NORMALIZE/SCALE THE DATA
# ============================================================================

print("\n[STEP 3] Feature Scaling (Normalization)...")
print("-" * 80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrames to maintain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Features have been scaled using StandardScaler")
print(f"Scaled X_train mean (should be close to 0): {X_train_scaled.mean().mean():.6f}")
print(f"Scaled X_train std (should be close to 1): {X_train_scaled.std().mean():.6f}")

# ============================================================================
# STEP 5: TRAIN A CLASSIFICATION MODEL
# ============================================================================

print("\n[STEP 4] Training Machine Learning Model...")
print("-" * 80)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

print("Random Forest Classifier trained successfully!")

# Evaluate model performance
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\nModel Performance:")
print(f"  - Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  - Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant']))

# ============================================================================
# STEP 6: INSTALL & IMPORT LIME
# ============================================================================

print("\n[STEP 5] Setting up LIME Explainer...")
print("-" * 80)

try:
    from lime.lime_tabular import LimeTabularExplainer
    print("✓ LIME library successfully imported!")
except ImportError:
    print("✗ LIME not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'lime'])
    from lime.lime_tabular import LimeTabularExplainer
    print("✓ LIME installed and imported successfully!")

# ============================================================================
# STEP 7: INITIALIZE LIME EXPLAINER
# ============================================================================

print("\nInitializing LimeTabularExplainer...")
print("-" * 80)

# Create LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train_scaled.values,
    feature_names=X_train_scaled.columns.tolist(),
    class_names=['Benign (Negative)', 'Malignant (Positive)'],
    mode='classification',
    verbose=False,
    random_state=42
)

print("✓ LimeTabularExplainer initialized successfully!")
print(f"  - Training data shape: {X_train_scaled.shape}")
print(f"  - Number of features: {len(X_train_scaled.columns)}")
print(f"  - Mode: Classification (Binary)")

# ============================================================================
# STEP 8: EXPLAIN INDIVIDUAL PREDICTIONS
# ============================================================================

print("\n[STEP 6] Explaining Individual Predictions with LIME...")
print("-" * 80)

# Function to make predictions (wrapper for LIME compatibility)
def predict_proba(data):
    """
    Wrapper function to convert numpy array to DataFrame and predict probabilities.
    LIME expects model.predict to accept numpy array and return probability predictions.
    """
    df = pd.DataFrame(data, columns=X_train_scaled.columns)
    return model.predict_proba(df)

# Select instances to explain - one benign and one malignant
benign_idx = np.where((y_test == 0).values)[0][0]
malignant_idx = np.where((y_test == 1).values)[0][0]

print(f"\nInstance 1 (Benign Case) - Index: {benign_idx}")
print(f"Instance 2 (Malignant Case) - Index: {malignant_idx}")

instances_to_explain = [
    ('Benign Case', benign_idx),
    ('Malignant Case', malignant_idx)
]

# Store explanations for visualization
explanations = {}

for case_name, idx in instances_to_explain:
    print(f"\n{'='*80}")
    print(f"LIME Explanation: {case_name}")
    print(f"{'='*80}")
    
    # Get the instance
    instance = X_test_scaled.iloc[idx]
    
    # Get model prediction
    proba = predict_proba(instance.values.reshape(1, -1))[0]
    prediction_class = np.argmax(proba)
    prediction_label = ['Benign', 'Malignant'][prediction_class]
    confidence = proba[prediction_class]
    
    print(f"\nInstance Information:")
    print(f"  - Actual Class: {'Benign' if y_test.iloc[idx] == 0 else 'Malignant'}")
    print(f"  - Model Prediction: {prediction_label}")
    print(f"  - Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"  - Probability (Benign): {proba[0]:.4f}")
    print(f"  - Probability (Malignant): {proba[1]:.4f}")
    
    # Generate LIME explanation
    lime_exp = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=predict_proba,
        num_features=10  # Top 10 most important features
    )
    
    explanations[case_name] = {
        'exp': lime_exp,
        'idx': idx,
        'instance': instance,
        'proba': proba,
        'prediction_label': prediction_label,
        'actual_label': 'Benign' if y_test.iloc[idx] == 0 else 'Malignant'
    }
    
    # Print LIME explanation
    print(f"\nTop 10 Features Contributing to Prediction:")
    print("-" * 80)
    
    exp_list = lime_exp.as_list()
    for i, (feature, contribution) in enumerate(exp_list, 1):
        direction = "↑" if contribution > 0 else "↓"
        prediction_direction = "toward Malignant" if contribution > 0 else "toward Benign"
        print(f"{i:2d}. {feature:50s} | Contribution: {contribution:+.4f} {direction} {prediction_direction}")

# ============================================================================
# STEP 9: VISUALIZE EXPLANATIONS
# ============================================================================

print(f"\n[STEP 7] Visualizing LIME Explanations...")
print("-" * 80)

# Create visualizations
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('LIME Explanations for Healthcare Dataset Predictions', fontsize=16, fontweight='bold')

case_list = list(explanations.keys())

for plot_idx, case_name in enumerate(case_list):
    ax = axes[plot_idx]
    exp_data = explanations[case_name]
    lime_exp = exp_data['exp']
    
    # Extract feature contributions
    exp_list = lime_exp.as_list()
    features = [item[0].split('<=')[0].split('>')[0].strip() for item in exp_list]
    contributions = [item[1] for item in exp_list]
    
    # Create bar plot
    colors = ['red' if c > 0 else 'green' for c in contributions]
    bars = ax.barh(range(len(features)), contributions, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Feature Contribution to Prediction', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add title with prediction info
    confidence_idx = 0 if exp_data['prediction_label'] == 'Benign' else 1
    confidence_value = exp_data['proba'][confidence_idx]
    title = (f"{case_name}\n"
             f"Prediction: {exp_data['prediction_label']} (Confidence: {confidence_value:.2%})\n"
             f"Actual: {exp_data['actual_label']}")
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, contributions)):
        x_pos = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=9)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('lime_explanations_healthcare.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'lime_explanations_healthcare.png'")
plt.show()

# ============================================================================
# STEP 10: GLOBAL FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print(f"\n[STEP 8] Global Feature Importance (Model-level)...")
print("-" * 80)

# Get feature importance from the model
feature_importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features (from Random Forest):")
print(feature_importance_df.head(10).to_string(index=False))

# Visualize global feature importance
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance_df.head(10)
ax.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue', edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax.set_title('Global Feature Importance (Random Forest Model)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle='--')

for i, val in enumerate(top_features['Importance'].values):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('global_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Global feature importance saved as 'global_feature_importance.png'")
plt.show()

# ============================================================================
# STEP 11: KEY INSIGHTS & SUMMARY
# ============================================================================

print(f"\n[STEP 9] Key Insights & Summary")
print("=" * 80)

print("\nLIME (Local Interpretable Model-agnostic Explanations) - Key Takeaways:")
print("-" * 80)

print("""
1. LOCAL vs GLOBAL INTERPRETABILITY:
   - LIME explains INDIVIDUAL predictions (Local)
   - Feature Importance explains the overall model behavior (Global)
   
2. HOW LIME WORKS:
   - Fits a local linear model around the instance to explain
   - Perturbs the instance and measures prediction changes
   - Uses weights to give more importance to similar instances
   - Identifies which features pushed the prediction in that direction
   
3. HEALTHCARE APPLICATION:
   - Doctors can understand WHY the model predicted Malignant vs Benign
   - Identifies which patient measurements were most influential
   - Helps build trust in AI-assisted diagnosis systems
   - Can highlight anomalies in patient data
   
4. PRACTICAL USE:
   - Use LIME when model transparency is critical (healthcare, finance, legal)
   - Explains black-box models (Random Forest, Neural Networks, etc.)
   - Complements global feature importance for complete picture
   
5. ADVANTAGES:
   ✓ Model-agnostic (works with any model)
   ✓ Provides local explanations for individual predictions
   ✓ Easy to understand visualizations
   ✓ Can identify which features caused specific predictions
   
6. LIMITATIONS:
   ✗ Computationally expensive for large datasets
   ✗ Depends on local interpretability assumption
   ✗ Results can be sensitive to perturbation method
""")

print("\n" + "=" * 80)
print("Program completed successfully!")
print("=" * 80)
