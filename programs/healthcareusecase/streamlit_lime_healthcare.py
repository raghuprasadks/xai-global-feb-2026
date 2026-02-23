"""
Streamlit App: XAI LIME Explainer for Healthcare
=================================================

An interactive web application to demonstrate Local Interpretable Model-agnostic 
Explanations (LIME) for a healthcare classification model using the Breast Cancer dataset.

Features:
- Interactive model training and evaluation
- Individual prediction explanations with LIME
- Feature contribution visualization
- Global vs Local interpretability comparison
- Patient case studies
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lime.lime_tabular import LimeTabularExplainer

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="XAI LIME Healthcare Explainer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING FUNCTIONS FOR PERFORMANCE
# ============================================================================

@st.cache_data
def load_data():
    """Load and prepare the breast cancer dataset"""
    cancer_data = load_breast_cancer()
    X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    y = pd.Series(cancer_data.target, name='diagnosis')
    return X, y, cancer_data.DESCR

@st.cache_data
def prepare_data(test_size=0.2):
    """Split and scale the data"""
    X, y, _ = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

@st.cache_resource
def train_model():
    """Train the Random Forest model"""
    X_train, X_test, y_train, y_test, _ = prepare_data()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

@st.cache_resource
def create_lime_explainer():
    """Initialize LIME explainer"""
    model, X_train, X_test, y_train, y_test = train_model()
    
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['Benign (Negative)', 'Malignant (Positive)'],
        mode='classification',
        verbose=False,
        random_state=42
    )
    
    return explainer, model, X_train, X_test, y_train, y_test

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def predict_proba(model, data, X_train):
    """Wrapper to make predictions compatible with LIME"""
    df = pd.DataFrame(data, columns=X_train.columns)
    return model.predict_proba(df)

def get_explanation_for_instance(explainer, model, X_train, X_test, idx):
    """Generate LIME explanation for a specific instance"""
    instance = X_test.iloc[idx]
    proba = predict_proba(model, instance.values.reshape(1, -1), X_train)[0]
    prediction_class = np.argmax(proba)
    
    lime_exp = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=lambda x: predict_proba(model, x, X_train),
        num_features=10
    )
    
    return {
        'instance': instance,
        'proba': proba,
        'prediction_class': prediction_class,
        'explanation': lime_exp,
        'exp_list': lime_exp.as_list()
    }

def plot_lime_explanation(exp_data, title):
    """Create visualization for LIME explanation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_list = exp_data['exp_list']
    features = [item[0].split('<=')[0].split('>')[0].strip() for item in exp_list]
    contributions = [item[1] for item in exp_list]
    
    colors = ['#ff6b6b' if c > 0 else '#51cf66' for c in contributions]
    bars = ax.barh(range(len(features)), contributions, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('Feature Contribution to Prediction', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    for i, (bar, val) in enumerate(zip(bars, contributions)):
        x_pos = val + (0.002 if val > 0 else -0.002)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig

def plot_global_feature_importance(model, X_train):
    """Create visualization for global feature importance"""
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance_df.head(10)
    
    bars = ax.barh(range(len(top_features)), top_features['Importance'].values, 
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Global Feature Importance (Random Forest)', fontsize=13, fontweight='bold', pad=20)
    
    for i, val in enumerate(top_features['Importance'].values):
        ax.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    return fig, feature_importance_df

def plot_confusion_matrix(model, X_test, y_test):
    """Create confusion matrix visualization"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title('Confusion Matrix - Model Performance', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    # 🏥 XAI LIME Explainer for Healthcare
    ## Understanding AI-Assisted Medical Diagnosis
    """)
    
    st.markdown("""
    This interactive application demonstrates **LIME (Local Interpretable Model-agnostic Explanations)** 
    for explaining machine learning predictions in medical diagnosis using the Breast Cancer dataset.
    """)
    
    # Sidebar Navigation
    st.sidebar.markdown("## 📋 Navigation")
    page = st.sidebar.radio(
        "Select a section:",
        ["📊 Overview", "🏋️ Model Training", "🔍 Single Prediction Explainer", 
         "📈 Case Studies", "🎓 Learning Resources"]
    )
    
    # ========================================================================
    # PAGE 1: OVERVIEW
    # ========================================================================
    if page == "📊 Overview":
        st.header("Dataset Overview")
        
        X, y, descr = load_data()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", X.shape[0])
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Benign Cases", (y == 0).sum())
        with col4:
            st.metric("Malignant Cases", (y == 1).sum())
        
        st.subheader("Class Distribution")
        class_dist = pd.DataFrame({
            'Class': ['Benign (0)', 'Malignant (1)'],
            'Count': [(y == 0).sum(), (y == 1).sum()],
            'Percentage': [(y == 0).sum()/len(y)*100, (y == 1).sum()/len(y)*100]
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(class_dist, width='stretch')
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = ['Benign', 'Malignant']
            counts = [(y == 0).sum(), (y == 1).sum()]
            colors = ['#51cf66', '#ff6b6b']
            ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Class Distribution', fontweight='bold')
            st.pyplot(fig)
        
        st.subheader("Dataset Features (First 10)")
        st.dataframe(X.head(), width='stretch')
        
        with st.expander("📄 Full Dataset Description"):
            st.text(descr)
    
    # ========================================================================
    # PAGE 2: MODEL TRAINING
    # ========================================================================
    elif page == "🏋️ Model Training":
        st.header("Model Training & Evaluation")
        
        st.info("""
        This section shows the training process and performance metrics of the Random Forest classifier.
        """)
        
        # Load and train model
        model, X_train, X_test, y_train, y_test = train_model()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Training Configuration")
            st.write(f"- **Model**: Random Forest Classifier")
            st.write(f"- **Number of Trees**: 100")
            st.write(f"- **Training Samples**: {X_train.shape[0]}")
            st.write(f"- **Test Samples**: {X_test.shape[0]}")
            st.write(f"- **Features**: {X_train.shape[1]}")
        
        with col2:
            st.subheader("Model Performance")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            st.metric("Training Accuracy", f"{train_acc:.2%}")
            st.metric("Test Accuracy", f"{test_acc:.2%}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(model, X_test, y_test)
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        class_report = classification_report(y_test, y_test_pred, 
                                             target_names=['Benign', 'Malignant'], 
                                             output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        st.dataframe(class_report_df, width='stretch')
        
        # Global Feature Importance
        st.subheader("Global Feature Importance")
        fig, importance_df = plot_global_feature_importance(model, X_train)
        st.pyplot(fig)
        
        with st.expander("📊 View All Feature Importance Scores"):
            st.dataframe(importance_df, width='stretch')
    
    # ========================================================================
    # PAGE 3: SINGLE PREDICTION EXPLAINER
    # ========================================================================
    elif page == "🔍 Single Prediction Explainer":
        st.header("Explain Individual Predictions with LIME")
        
        st.info("""
        Select a patient case and LIME will explain which features contributed most to 
        the model's prediction (Benign or Malignant).
        """)
        
        # Load components
        explainer, model, X_train, X_test, y_train, y_test = create_lime_explainer()
        
        # Instance Selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_idx = st.slider(
                "Select a test case (patient) to explain:",
                min_value=0,
                max_value=len(X_test) - 1,
                value=0,
                step=1
            )
        with col2:
            st.metric("Total Test Cases", len(X_test))
        
        # Get explanation
        exp_data = get_explanation_for_instance(explainer, model, X_train, X_test, selected_idx)
        
        # Display Results
        col1, col2, col3 = st.columns(3)
        
        prediction_label = ['Benign', 'Malignant'][exp_data['prediction_class']]
        actual_label = ['Benign', 'Malignant'][y_test.iloc[selected_idx]]
        confidence = exp_data['proba'][exp_data['prediction_class']]
        
        with col1:
            st.metric("Prediction", prediction_label)
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        with col3:
            is_correct = prediction_label == actual_label
            status = "✅ Correct" if is_correct else "❌ Incorrect"
            st.metric("Actual Label", f"{actual_label} {status}")
        
        # Probability Distribution
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Class': ['Benign', 'Malignant'],
            'Probability': exp_data['proba']
        })
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(prob_df, width='stretch')
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#51cf66' if c == 'Benign' else '#ff6b6b' for c in prob_df['Class']]
            ax.bar(prob_df['Class'], prob_df['Probability'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_ylabel('Probability', fontweight='bold')
            ax.set_title('Prediction Probabilities', fontweight='bold')
            ax.set_ylim(0, 1)
            for i, v in enumerate(prob_df['Probability']):
                ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
            st.pyplot(fig)
        
        # LIME Explanation
        st.subheader("LIME Explanation - Feature Contributions")
        st.write(f"*Top 10 features that influenced the **{prediction_label}** prediction:*")
        
        fig = plot_lime_explanation(exp_data, 
                                   f"LIME Explanation - Predicted: {prediction_label} (Confidence: {confidence:.2%})")
        st.pyplot(fig)
        
        # Feature Contributions Table
        st.subheader("Feature Contribution Details")
        exp_details = []
        for i, (feature, contribution) in enumerate(exp_data['exp_list'], 1):
            direction = "↑ Malignant" if contribution > 0 else "↓ Benign"
            exp_details.append({
                'Rank': i,
                'Feature': feature,
                'Contribution': f"{contribution:+.4f}",
                'Direction': direction
            })
        
        st.dataframe(pd.DataFrame(exp_details), width='stretch')
        
        # Patient Details
        st.subheader("Patient Measurements (Normalized)")
        instance_df = pd.DataFrame({
            'Feature': exp_data['instance'].index,
            'Scaled Value': exp_data['instance'].values
        })
        st.dataframe(instance_df, width='stretch')
    
    # ========================================================================
    # PAGE 4: CASE STUDIES
    # ========================================================================
    elif page == "📈 Case Studies":
        st.header("Case Studies - Real Patient Examples")
        
        st.info("""
        Compare LIME explanations for both Benign and Malignant cases to understand 
        how the model makes different decisions.
        """)
        
        explainer, model, X_train, X_test, y_train, y_test = create_lime_explainer()
        
        # Find one benign and one malignant case
        benign_idx = np.where((y_test == 0).values)[0][0]
        malignant_idx = np.where((y_test == 1).values)[0][0]
        
        cases = [
            ('Benign Case Study', benign_idx, y_test.iloc[benign_idx]),
            ('Malignant Case Study', malignant_idx, y_test.iloc[malignant_idx])
        ]
        
        tabs = st.tabs([c[0] for c in cases])
        
        for tab, (case_name, idx, actual_class) in zip(tabs, cases):
            with tab:
                exp_data = get_explanation_for_instance(explainer, model, X_train, X_test, idx)
                
                prediction_label = ['Benign', 'Malignant'][exp_data['prediction_class']]
                confidence = exp_data['proba'][exp_data['prediction_class']]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Predicted", prediction_label)
                with col2:
                    st.metric("Confidence", f"{confidence:.2%}")
                with col3:
                    actual_label = ['Benign', 'Malignant'][actual_class]
                    st.metric("Actual", actual_label)
                with col4:
                    is_correct = prediction_label == actual_label
                    st.metric("Correct", "✅" if is_correct else "❌")
                
                st.subheader("LIME Explanation")
                fig = plot_lime_explanation(exp_data, 
                                           f"{case_name} - Predicted: {prediction_label}")
                st.pyplot(fig)
                
                st.subheader("Key Findings")
                top_3_features = exp_data['exp_list'][:3]
                for i, (feature, contribution) in enumerate(top_3_features, 1):
                    direction = "positive (toward Malignant)" if contribution > 0 else "negative (toward Benign)"
                    st.write(f"{i}. **{feature}** - {direction} contribution of **{contribution:+.4f}**")
    
    # ========================================================================
    # PAGE 5: LEARNING RESOURCES
    # ========================================================================
    elif page == "🎓 Learning Resources":
        st.header("Learning Resources - Understanding LIME")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("What is LIME?")
            st.write("""
            **LIME** stands for **Local Interpretable Model-agnostic Explanations**.
            
            It's a technique to explain individual predictions of any machine learning model 
            by approximating the model locally with an interpretable one.
            """)
            
            st.subheader("How LIME Works")
            st.write("""
            1. **Select an instance** to explain
            2. **Perturb the instance** slightly and observe prediction changes
            3. **Weight predictions** based on similarity to original instance
            4. **Fit a linear model** using weighted predictions
            5. **Extract feature contributions** from the linear model
            """)
        
        with col2:
            st.subheader("Key Advantages")
            st.write("""
            ✅ **Model-Agnostic** - Works with any model
            ✅ **Interpretable** - Produces easy-to-understand explanations
            ✅ **Local** - Explains individual predictions
            ✅ **Flexible** - Can explain any type of prediction
            """)
            
            st.subheader("Key Limitations")
            st.write("""
            ❌ **Computational Cost** - Can be slow for large datasets
            ❌ **Sensitivity** - Results depend on perturbation method
            ❌ **Limited Scope** - Only explains local behavior
            ❌ **Stability** - Explanations can vary between runs
            """)
        
        st.divider()
        
        st.subheader("Comparison: Local vs Global Interpretability")
        
        comparison_df = pd.DataFrame({
            'Aspect': ['What it explains', 'Scope', 'Use Case', 'Example', 'Tools'],
            'Local (LIME)': [
                'Individual predictions',
                'Single instance',
                'Understanding specific decisions',
                'Why patient X classified as Malignant?',
                'LIME, SHAP'
            ],
            'Global (Feature Importance)': [
                'Overall model behavior',
                'Entire dataset',
                'Understanding model patterns',
                'Which measurements matter most?',
                'Feature Importance, Permutation'
            ]
        })
        
        st.dataframe(comparison_df, width='stretch')
        
        st.subheader("Healthcare Applications")
        st.write("""
        🏥 **Medical Diagnosis Support**: Doctors can understand AI reasoning
        
        🔍 **Quality Assurance**: Verify model behavior on edge cases
        
        📊 **Regulatory Compliance**: Explain decisions to regulators
        
        🎓 **Model Development**: Identify which features should be collected
        
        ⚠️ **Anomaly Detection**: Spot unusual patient measurements
        """)
        
        st.subheader("Further Reading")
        st.write("""
        - [LIME Paper](https://arxiv.org/abs/1602.04938) - Original research paper
        - [LIME GitHub](https://github.com/marcotcr/lime) - Official repository
        - [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) - Free book
        - [SHAP Documentation](https://shap.readthedocs.io/) - Alternative explainability method
        """)

if __name__ == "__main__":
    main()
