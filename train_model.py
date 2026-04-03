import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import shap
import pickle

dataset_path = 'dataset/tox21.csv'
model_path = 'model/toxicity_model.pkl'

def generate_synthetic_data(path, num_samples=2000):
    print("Generating synthetic tox21 dataset...")
    np.random.seed(42)
    # Generate features
    mw = np.random.normal(350, 100, num_samples) # Molecular Weight
    logp = np.random.normal(2.5, 1.5, num_samples) # LogP
    tpsa = np.random.normal(80, 30, num_samples) # TPSA
    num_h_donors = np.random.poisson(1.5, num_samples)
    num_h_acceptors = np.random.poisson(4.0, num_samples)
    num_rotatable_bonds = np.random.poisson(4.5, num_samples)
    aromatic_rings = np.random.poisson(1.5, num_samples)
    
    # Generate toxicity based on some non-linear combination
    toxicity_prob = 1 / (1 + np.exp(-(0.05*mw + 1.2*logp - 0.03*tpsa + 0.5*num_h_donors - 0.2*num_h_acceptors - 6)))
    toxicity = np.where(toxicity_prob > 0.5, 1, 0)
    
    df = pd.DataFrame({
        'MolecularWeight': mw,
        'LogP': logp,
        'TPSA': tpsa,
        'NumHDonors': num_h_donors,
        'NumHAcceptors': num_h_acceptors,
        'NumRotatableBonds': num_rotatable_bonds,
        'AromaticRings': aromatic_rings,
        'Toxicity': toxicity
    })
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset generated at {path}")

def run_pipeline():
    if not os.path.exists(dataset_path):
        generate_synthetic_data(dataset_path)
    
    print("\n--- 1. Data Loading ---")
    df = pd.read_csv(dataset_path)
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDataset Statistics:\n", df.describe())
    
    print("\n--- 2. Data Preprocessing ---")
    df = df.dropna()
    X = df.drop('Toxicity', axis=1)
    y = df['Toxicity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("\n--- 3. Exploratory Data Analysis ---")
    sns.set_theme(style="whitegrid")
    
    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('static/images/correlation_heatmap.png')
    plt.close()
    
    # Distribution of toxicity labels
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Toxicity', data=df)
    plt.title("Distribution of Toxicity Labels")
    plt.savefig('static/images/toxicity_dist.png')
    plt.close()
    
    # Feature pairs
    sns.pairplot(df, vars=['MolecularWeight', 'LogP', 'TPSA'], hue='Toxicity')
    plt.savefig('static/images/pairplot.png')
    plt.close()
    
    print("\n--- 4. Feature Selection & 5. Machine Learning Models ---")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_f1 = 0
    results = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else preds
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, prob)
        
        results.append((name, acc, prec, rec, f1, roc))
        print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            
    print("\nBest Model selected based on F1-Score:", best_model.__class__.__name__)
    
    print("\n--- 6. Model Explainability ---")
    # Using best model (e.g., XGBoost or RF) for SHAP
    if hasattr(best_model, "feature_importances_"):
        plt.figure(figsize=(8, 6))
        sns.barplot(x=best_model.feature_importances_, y=X.columns)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig('static/images/feature_importance.png')
        plt.close()
    
    try:
        explainer = shap.Explainer(best_model)
        shap_values = explainer(X_test_scaled)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test_scaled, show=False)
        plt.savefig('static/images/shap_summary.png', bbox_inches='tight')
        plt.close()
    except Exception as e:
        print("SHAP plot generation encountered an issue:", str(e))
    
    print("\n--- 7. Save Model ---")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': best_model, 'scaler': scaler, 'features': list(X.columns)}, f)
    print(f"Model and scaler saved to {model_path}")

if __name__ == "__main__":
    os.makedirs('static/images', exist_ok=True)
    run_pipeline()
