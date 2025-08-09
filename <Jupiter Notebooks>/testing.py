import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support,
                           roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class InjurySeverityAnalysis:
    def __init__(self, uk_data, us_data):
        self.uk_data = uk_data.copy()
        self.us_data = us_data.copy()
        self.models = {}
        self.best_params = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, dataset='UK'):
        """Prepare data for modeling"""
        print(f"Preparing {dataset} dataset...")
        
        if dataset == 'UK':
            data = self.uk_data.copy()
        else:
            data = self.us_data.copy()
        
        # Identify target column (assuming it's related to injury severity)
        # Look for columns that might represent injury severity
        potential_targets = [col for col in data.columns if 
                           any(word in col.lower() for word in 
                               ['severity', 'injury', 'casualty', 'fatal', 'killed', 'serious'])]
        
        print(f"Potential target columns: {potential_targets}")
        
        # For this example, let's assume the last column is the target
        # You should adjust this based on your actual target column
        if potential_targets:
            target_col = potential_targets[0]  # Use first match
        else:
            # If no obvious target column, use the last column
            target_col = data.columns[-1]
            
        print(f"Using '{target_col}' as target variable")
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Encode target if it's not already numeric
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
            print(f"Target classes after encoding: {np.unique(y)}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Target distribution in training set:")
        print(pd.Series(self.y_train).value_counts().sort_index())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def initialize_models(self):
        """Initialize all models"""
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
            'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
            'SVM': SVC(random_state=RANDOM_STATE, probability=True),
            'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        }
    
    def define_param_grids(self):
        """Define parameter grids for RandomizedSearchCV"""
        return {
            'Decision Tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'l1_ratio': [0.1, 0.5, 0.9]
            }
        }
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning using RandomizedSearchCV"""
        print("Starting hyperparameter tuning...")
        param_grids = self.define_param_grids()
        
        for name, model in self.models.items():
            print(f"\nTuning {name}...")
            
            # Special handling for Logistic Regression elasticnet
            if name == 'Logistic Regression':
                param_grid = param_grids[name].copy()
                # Remove elasticnet combinations that don't work with certain solvers
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            else:
                param_grid = param_grids[name]
            
            random_search = RandomizedSearchCV(
                model, param_grid, n_iter=20, cv=3, 
                scoring='accuracy', n_jobs=-1, random_state=RANDOM_STATE
            )
            
            random_search.fit(self.X_train, self.y_train)
            
            self.best_params[name] = random_search.best_params_
            self.models[name] = random_search.best_estimator_
            
            print(f"Best parameters for {name}: {random_search.best_params_}")
            print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\nTraining and evaluating models...")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(self.y_train, y_train_pred)
            test_accuracy = accuracy_score(self.y_test, y_test_pred)
            
            # Calculate precision, recall, f1-score
            train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
                self.y_train, y_train_pred, average='weighted'
            )
            test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
                self.y_test, y_test_pred, average='weighted'
            )
            
            # Store results
            self.results[name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'train_precision': train_prec,
                'test_precision': test_prec,
                'train_recall': train_rec,
                'test_recall': test_rec,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            }
            
            print(f"Training Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Test Precision: {test_prec:.4f}")
            print(f"Test Recall: {test_rec:.4f}")
            print(f"Test F1-Score: {test_f1:.4f}")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (name, results) in enumerate(self.results.items()):
            if i >= len(axes):
                break
                
            cm = confusion_matrix(self.y_test, results['y_test_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name} - Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Remove empty subplot
        if len(self.models) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self):
        """Plot model comparison metrics"""
        metrics = ['train_accuracy', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        model_names = list(self.results.keys())
        
        # Create comparison dataframe
        comparison_data = []
        for model_name in model_names:
            row = [model_name]
            for metric in metrics:
                row.append(self.results[model_name][metric])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data, columns=['Model'] + metrics)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
            
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=axes[i], 
                             color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].legend().remove()
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def explain_with_shap(self, model_name='Random Forest'):
        """Generate SHAP explanations"""
        print(f"\nGenerating SHAP explanations for {model_name}...")
        
        model = self.models[model_name]
        
        # Create SHAP explainer
        if model_name in ['Random Forest', 'XGBoost', 'Decision Tree']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(self.X_test.iloc[:100])  # Use first 100 samples
            
            # For multi-class, take the first class
            if len(shap_values) > 1:
                shap_values = shap_values[1]  # Usually the positive class
        else:
            # For linear models
            explainer = shap.LinearExplainer(model, self.X_train)
            shap_values = explainer.shap_values(self.X_test.iloc[:100])
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test.iloc[:100], 
                         feature_names=self.feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.show()
        
        # Feature importance plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test.iloc[:100], 
                         plot_type="bar", feature_names=self.feature_names, show=False)
        plt.title(f'SHAP Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()
    
    def explain_with_lime(self, model_name='Random Forest', instance_idx=0):
        """Generate LIME explanations"""
        print(f"\nGenerating LIME explanation for {model_name}, instance {instance_idx}...")
        
        model = self.models[model_name]
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=['Low Severity', 'High Severity'],  # Adjust based on your classes
            discretize_continuous=True
        )
        
        # Explain instance
        instance = self.X_test.iloc[instance_idx].values
        explanation = explainer.explain_instance(
            instance, model.predict_proba, num_features=10
        )
        
        # Show explanation
        explanation.show_in_notebook(show_table=True)
        
        return explanation
    
    def print_results_summary(self):
        """Print comprehensive results summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
        print("="*80)
        
        # Create comparison dataframe
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Train Accuracy': f"{results['train_accuracy']:.4f}",
                'Test Accuracy': f"{results['test_accuracy']:.4f}",
                'Test Precision': f"{results['test_precision']:.4f}",
                'Test Recall': f"{results['test_recall']:.4f}",
                'Test F1-Score': f"{results['test_f1']:.4f}",
                'Overfitting': f"{results['train_accuracy'] - results['test_accuracy']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Best performing model
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['test_accuracy'])
        print(f"\nBest performing model: {best_model}")
        print(f"Test Accuracy: {self.results[best_model]['test_accuracy']:.4f}")
        
        return comparison_df

# Usage Example
def run_analysis(uk_data, us_data, dataset='UK'):
    """Run the complete analysis pipeline"""
    
    # Initialize analysis
    analysis = InjurySeverityAnalysis(uk_data, us_data)
    
    # Prepare data
    X_train, X_test, y_train, y_test = analysis.prepare_data(dataset)
    
    # Initialize models
    analysis.initialize_models()
    
    # Perform hyperparameter tuning
    analysis.hyperparameter_tuning()
    
    # Train and evaluate models
    analysis.train_and_evaluate_models()
    
    # Print results summary
    results_df = analysis.print_results_summary()
    
    # Plot confusion matrices
    analysis.plot_confusion_matrices()
    
    # Plot model comparison
    comparison_df = analysis.plot_model_comparison()
    
    # SHAP explanations
    try:
        analysis.explain_with_shap('Random Forest')
        analysis.explain_with_shap('XGBoost')
    except Exception as e:
        print(f"SHAP explanation error: {e}")
    
    # LIME explanations
    try:
        lime_explanation = analysis.explain_with_lime('Random Forest', 0)
    except Exception as e:
        print(f"LIME explanation error: {e}")
    
    return analysis, results_df

# Run the analysis
# Make sure your datasets are properly loaded
print("Starting ML Analysis Pipeline...")
print("Note: Make sure your datasets have the correct target variable column")

# Example usage (uncomment and modify based on your actual data):
# UK_data = pd.read_csv("path_to_your_uk_data.csv")
# US_data = pd.read_csv("path_to_your_us_data.csv")
# analysis, results = run_analysis(UK_data, US_data, dataset='UK')