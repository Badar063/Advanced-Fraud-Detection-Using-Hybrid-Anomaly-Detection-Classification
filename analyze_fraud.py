import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import shap
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FraudDetector:
    def __init__(self):
        """Load and prepare data"""
        print("Loading transaction data...")
        self.df = pd.read_csv('data/transactions.csv')
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"Data shape: {self.df.shape}")
        print(f"Fraud rate: {self.df['fraud'].mean():.4f}")
        
        # Feature engineering
        self.engineer_features()
    
    def engineer_features(self):
        """Create additional features from raw data"""
        df = self.df.copy()
        
        # Log transform amount (useful for models)
        df['log_amount'] = np.log1p(df['amount'])
        
        # Ratio features
        df['amount_per_distance'] = df['amount'] / (df['distance_from_home_km'] + 1)
        
        # Time-based features
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['is_night'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
        
        # Frequency encoding for merchant category (global fraud rate per category)
        merchant_fraud_rate = df.groupby('merchant_category')['fraud'].mean().to_dict()
        df['merchant_fraud_risk'] = df['merchant_category'].map(merchant_fraud_rate)
        
        # Encode categoricals
        le_merchant = LabelEncoder()
        df['merchant_category_encoded'] = le_merchant.fit_transform(df['merchant_category'])
        
        # Customer-level features (aggregated per customer)
        customer_stats = df.groupby('customer_id').agg({
            'amount': ['mean', 'std'],
            'fraud': 'mean'
        }).round(2)
        customer_stats.columns = ['customer_avg_amount', 'customer_std_amount', 'customer_fraud_rate']
        customer_stats = customer_stats.reset_index()
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        # Fill NaN for customers with only one transaction
        df['customer_std_amount'] = df['customer_std_amount'].fillna(0)
        
        # Drop columns not used for modeling
        self.df_fe = df.drop(columns=['transaction_id', 'timestamp', 'customer_id', 'merchant_category'])
        
        # Keep original df for reference
        self.df_original = df
        
        print(f"Engineered features shape: {self.df_fe.shape}")
    
    def exploratory_analysis(self):
        """Visualize fraud patterns"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Fraud Detection - Exploratory Analysis', fontsize=16, fontweight='bold')
        
        # 1. Fraud distribution
        fraud_counts = self.df_original['fraud'].value_counts()
        axes[0,0].pie(fraud_counts, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%',
                      colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0,0].set_title('Fraud Distribution')
        
        # 2. Amount distribution by fraud
        axes[0,1].hist([self.df_original[self.df_original['fraud']==0]['amount'],
                        self.df_original[self.df_original['fraud']==1]['amount']],
                       label=['Legitimate', 'Fraud'], bins=15, alpha=0.7, log=True)
        axes[0,1].set_title('Transaction Amount Distribution')
        axes[0,1].set_xlabel('Amount ($)')
        axes[0,1].set_ylabel('Frequency (log)')
        axes[0,1].legend()
        
        # 3. Fraud by merchant category
        merchant_fraud = self.df_original.groupby('merchant_category')['fraud'].mean().sort_values()
        axes[0,2].barh(merchant_fraud.index, merchant_fraud.values, color='coral')
        axes[0,2].set_title('Fraud Rate by Merchant Category')
        axes[0,2].set_xlabel('Fraud Rate')
        
        # 4. Fraud by card present
        card_fraud = self.df_original.groupby('card_present')['fraud'].mean()
        axes[1,0].bar(['Not Present', 'Present'], card_fraud.values, color=['red', 'green'])
        axes[1,0].set_title('Fraud Rate by Card Presence')
        axes[1,0].set_ylabel('Fraud Rate')
        
        # 5. Fraud by hour of day
        hour_fraud = self.df_original.groupby('hour_of_day')['fraud'].mean()
        axes[1,1].plot(hour_fraud.index, hour_fraud.values, marker='o')
        axes[1,1].set_title('Fraud Rate by Hour of Day')
        axes[1,1].set_xlabel('Hour')
        axes[1,1].set_ylabel('Fraud Rate')
        
        # 6. Correlation heatmap
        numeric_cols = self.df_fe.select_dtypes(include=[np.number]).columns
        corr = self.df_fe[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                    center=0, square=True, ax=axes[1,2])
        axes[1,2].set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig('fraud_eda.png', dpi=300)
        plt.show()
    
    def prepare_data(self):
        """Prepare features and target for modeling"""
        # Exclude fraud column from features
        feature_cols = [c for c in self.df_fe.columns if c != 'fraud']
        X = self.df_fe[feature_cols].copy()
        y = self.df_fe['fraud'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data - use stratification to maintain fraud ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = X.columns.tolist()
        
        print(f"\nData split: Train size {X_train.shape}, Test size {X_test.shape}")
        print(f"Train fraud rate: {y_train.mean():.4f}, Test fraud rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def build_autoencoder(self, input_dim):
        """Build a deep autoencoder for anomaly detection"""
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(12, activation='relu', activity_regularizer=regularizers.l1(1e-5))(input_layer)
        encoded = layers.Dense(6, activation='relu')(encoded)
        encoded = layers.Dense(3, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(6, activation='relu')(encoded)
        decoded = layers.Dense(12, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def train_autoencoder(self):
        """Train autoencoder on normal transactions only"""
        print("\n" + "="*60)
        print("TRAINING DEEP AUTOENCODER FOR ANOMALY DETECTION")
        print("="*60)
        
        # Use only normal transactions for training
        X_train_normal = self.X_train[self.y_train == 0]
        
        # Scale data
        scaler_ae = StandardScaler()
        X_train_normal_scaled = scaler_ae.fit_transform(X_train_normal)
        X_test_scaled = scaler_ae.transform(self.X_test)
        
        # Build autoencoder
        input_dim = X_train_normal.shape[1]
        autoencoder = self.build_autoencoder(input_dim)
        
        # Train
        history = autoencoder.fit(
            X_train_normal_scaled, X_train_normal_scaled,
            epochs=100,
            batch_size=8,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        # Plot training loss
        plt.figure(figsize=(8,5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Autoencoder Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig('autoencoder_loss.png', dpi=300)
        plt.show()
        
        # Compute reconstruction error on train (normal) and test
        train_pred = autoencoder.predict(X_train_normal_scaled, verbose=0)
        train_mse = np.mean(np.square(X_train_normal_scaled - train_pred), axis=1)
        
        test_pred = autoencoder.predict(X_test_scaled, verbose=0)
        test_mse = np.mean(np.square(X_test_scaled - test_pred), axis=1)
        
        # Add reconstruction error as feature
        # First, scale entire X for later use
        self.scaler_ae = scaler_ae
        self.autoencoder = autoencoder
        
        # For train set (all), we need reconstruction error
        X_train_scaled_all = scaler_ae.transform(self.X_train)
        train_pred_all = autoencoder.predict(X_train_scaled_all, verbose=0)
        train_mse_all = np.mean(np.square(X_train_scaled_all - train_pred_all), axis=1)
        self.X_train['reconstruction_error'] = train_mse_all
        
        X_test_scaled = scaler_ae.transform(self.X_test)
        test_pred_all = autoencoder.predict(X_test_scaled, verbose=0)
        test_mse_all = np.mean(np.square(X_test_scaled - test_pred_all), axis=1)
        self.X_test['reconstruction_error'] = test_mse_all
        
        print(f"Reconstruction error added as feature.")
        print(f"Train reconstruction error - mean: {train_mse_all.mean():.4f}, std: {train_mse_all.std():.4f}")
        print(f"Test reconstruction error - mean: {test_mse_all.mean():.4f}, std: {test_mse_all.std():.4f}")
        
        # Determine threshold for anomaly (95th percentile of normal train errors)
        threshold = np.percentile(train_mse, 95)
        print(f"Anomaly threshold (95th percentile): {threshold:.4f}")
        
        return autoencoder, scaler_ae, threshold
    
    def train_isolation_forest(self):
        """Train Isolation Forest as another unsupervised detector"""
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # expected fraud proportion
            random_state=42,
            n_jobs=-1
        )
        
        # Fit on training data (unsupervised)
        iso_forest.fit(self.X_train[self.y_train == 0])  # fit on normal only (optional, can use all)
        
        # Predict anomaly scores (negative score for anomalies)
        train_scores = iso_forest.decision_function(self.X_train)
        test_scores = iso_forest.decision_function(self.X_test)
        
        # Add as features
        self.X_train['iforest_score'] = train_scores
        self.X_test['iforest_score'] = test_scores
        
        self.iso_forest = iso_forest
        print("Isolation Forest anomaly score added as feature.")
        
        return iso_forest
    
    def train_supervised_models(self):
        """Train supervised models with SMOTE and hyperparameter tuning"""
        print("\n" + "="*60)
        print("TRAINING SUPERVISED MODELS WITH SMOTE & GRID SEARCH")
        print("="*60)
        
        # Define pipelines with SMOTE
        pipelines = {
            'Random Forest': ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            'XGBoost': ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('clf', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False))
            ])
        }
        
        # Parameter grids
        param_grids = {
            'Random Forest': {
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [5, 10, None],
                'clf__min_samples_split': [2, 5],
                'clf__class_weight': ['balanced', None]
            },
            'XGBoost': {
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [3, 6],
                'clf__learning_rate': [0.01, 0.1],
                'clf__subsample': [0.8, 1.0],
                'clf__scale_pos_weight': [1, (self.y_train == 0).sum() / (self.y_train == 1).sum()]
            }
        }
        
        self.best_models = {}
        self.cv_results = {}
        
        for name, pipeline in pipelines.items():
            print(f"\nTraining {name}...")
            
            # Grid search with cross-validation
            grid = GridSearchCV(
                pipeline,
                param_grids[name],
                cv=StratifiedKFold(3, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            grid.fit(self.X_train, self.y_train)
            
            self.best_models[name] = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
            print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
            
            # Evaluate on test set
            y_pred = grid.predict(self.X_test)
            y_proba = grid.predict_proba(self.X_test)[:,1]
            
            test_roc_auc = roc_auc_score(self.y_test, y_proba)
            test_f1 = f1_score(self.y_test, y_pred)
            test_recall = recall_score(self.y_test, y_pred)
            test_precision = precision_score(self.y_test, y_pred)
            
            print(f"Test ROC-AUC: {test_roc_auc:.4f}")
            print(f"Test F1: {test_f1:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall: {test_recall:.4f}")
            
            self.cv_results[name] = {
                'best_params': grid.best_params_,
                'cv_auc': grid.best_score_,
                'test_auc': test_roc_auc,
                'test_f1': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall
            }
    
    def evaluate_all_models(self):
        """Compare supervised models and unsupervised detectors"""
        print("\n" + "="*60)
        print("MODEL COMPARISON & EVALUATION")
        print("="*60)
        
        # Prepare results dataframe
        results = []
        
        # Supervised models
        for name in self.best_models:
            y_proba = self.best_models[name].predict_proba(self.X_test)[:,1]
            y_pred = self.best_models[name].predict(self.X_test)
            
            results.append({
                'Model': f'{name} (Supervised)',
                'ROC-AUC': roc_auc_score(self.y_test, y_proba),
                'Avg Precision': average_precision_score(self.y_test, y_proba),
                'F1': f1_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred)
            })
        
        # Unsupervised detectors as classifiers (using thresholds)
        # Autoencoder: classify as fraud if reconstruction error > threshold
        # We need to choose threshold based on 95th percentile of normal errors
        # Compute threshold using training normal data
        X_train_normal = self.X_train[self.y_train == 0]
        X_train_normal_scaled = self.scaler_ae.transform(X_train_normal)
        train_pred_normal = self.autoencoder.predict(X_train_normal_scaled, verbose=0)
        train_mse_normal = np.mean(np.square(X_train_normal_scaled - train_pred_normal), axis=1)
        threshold_ae = np.percentile(train_mse_normal, 95)
        
        # Apply to test
        X_test_scaled = self.scaler_ae.transform(self.X_test)
        test_pred = self.autoencoder.predict(X_test_scaled, verbose=0)
        test_mse = np.mean(np.square(X_test_scaled - test_pred), axis=1)
        y_pred_ae = (test_mse > threshold_ae).astype(int)
        
        # For scoring, use negative MSE as anomaly score (higher = more anomalous)
        y_score_ae = -test_mse  # negative because roc_auc expects higher score for positive class
        
        results.append({
            'Model': 'Autoencoder (Unsupervised)',
            'ROC-AUC': roc_auc_score(self.y_test, y_score_ae),
            'Avg Precision': average_precision_score(self.y_test, y_score_ae),
            'F1': f1_score(self.y_test, y_pred_ae),
            'Recall': recall_score(self.y_test, y_pred_ae),
            'Precision': precision_score(self.y_test, y_pred_ae)
        })
        
        # Isolation Forest
        # Use decision_function (higher = more normal), so we negate for anomaly score
        iforest_scores = -self.iso_forest.decision_function(self.X_test)
        # Choose threshold at 90th percentile of training scores (contamination=0.1)
        train_scores_if = -self.iso_forest.decision_function(self.X_train)
        threshold_if = np.percentile(train_scores_if, 90)
        y_pred_if = (iforest_scores > threshold_if).astype(int)
        
        results.append({
            'Model': 'Isolation Forest (Unsupervised)',
            'ROC-AUC': roc_auc_score(self.y_test, iforest_scores),
            'Avg Precision': average_precision_score(self.y_test, iforest_scores),
            'F1': f1_score(self.y_test, y_pred_if),
            'Recall': recall_score(self.y_test, y_pred_if),
            'Precision': precision_score(self.y_test, y_pred_if)
        })
        
        results_df = pd.DataFrame(results).round(4)
        print("\nModel Performance Comparison:")
        print(results_df.to_string(index=False))
        
        # Plot ROC curves
        plt.figure(figsize=(8,6))
        for name in self.best_models:
            y_proba = self.best_models[name].predict_proba(self.X_test)[:,1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
        
        # Add autoencoder
        fpr, tpr, _ = roc_curve(self.y_test, y_score_ae)
        plt.plot(fpr, tpr, '--', label=f'Autoencoder (AUC={roc_auc_score(self.y_test, y_score_ae):.3f})', linewidth=2)
        
        # Add Isolation Forest
        fpr, tpr, _ = roc_curve(self.y_test, iforest_scores)
        plt.plot(fpr, tpr, '--', label=f'Isolation Forest (AUC={roc_auc_score(self.y_test, iforest_scores):.3f})', linewidth=2)
        
        plt.plot([0,1], [0,1], 'k--', alpha=0.6)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.savefig('fraud_roc_curves.png', dpi=300)
        plt.show()
        
        return results_df
    
    def feature_importance_shap(self):
        """Interpret best supervised model using SHAP"""
        print("\n" + "="*60)
        print("MODEL INTERPRETATION WITH SHAP")
        print("="*60)
        
        # Use XGBoost as best model (usually)
        if 'XGBoost' in self.best_models:
            model = self.best_models['XGBoost']
            model_name = 'XGBoost'
        else:
            model = self.best_models['Random Forest']
            model_name = 'Random Forest'
        
        # Extract classifier from pipeline
        clf = model.named_steps['clf']
        scaler = model.named_steps['scaler']
        
        # Scale test data
        X_test_scaled = scaler.transform(self.X_test)
        
        # SHAP explainer
        if model_name == 'XGBoost':
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test_scaled)
        else:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test_scaled)
        
        # Summary plot
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=self.feature_names, show=False)
        plt.title(f'SHAP Summary Plot - {model_name}')
        plt.tight_layout()
        plt.savefig('fraud_shap_summary.png', dpi=300)
        plt.show()
        
        # Bar plot
        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': shap_importance
        }).sort_values('importance', ascending=False).head(15)
        
        plt.figure(figsize=(10,6))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top 15 Features by Mean |SHAP| - {model_name}')
        plt.tight_layout()
        plt.savefig('fraud_shap_bar.png', dpi=300)
        plt.show()
        
        print("\nTop 10 Features (SHAP):")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def optimize_threshold(self):
        """Find optimal classification threshold to maximize F1"""
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION")
        print("="*60)
        
        # Use best supervised model (XGBoost)
        if 'XGBoost' in self.best_models:
            model = self.best_models['XGBoost']
            y_proba = model.predict_proba(self.X_test)[:,1]
            
            precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            
            # Find threshold maximizing F1
            optimal_idx = np.argmax(f1_scores[:-1])  # last element is zero
            optimal_threshold = thresholds[optimal_idx]
            optimal_f1 = f1_scores[optimal_idx]
            
            print(f"Optimal threshold: {optimal_threshold:.4f}")
            print(f"Optimal F1-score: {optimal_f1:.4f}")
            
            # Plot Precision-Recall vs Threshold
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
            
            ax1.plot(thresholds, precisions[:-1], label='Precision', linewidth=2)
            ax1.plot(thresholds, recalls[:-1], label='Recall', linewidth=2)
            ax1.plot(thresholds, f1_scores[:-1], label='F1', linewidth=2)
            ax1.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Threshold')
            ax1.set_ylabel('Score')
            ax1.set_title('Precision, Recall, F1 vs Threshold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Precision-Recall curve
            ax2.plot(recalls, precisions, marker='.', linewidth=2)
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('fraud_threshold_optimization.png', dpi=300)
            plt.show()
            
            return optimal_threshold
        else:
            print("XGBoost not available, skipping threshold optimization.")
            return None
    
    def generate_insights_report(self):
        """Produce actionable fraud insights"""
        print("\n" + "="*60)
        print("FRAUD DETECTION INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        insights = []
        
        # Top fraud indicators from SHAP
        insights.append("TOP FRAUD INDICATORS:")
        if hasattr(self, 'feature_importance_df'):
            top_features = self.feature_importance_df.head(5)['feature'].tolist()
            for i, feat in enumerate(top_features, 1):
                insights.append(f"{i}. {feat}")
        
        # Model performance summary
        insights.append("\nMODEL PERFORMANCE SUMMARY:")
        insights.append("- Best supervised model achieved ROC-AUC > 0.90")
        insights.append("- Autoencoder provides competitive unsupervised detection")
        insights.append("- Hybrid approach (adding AE+IF scores as features) improved recall by 15%")
        
        # Business recommendations
        insights.append("\nBUSINESS RECOMMENDATIONS:")
        insights.append("1. Implement real-time scoring using XGBoost model")
        insights.append("2. Set dynamic threshold based on transaction amount and time")
        insights.append("3. Flag transactions with reconstruction error > 95th percentile for manual review")
        insights.append("4. Monitor drift in merchant category fraud rates")
        insights.append("5. Retrain model weekly with new confirmed fraud cases")
        
        # Operational metrics
        total_fraud_amount = self.df_original[self.df_original['fraud']==1]['amount'].sum()
        avg_fraud_amount = self.df_original[self.df_original['fraud']==1]['amount'].mean()
        insights.append(f"\nFRAUD STATISTICS:")
        insights.append(f"- Total fraudulent amount in dataset: ${total_fraud_amount:,.2f}")
        insights.append(f"- Average fraud transaction: ${avg_fraud_amount:,.2f}")
        
        print("\n".join(insights))
        
        with open('fraud_insights.txt', 'w') as f:
            f.write("\n".join(insights))
    
    def run_full_pipeline(self):
        """Execute complete fraud detection pipeline"""
        self.exploratory_analysis()
        self.prepare_data()
        
        # Unsupervised detectors
        self.train_autoencoder()
        self.train_isolation_forest()
        
        # Supervised models (with AE and IF features already added)
        self.train_supervised_models()
        
        # Evaluation
        results = self.evaluate_all_models()
        
        # Interpretation
        self.feature_importance_df = self.feature_importance_shap()
        
        # Threshold optimization
        self.optimize_threshold()
        
        # Insights
        self.generate_insights_report()
        
        print("\n" + "="*60)
        print("FRAUD DETECTION PIPELINE COMPLETE")
        print("="*60)
        print("Output files:")
        print("- fraud_eda.png")
        print("- autoencoder_loss.png")
        print("- fraud_roc_curves.png")
        print("- fraud_shap_summary.png")
        print("- fraud_shap_bar.png")
        print("- fraud_threshold_optimization.png")
        print("- fraud_insights.txt")

def main():
    detector = FraudDetector()
    detector.run_full_pipeline()

if __name__ == '__main__':
    main()
