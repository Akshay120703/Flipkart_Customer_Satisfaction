import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    """Machine Learning models for customer satisfaction prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def prepare_data(self, df):
        """Prepare data for machine learning"""
        try:
            # Create a copy of the dataframe
            df_ml = df.copy()
            
            # Target variable
            if 'CSAT_Score' in df_ml.columns:
                # Convert to binary classification (satisfied vs not satisfied)
                df_ml['satisfied'] = (df_ml['CSAT_Score'] >= 4).astype(int)
                target = 'satisfied'
            else:
                print("CSAT Score column not found")
                return None
            
            # Select features for modeling
            feature_columns = [
                'channel_name', 'category', 'Sub_category', 'Tenure_Bucket', 
                'Agent_Shift', 'connected_handling_time', 'Item_price'
            ]
            
            # Filter existing columns
            available_features = [col for col in feature_columns if col in df_ml.columns]
            
            if len(available_features) == 0:
                print("No suitable features found for modeling")
                return None
            
            # Prepare features
            X = df_ml[available_features].copy()
            y = df_ml[target]
            
            # Handle missing values
            X = self._handle_missing_values(X)
            
            # Encode categorical variables
            X_encoded = self._encode_categorical_features(X)
            
            # Store feature names
            self.feature_names = X_encoded.columns.tolist()
            
            return X_encoded, y
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            return None
    
    def _handle_missing_values(self, X):
        """Handle missing values in features"""
        # Fill categorical missing values with 'Unknown'
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('Unknown')
        
        # Fill numerical missing values with median
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            X[col] = X[col].fillna(X[col].median())
        
        return X
    
    def _encode_categorical_features(self, X):
        """Encode categorical features"""
        X_encoded = X.copy()
        
        # Get categorical columns
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        # One-hot encode categorical variables
        for col in categorical_cols:
            # Create dummy variables
            dummies = pd.get_dummies(X_encoded[col], prefix=col, drop_first=True)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded.drop(col, axis=1, inplace=True)
        
        return X_encoded
    
    def train_models(self, X, y):
        """Train multiple models and return performance metrics"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler
            self.scalers['standard'] = scaler
            
            # Define models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42, probability=True)
            }
            
            results = {}
            
            # Train and evaluate each model
            for name, model in models.items():
                try:
                    # Use scaled data for models that benefit from scaling
                    if name in ['Logistic Regression', 'SVM']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                    
                    results[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    # Store model
                    self.models[name] = model
                    
                except Exception as e:
                    print(f"Error training {name}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            return {}
    
    def hyperparameter_tuning(self, X, y, model_name='Random Forest'):
        """Perform hyperparameter tuning for specified model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Define parameter grids
            param_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'Logistic Regression': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
            
            if model_name not in param_grids:
                print(f"Hyperparameter tuning not defined for {model_name}")
                return None
            
            # Define model
            if model_name == 'Random Forest':
                model = RandomForestClassifier(random_state=42)
            elif model_name == 'Gradient Boosting':
                model = GradientBoostingClassifier(random_state=42)
            elif model_name == 'Logistic Regression':
                model = LogisticRegression(random_state=42, max_iter=1000)
            
            # Perform grid search
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit with appropriate data
            if model_name == 'Logistic Regression':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                grid_search.fit(X_train_scaled, y_train)
                y_pred = grid_search.predict(X_test_scaled)
            else:
                grid_search.fit(X_train, y_train)
                y_pred = grid_search.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1_score': f1
            }
            
            # Store best model
            self.models[f'{model_name}_tuned'] = grid_search.best_estimator_
            
            return results
            
        except Exception as e:
            print(f"Error in hyperparameter tuning: {str(e)}")
            return None
    
    def get_feature_importance(self, X, y):
        """Get feature importance from Random Forest model"""
        try:
            # Train Random Forest
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            
            # Get feature importance
            importance = rf_model.feature_importances_
            feature_names = X.columns if hasattr(X, 'columns') else self.feature_names
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None
    
    def predict_satisfaction(self, X, model_name='Random Forest'):
        """Predict customer satisfaction"""
        try:
            if model_name not in self.models:
                print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
                return None
            
            model = self.models[model_name]
            
            # Prepare input data
            X_processed = self._prepare_input_data(X)
            
            # Make predictions
            predictions = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)[:, 1]
            
            return predictions, probabilities
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None, None
    
    def _prepare_input_data(self, X):
        """Prepare input data for prediction"""
        # This would need to be implemented based on the specific preprocessing
        # steps used during training
        return X
    
    def get_model_interpretation(self, model_name='Random Forest'):
        """Get model interpretation and insights"""
        try:
            if model_name not in self.models:
                return None
            
            model = self.models[model_name]
            interpretation = {}
            
            # For tree-based models, get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_names = self.feature_names
                
                interpretation['feature_importance'] = dict(zip(feature_names, importance))
                
                # Get top 10 most important features
                top_features = sorted(
                    zip(feature_names, importance), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                
                interpretation['top_features'] = top_features
            
            # For logistic regression, get coefficients
            if hasattr(model, 'coef_'):
                coef = model.coef_[0]
                feature_names = self.feature_names
                
                interpretation['coefficients'] = dict(zip(feature_names, coef))
                
                # Get most influential features (highest absolute coefficients)
                influential_features = sorted(
                    zip(feature_names, coef), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:10]
                
                interpretation['influential_features'] = influential_features
            
            return interpretation
            
        except Exception as e:
            print(f"Error getting model interpretation: {str(e)}")
            return None
    
    def evaluate_model_performance(self, X, y, model_name='Random Forest'):
        """Detailed model performance evaluation"""
        try:
            if model_name not in self.models:
                return None
            
            model = self.models[model_name]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate detailed metrics
            evaluation = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            return evaluation
            
        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            return None
