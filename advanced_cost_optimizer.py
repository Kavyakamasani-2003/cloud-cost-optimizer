import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime  
from dotenv import load_dotenv


load_dotenv()


# Logging Configuration
def setup_logging():
    """
    Configure comprehensive logging with file and console handlers
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(os.getenv('LOG_FILE', './logs/cost_optimizer.log'))
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        handlers=[
            # File handler
            logging.FileHandler(
                os.getenv('LOG_FILE', './logs/cost_optimizer.log')
            ),
            # Console handler
            logging.StreamHandler()
        ]
    )

    # Create and return logger
    logger = logging.getLogger('CloudCostOptimizer')
    return logger

# Global logger
logger = setup_logging()


class AdvancedCostOptimizer:
    def __init__(self, csv_path: str = None):
        """
        Advanced Cost Optimization Engine
        
        :param csv_path: Path to cloud cost CSV file
        """
        # Use environment variable or default path
        self.csv_path = csv_path or os.getenv('DATA_PATH', './data/augmented_cloud_costs.csv')
        
        # Create output directory
        self.output_dir = os.getenv('OUTPUT_DIRECTORY', './reports')
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # Log initialization
            logger.info(f"Initializing Cost Optimizer with data from {self.csv_path}")
            
            # Read and process data
            self.data = pd.read_csv(self.csv_path)
            self.data['UsageDate'] = pd.to_datetime(self.data['UsageDate'])
            
            # Log data details
            logger.info(f"Loaded {len(self.data)} records")
        
        except Exception as e:
            logger.error(f"Error during initialization: {e}", exc_info=True)
            raise
    
    def _enhanced_feature_engineering(self, service_data):
        """
        Advanced feature engineering with more sophisticated transformations
        
        :param service_data: DataFrame for a specific service
        :return: Enhanced feature matrix
        """
        # Ensure data is sorted by date
        service_data = service_data.sort_values('UsageDate')
        
        # Existing feature engineering
        service_data['Month'] = service_data['UsageDate'].dt.month
        service_data['DayOfWeek'] = service_data['UsageDate'].dt.dayofweek
        service_data['Quarter'] = service_data['UsageDate'].dt.quarter
        
        # Advanced time-based features
        service_data['DayOfYear'] = service_data['UsageDate'].dt.dayofyear
        service_data['IsWeekend'] = service_data['UsageDate'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Rolling window features with safe handling
        def safe_rolling_mean(x, window):
            """Handle rolling mean with minimum samples"""
            return x.rolling(window=window, min_periods=1).mean()
        
        service_data['CostMovingAverage_3M'] = service_data.groupby('ServiceName')['CostUSD'].transform(lambda x: safe_rolling_mean(x, 3))
        service_data['CostMovingAverage_6M'] = service_data.groupby('ServiceName')['CostUSD'].transform(lambda x: safe_rolling_mean(x, 6))
        
        # Lagged features with safe handling
        service_data['PreviousMonthCost'] = service_data.groupby('ServiceName')['CostUSD'].shift(1).fillna(service_data['CostUSD'].mean())
        
        # Cost change rate with safe handling
        service_data['CostChangeRate'] = service_data.groupby('ServiceName')['CostUSD'].pct_change(fill_method=None).fillna(0)
        
        return service_data
    
    def _preprocess_service_data(self, service_name: str):
        """
        Advanced preprocessing for service-specific data
        
        :param service_name: Name of cloud service
        :return: Preprocessed feature matrix and target
        """
        # Minimum data points check
        MIN_REQUIRED_SAMPLES = 5
        service_data = self.data[self.data['ServiceName'] == service_name].copy()
        
        if len(service_data) < MIN_REQUIRED_SAMPLES:
            logging.warning(f"Insufficient data for {service_name}. Skipping model training.")
            return None, None
        
        # Enhanced feature engineering
        try:
            service_data = self._enhanced_feature_engineering(service_data)
        except Exception as e:
            logging.error(f"Feature engineering failed for {service_name}: {e}")
            return None, None
        
        # Drop rows with NaN, but keep at least MIN_REQUIRED_SAMPLES
        service_data.dropna(inplace=True)
        
        # Recheck data points after preprocessing
        if len(service_data) < MIN_REQUIRED_SAMPLES:
            logging.warning(f"Insufficient data for {service_name} after preprocessing. Skipping model training.")
            return None, None
        
        # Select features with safe handling
        features = [
            'Month', 'DayOfWeek', 'Quarter', 'DayOfYear', 'IsWeekend', 
            'PreviousMonthCost', 'CostMovingAverage_3M', 'CostMovingAverage_6M', 
            'CostChangeRate'
        ]
        target = 'CostUSD'
        
        # Ensure all features exist
        missing_features = [f for f in features if f not in service_data.columns]
        if missing_features:
            logging.error(f"Missing features for {service_name}: {missing_features}")
            return None, None
        
        # Handle potential infinite values
        X = service_data[features]
        y = service_data[target]
        
        # Replace infinite values with median
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        
        return X, y
    
    def _preprocess_features(self, features):
        """
        Robust feature preprocessing to handle various data quality issues
        
        :param features: DataFrame of input features
        :return: Preprocessed and cleaned features
        """
        # Create a copy to avoid modifying original data
        processed_features = features.copy()
        
        # Replace infinite values with NaN
        processed_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Imputation strategies
        for column in processed_features.columns:
            # Handle NaN values
            if processed_features[column].isnull().any():
                # Use median for numerical columns
                if processed_features[column].dtype in ['float64', 'int64']:
                    processed_features[column].fillna(processed_features[column].median(), inplace=True)
                # Use mode for categorical columns
                else:
                    processed_features[column].fillna(processed_features[column].mode()[0], inplace=True)
        
        # Clip extreme values to prevent scaling issues
        for column in processed_features.columns:
            if processed_features[column].dtype in ['float64', 'int64']:
                q1 = processed_features[column].quantile(0.01)
                q99 = processed_features[column].quantile(0.99)
                processed_features[column] = processed_features[column].clip(q1, q99)
        
        return processed_features

    def _generate_prediction_features(self, service_data, recent_data):
        """
        Generate robust prediction features with comprehensive fallback
        
        :param service_data: DataFrame of service data
        :param recent_data: Most recent data point
        :return: DataFrame of prediction features
        """
        # Comprehensive feature list with default generation strategies
        feature_generation_strategies = {
            'Month': lambda: recent_data['UsageDate'].month if hasattr(recent_data, 'UsageDate') else service_data['UsageDate'].dt.month.iloc[-1],
            'DayOfWeek': lambda: recent_data['UsageDate'].dayofweek if hasattr(recent_data, 'UsageDate') else service_data['UsageDate'].dt.dayofweek.iloc[-1],
            'Quarter': lambda: recent_data['UsageDate'].quarter if hasattr(recent_data, 'UsageDate') else service_data['UsageDate'].dt.quarter.iloc[-1],
            'DayOfYear': lambda: recent_data['UsageDate'].dayofyear if hasattr(recent_data, 'UsageDate') else service_data['UsageDate'].dt.dayofyear.iloc[-1],
            'IsWeekend': lambda: int(recent_data['UsageDate'].dayofweek in [5, 6]) if hasattr(recent_data, 'UsageDate') else int(service_data['UsageDate'].dt.dayofweek.iloc[-1] in [5, 6]),
            'PreviousMonthCost': lambda: service_data['CostUSD'].mean(),
            'CostMovingAverage_3M': lambda: service_data['CostUSD'].rolling(window=3, min_periods=1).mean().iloc[-1],
            'CostMovingAverage_6M': lambda: service_data['CostUSD'].rolling(window=6, min_periods=1).mean().iloc[-1],
            'CostChangeRate': lambda: service_data['CostUSD'].pct_change().fillna(0).iloc[-1]
        }
        
        # Generate features
        prediction_features = {}
        for feature, strategy in feature_generation_strategies.items():
            try:
                prediction_features[feature] = [strategy()]
            except Exception as e:
                logging.warning(f"Error generating feature {feature}: {e}")
                prediction_features[feature] = [0]
        
        # Convert to DataFrame and preprocess
        features_df = pd.DataFrame(prediction_features)
        return self._preprocess_features(features_df)

    def train_optimization_model(self, service_name: str):
        """
        Enhanced model training with more robust validation
        
        :param service_name: Name of cloud service
        :return: Enhanced model performance metrics
        """
        try:
            # Preprocess data
            X, y = self._preprocess_service_data(service_name)
            
            # Comprehensive data validation
            if X is None or y is None:
                logging.warning(f"Cannot train model for {service_name} due to data preprocessing issues.")
                return {
                    'service': service_name,
                    'status': 'insufficient_data',
                    'message': 'Data preprocessing failed'
                }
            
            # Adjust train-test split dynamically
            test_size = min(0.2, max(0.1, 1 / len(X)))
            
            # Split data with stratification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Feature scaling with robust method
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Polynomial feature expansion
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_train_poly = poly.fit_transform(X_train_scaled)
            X_test_poly = poly.transform(X_test_scaled)
            
            # Ensemble of models for more robust prediction
            models = [
                RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
                GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            ]
            
            # Cross-validation for more reliable performance estimation
            cv_scores = {}
            final_model = None
            best_cv_score = float('-inf')
            
            for model in models:
                try:
                    model.fit(X_train_poly, y_train)
                    
                    # Perform cross-validation
                    cv_score = np.mean(cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2'))
                    cv_scores[type(model).__name__] = cv_score
                    
                    # Select best performing model
                    if cv_score > best_cv_score:
                        best_cv_score = cv_score
                        final_model = model
                except Exception as model_error:
                    logging.error(f"Model training error for {service_name}: {model_error}")
                    continue
            
            # Validate model selection
            if final_model is None:
                logging.error(f"No valid model could be trained for {service_name}")
                return {
                    'service': service_name,
                    'status': 'error',
                    'message': 'Model training failed completely'
                }
            
            # Model evaluation
            train_score = final_model.score(X_train_poly, y_train)
            test_score = final_model.score(X_test_poly, y_test)
            
            # Feature importance from the best model
            feature_names = poly.get_feature_names_out(X.columns)
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save model and artifacts
            os.makedirs(self.output_dir, exist_ok=True)
            joblib.dump(final_model, os.path.join(self.output_dir, f'{service_name}_optimization_model.joblib'))
            joblib.dump(scaler, os.path.join(self.output_dir, f'{service_name}_scaler.joblib'))
            joblib.dump(poly, os.path.join(self.output_dir, f'{service_name}_polynomial_transformer.joblib'))
            feature_importance.to_csv(os.path.join(self.output_dir, f'{service_name}_feature_importance.csv'), index=False)
            
            return {
                'service': service_name,
                'status': 'success',
                'train_r2_score': train_score,
                'test_r2_score': test_score,
                'cv_scores': cv_scores,
                'feature_importance': feature_importance.head(10).to_dict(orient='records')
            }
        
        except Exception as e:
            logging.error(f"Comprehensive error in training model for {service_name}: {e}")
            return {
                'service': service_name,
                'status': 'error',
                'message': str(e)
            }
    
    def _generate_detailed_recommendations(self, service_name, service_data, top_features, predicted_cost, avg_cost):
        """
        Generate comprehensive and nuanced recommendations
        
        :param service_name: Name of the cloud service
        :param service_data: DataFrame of service cost data
        :param top_features: Top influential features
        :param predicted_cost: Predicted future cost
        :param avg_cost: Average historical cost
        :return: List of detailed recommendations
        """
        recommendations = []
        
        # Cost Variability Analysis
        cost_std = service_data['CostUSD'].std()
        cost_variability = cost_std / max(avg_cost, 0.01)
        
        # Severity Classification
        if predicted_cost > avg_cost * (1 + cost_variability):
            severity = 'high'
            cost_increase_percentage = ((predicted_cost - avg_cost) / avg_cost) * 100
        elif predicted_cost > avg_cost * (1 + cost_variability * 0.5):
            severity = 'medium'
            cost_increase_percentage = ((predicted_cost - avg_cost) / avg_cost) * 100
        else:
            severity = 'low'
            cost_increase_percentage = 0
        
        # Detailed Cost Reduction Recommendation
        if severity in ['medium', 'high']:
            cost_reduction_rec = {
                'type': 'cost_reduction',
                'severity': severity,
                'description': f'Potential cost increase of {cost_increase_percentage:.2f}% detected for {service_name}',
                'predicted_cost': predicted_cost,
                'current_avg_cost': avg_cost,
                'suggested_actions': [
                    f'Investigate primary cost driver: {top_features.iloc[0]["feature"]} (Impact: {top_features.iloc[0].get("importance", 0):.2%})',
                    f'Secondary analysis: {top_features.iloc[1]["feature"]} (Impact: {top_features.iloc[1].get("importance", 0):.2%})',
                    'Conduct comprehensive resource utilization audit',
                    'Explore cost-optimization cloud strategies'
                ]
            }
            recommendations.append(cost_reduction_rec)
        
        # Resource Optimization Recommendation
        resource_optimization_rec = {
            'type': 'resource_optimization',
            'severity': 'medium',
            'description': 'Comprehensive cloud resource optimization insights',
            'top_features': top_features.to_dict(orient='records'),
            'suggested_actions': [
                f'Prioritize optimization of {top_features.iloc[0]["feature"]} (Strategic Impact: {top_features.iloc[0].get("importance", 0):.2%})',
                f'Secondary optimization focus: {top_features.iloc[1]["feature"]} (Tactical Impact: {top_features.iloc[1].get("importance", 0):.2%})',
                'Implement dynamic resource scaling',
                'Develop predictive cost management framework'
            ]
        }
        recommendations.append(resource_optimization_rec)
        
        # Performance Efficiency Recommendation
        performance_rec = {
            'type': 'performance_efficiency',
            'severity': 'low',
            'description': 'Long-term cloud infrastructure optimization',
            'insights': [
                f'Analyze temporal patterns in {top_features.iloc[0]["feature"]}',
                'Identify potential over-provisioning opportunities',
                'Evaluate alternative service configurations'
            ]
        }
        recommendations.append(performance_rec)
        
        return recommendations

    def generate_cost_optimization_recommendations(self, service_name: str):
        """
        Generate advanced cost optimization recommendations
        
        :param service_name: Name of cloud service
        :return: Optimization recommendations
        """
        try:
            # Load model and scaler
            model_path = os.path.join(self.output_dir, f'{service_name}_optimization_model.joblib')
            scaler_path = os.path.join(self.output_dir, f'{service_name}_scaler.joblib')
            poly_path = os.path.join(self.output_dir, f'{service_name}_polynomial_transformer.joblib')
            
            # Check if model files exist
            if not all(os.path.exists(path) for path in [model_path, scaler_path, poly_path]):
                logging.warning(f"No pre-trained model found for {service_name}")
                return None
            
            # Load saved artifacts
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            poly = joblib.load(poly_path)
            
            # Load feature importance
            feature_importance_path = os.path.join(self.output_dir, f'{service_name}_feature_importance.csv')
            feature_importance = pd.read_csv(feature_importance_path)
            
            # Recent service data
            service_data = self.data[self.data['ServiceName'] == service_name].copy()
            
            # Ensure sufficient data
            if len(service_data) < 2:
                logging.warning(f"Insufficient data for generating recommendations for {service_name}")
                return None
            
            # Ensure data is sorted and preprocessed
            service_data = self._enhanced_feature_engineering(service_data)
            
            # Select most recent data point
            recent_data = service_data.sort_values('UsageDate').iloc[-1]
            
            # Generate prediction features with robust method
            prediction_features = self._generate_prediction_features(service_data, recent_data)
            
            # Ensure correct feature set
            all_possible_features = [
                'Month', 'DayOfWeek', 'Quarter', 'DayOfYear', 'IsWeekend', 
                'PreviousMonthCost', 'CostMovingAverage_3M', 'CostMovingAverage_6M', 
                'CostChangeRate'
            ]
            
            # Validate and adjust features
            for feature in all_possible_features:
                if feature not in prediction_features.columns:
                    prediction_features[feature] = 0
            
            # Reorder columns to match expected order
            prediction_features = prediction_features[all_possible_features]
            
            # Robust scaling with error handling
            try:
                prediction_features_scaled = scaler.transform(prediction_features)
            except ValueError as scale_error:
                logging.error(f"Scaling error for {service_name}: {scale_error}")
                # Fallback: use mean scaling
                prediction_features_scaled = (prediction_features - prediction_features.mean()) / prediction_features.std()
            
            # Polynomial transformation with error handling
            try:
                prediction_features_poly = poly.transform(prediction_features_scaled)
            except ValueError as poly_error:
                logging.error(f"Polynomial transformation error for {service_name}: {poly_error}")
                # Fallback: use original scaled features
                prediction_features_poly = prediction_features_scaled
            
            # Predict cost with error handling
            try:
                predicted_cost = model.predict(prediction_features_poly)[0]
            except Exception as pred_error:
                logging.error(f"Cost prediction error for {service_name}: {pred_error}")
                # Fallback: use historical average
                predicted_cost = service_data['CostUSD'].mean()
            
            # Top features for recommendations with error handling
            try:
                top_features = feature_importance.head(3)
            except Exception:
                # Fallback if feature importance is unavailable
                top_features = pd.DataFrame({
                    'feature': ['Unknown Feature 1', 'Unknown Feature 2', 'Unknown Feature 3'],
                    'importance': [0.5, 0.3, 0.2]
                })
            
            # Cost Reduction Strategies
            avg_cost = service_data['CostUSD'].mean()
            
            # Generate comprehensive recommendations
            recommendations = self._generate_detailed_recommendations(
                service_name, 
                service_data, 
                top_features, 
                predicted_cost, 
                avg_cost
            )
            
            return recommendations
        
        except Exception as e:
            logging.error(f"Comprehensive error generating recommendations for {service_name}: {e}")
            return None
    
    def generate_comprehensive_optimization_report(self):
        """
        Generate comprehensive optimization report for all services
        
        :return: Detailed optimization insights
        """
        comprehensive_report = {
            'services': {},
            'overall_insights': {
                'total_potential_savings': 0,
                'services_analyzed': 0,
                'services_with_insufficient_data': []
            }
        }
        
        for service in self.data['ServiceName'].unique():
            # Train optimization model
            model_performance = self.train_optimization_model(service)
            
            # Add to comprehensive report
            comprehensive_report['services'][service] = model_performance
            
            # Track services with insufficient data
            if model_performance['status'] in ['insufficient_data', 'limited_data']:
                comprehensive_report['overall_insights']['services_with_insufficient_data'].append(service)
            elif model_performance['status'] == 'success':
                comprehensive_report['overall_insights']['services_analyzed'] += 1
            
            # Generate recommendations only for successful models
            if model_performance['status'] == 'success':
                recommendations = self.generate_cost_optimization_recommendations(service)
                if recommendations:
                    comprehensive_report['services'][service]['recommendations'] = recommendations
        
        # Save report
        report_path = os.path.join(self.output_dir, 'comprehensive_optimization_report.json')
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        logging.info(f"Comprehensive Optimization Report generated: {report_path}")
        return comprehensive_report

def main():
    try:
        # Initialize optimizer
        csv_path = r'C:\Users\kavya\CascadeProjects\cloud-cost-optimizer\augmented_cloud_costs.csv'
        optimizer = AdvancedCostOptimizer(csv_path)
        
        # Log start of report generation
        logger.info("Starting comprehensive optimization report generation")
        
        # Generate comprehensive report
        report = optimizer.generate_comprehensive_optimization_report()
        
        # Prepare report path
        report_path = os.path.join(
            optimizer.output_dir, 
            f'comprehensive_optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log successful report generation
        logger.info(f"Comprehensive Optimization Report generated: {report_path}")
        
        # Print key insights
        print("\n🚀 Comprehensive Optimization Insights 🚀")
        print(f"Total Services Analyzed: {report['overall_insights']['services_analyzed']}")
        print(f"Services with Insufficient Data: {report['overall_insights']['services_with_insufficient_data']}")
    
    except Exception as e:
        # Log any errors during the process
        logger.error(f"Error in optimization process: {e}", exc_info=True)

if __name__ == "__main__":
    main()