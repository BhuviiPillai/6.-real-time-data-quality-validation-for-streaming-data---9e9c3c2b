import os
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.ensemble import IsolationForest
import joblib
from .utils import logger, calculate_statistics

class ValidationRules:
    """Defines data validation rules for real-time streaming data."""
    
    def __init__(self):
        self.anomaly_models = {}
        self.baseline_stats = {}
    
    def load_anomaly_model(self, sensor_id: str, model_path: Optional[str] = None) -> bool:
        """
        Load a pre-trained anomaly detection model for a specific sensor.
        If no model exists, create a new one.
        """
        if model_path and os.path.exists(model_path):
            try:
                self.anomaly_models[sensor_id] = joblib.load(model_path)
                logger.info(f"Loaded anomaly model for sensor {sensor_id} from {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error loading model for {sensor_id}: {e}")
        
        # Create a new model if none exists
        self.anomaly_models[sensor_id] = IsolationForest(
            n_estimators=100, 
            contamination=0.05,  # Assume 5% of data points are anomalies
            random_state=42
        )
        logger.info(f"Created new anomaly model for sensor {sensor_id}")
        return False
    
    def save_anomaly_model(self, sensor_id: str, model_path: str) -> bool:
        """Save the trained anomaly detection model."""
        if sensor_id in self.anomaly_models:
            try:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(self.anomaly_models[sensor_id], model_path)
                logger.info(f"Saved anomaly model for sensor {sensor_id} to {model_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving model for {sensor_id}: {e}")
        return False
    
    def train_anomaly_model(self, sensor_id: str, values: List[float]) -> None:
        """Train the anomaly detection model with historical data."""
        if sensor_id not in self.anomaly_models:
            self.load_anomaly_model(sensor_id)
        
        # Filter out None values
        valid_values = np.array([v for v in values if v is not None]).reshape(-1, 1)
        if len(valid_values) > 10:  # Need enough data points to train
            self.anomaly_models[sensor_id].fit(valid_values)
            logger.info(f"Trained anomaly model for sensor {sensor_id} with {len(valid_values)} data points")
    
    def update_baseline_stats(self, sensor_id: str, values: List[float]) -> None:
        """Update baseline statistics for a sensor."""
        self.baseline_stats[sensor_id] = calculate_statistics(values)
        logger.info(f"Updated baseline stats for sensor {sensor_id}: {self.baseline_stats[sensor_id]}")
    
    def check_completeness(self, data_point: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if all required fields are present and non-null."""
        required_fields = ['timestamp', 'sensor_id', 'value', 'unit']
        
        for field in required_fields:
            if field not in data_point:
                return False, f"Missing required field: {field}"
            
        if data_point['value'] is None:
            return False, "Value is null (missing data)"
            
        return True, "Data is complete"
    
    def check_range(self, 
                   data_point: Dict[str, Any], 
                   min_value: Optional[float] = None, 
                   max_value: Optional[float] = None) -> Tuple[bool, str]:
        """Check if the value is within an acceptable range."""
        value = data_point.get('value')
        
        if value is None:
            return False, "Cannot check range: value is null"
        
        sensor_id = data_point.get('sensor_id')
        if sensor_id in self.baseline_stats and not (min_value and max_value):
            stats = self.baseline_stats[sensor_id]
            if stats['mean'] is not None and stats['std'] is not None:
                # Use mean ± 3 standard deviations as range if not provided
                effective_min = min_value if min_value is not None else stats['mean'] - 3 * stats['std']
                effective_max = max_value if max_value is not None else stats['mean'] + 3 * stats['std']
            else:
                # Fallback to min/max if mean/std aren't available
                effective_min = min_value if min_value is not None else stats.get('min')
                effective_max = max_value if max_value is not None else stats.get('max')
        else:
            effective_min = min_value
            effective_max = max_value
        
        # Perform range check if we have limits
        if effective_min is not None and value < effective_min:
            return False, f"Value {value} is below minimum {effective_min}"
        
        if effective_max is not None and value > effective_max:
            return False, f"Value {value} exceeds maximum {effective_max}"
        
        return True, "Value is within acceptable range"
    
    def detect_anomaly(self, data_point: Dict[str, Any]) -> Tuple[bool, str]:
        """Detect anomalies using the trained model."""
        value = data_point.get('value')
        sensor_id = data_point.get('sensor_id')
        
        if value is None or sensor_id is None:
            return False, "Cannot detect anomaly: missing value or sensor_id"
        
        if sensor_id not in self.anomaly_models:
            return True, "No anomaly model available for this sensor"
        
        try:
            # Reshape for sklearn
            prediction = self.anomaly_models[sensor_id].predict(np.array([[value]]))
            if prediction[0] == -1:  # -1 indicates anomaly in IsolationForest
                return False, f"Anomaly detected: value {value} is unusual for sensor {sensor_id}"
            return True, "No anomaly detected"
        except Exception as e:
            logger.error(f"Error detecting anomaly: {e}")
            return True, f"Error in anomaly detection: {e}"
    
    def apply_validation_rules(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all validation rules to a data point and return results."""
        results = {
            "data_point": data_point,
            "validations": {},
            "is_valid": True
        }
        
        # Check completeness
        complete, msg = self.check_completeness(data_point)
        results["validations"]["completeness"] = {"passed": complete, "message": msg}
        if not complete:
            results["is_valid"] = False
        
        # Only proceed with other checks if we have a value
        if data_point.get('value') is not None:
            # Check range
            in_range, range_msg = self.check_range(data_point)
            results["validations"]["range"] = {"passed": in_range, "message": range_msg}
            if not in_range:
                results["is_valid"] = False
            
            # Check for anomalies
            normal, anomaly_msg = self.detect_anomaly(data_point)
            results["validations"]["anomaly"] = {"passed": normal, "message": anomaly_msg}
            if not normal:
                results["is_valid"] = False
        
        return results 