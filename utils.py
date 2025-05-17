import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_quality_validator')

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return []

def save_json_data(data: List[Dict[str, Any]], file_path: str) -> bool:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        return False

def timestamp_to_datetime(timestamp: str) -> datetime:
    """Convert ISO timestamp string to datetime object."""
    try:
        return datetime.fromisoformat(timestamp)
    except ValueError:
        # Remove Z if present for UTC time
        if timestamp.endswith('Z'):
            return datetime.fromisoformat(timestamp[:-1])
        raise

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    values_array = np.array([v for v in values if v is not None])
    if len(values_array) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    
    return {
        "count": len(values_array),
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array))
    }

def get_sensor_groups(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group data points by sensor ID."""
    groups = {}
    for item in data:
        sensor_id = item.get("sensor_id")
        if sensor_id:
            if sensor_id not in groups:
                groups[sensor_id] = []
            groups[sensor_id].append(item)
    return groups 