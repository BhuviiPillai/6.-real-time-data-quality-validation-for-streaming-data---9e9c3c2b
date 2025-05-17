import unittest
import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.validation_rules import ValidationRules
from src.data_validator import StreamingDataValidator
from src.utils import load_json_data

class TestValidationRules(unittest.TestCase):
    """Tests for the ValidationRules class."""
    
    def setUp(self):
        self.validation_rules = ValidationRules()
        
        # Sample data points
        self.normal_data = {
            "timestamp": "2023-07-01T08:00:00", 
            "sensor_id": "temp_001", 
            "value": 22.5, 
            "unit": "celsius", 
            "status": "normal"
        }
        
        self.anomalous_data = {
            "timestamp": "2023-07-01T08:00:00", 
            "sensor_id": "temp_001", 
            "value": 45.0,  # Anomalous value
            "unit": "celsius", 
            "status": "anomaly"
        }
        
        self.missing_data = {
            "timestamp": "2023-07-01T08:00:00", 
            "sensor_id": "temp_001", 
            "value": None,  # Missing value
            "unit": "celsius", 
            "status": "missing"
        }
        
        self.incomplete_data = {
            "timestamp": "2023-07-01T08:00:00", 
            "sensor_id": "temp_001"
            # Missing value and unit
        }
    
    def test_check_completeness(self):
        """Test the completeness check."""
        # Test complete data
        complete, _ = self.validation_rules.check_completeness(self.normal_data)
        self.assertTrue(complete)
        
        # Test missing value
        complete, _ = self.validation_rules.check_completeness(self.missing_data)
        self.assertFalse(complete)
        
        # Test incomplete data
        complete, _ = self.validation_rules.check_completeness(self.incomplete_data)
        self.assertFalse(complete)
    
    def test_check_range(self):
        """Test the range check."""
        # Test in-range data
        in_range, _ = self.validation_rules.check_range(self.normal_data, 20.0, 25.0)
        self.assertTrue(in_range)
        
        # Test out-of-range data
        in_range, _ = self.validation_rules.check_range(self.anomalous_data, 20.0, 25.0)
        self.assertFalse(in_range)
        
        # Test with missing value
        in_range, _ = self.validation_rules.check_range(self.missing_data)
        self.assertFalse(in_range)
    
    def test_apply_validation_rules(self):
        """Test applying all validation rules."""
        # Update baseline stats for better testing
        self.validation_rules.update_baseline_stats("temp_001", [22.0, 22.5, 23.0, 22.8])
        
        # Test normal data
        result = self.validation_rules.apply_validation_rules(self.normal_data)
        self.assertTrue(result["is_valid"])
        
        # Test anomalous data
        result = self.validation_rules.apply_validation_rules(self.anomalous_data)
        self.assertFalse(result["is_valid"])
        
        # Test missing data
        result = self.validation_rules.apply_validation_rules(self.missing_data)
        self.assertFalse(result["is_valid"])
        
        # Test incomplete data
        result = self.validation_rules.apply_validation_rules(self.incomplete_data)
        self.assertFalse(result["is_valid"])

class TestStreamingDataValidator(unittest.TestCase):
    """Tests for the StreamingDataValidator class."""
    
    def setUp(self):
        # Create a temporary directory for models
        self.temp_dir = tempfile.mkdtemp()
        self.validator = StreamingDataValidator(model_dir=self.temp_dir)
        
        # Sample data for testing
        self.sample_data = [
            {
                "timestamp": "2023-07-01T08:00:00", 
                "sensor_id": "temp_001", 
                "value": 22.5, 
                "unit": "celsius", 
                "status": "normal"
            },
            {
                "timestamp": "2023-07-01T08:00:10", 
                "sensor_id": "temp_001", 
                "value": 22.7, 
                "unit": "celsius", 
                "status": "normal"
            },
            {
                "timestamp": "2023-07-01T08:00:20", 
                "sensor_id": "temp_001", 
                "value": 35.2, 
                "unit": "celsius", 
                "status": "anomaly"
            }
        ]
        
        # Create a temporary file with sample data
        self.temp_file = os.path.join(self.temp_dir, "sample_data.json")
        with open(self.temp_file, "w") as f:
            json.dump(self.sample_data, f)
    
    def tearDown(self):
        # Clean up temporary files
        try:
            os.remove(self.temp_file)
        except:
            pass
            
        try:
            os.rmdir(self.temp_dir)
        except:
            pass
    
    def test_load_historical_data(self):
        """Test loading historical data and training models."""
        # Test with valid data file
        success = self.validator.load_historical_data(self.temp_file)
        self.assertTrue(success)
        
        # Check that models were created
        self.assertIn("temp_001", self.validator.validation_rules.baseline_stats)
        self.assertIn("temp_001", self.validator.validation_rules.anomaly_models)
        
        # Test with non-existent file
        success = self.validator.load_historical_data("non_existent_file.json")
        self.assertFalse(success)
    
    def test_validate_data_point(self):
        """Test validating individual data points."""
        # First load historical data for baseline stats
        self.validator.load_historical_data(self.temp_file)
        
        # Test normal data point
        result = self.validator.validate_data_point(self.sample_data[0])
        self.assertTrue(result["is_valid"])
        
        # Test anomalous data point
        result = self.validator.validate_data_point(self.sample_data[2])
        self.assertFalse(result["is_valid"])
        
        # Test with incomplete data
        result = self.validator.validate_data_point({"sensor_id": "temp_001"})
        self.assertFalse(result["is_valid"])
    
    @patch('socket.socket')
    def test_connect_to_stream(self, mock_socket):
        """Test connecting to a data stream."""
        # Mock socket connection
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance
        
        # Test successful connection
        success = self.validator.connect_to_stream()
        self.assertTrue(success)
        mock_socket_instance.connect.assert_called_once()
        
        # Test connection failure
        mock_socket_instance.connect.side_effect = Exception("Connection failed")
        success = self.validator.connect_to_stream()
        self.assertFalse(success)
    
    def test_generate_report(self):
        """Test generating validation report."""
        # Add some fake validation results to the buffer
        self.validator.data_buffer = [
            {
                "data_point": {"sensor_id": "temp_001", "value": 22.5},
                "is_valid": True,
                "validations": {
                    "completeness": {"passed": True},
                    "range": {"passed": True},
                    "anomaly": {"passed": True}
                }
            },
            {
                "data_point": {"sensor_id": "temp_001", "value": 45.0},
                "is_valid": False,
                "validations": {
                    "completeness": {"passed": True},
                    "range": {"passed": False},
                    "anomaly": {"passed": False}
                }
            }
        ]
        
        # Generate report
        report = self.validator.generate_report()
        
        # Check report content
        self.assertEqual(report["total_data_points"], 2)
        self.assertEqual(report["valid_data_points"], 1)
        self.assertIn("sensor_stats", report)
        self.assertIn("temp_001", report["sensor_stats"])
        self.assertEqual(report["sensor_stats"]["temp_001"]["valid"], 1)
        self.assertEqual(report["sensor_stats"]["temp_001"]["total"], 2)

if __name__ == "__main__":
    unittest.main() 