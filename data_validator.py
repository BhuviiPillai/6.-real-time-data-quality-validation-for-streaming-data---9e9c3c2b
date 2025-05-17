import os
import json
import time
import threading
import socket
import argparse
import signal
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from .validation_rules import ValidationRules
from .utils import logger, load_json_data, get_sensor_groups

class StreamingDataValidator:
    """Real-time streaming data validator using AI-based anomaly detection."""
    
    def __init__(self, model_dir: str = "../models"):
        self.model_dir = model_dir
        self.validation_rules = ValidationRules()
        self.data_buffer = []
        self.running = False
        self.socket = None
        self.client_sockets = []
        self.threads = []
        self.callback = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def load_historical_data(self, data_file: str) -> bool:
        """Load historical data and train models with it."""
        try:
            data = load_json_data(data_file)
            if not data:
                logger.warning(f"No data found in {data_file}")
                return False
                
            logger.info(f"Loaded {len(data)} historical data points")
            
            # Group data by sensor
            sensor_groups = get_sensor_groups(data)
            
            # Train models for each sensor
            for sensor_id, sensor_data in sensor_groups.items():
                values = [item.get('value') for item in sensor_data 
                          if item.get('value') is not None]
                
                if len(values) > 10:  # Minimum data points needed
                    # Update baseline statistics
                    self.validation_rules.update_baseline_stats(sensor_id, values)
                    
                    # Train anomaly detection model
                    self.validation_rules.train_anomaly_model(sensor_id, values)
                    
                    # Save model
                    model_path = os.path.join(self.model_dir, f"{sensor_id}_model.pkl")
                    self.validation_rules.save_anomaly_model(sensor_id, model_path)
                    
                    logger.info(f"Trained and saved model for sensor {sensor_id}")
                else:
                    logger.warning(f"Not enough data points for sensor {sensor_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return False
    
    def validate_data_point(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single data point."""
        try:
            return self.validation_rules.apply_validation_rules(data_point)
        except Exception as e:
            logger.error(f"Error validating data point: {e}")
            return {
                "data_point": data_point,
                "validations": {"error": {"passed": False, "message": str(e)}},
                "is_valid": False
            }
    
    def connect_to_stream(self, host: str = 'localhost', port: int = 5555):
        """Connect to a streaming data source via socket."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((host, port))
            logger.info(f"Connected to data stream at {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to data stream: {e}")
            return False
    
    def setup_server(self, port: int = 5556):
        """Set up a server to broadcast validation results."""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', port))
            server_socket.listen(5)
            logger.info(f"Server started for validation results on port {port}")
            
            # Thread to accept client connections
            def accept_clients():
                while self.running:
                    try:
                        client, addr = server_socket.accept()
                        logger.info(f"Client connected to validation stream: {addr}")
                        self.client_sockets.append(client)
                    except Exception as e:
                        if self.running:  # Only log if still running
                            logger.error(f"Error accepting validation client: {e}")
                        break
            
            accept_thread = threading.Thread(target=accept_clients, daemon=True)
            accept_thread.start()
            self.threads.append(accept_thread)
            
            return server_socket
        except Exception as e:
            logger.error(f"Error setting up server: {e}")
            return None
    
    def broadcast_result(self, result: Dict[str, Any]):
        """Broadcast validation result to connected clients."""
        message = json.dumps(result) + "\n"
        disconnected = []
        
        for i, client in enumerate(self.client_sockets):
            try:
                client.sendall(message.encode())
            except Exception as e:
                logger.error(f"Error sending to validation client: {e}")
                disconnected.append(i)
        
        # Remove disconnected clients
        for idx in sorted(disconnected, reverse=True):
            try:
                self.client_sockets[idx].close()
            except:
                pass
            del self.client_sockets[idx]
    
    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set a callback function to be called with each validation result."""
        self.callback = callback
    
    def start_validation(self, host: str = 'localhost', port: int = 5555, 
                         result_port: int = 5556, buffer_size: int = 1024):
        """Start real-time validation of streaming data."""
        if not self.connect_to_stream(host, port):
            return False
        
        server_socket = self.setup_server(result_port)
        self.running = True
        
        # Thread to process incoming data
        def process_stream():
            buffer = ""
            while self.running:
                try:
                    data = self.socket.recv(buffer_size).decode()
                    if not data:  # Connection closed
                        logger.warning("Stream connection closed")
                        break
                        
                    buffer += data
                    
                    # Process complete JSON objects
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        try:
                            data_point = json.loads(line)
                            
                            # Validate data point
                            result = self.validate_data_point(data_point)
                            
                            # Broadcast result
                            self.broadcast_result(result)
                            
                            # Store result
                            self.data_buffer.append(result)
                            
                            # Call callback if set
                            if self.callback:
                                self.callback(result)
                            
                            # Log validation result
                            sensor_id = data_point.get('sensor_id', 'unknown')
                            status = "VALID" if result["is_valid"] else "INVALID"
                            logger.info(f"Validated data from {sensor_id}: {status}")
                            
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON: {line}")
                except Exception as e:
                    if self.running:  # Only log if still running
                        logger.error(f"Error processing stream: {e}")
                        time.sleep(1)  # Avoid tight loop on error
                    break
        
        proc_thread = threading.Thread(target=process_stream, daemon=True)
        proc_thread.start()
        self.threads.append(proc_thread)
        
        logger.info("Started real-time data validation")
        return True
    
    def stop_validation(self):
        """Stop the validation process."""
        self.running = False
        
        # Close socket connection
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        # Close all client connections
        for client in self.client_sockets:
            try:
                client.close()
            except:
                pass
        
        logger.info("Stopped real-time data validation")
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate a summary report of validation results."""
        if not self.data_buffer:
            logger.warning("No data to generate report")
            return {}
        
        report = {
            "total_data_points": len(self.data_buffer),
            "valid_data_points": sum(1 for r in self.data_buffer if r.get("is_valid", False)),
            "timestamp": datetime.now().isoformat(),
            "sensor_stats": {},
            "validation_stats": {
                "completeness": {"passed": 0, "failed": 0},
                "range": {"passed": 0, "failed": 0},
                "anomaly": {"passed": 0, "failed": 0}
            }
        }
        
        # Group by sensor
        sensor_results = {}
        for result in self.data_buffer:
            data_point = result.get("data_point", {})
            sensor_id = data_point.get("sensor_id")
            if not sensor_id:
                continue
                
            if sensor_id not in sensor_results:
                sensor_results[sensor_id] = []
            sensor_results[sensor_id].append(result)
        
        # Calculate per-sensor statistics
        for sensor_id, results in sensor_results.items():
            valid_count = sum(1 for r in results if r.get("is_valid", False))
            
            sensor_report = {
                "total": len(results),
                "valid": valid_count,
                "valid_percentage": round(valid_count / len(results) * 100, 2) if results else 0
            }
            
            # Collect validation stats
            for validation_type in ["completeness", "range", "anomaly"]:
                passed = sum(1 for r in results if r.get("validations", {}).get(validation_type, {}).get("passed", False))
                failed = len(results) - passed
                
                sensor_report[validation_type] = {
                    "passed": passed,
                    "failed": failed,
                    "pass_percentage": round(passed / len(results) * 100, 2) if results else 0
                }
                
                # Update overall stats
                report["validation_stats"][validation_type]["passed"] += passed
                report["validation_stats"][validation_type]["failed"] += failed
            
            report["sensor_stats"][sensor_id] = sensor_report
        
        # Calculate overall percentages
        total = report["total_data_points"]
        for validation_type in report["validation_stats"]:
            stats = report["validation_stats"][validation_type]
            if total > 0:
                stats["pass_percentage"] = round(stats["passed"] / total * 100, 2)
            else:
                stats["pass_percentage"] = 0
        
        # Save report to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Saved validation report to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report
    
    def visualize_results(self, 
                         show_plot: bool = True, 
                         save_path: Optional[str] = None,
                         time_window: Optional[int] = None):
        """Visualize validation results."""
        if not self.data_buffer:
            logger.warning("No data to visualize")
            return
        
        # Convert to DataFrame for easier manipulation
        records = []
        for result in self.data_buffer:
            data_point = result.get("data_point", {})
            is_valid = result.get("is_valid", False)
            timestamp = data_point.get("timestamp")
            sensor_id = data_point.get("sensor_id")
            value = data_point.get("value")
            
            if None in (timestamp, sensor_id):
                continue
                
            record = {
                "timestamp": timestamp,
                "sensor_id": sensor_id,
                "value": value,
                "is_valid": is_valid
            }
            
            # Add specific validation results
            for validation_type, validation_result in result.get("validations", {}).items():
                record[f"{validation_type}_valid"] = validation_result.get("passed", False)
            
            records.append(record)
        
        if not records:
            logger.warning("No valid records to visualize")
            return
            
        df = pd.DataFrame(records)
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Filter by time window if specified
        if time_window:
            cutoff = datetime.now() - timedelta(minutes=time_window)
            df = df[df["timestamp"] >= pd.Timestamp(cutoff)]
        
        # Group by sensor
        sensor_groups = df.groupby("sensor_id")
        
        # Create plots
        fig, axs = plt.subplots(len(sensor_groups), 1, figsize=(12, 4 * len(sensor_groups)))
        if len(sensor_groups) == 1:
            axs = [axs]
        
        for i, (sensor_id, group) in enumerate(sensor_groups):
            ax = axs[i]
            
            # Plot values
            valid_data = group[group["is_valid"]]
            invalid_data = group[~group["is_valid"]]
            
            if not valid_data.empty:
                ax.plot(valid_data["timestamp"], valid_data["value"], 'o-', color='green', label='Valid')
            if not invalid_data.empty:
                ax.plot(invalid_data["timestamp"], invalid_data["value"], 'x', color='red', label='Invalid')
            
            ax.set_title(f"Sensor: {sensor_id}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        if show_plot:
            plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time data validation')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host of the data stream')
    parser.add_argument('--port', type=int, default=5555,
                        help='Port of the data stream')
    parser.add_argument('--result-port', type=int, default=5556,
                        help='Port for validation results')
    parser.add_argument('--historical-data', type=str, required=False,
                        help='Path to historical data for model training')
    parser.add_argument('--model-dir', type=str, default='../models',
                        help='Directory to store trained models')
    parser.add_argument('--report-file', type=str, default=None,
                        help='Path to save the validation report')
    parser.add_argument('--timeout', type=int, default=0,
                        help='Automatically stop after this many seconds (0 = run indefinitely)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization when stopping')
    args = parser.parse_args()
    
    # Print clear startup message
    print(f"\n=== STARTING DATA VALIDATOR ===")
    print(f"- Host: {args.host}")
    print(f"- Port: {args.port}")
    print(f"- Result port: {args.result_port}")
    print(f"- Historical data: {args.historical_data if args.historical_data else 'None'}")
    print(f"- Model directory: {args.model_dir}")
    print(f"- Report file: {args.report_file if args.report_file else 'None'}")
    print(f"- Timeout: {args.timeout if args.timeout > 0 else 'None (running until stopped)'}")
    print(f"- Visualize: {'Yes' if args.visualize else 'No'}")
    
    # Create validator instance
    validator = StreamingDataValidator(model_dir=args.model_dir)
    
    # Load historical data if provided
    if args.historical_data:
        print(f"Loading historical data from {args.historical_data}...")
        success = validator.load_historical_data(args.historical_data)
        if success:
            print(f"Successfully loaded historical data and trained models")
        else:
            print(f"WARNING: Problems loading historical data")
    
    # Helper function to generate report and cleanup
    def generate_report_and_cleanup():
        validator.stop_validation()
        
        # Generate and save report if needed
        if args.report_file and validator.data_buffer:
            print(f"Generating validation report...")
            report = validator.generate_report()
            with open(args.report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Validation report saved to {args.report_file}")
            
            # Generate visualization if requested
            if args.visualize:
                print(f"Generating visualization...")
                viz_path = args.report_file.replace('.json', '.png') if args.report_file.endswith('.json') else 'validation_results.png'
                validator.visualize_results(show_plot=False, save_path=viz_path)
                print(f"Visualization saved to {viz_path}")
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nStopping validator due to signal...")
        generate_report_and_cleanup()
        print("Validator stopped")
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start validation
    print(f"Connecting to data stream at {args.host}:{args.port}...")
    if not validator.start_validation(host=args.host, port=args.port, result_port=args.result_port):
        print("Failed to connect to data stream. Make sure the data generator is running.")
        return
    
    print(f"Successfully connected to data stream")
    
    # Add timer for automatic shutdown if timeout is specified
    if args.timeout > 0:
        print(f"Validator will automatically stop after {args.timeout} seconds")
        
        def timeout_handler():
            print(f"\nTimeout of {args.timeout} seconds reached. Stopping validator...")
            generate_report_and_cleanup()
            print("Validator stopped due to timeout")
            exit(0)
            
        timer = threading.Timer(args.timeout, timeout_handler)
        timer.daemon = True
        timer.start()
    
    # Main loop to keep the program running
    try:
        print("Validator running. Press Ctrl+C to stop and generate report.")
        start_time = time.time()
        while validator.running:
            # Log how many data points we've processed periodically
            elapsed = time.time() - start_time
            if elapsed % 10 < 0.1 and elapsed > 1:  # Log every ~10 seconds
                if args.timeout > 0:
                    remaining = args.timeout - int(elapsed)
                    if remaining > 0:
                        print(f"Validator running... {len(validator.data_buffer)} data points processed. {remaining} seconds remaining until timeout")
                else:
                    print(f"Validator running... {len(validator.data_buffer)} data points processed")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping validator due to keyboard interrupt...")
    finally:
        generate_report_and_cleanup()
        print("Validator stopped")

if __name__ == "__main__":
    main() 