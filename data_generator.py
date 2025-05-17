import json
import time
import random
import argparse
import socket
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from .utils import logger, save_json_data
import os
import signal

class StreamingDataGenerator:
    """Generates simulated streaming data from IoT sensors."""
    
    def __init__(self, output_file: Optional[str] = None, port: int = 5555):
        self.output_file = output_file
        self.port = port
        self.running = False
        self.thread = None
        self.socket = None
        self.clients = []
        
        # Define sensor configurations
        self.sensors = {
            "temp_001": {
                "baseline": 22.5,
                "noise_level": 0.5,
                "unit": "celsius",
                "anomaly_prob": 0.05
            },
            "pressure_001": {
                "baseline": 1013.0,
                "noise_level": 0.5,
                "unit": "hPa",
                "anomaly_prob": 0.05
            },
            "humidity_001": {
                "baseline": 45.0,
                "noise_level": 1.0,
                "unit": "%",
                "anomaly_prob": 0.05,
                "missing_prob": 0.03  # Probability of missing data
            }
        }
        
        self.data_buffer = []
    
    def generate_data_point(self, sensor_id: str) -> Dict[str, Any]:
        """Generate a single data point for a given sensor."""
        config = self.sensors.get(sensor_id, {})
        baseline = config.get("baseline", 0)
        noise_level = config.get("noise_level", 1.0)
        unit = config.get("unit", "")
        anomaly_prob = config.get("anomaly_prob", 0.05)
        missing_prob = config.get("missing_prob", 0.0)
        
        timestamp = datetime.now().isoformat()
        
        # Check for missing data
        if random.random() < missing_prob:
            return {
                "timestamp": timestamp,
                "sensor_id": sensor_id,
                "value": None,
                "unit": unit,
                "status": "missing"
            }
        
        # Determine if this will be an anomaly
        is_anomaly = random.random() < anomaly_prob
        
        if is_anomaly:
            # Generate an anomalous value (far from baseline)
            value = baseline + (random.choice([-1, 1]) * 
                               (noise_level * 10 + random.uniform(5, 15)))
            status = "anomaly"
        else:
            # Generate normal value with noise
            value = baseline + random.uniform(-noise_level, noise_level)
            status = "normal"
        
        return {
            "timestamp": timestamp,
            "sensor_id": sensor_id,
            "value": value,
            "unit": unit,
            "status": status
        }
    
    def generate_batch(self) -> List[Dict[str, Any]]:
        """Generate a batch of data points for all sensors."""
        batch = []
        for sensor_id in self.sensors:
            batch.append(self.generate_data_point(sensor_id))
        return batch
    
    def setup_socket_server(self):
        """Set up a socket server to send data to connected clients."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('localhost', self.port))
        self.socket.listen(5)
        logger.info(f"Socket server started on port {self.port}")
        
        # Thread to accept client connections
        def accept_clients():
            while self.running:
                try:
                    client, addr = self.socket.accept()
                    logger.info(f"Client connected: {addr}")
                    self.clients.append(client)
                except Exception as e:
                    if self.running:  # Only log if still running
                        logger.error(f"Error accepting client: {e}")
                    break
        
        threading.Thread(target=accept_clients, daemon=True).start()
    
    def send_to_clients(self, data_point: Dict[str, Any]):
        """Send data to all connected clients."""
        message = json.dumps(data_point) + "\n"
        disconnected = []
        
        for i, client in enumerate(self.clients):
            try:
                client.sendall(message.encode())
            except Exception as e:
                logger.error(f"Error sending to client: {e}")
                disconnected.append(i)
        
        # Remove disconnected clients
        for idx in sorted(disconnected, reverse=True):
            try:
                self.clients[idx].close()
            except:
                pass
            del self.clients[idx]
    
    def start(self, interval: float = 1.0):
        """Start generating streaming data at the specified interval (seconds)."""
        self.running = True
        
        # Setup socket server if needed
        self.setup_socket_server()
        
        # Run in a separate thread
        def run_generator():
            while self.running:
                try:
                    data_batch = self.generate_batch()
                    
                    # Send each data point to connected clients
                    for data_point in data_batch:
                        self.send_to_clients(data_point)
                        self.data_buffer.append(data_point)
                    
                    # Save to file if specified
                    if self.output_file and len(self.data_buffer) >= 100:
                        save_json_data(self.data_buffer, self.output_file)
                        logger.info(f"Saved {len(self.data_buffer)} data points to {self.output_file}")
                        self.data_buffer = []
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in data generation: {e}")
                    if not self.running:
                        break
                    time.sleep(1)  # Wait before retrying
        
        self.thread = threading.Thread(target=run_generator, daemon=True)
        self.thread.start()
        logger.info(f"Data generator started with interval {interval}s")
    
    def stop(self):
        """Stop the data generator."""
        self.running = False
        
        # Save any remaining data
        if self.output_file and self.data_buffer:
            save_json_data(self.data_buffer, self.output_file)
            logger.info(f"Saved final {len(self.data_buffer)} data points to {self.output_file}")
        
        # Close all client connections
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        
        # Close server socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        logger.info("Data generator stopped")

def parse_args():
    parser = argparse.ArgumentParser(description='Generate streaming IoT sensor data')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Time interval between data points in seconds')
    parser.add_argument('--output', type=str, default=None,
                        help='File to save generated data')
    parser.add_argument('--port', type=int, default=5555,
                        help='Port for the socket server')
    parser.add_argument('--timeout', type=int, default=0,
                        help='Automatically stop after this many seconds (0 = run indefinitely)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Print clear startup message with timeout info
    print(f"\n=== STARTING DATA GENERATOR ===")
    print(f"- Interval: {args.interval} seconds")
    print(f"- Output file: {args.output if args.output else 'None'}")
    print(f"- Server port: {args.port}")
    print(f"- Timeout: {args.timeout if args.timeout > 0 else 'None (running until stopped)'}")
    
    generator = StreamingDataGenerator(args.output, args.port)
    
    # Start the generator first
    try:
        generator.start(args.interval)
        print(f"Data generator running with interval {args.interval}s.")
    except Exception as e:
        print(f"ERROR starting generator: {e}")
        return
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        print("\nStopping data generator due to signal...")
        generator.stop()
        print("Data generator stopped")
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Add timer for automatic shutdown if timeout is specified
    if args.timeout > 0:
        print(f"Data generator will automatically stop after {args.timeout} seconds")
        
        def timeout_handler():
            print(f"\nTimeout of {args.timeout} seconds reached. Stopping data generator...")
            generator.stop()
            print("Data generator stopped due to timeout")
            exit(0)
            
        timer = threading.Timer(args.timeout, timeout_handler)
        timer.daemon = True
        timer.start()
    
    # Main loop to keep the program running
    try:
        start_time = time.time()
        while generator.running:
            elapsed = time.time() - start_time
            if args.timeout > 0 and elapsed % 10 < 0.1 and elapsed > 1:  # Log every ~10 seconds
                remaining = args.timeout - int(elapsed)
                if remaining > 0:
                    print(f"Generator running... {remaining} seconds remaining until timeout")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping data generator due to keyboard interrupt...")
    finally:
        generator.stop()
        print("Data generator stopped")

if __name__ == "__main__":
    main() 