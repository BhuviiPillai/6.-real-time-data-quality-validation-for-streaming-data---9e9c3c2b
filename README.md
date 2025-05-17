# Real-Time Data Quality Validation for Streaming Data

This project implements a real-time data validation system for streaming data from IoT sensors or other sources. It uses AI-based anomaly detection and statistical methods to validate incoming data and ensure its quality.

## Features

- **Real-time data validation** for streams of sensor data
- **Multiple validation methods**:
  - Completeness validation (ensure all required fields are present)
  - Range validation (ensure values are within acceptable ranges)
  - Anomaly detection using Isolation Forest algorithm
- **Automatic baseline statistics** calculation from historical data
- **Streaming architecture** with socket-based communication
- **Visualization tools** for monitoring data quality
- **Detailed reporting** of validation results
- **Sample data generator** for testing and demonstration

## Project Structure

```
realtime_data_quality/
├── data/
│   └── sample_data.json        # Example data for testing
├── models/                     # Directory for saved models
├── src/
│   ├── data_generator.py       # Script to generate sample streaming data
│   ├── data_validator.py       # Main script for real-time data validation
│   ├── validation_rules.py     # Defines data validation rules and AI model loading
│   ├── utils.py                # Utility functions
│   └── __init__.py
├── tests/
│   └── test_data_validator.py  # Unit tests
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd realtime_data_quality
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Data Generator

To simulate streaming data from IoT sensors:

```bash
python -m src.data_generator --interval 0.5 --output data/generated_data.json
```

Options:
- `--interval`: Time interval between data points in seconds (default: 1.0)
- `--output`: File to save generated data (optional)
- `--port`: Port for the socket server (default: 5555)

### Running the Data Validator

To validate incoming streaming data:

```bash
python -m src.data_validator --historical-data data/sample_data.json --report-file validation_report.json
```

Options:
- `--host`: Host of the data stream (default: localhost)
- `--port`: Port of the data stream (default: 5555)
- `--result-port`: Port for validation results (default: 5556)
- `--historical-data`: Path to historical data for model training
- `--model-dir`: Directory to store trained models (default: ../models)
- `--report-file`: Path to save the validation report

### Example Workflow

1. Start the data generator to simulate IoT sensors:
```bash
python -m src.data_generator
```

2. In another terminal, start the validator:
```bash
python -m src.data_validator --historical-data data/sample_data.json
```

3. The validator will connect to the generator, validate data in real-time, and output results.

## Implementation Details

### Data Generator
Simulates data from IoT sensors with configurable noise levels and anomaly probabilities. It serves the data via a socket server to enable real-time streaming.

### Validation Rules
- **Completeness Check**: Ensures all required fields are present and non-null
- **Range Check**: Verifies values are within acceptable ranges based on historical data
- **Anomaly Detection**: Uses Isolation Forest to detect unusual values based on learned patterns

### Streaming Architecture
- Socket-based communication allows real-time processing of data
- Data generator acts as a data source/producer
- Validator connects as a client and processes incoming data
- Results can be broadcast to multiple clients for visualization or further processing

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 