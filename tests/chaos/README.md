# Chaos Testing Framework

This directory contains tools and scripts for running chaos tests on the AI Trading Agent system. These tests deliberately introduce failures into the system to verify that it can recover properly.

## Requirements

- Kubernetes cluster with the AI Trading Agent deployed
- kubectl configured to access the cluster
- Python 3.9+ with the following packages:
  - kubernetes
  - pytest
  - chaostoolkit
  - chaostoolkit-kubernetes

## Installation

```bash
pip install -r requirements.txt
```

## Running Tests

To run all chaos tests:

```bash
python -m pytest -v
```

To run a specific chaos test:

```bash
python -m pytest -v test_network_partition.py
```

## Available Chaos Tests

1. **Pod Termination**: Randomly terminates pods to test recovery
2. **Network Delay**: Introduces network latency between components
3. **Network Partition**: Simulates network partitions between services
4. **Resource Exhaustion**: Deliberately exhausts CPU/memory to test throttling
5. **Database Failures**: Simulates database outages
6. **Redis Failures**: Simulates Redis outages
7. **Component Restarts**: Forces component restarts during operation

## Adding New Chaos Tests

1. Create a new Python file with your test
2. Subclass `BaseChaosTest` from `base.py`
3. Implement the required methods:
   - `setup()`: Set up the chaos experiment
   - `execute()`: Execute the chaos operation
   - `verify()`: Verify the system recovers properly
   - `cleanup()`: Clean up after the test
