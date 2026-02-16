#!/bin/bash

echo "Starting NEPSE Analysis App..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed"
    echo "Please install Python3 from https://python.org"
    exit 1
fi

# Run the setup script
python3 setup.py
