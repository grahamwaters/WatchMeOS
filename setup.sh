#!/bin/bash
# setup.sh – Create a virtual environment and install dependencies

# Create a virtual environment named "venv"
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install opencv-python mediapipe flask scikit-learn numpy

echo "Setup complete."
echo "To run the application, execute:"
echo "    source venv/bin/activate"
echo "    python main.py"
