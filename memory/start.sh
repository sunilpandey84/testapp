#!/bin/bash

# This script starts both the Flask API and Angular UI for the Lineage Assistant

# Function to check if a command exists
command_exists() {
  command -v "$1" &> /dev/null
}

# Check for Python (macOS might have python3 instead of python)
PYTHON_CMD=""
if command_exists python; then
  PYTHON_CMD="python"
elif command_exists python3; then
  PYTHON_CMD="python3"
else
  echo "Error: Python is required but not installed."
  echo "Please install Python from https://www.python.org/downloads/"
  exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# Check for Node.js in multiple ways (macOS might have it in different locations)
NODE_FOUND=false
if command_exists node; then
  NODE_FOUND=true
elif [ -d "/usr/local/opt/node" ]; then
  export PATH="/usr/local/opt/node/bin:$PATH"
  NODE_FOUND=true
elif [ -d "/opt/homebrew/bin" ] && [ -f "/opt/homebrew/bin/node" ]; then
  export PATH="/opt/homebrew/bin:$PATH"
  NODE_FOUND=true
elif [ -d "$HOME/.nvm" ] && [ -f "$HOME/.nvm/nvm.sh" ]; then
  echo "Found NVM installation. Attempting to use it..."
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
  NODE_FOUND=true
fi

if [ "$NODE_FOUND" = false ]; then
  echo "Error: Node.js is required but not installed or not found in PATH."
  echo "Please install Node.js using one of these methods:"
  echo "  - Download from https://nodejs.org/"
  echo "  - Using Homebrew: brew install node"
  echo "  - Using NVM: nvm install node"
  exit 1
fi

# Check for npm
if ! command_exists npm; then
  echo "Error: npm is required but not found."
  echo "npm usually comes with Node.js. Try reinstalling Node.js."
  exit 1
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  echo "Activating virtual environment..."
  source .venv/bin/activate
else
  echo "No virtual environment found. Creating one..."
  $PYTHON_CMD -m venv .venv
  source .venv/bin/activate
  echo "Installing Python dependencies..."
  pip install -r requirements.txt
fi

# Install frontend dependencies if needed
if [ ! -d "lineage-ui/node_modules" ]; then
  echo "Installing frontend dependencies..."
  cd lineage-ui
  npm install
  cd ..
fi

# Start the Flask API in the background
echo "Starting Flask API server..."
$PYTHON_CMD backup/api.py &
API_PID=$!

# Give the API a moment to start
sleep 2

# Start the Angular development server
echo "Starting Angular development server..."
cd lineage-ui
npm start || {
  echo "Failed to start Angular development server."
  cd ..
  echo "You can still use the API directly at http://localhost:5000/api/query"
  echo "Example usage with curl:"
  echo 'curl -X POST http://localhost:5000/api/query -H "Content-Type: application/json" -d '\''{"message": "Show me the available data contracts"}'\'''
  echo "Press Ctrl+C to stop the API server."
  wait $API_PID
  exit 1
}

# When the Angular server is stopped, also stop the Flask API
kill $API_PID

echo "All services stopped."
