#!/bin/bash

# This script helps install Node.js on macOS for the Lineage Assistant UI

echo "Node.js Installer for Data Lineage Assistant"
echo "=========================================="

# Function to check if a command exists
command_exists() {
  command -v "$1" &> /dev/null
}

# Check if Node.js is already installed
if command_exists node; then
  NODE_VERSION=$(node -v)
  echo "Node.js is already installed (version $NODE_VERSION)."
  exit 0
fi

# Check if Homebrew is installed
if command_exists brew; then
  echo "Homebrew found. Installing Node.js using Homebrew..."
  brew update
  brew install node
  
  if command_exists node; then
    NODE_VERSION=$(node -v)
    echo "Node.js installed successfully (version $NODE_VERSION)."
    echo "You can now run ./start.sh to start the application."
    exit 0
  else
    echo "Failed to install Node.js using Homebrew."
  fi
else
  echo "Homebrew is not installed. You can install it by following instructions at:"
  echo "https://brew.sh/"
  echo ""
  echo "Or install Node.js directly from the official website:"
  echo "https://nodejs.org/en/download/"
  
  read -p "Would you like to install Homebrew now? (y/n): " INSTALL_HOMEBREW
  
  if [[ $INSTALL_HOMEBREW == "y" ]]; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    if command_exists brew; then
      echo "Homebrew installed successfully. Installing Node.js..."
      brew install node
      
      if command_exists node; then
        NODE_VERSION=$(node -v)
        echo "Node.js installed successfully (version $NODE_VERSION)."
        echo "You can now run ./start.sh to start the application."
        exit 0
      else
        echo "Failed to install Node.js using Homebrew."
      fi
    else
      echo "Failed to install Homebrew."
    fi
  fi
fi

echo ""
echo "Alternative options:"
echo "1. Download and install Node.js directly from https://nodejs.org/en/download/"
echo "2. Install NVM (Node Version Manager) and then install Node.js using NVM:"
echo "   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash"
echo "   nvm install node"

exit 1
