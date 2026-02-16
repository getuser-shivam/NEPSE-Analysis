#!/usr/bin/env python3
"""
NEPSE Analysis App Setup Script
Automatically installs dependencies and runs the application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def run_app():
    """Run the NEPSE Analysis application"""
    print("Starting NEPSE Analysis App...")
    try:
        subprocess.check_call([sys.executable, "main.py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Application closed by user")
        return True

def main():
    """Main setup function"""
    print("=" * 50)
    print("🚀 NEPSE Analysis App Setup")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install dependencies
    if not install_requirements():
        print("\n❌ Setup failed. Please install dependencies manually:")
        print("pip install -r requirements.txt")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete! Starting application...")
    print("=" * 50)
    
    # Run the application
    run_app()

if __name__ == "__main__":
    main()
