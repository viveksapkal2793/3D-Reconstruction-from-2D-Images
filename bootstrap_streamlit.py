import os
import sys
import subprocess

def install_opencv():
    print("Checking for OpenCV...")
    try:
        import cv2
        print(f"OpenCV already installed: {cv2.__version__}")
        return True
    except ImportError:
        print("OpenCV not found. Installing...")
        
        # Try pip installation
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                  "opencv-contrib-python-headless==4.8.0.76"])
            import cv2
            print(f"Successfully installed OpenCV: {cv2.__version__}")
            return True
        except Exception as e:
            print(f"Failed to install OpenCV via pip: {e}")
            return False

def run_app():
    # Install OpenCV if needed
    if install_opencv():
        import streamlit as st
        # Import and run the actual app
        import app
        app.main()
    else:
        print("ERROR: Failed to install OpenCV. Application cannot run.")
        sys.exit(1)

if __name__ == "__main__":
    run_app()