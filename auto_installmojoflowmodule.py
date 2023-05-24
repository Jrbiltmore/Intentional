import subprocess
import sys

def install_mojoflow_module():
    try:
        # Install the mojoflow module using pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mojoflow"])
        print("mojoflow module installed successfully!")
    except subprocess.CalledProcessError:
        print("Failed to install mojoflow module.")

if __name__ == '__main__':
    install_mojoflow_module()
