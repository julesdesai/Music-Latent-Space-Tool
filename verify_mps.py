import torch
import platform
import subprocess

def get_macos_version():
    """Get detailed macOS version info"""
    try:
        return subprocess.check_output(['sw_vers']).decode()
    except:
        return platform.platform()

def verify_mps():
    print("\n=== System Information ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"macOS Info:\n{get_macos_version()}")
    
    print("\n=== MPS Availability ===")
    print(f"Is MPS built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_built():
        print("\nTrying to create MPS device...")
        try:
            device = torch.device("mps")
            # Try to perform a simple operation
            x = torch.ones(1, device=device)
            y = x + 1
            print("Successfully created and used MPS device!")
            print(f"Test tensor result: {y}")
        except Exception as e:
            print(f"Error using MPS device: {e}")
    
    print("\n=== CPU Information ===")
    try:
        cpu_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
        print(f"CPU: {cpu_info}")
    except:
        print("Couldn't get CPU info")

if __name__ == "__main__":
    verify_mps()