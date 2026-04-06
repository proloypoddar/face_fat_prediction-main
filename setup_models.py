import os
import torch
import warnings

def download_midas():
    print("[INFO] Setting up models directory...")
    os.makedirs("models", exist_ok=True)
    
    model_path = os.path.join("models", "midas_small.pt")
    
    if os.path.exists(model_path):
        print(f"[INFO] Model already exists at {model_path}")
        return

    print("[INFO] Downloading MiDaS small model (~80 MB)...")
    try:
        # Suppress warnings during download
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # Download via torch hub
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        
        # Save locally for offline deployment
        torch.save(model.state_dict(), model_path)
        print(f"[SUCCESS] Model saved to {model_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")

if __name__ == "__main__":
    download_midas()
