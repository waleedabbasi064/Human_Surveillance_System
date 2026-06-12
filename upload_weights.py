from huggingface_hub import HfApi
import glob
import os

api = HfApi()

# Get username and setup repository ID
username = api.whoami(token=os.environ["HF_TOKEN"])["name"]
repo_id = f"{username}/pose-weights"
token = os.environ["HF_TOKEN"]

# Ensure the repo exists (create if missing)
try:
    api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)
    print(f"Ensured repo exists: {repo_id}")
except Exception as e:
    print(f"Warning: could not create or verify repo {repo_id}: {e}")

# Define the root path of your project
BASE_DIR = "/home/waleed64/Documents/Human_Centric_Anomaly_Detection_Agent"

# 1. Explicitly list the exact files to upload from PoseEstimationModel
explicit_files = [
    os.path.join(BASE_DIR, "PoseEstimationModel/yolo26n.pt"),
    os.path.join(BASE_DIR, "PoseEstimationModel/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"),
    os.path.join(BASE_DIR, "PoseEstimationModel/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"),
]

# 2. Grab all files dynamically inside the Trained_Models/CHAD directory
chad_dir_path = os.path.join(BASE_DIR, "Trained_Models/CHAD")
chad_files = glob.glob(os.path.join(chad_dir_path, "*"))

# Combine both target sets into a single upload queue
final_upload_list = explicit_files + chad_files

# Upload processing loop
for local_path in sorted(set(final_upload_list)):
    # Double check file exists locally before trying to upload
    if not os.path.exists(local_path):
        print(f"❌ File not found locally, skipping: {local_path}")
        continue
    
    if os.path.isdir(local_path):
        continue # Skip directory references picked up by glob

    # Calculate relative path from BASE_DIR to preserve clean folder structure on Hugging Face
    repo_path = os.path.relpath(local_path, start=BASE_DIR)

    print(f"Uploading: {repo_path}")

    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,  # Saves exactly as PoseEstimationModel/... or Trained_Models/...
        repo_id=repo_id,
        token=token
    )

print("Done 🚀 Everything specified has been safely uploaded.")