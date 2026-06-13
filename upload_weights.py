from huggingface_hub import HfApi
import glob, os

api = HfApi()

username = api.whoami(token=os.environ["HF_TOKEN"])["name"]
repo_id = f"{username}/pose-weights"
token = os.environ["HF_TOKEN"]

# Ensure the repo exists (create if missing)
try:
    api.create_repo(repo_id=repo_id, token=token, private=False, exist_ok=True)
    print(f"Ensured repo exists: {repo_id}")
except Exception as e:
    print(f"Warning: could not create or verify repo {repo_id}: {e}")

ALLOWED_EXTENSIONS = (".pt", ".pth", ".pth.tar")

patterns = [
    "PoseEstimationModel/",
    "Trained_Models/",
    "results-sparta-c/",
    "results-sparta-f/",
    "mmpose/",
]

files = []

# Collect all files recursively
for p in patterns:
    for ext in ALLOWED_EXTENSIONS:
        files += glob.glob(os.path.join(p, f"**/*{ext}"), recursive=True)

# ==================== ADD THIS UPLOAD LOGIC BELOW ====================

if not files:
    print("No weight files found matching the patterns and extensions.")
else:
    print(f"Found {len(files)} files to upload. Starting upload...")
    
    for file_path in files:
        print(f"Uploading: {file_path}...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,  # Keeps the same folder structure in HF
                repo_id=repo_id,
                token=token
            )
            print(f"Successfully uploaded: {file_path}")
        except Exception as e:
            print(f"Failed to upload {file_path}: {e}")
            
    print("All uploads completed!")