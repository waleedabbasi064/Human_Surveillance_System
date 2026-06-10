from huggingface_hub import HfApi
import glob, os

api = HfApi()

username = api.whoami(token=os.environ["HF_TOKEN"]) ["name"]
# Use username-qualified repo id so uploads target your account namespace
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

# collect all files recursively
for p in patterns:
    for ext in ALLOWED_EXTENSIONS:
        files += glob.glob(os.path.join(p, f"**/*{ext}"), recursive=True)

# upload with FULL relative path (THIS preserves structure)
for path in sorted(set(files)):

    # THIS is the key fix 👇
    repo_path = os.path.relpath(path, start=".")

    print("Uploading:", repo_path)

    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=repo_path,   # 👈 preserves folder structure
        repo_id=repo_id,
        token=token
    )

print("Done 🚀")