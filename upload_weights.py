# upload_weights.py
from huggingface_hub import HfApi
import glob, os

api = HfApi()
username = api.whoami(token=os.environ['HF_TOKEN'])['name']
repo_id = f"{username}/pose-weights"
token = os.environ['HF_TOKEN']

patterns = [
    "PoseEstimationModel/*.pt",
    "PoseEstimationModel/*.pth*",
    "results-sparta-c/*.pth.tar",
    "results-sparta-f/*.pth.tar",
    "Trained_Models/**/*.pth.tar",
    "mmpose/**/*.pth",
]
files = []
for p in patterns:
    files += glob.glob(p, recursive=True)

for path in sorted(set(files)):
    fname = os.path.basename(path)               # upload as root filename (recommended)
    print("Uploading", fname, "from", path)
    api.upload_file(path_or_fileobj=path, path_in_repo=fname, repo_id=repo_id, token=token)
print("Done.")