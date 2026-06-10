import os
import sys
from typing import Optional

def ensure_pose_config_uploaded(local_path: str, repo_id: Optional[str] = None) -> bool:
    """If `HF_TOKEN` is set, upload `local_path` to `repo_id` (a Space repo).

    Returns True on success (or if token not set but file exists locally), False on failure.
    """
    if not os.path.exists(local_path):
        print(f"[UPLOAD] Local config not found: {local_path}")
        return False

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[UPLOAD] HF_TOKEN not set; skipping remote upload.")
        return True

    if repo_id is None:
        repo_id = os.environ.get("HF_SPACE_REPO", "shahzaib7788/human-centric-anomaly-detection")

    try:
        from huggingface_hub import upload_file

        print(f"[UPLOAD] Uploading {local_path} -> {repo_id}:PoseEstimationModel/config.yaml")
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo="PoseEstimationModel/config.yaml",
            repo_id=repo_id,
            repo_type="space",
            token=hf_token,
            commit_message="Ensure PoseEstimationModel/config.yaml present",
        )
        print("[UPLOAD] Upload succeeded.")
        return True
    except Exception as e:
        print(f"[UPLOAD] Upload failed: {e}")
        return False


if __name__ == '__main__':
    ok = ensure_pose_config_uploaded(sys.argv[1] if len(sys.argv) > 1 else 'PoseEstimationModel/config.yaml')
    if not ok:
        sys.exit(1)
