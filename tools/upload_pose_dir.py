import os
import fnmatch
from typing import List

EXCLUDE_EXTS = {'.pt', '.pth', '.pth.tar', '.ckpt'}


def is_weight_file(fname: str) -> bool:
    return any(fname.endswith(ext) for ext in EXCLUDE_EXTS)


def gather_files(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            rel = os.path.relpath(os.path.join(dirpath, f), start=os.getcwd())
            # Skip weights
            if is_weight_file(f):
                continue
            files.append(rel.replace('\\', '/'))
    return sorted(files)


def check_gitignore_for_pose_config():
    gi = os.path.join(os.getcwd(), '.gitignore')
    if not os.path.exists(gi):
        return None
    with open(gi, 'r') as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.strip().startswith('#')]
    # quick heuristic: if PoseEstimationModel/* is present and !PoseEstimationModel/*.py present, YAMLs will be ignored
    has_all = any(l == 'PoseEstimationModel/*' for l in lines)
    has_allow_py = any(l == '!PoseEstimationModel/*.py' for l in lines)
    return {'has_all': has_all, 'has_allow_py': has_allow_py, 'lines': lines}


def upload_files(files: List[str], repo_id: str, token: str):
    from huggingface_hub import upload_file
    skipped = []
    for f in files:
        dest = f  # keep relative path
        try:
            print(f"Uploading {f} -> {repo_id}:{dest}")
            upload_file(path_or_fileobj=f, path_in_repo=dest, repo_id=repo_id, repo_type='space', token=token)
        except Exception as e:
            print(f"Failed to upload {f}: {e}")
            skipped.append((f, str(e)))
    return skipped


def main():
    root = os.path.join(os.getcwd(), 'PoseEstimationModel')
    if not os.path.isdir(root):
        print('PoseEstimationModel directory not found in cwd')
        return 2

    files = gather_files(root)
    print(f'Found {len(files)} files to consider (weights excluded).')

    gi_check = check_gitignore_for_pose_config()
    if gi_check:
        if gi_check['has_all'] and gi_check['has_allow_py']:
            print('[WARNING] .gitignore contains patterns that will prevent non-.py files under PoseEstimationModel from being tracked by git.')
            print('         You may want to add a line: !PoseEstimationModel/config.yaml to allow committing the YAML.')
        else:
            print('[INFO] .gitignore present; lines relevant to PoseEstimationModel:')
            for l in gi_check['lines']:
                if 'PoseEstimationModel' in l:
                    print('  ', l)

    hf_token = os.environ.get('HF_TOKEN')
    repo_id = os.environ.get('HF_SPACE_REPO', 'shahzaib7788/human-centric-anomaly-detection')
    if not hf_token:
        print('[ERROR] HF_TOKEN not set in environment; cannot upload. Set HF_TOKEN and retry.')
        # Just print the list of files for manual upload
        for f in files:
            print('  ', f)
        return 3

    skipped = upload_files(files, repo_id=repo_id, token=hf_token)
    if skipped:
        print('\nSome files failed to upload:')
        for f, err in skipped:
            print(' -', f, err)
        return 4

    print('\nUpload completed successfully.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
