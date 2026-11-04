"""Create/update the Hugging Face Space (Docker SDK) from this repo.

Usage: python deploy/hf_space.py [owner/space-name]   (needs `hf auth login`)
"""
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
FRONTMATTER = """---
title: CardioCompass
emoji: 🫀
colorFrom: red
colorTo: pink
sdk: docker
app_port: 8501
pinned: false
short_description: Heart-risk estimate and hybrid recommender (demo)
---

"""


def main(repo_id: str):
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "README.md").write_text(FRONTMATTER + (ROOT / "README.md").read_text())
        api.upload_folder(folder_path=str(ROOT), repo_id=repo_id, repo_type="space",
                          ignore_patterns=[".git/*", ".venv/*", "__pycache__/*", "*/__pycache__/*", "mlruns/*",
                                           "tests/*", ".pytest_cache/*", "data/feedback.csv", "README.md"],
                          commit_message="Deploy CardioCompass")
        api.upload_file(path_or_fileobj=str(Path(tmp) / "README.md"), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="space", commit_message="Space README")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else f"{HfApi().whoami()['name']}/cardiocompass")
