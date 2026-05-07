import base64
import os

import requests
import streamlit as st

PERSISTED_FILES = [
    "models_config.json",
    "current_prompts.csv",
    "model_responses.csv",
    "final_judge_responses.csv",
    "final_judge_responses_parsed.csv",
]


def _config() -> tuple[str, str, str] | tuple[None, None, None]:
    """Return (token, repo, branch) from st.secrets or env vars, or (None, None, None)."""
    try:
        token  = st.secrets["GITHUB_TOKEN"]
        repo   = st.secrets["GITHUB_REPO"]    # format: "owner/repo-name"
        branch = st.secrets.get("GITHUB_BRANCH", "streamlit-deployment")
    except Exception:
        token  = os.getenv("GITHUB_TOKEN")
        repo   = os.getenv("GITHUB_REPO")
        branch = os.getenv("GITHUB_BRANCH", "streamlit-deployment")
    if token and repo:
        return token, repo, branch
    return None, None, None


def is_configured() -> bool:
    token, _, _ = _config()
    return token is not None


def pull(path: str) -> bool:
    """Download a file from GitHub and write it to local disk. Returns True on success."""
    token, repo, branch = _config()
    if not token:
        return False
    r = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers={"Authorization": f"token {token}"},
        params={"ref": branch},
        timeout=10,
    )
    if r.status_code == 404:
        return False
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(base64.b64decode(r.json()["content"]))
    return True


def push(path: str, message: str = "Update via Streamlit app") -> bool:
    """Push a local file to GitHub. Returns True on success."""
    token, repo, branch = _config()
    if not token:
        return False

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    # Fetch current SHA — required by GitHub API to update an existing file
    r = requests.get(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers={"Authorization": f"token {token}"},
        params={"ref": branch},
        timeout=10,
    )
    sha = r.json().get("sha") if r.status_code == 200 else None

    payload = {"message": message, "content": encoded, "branch": branch}
    if sha:
        payload["sha"] = sha

    r = requests.put(
        f"https://api.github.com/repos/{repo}/contents/{path}",
        headers={"Authorization": f"token {token}"},
        json=payload,
        timeout=15,
    )
    r.raise_for_status()
    return True


def sync_from_github() -> None:
    """Pull all persisted files from GitHub once per session."""
    if st.session_state.get("github_synced"):
        return
    if not is_configured():
        st.session_state["github_synced"] = True
        return
    for path in PERSISTED_FILES:
        pull(path)
    st.session_state["github_synced"] = True
