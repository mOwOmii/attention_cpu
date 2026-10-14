#!/usr/bin/env bash
# Run in a regular terminal: .git is read-only in the restricted agent session.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
GIT_AUTHOR_LABEL="${ATTENTION_GIT_NAME:-renhj}"
GIT_AUTHOR_MAIL="2677703123@qq.com"
REPO_NAME="${ATTENTION_GITHUB_REPO:-attention_cpu}"
LOCAL_ONLY=false
case "${1:-}" in
  --local-only) LOCAL_ONLY=true ;;
  '') ;;
  *) printf 'Usage: bash scripts/publish_github.sh [--local-only]\n' >&2; exit 2 ;;
esac
command -v git >/dev/null || { printf 'git is required.\n' >&2; exit 1; }
if git rev-parse --show-toplevel >/dev/null 2>&1; then
  EXISTING_ROOT="$(git rev-parse --show-toplevel)"
  [[ "$EXISTING_ROOT" == "$PROJECT_ROOT" ]] || { printf 'Refusing to modify a parent repository.\n' >&2; exit 1; }
else
  git init -b main
fi
# Identity is local to this repository; no password/token is written to project files.
git config --local user.name "$GIT_AUTHOR_LABEL"
git config --local user.email "$GIT_AUTHOR_MAIL"
git add -- .
if ! git diff --cached --quiet; then
  git -c commit.gpgsign=false commit -m "Implement CPU attention with TP/DP acceptance and scheduler integration"
fi
COMMIT_ID="$(git rev-parse HEAD)"
printf 'Local commit: %s\n' "$COMMIT_ID"
if "$LOCAL_ONLY"; then
  exit 0
fi
command -v gh >/dev/null || {
  printf 'Local commit completed. Install GitHub CLI (gh), then run this script again to publish.\n' >&2
  exit 1
}
export GH_HOST=github.com
if ! gh auth status --hostname github.com >/dev/null 2>&1; then
  gh auth login --hostname github.com --git-protocol https --web
fi
gh auth setup-git --hostname github.com
OWNER="$(gh api --hostname github.com user --jq '.login')"
[[ "$REPO_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'Invalid repository name.\n' >&2; exit 1; }
REMOTE_REPO="$OWNER/$REPO_NAME"
HTTPS_URL="https://github.com/$REMOTE_REPO.git"
if git remote get-url origin >/dev/null 2>&1; then
  ORIGIN_URL="$(git remote get-url origin)"
  case "$ORIGIN_URL" in
    "$HTTPS_URL"|"https://github.com/$REMOTE_REPO"|"git@github.com:$REMOTE_REPO.git") ;;
    *) printf 'Existing origin points to another repository; not changing it.\n' >&2; exit 1 ;;
  esac
  [[ "$(gh repo view "$REMOTE_REPO" --json isPrivate --jq '.isPrivate')" == true ]] || {
    printf 'Expected a private repository; publication stopped.\n' >&2; exit 1;
  }
else
  if gh repo view "$REMOTE_REPO" --json nameWithOwner >/dev/null 2>&1; then
    [[ "$(gh repo view "$REMOTE_REPO" --json isPrivate --jq '.isPrivate')" == true ]] || {
      printf 'An existing public repository has this name; choose another ATTENTION_GITHUB_REPO.\n' >&2; exit 1;
    }
  else
    gh repo create "$REMOTE_REPO" --private --description 'CPU Attention, TP/DP training, and multi-node acceptance'
  fi
  git remote add origin "$HTTPS_URL"
fi
BRANCH_NAME="$(git symbolic-ref --short HEAD)"
git push --set-upstream origin "$BRANCH_NAME"
printf 'Published commit %s to https://github.com/%s\n' "$COMMIT_ID" "$REMOTE_REPO"
