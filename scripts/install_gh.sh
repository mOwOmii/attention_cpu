#!/usr/bin/env bash
# Install the official GitHub CLI Linux binary without sudo.
set -euo pipefail

version=2.102.0
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This installer requires Linux x86_64.' >&2
    exit 1
fi
for command in curl tar sha256sum install mktemp awk; do
    command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done

archive="gh_${version}_linux_amd64.tar.gz"
base="https://github.com/cli/cli/releases/download/v${version}"
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT
cd "$work_dir"
curl --fail --location --retry 2 --connect-timeout 15 --max-time 300 \
    --output "$archive" "$base/$archive"
curl --fail --location --retry 2 --connect-timeout 15 --max-time 60 \
    --output checksums.txt "$base/gh_${version}_checksums.txt"
awk -v name="$archive" '$2 == name { print }' checksums.txt > selected.sha256
if [[ ! -s selected.sha256 ]]; then
    echo 'Official checksum for this archive was not found.' >&2
    exit 1
fi
sha256sum --check selected.sha256
tar -xzf "$archive"
"gh_${version}_linux_amd64/bin/gh" --version
install -d "$HOME/.local/bin"
install -m 0755 "gh_${version}_linux_amd64/bin/gh" "$HOME/.local/bin/gh"
"$HOME/.local/bin/gh" --version
printf '\nInstalled to %s/.local/bin/gh\n' "$HOME"
printf 'For this shell, run: export PATH="$HOME/.local/bin:$PATH"\n'
