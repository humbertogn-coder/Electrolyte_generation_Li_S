#!/bin/bash
# Install CREST 3.0.2 (official static Linux build from GitHub) into /scratch/user/$USER/bin.
# The binary is statically linked: it needs no module and runs on Grace's RHEL as is.
# Usage (once, on a Grace login node, from the repo root):
#   bash scripts/grace/install_crest.sh
#   echo 'export PATH=/scratch/user/$USER/bin:$PATH' >> ~/.bashrc     # optional
set -euo pipefail
VERSION=${VERSION:-3.0.2}
DEST=/scratch/user/$USER/bin
URL="https://github.com/crest-lab/crest/releases/download/v${VERSION}/crest-gnu-12-ubuntu-latest.tar.xz"
mkdir -p "$DEST"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "downloading $URL"
curl -fsSL -o "$TMP/crest.tar.xz" "$URL"
tar -xf "$TMP/crest.tar.xz" -C "$TMP"
install -m 755 "$TMP/crest/crest" "$DEST/crest"
"$DEST/crest" --version 2>&1 | grep -m1 "Version" || true
echo "installed $DEST/crest"
