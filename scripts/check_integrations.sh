#!/usr/bin/env bash
# Run tests across specified integration packages.
# Usage: ./scripts/check_integrations.sh llms/llama-index-llms-openai embeddings/llama-index-embeddings-openai
# Or without args to run core only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

run_tests() {
  local pkg_path="$1"
  local full_path="$ROOT/llama-index-integrations/$pkg_path"
  if [ ! -d "$full_path" ]; then
    echo "Package not found: $full_path" >&2
    return 1
  fi
  echo "==> Testing $pkg_path"
  pushd "$full_path" > /dev/null
  pip install -e ".[dev]" -q
  pytest tests/ -v --tb=short
  popd > /dev/null
}

# Always test core first
echo "==> Testing llama-index-core"
pushd "$ROOT/llama-index-core" > /dev/null
pip install -e ".[dev]" -q
pytest tests/core/ -v --tb=short -m "not integration"
popd > /dev/null

# Test any additional packages specified as args
for pkg in "$@"; do
  run_tests "$pkg"
done

echo ""
echo "All tests passed."
