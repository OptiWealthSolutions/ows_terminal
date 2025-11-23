# -----------------------------------------------------------------------------
# setup.sh (Fixed for Python < 3.14 compatibility)
# -----------------------------------------------------------------------------
set -e

# Constraints
MIN_VERSION="3.10"
# Numba currently fails on 3.14, so we must block it
UNSUPPORTED_VERSION="3.14" 

# 1. Attempt to find a preferred stable binary first
if command -v python3.11 &> /dev/null; then
    PY_CMD="python3.11"
elif command -v python3.12 &> /dev/null; then
    PY_CMD="python3.12"
elif command -v python3.10 &> /dev/null; then
    PY_CMD="python3.10"
else
    # Fallback to system default
    PY_CMD="python3"
fi

if ! command -v $PY_CMD &> /dev/null; then
    echo "Error: Could not find a suitable python3 executable."
    echo "Please install Python 3.11 or 3.12 (e.g., 'brew install python@3.11')."
    exit 1
fi

# 2. Check the version of the selected binary
CURRENT_VER=$($PY_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

echo "Using Python executable: $PY_CMD (Version $CURRENT_VER)"

# Check Minimum (>= 3.10)
if [[ $(printf '%s\n' "$MIN_VERSION" "$CURRENT_VER" | sort -V | head -n1) != "$MIN_VERSION" ]]; then
    echo "Error: Python $MIN_VERSION or higher is required."
    exit 1
fi

# Check Maximum (< 3.14)
# If the 'head -n1' of (3.14, Current) is 3.14, it means Current >= 3.14.
if [[ $(printf '%s\n' "$UNSUPPORTED_VERSION" "$CURRENT_VER" | sort -V | head -n1) == "$UNSUPPORTED_VERSION" ]]; then
    echo "Error: Python $CURRENT_VER is too new. Libraries like Numba require Python < $UNSUPPORTED_VERSION."
    echo "Please install Python 3.11 or 3.12."
    exit 1
fi

# 3. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv) using $PY_CMD ..."
    $PY_CMD -m venv .venv
fi

# 4. Activate and Install
source .venv/bin/activate

echo "Upgrading pip ..."
pip install --upgrade pip setuptools wheel

echo "Installing project dependencies ..."
pip install -r requirements.txt

echo "------------------------------------------------"
echo "Setup complete. Activate the virtual environment with:"
echo "  source .venv/bin/activate"