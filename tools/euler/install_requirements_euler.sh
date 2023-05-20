# Get the full path of the current script
SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Navigate to the parent directory
cd "$SCRIPT_PATH/.."

# Ignore open3d and torchvision (open3d because CentOS, torchvision because cuda 11.3)
grep -v -e 'open3d' -e 'torchvision' requirements.txt > temp-requirements.txt
pip install -r temp-requirements.txt
rm temp-requirements.txt

pip install -r requirements-dev.txt
pip install -r requirements-euler.txt
pip install -e .