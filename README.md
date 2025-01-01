curl -LsSf https://astral.sh/uv/install.sh | sh
python -m venv .venv
source ./venv/bin/activate
uv pip install -r requirements.txt