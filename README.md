# Monogram Retrieval
## Cross modal retrieval between Monogram Schema and Seal Images

## Environment Setup

This project uses **uv** for dependency management and reproducible environments.

### 1. Install `uv`

If `uv` is not installed on your system:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify installation:

```bash
uv --version
```

### 2. Clone the repository

```bash
git clone https://github.com/<username>/cross_modal_retrieval.git
cd cross_modal_retrieval
```

### 3. Create the virtual environment and install dependencies

Run:

```bash
uv sync
```

This will:

* create a project virtual environment (`.venv/`)
* install all dependencies from `pyproject.toml`
* reproduce the exact dependency versions specified in `uv.lock`

### 4. Activate the environment (optional)

```bash
source .venv/bin/activate
```

You can also run commands without activating the environment:

```bash
uv run python main.py
```
