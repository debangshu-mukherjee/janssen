.PHONY: requirements install

# Generate requirements.txt from pyproject.toml
requirements:
	uv pip compile pyproject.toml --all-extras -o requirements.txt

# Install dependencies
install: requirements
	uv pip install -r requirements.txt