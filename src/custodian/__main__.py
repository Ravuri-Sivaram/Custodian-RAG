"""`python -m custodian <cmd>` entry point (works even without pip install, using the same path-dependency mode as the engine)."""
from .cli import main

if __name__ == "__main__":
    main()
