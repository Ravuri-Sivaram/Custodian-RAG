"""pytest root config: ensures `import custodian` (and the engine packages
chunker/embedder/generator) resolves even without a pip install. A src-layout
transitional artifact -- this file shrinks/goes away once W0b's
`pip install -e '.[dev]'` lands."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
