"""
Veritas Agent Package
Verified research agent with NLI entailment citations and abstention gates.
"""

import os

# Keep transformers on the torch backend. setdefault so importing this package cannot
# override a choice the surrounding process already made.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Load API keys from the repo-root .env, resolved from this file rather than the working
# directory so `py -m veritas ...` works from anywhere. Real environment variables win.
try:
    from dotenv import load_dotenv
    load_dotenv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        override=False,
    )
except ImportError:
    pass  # python-dotenv is optional; exporting the vars by hand works just as well

__version__ = "0.1.0"
