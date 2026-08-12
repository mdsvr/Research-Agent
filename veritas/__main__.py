import sys

from veritas.cli import main
from veritas.generate import QuotaExhausted

if __name__ == "__main__":
    try:
        main()
    except QuotaExhausted as e:
        # One handler for every command: an exhausted quota is a stop, not a traceback.
        print(f"\nAborted: {e}", file=sys.stderr)
        sys.exit(2)
