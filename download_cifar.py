#!/usr/bin/env python3
"""Convenience root wrapper for scripts/download_cifar.py."""
import sys
from scripts.download_cifar import main

if __name__ == "__main__":
    sys.exit(main())
