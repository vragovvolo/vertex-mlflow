import pytest
import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This file is intentionally left mostly empty
# It serves to make the 'tests' directory a proper pytest package
# You can add fixtures or setup/teardown code here if needed 