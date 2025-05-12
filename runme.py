# runme.py
import sys
import os
import argparse

# Ensure src is in the path...
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.main import main
    
if __name__ == "__main__":
    main()