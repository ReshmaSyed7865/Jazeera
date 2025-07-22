#!/usr/bin/env python3
"""
Script to run the QA pair generation process.
This script allows you to easily start or resume the QA generation process.
"""

import os
import sys
from generate_qa_pairs import main as generate_qa

if __name__ == "__main__":
    print("=" * 60)
    print("Jazeera Airways QA Pair Generator")
    print("=" * 60)
    print("This script will process text files from the 'India Arabic' folder")
    print("and generate 5 Q&A pairs for each file using Azure OpenAI.")
    print("\nThe results will be saved to 'jazeera_qa_pairs.xlsx'.")
    print("Progress will be tracked, so you can resume if the process is interrupted.")
    print("=" * 60)
    
    try:
        input("Press Enter to start (or Ctrl+C to cancel)...")
        generate_qa()
    except KeyboardInterrupt:
        print("\nProcess cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
