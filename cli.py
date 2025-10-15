"""
Command-line interface for SUID generation.
"""
import argparse

def main():
    """
    Main function for the CLI.
    """
    parser = argparse.ArgumentParser(description="Generate SUIDs.")
    parser.add_argument("--pos-prefer", dest="pos_prefer", action="store_true",
                        help="Prefer certain parts of speech.")
    args = parser.parse_args()

    if args.pos_prefer:
        print("POS preference enabled.")
    else:
        print("POS preference disabled.")

if __name__ == "__main__":
    main()
