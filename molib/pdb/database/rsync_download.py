"""
Rsync download
"""

import argparse
import logging
import os
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdb_interaction_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """main"""

    parser = argparse.ArgumentParser(description="Download all PDB pdb files")
    parser.add_argument(
        "-o",
        "--output_dir",
        default="mmCIF",
        required=False,
        help="Directory to save PDB files",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    if args.output_dir is None:
        args.output_dir = "mmCIF"

    # Validate input directory
    if not os.path.exists(args.output_dir):
        logger.error(f"PDB directory does not exist: {args.output_dir}")
        sys.exit(1)

    # Current coordinates (divided mmCIF), excluding obsolete
    rsync_command = f"""rsync -av --partial --timeout=300 --delete \
      --prune-empty-dirs \
      --include='*/' \
      --include='*.cif.gz' \
      --exclude='*' \
      rsync.ebi.ac.uk::pub/databases/pdb/data/structures/divided/mmCIF/ \
      {args.output_dir}/"""

    os.system(rsync_command)

    arg = argparse.ArgumentParser


if __name__ == "__main__":
    main()
