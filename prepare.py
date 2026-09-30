#!/usr/bin/env python3
"""Run VGGT on raw cases and save reusable features, cameras and rendered views."""
import argparse
from pathlib import Path

from viewweaver.preprocessing import add_preparation_arguments, preparation_options, prepare_one_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prepared"))
    add_preparation_arguments(parser)
    args = parser.parse_args()
    prepare_one_case(args.case, args.output_dir, args.vggt_model, preparation_options(args))


if __name__ == "__main__":
    main()
