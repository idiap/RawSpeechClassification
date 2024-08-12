# coding=utf-8

import argparse


def add_default_options(parser):
    """Add command line arguments which are common to all the scripts"""
    # fmt: off
    parser.add_argument(
        "--feature-dir", required=True,
        help="Path to the directory containing the features"
    )
    parser.add_argument(
        "--model-filename", required=True,
        help="Path to the .keras model"
    )
    parser.add_argument(
        "--output-dir", default="output-results",
        help="Output directory"
    )
    parser.add_argument(
        "--slice-split", type=int, default=25,
        help="Slice size for feature context"
    )
    parser.add_argument(
        "--verbose", type=int, default=0,
        help="Keras verbose level for fit and predict"
    )
    # fmt: on
