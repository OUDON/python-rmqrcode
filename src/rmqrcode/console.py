#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
from typing import Union

from rmqrcode import (
    DataTooLongError,
    ErrorCorrectionLevel,
    FitStrategy,
    IllegalVersionError,
    QRImage,
    rMQR,
)


def _show_error_and_exit(msg: str):
    print(msg, file=sys.stderr)
    sys.exit(1)


def _make_qr(data: str, ecc: ErrorCorrectionLevel, version: Union[str, None], fit_strategy: FitStrategy):
    if version is None:
        qr = rMQR.fit(data, ecc=ecc, fit_strategy=fit_strategy)
    else:
        try:
            qr = rMQR(version, ecc)
        except IllegalVersionError:
            _show_error_and_exit("Error: Illegal version.")
        qr.add_segment(data)
        qr.make()
    return qr


def _save_image(qr: rMQR, output: Union[str, bytes, Path]):
    image = QRImage(qr)
    try:
        image.save(output)
    except FileNotFoundError as e:
        _show_error_and_exit(f"Error: {e}")


def main():
    parser = _init_argparser()
    args = parser.parse_args()

    if args.ecc == "M":
        ecc = ErrorCorrectionLevel.M
    elif args.ecc == "H":
        ecc = ErrorCorrectionLevel.H

    fit_strategy = FitStrategy.BALANCED
    if args.fit_strategy == "min_width":
        fit_strategy = FitStrategy.MINIMIZE_WIDTH
    elif args.fit_strategy == "min_height":
        fit_strategy = FitStrategy.MINIMIZE_HEIGHT

    try:
        qr = _make_qr(args.DATA, ecc=ecc, version=args.version, fit_strategy=fit_strategy)
    except DataTooLongError:
        _show_error_and_exit("Error: The data is too long.")

    _save_image(qr, args.OUTPUT)


def _init_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("DATA", type=str, help="Data to encode.")
    parser.add_argument("OUTPUT", type=str, help="Output file path")
    parser.add_argument(
        "--ecc", help="Error correction level. (default: M)", type=str, choices=["M", "H"], default="M"
    )
    parser.add_argument("--version", help="rMQR Code version like 'R11x139'.")
    parser.add_argument(
        "--fit-strategy",
        choices=["min_width", "min_height", "balanced"],
        help="Strategy how to determine rMQR Code size.",
        dest="fit_strategy",
    )
    return parser


if __name__ == "__main__":
    main()
