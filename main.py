import argparse
import logging
from extractor import Extractor


def main():
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(
        description="Video and Audio Processor",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=45),
    )

    parser.add_argument("video_dir", nargs="?", help="The input video files directory.", default=argparse.SUPPRESS)

    parser.add_argument(
        "--log_level",
        default="info",
        help="Optional: logging level, e.g. info, debug, warning (default: %(default)s). Example: --log_level=debug",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional: directory to write output files (default: <input video dir>). Example: --output_dir=/app/output",
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    if not hasattr(args, "video_dir"):
        parser.print_help()
        exit(1)

    logger.info(f"Media processor beginning with video directory: {args.video_dir}")
    output_dir = args.output_dir if not args.output_dir else args.video_dir
    extractor = Extractor(
        video_dir=args.video_dir,
        output_dir=output_dir,
        log_level=log_level,
        log_formatter=log_formatter,
    )
    extractor.start()
    logger.info(f"Separation complete! Output directory(s): {' '.join(args.output_dir)}")


if __name__ == "__main__":
    main()
