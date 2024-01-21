import os
import argparse
import logging
import time
from extractor import Extractor, run_duration


def main():
    logger = logging.getLogger(__name__)
    log_handler = logging.StreamHandler()
    log_formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d - %(levelname)s - %(module)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)

    parser = argparse.ArgumentParser(
        description="Video and Audio Extractor",
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
        help="Optional: directory to write output files (default: <video's parent dir>). Example: --output_dir=/output"
    )

    parser.add_argument(
        "--subtitle_box",
        default="700:80:10:860",
        help="Optional: cropped box of subtitle in video (default: %(default)s). Example: --subtitle_box=700:80:10:860"
    )

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)

    if not hasattr(args, "video_dir"):
        parser.print_help()
        exit(1)

    logger.info(f"Media extractor beginning with video directory: {args.video_dir}")
    start = time.perf_counter()
    output_dir = args.video_dir[:-1] if args.video_dir.endswith("/") else args.video_dir
    output_dir = args.output_dir if args.output_dir else os.path.dirname(output_dir)
    extractor = Extractor(
        video_dir=args.video_dir,
        output_dir=output_dir,
        subtitle_box=args.subtitle_box,
        log_level=log_level,
        log_formatter=log_formatter,
    )
    extractor.start()
    logger.info(f"Extraction complete with duration: {run_duration(start)}! Output directory(s): {' '.join(output_dir)}")


if __name__ == "__main__":
    main()
