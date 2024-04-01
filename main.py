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
        "--output_dir",
        default=None,
        help="Optional: directory to write output files (default: <video's parent dir>). Example: --output_dir=/output"
    )

    parser.add_argument(
        "--subtitle_box",
        default="700:100:10:848",
        help="Optional: cropped box of subtitle in video (default: %(default)s). Example: --subtitle_box=700:100:10:848"
    )

    parser.add_argument(
        "--multi_lines_subtitle",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Optional: video has multiple lines subtitle (default: %(default)s). Example: --multi_lines_subtitle=True",
    )

    parser.add_argument(
        "--use_gpu",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Optional: enable GPU acceleration (default: %(default)s). Example: --use_gpu=False",
    )

    parser.add_argument(
        "--reencode",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Optional: re-encode the video with libx264 and aac audio (default: %(default)s). Example: --reencode=True",
    )

    parser.add_argument(
        "--encode_res",
        default=None,
        help="Optional: re-encode video with resolution (default: %(default)s). Example: --encode_res=720:1280"
    )

    parser.add_argument(
        "--encode_bitrate",
        default=None,
        help="Optional: re-encode video with bitrate (default: %(default)s). Example: --encode_bitrate=2048"
    )

    parser.add_argument(
        "--preprocess",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Optional: Preprocess the video - Merging, Separating (default: %(default)s). Example: --preprocess=False",
    )

    parser.add_argument(
        "--sample_duration",
        default=None,
        help="Optional: sample duration in seconds (default: %(default)s). Example: --sample_mode=60",
    )

    parser.add_argument(
        "--gen_multi_langs",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Optional: only generate multi-language subtitles (default: %(default)s). Example: --gen_multi_langs=True",
    )

    parser.add_argument(
        "--log_level",
        default="info",
        help="Optional: logging level, e.g. info, debug, warning (default: %(default)s). Example: --log_level=debug",
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
        multi_lines_subtitle=args.multi_lines_subtitle,
        use_gpu=args.use_gpu,
        reencode=args.reencode,
        encode_res=args.encode_res,
        encode_bitrate=args.encode_bitrate,
        preprocess=args.preprocess,
        sample_duration=args.sample_duration,
        gen_multi_langs=args.gen_multi_langs,
        log_level=log_level,
        log_formatter=log_formatter,
    )
    extractor.start()
    logger.info(f"Extraction complete with duration: {run_duration(start)}! Output directory(s): {' '.join(output_dir)}")


if __name__ == "__main__":
    main()
