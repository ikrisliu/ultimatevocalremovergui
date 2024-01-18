import os
import logging
import warnings
import ffmpeg
from uvr_cli import UVR


class Extractor:
    def __init__(self, video_dir: str, output_dir: str, log_level=logging.DEBUG, log_formatter=None):
        self.video_files = files = os.listdir(video_dir)
        self.output_dir = output_dir
        self.video_file = os.path.join(output_dir, "video.mp4")
        self.audio_file = os.path.join(output_dir, "audio.aac")
        self.vocal_file = os.path.join(output_dir, "1_audio_(Vocals).flac")
        self.subtitle_file = os.path.join(output_dir, "chinese.txt")
        self.timestamp_file = os.path.join(output_dir, "timestamp.txt")
        self.timestamps = [str]
        self.log_level = log_level
        self.log_formatter = log_formatter
        self.logger = self.get_logger()

    def start(self):
        self.merge_videos()
        self.extract_audio()
        self.separate_audio()
        self.extract_audio_times()
        self.ocr_subtitle()
        self.translate_subtitle()

    def merge_videos(self):
        self.logger.info(f"Merging multiple video files: {self.video_files}")
        input_options = []
        for file in self.video_files:
            input_options.extend(['-i', file])
        ffmpeg.input(*input_options).output(self.video_file).run()
        self.logger.info(f"Merged video file: {self.video_file}")

    def extract_audio(self):
        print(self)

    def separate_audio(self):
        self.logger.info(f"Separating audio file: {self.audio_file}")
        uvr = UVR(
            input_paths=[self.audio_file],
            export_path=self.output_dir,
        )
        uvr.process_start()
        self.logger.info(f"Separated audio file to Instrumental and Vocal in: {self.output_dir}")

    def extract_audio_times(self):
        self.logger.info(f"Extracting non-silence timestamps from vocal file: {self.vocal_file}")
        self.logger.info(f"Extracted timestamps successfully.")

    def ocr_subtitle(self):
        self.logger.info(f"Extracting subtitle by OCR from video: {self.video_file}")
        self.logger.info(f"Extracted subtitle in text file: {self.subtitle_file}")
        self.logger.info(f"Extracted subtitle timestamps in text file: {self.timestamp_file}")

    def translate_subtitle(self):
        print(self)

    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)
        log_level = self.log_level
        log_formatter = self.log_formatter

        log_handler = logging.StreamHandler()

        if log_formatter is None:
            log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        log_handler.setFormatter(log_formatter)

        if not logger.hasHandlers():
            logger.addHandler(log_handler)

        # Filter out noisy warnings from PyTorch for users who don't care about them
        if log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")
        return logger
