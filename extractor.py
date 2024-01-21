import os
import re
import time
import logging
import warnings
import shutil
import ffmpeg
from dataclasses import dataclass
from itertools import chain
from datetime import datetime, timedelta
from uvr_cli import UVR
from paddleocr import PaddleOCR
from webvtt_template import WEBVTT_STYLE


def run_duration(start: float):
    return time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - start)))


@dataclass
class Timecode:
    start: str
    end: str


class Extractor:
    def __init__(
            self, video_dir: str,
            output_dir: str,
            subtitle_box: str,
            log_level=logging.DEBUG,
            log_formatter=None
    ):
        self.video_dir = video_dir
        self.video_clips = os.listdir(video_dir)
        self.output_dir = output_dir
        self.subtitle_box = subtitle_box
        self.video_file = os.path.join(output_dir, "video.mp4")
        self.audio_file = os.path.join(output_dir, "audio.aac")
        self.vocal_file = os.path.join(output_dir, "1_audio_(Vocals).wav")
        self.subtitle_file = os.path.join(output_dir, "subtitle.txt")
        self.timecode_file = os.path.join(output_dir, "timecode.txt")
        self.timecodes: [Timecode] = []
        self.subtitles: [str] = []
        self.log_level = log_level
        self.log_formatter = log_formatter
        self.logger = self.get_logger()

    def start(self):
        self.merge_videos()
        self.separate_audio()
        self.separate_vocal()
        tc = self.detect_audio_timecode()
        self.ocr_subtitle(tc)
        self.translate_subtitle()

    def merge_videos(self):
        start = time.perf_counter()
        self.logger.info(f"Merging multiple video files: {self.video_clips}")
        list_file = "list.txt"
        try:
            with open(list_file, 'w') as file:
                for vc in self.video_clips:
                    file.write(f"file '{os.path.join(self.video_dir, vc)}'\n")
            ffmpeg.input(list_file, f='concat', safe=0).output(self.video_file, c='copy').run(overwrite_output=True)
            self.logger.info(f"Merged video file: {self.video_file}, with duration: {run_duration(start)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Merge video clips with error: {ex.stderr.decode('utf-8')}")
            raise ex
        finally:
            os.remove(list_file)

    def separate_audio(self):
        start = time.perf_counter()
        self.logger.info(f"Separating audio from video")
        try:
            ffmpeg.input(self.video_file).output(self.audio_file, acodec='acc', c='copy').run(overwrite_output=True)
            self.logger.info(f"Separated audio file: {self.audio_file}, with duration: {run_duration(start)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Separate audio from video with error: {ex.stderr.decode('utf-8')}")
            raise ex

    def separate_vocal(self):
        start = time.perf_counter()
        self.logger.info(f"Separating vocal from audio file: {self.audio_file}")
        try:
            uvr = UVR(
                input_paths=[self.audio_file],
                export_path=self.output_dir,
            )
            uvr.process_start()
            self.logger.info(f"Separated vocal file: {self.vocal_file}, with duratoin: {run_duration(start)}")
        except Exception as ex:
            self.logger.error(f"Separate vocal with error: {ex}")
            raise ex

    def detect_audio_timecode(self):
        start_time = time.perf_counter()
        self.logger.info(f"Detecting vocal event times from vocal file: {self.vocal_file}")
        tc_seconds: [Timecode] = []
        try:
            cmd = ffmpeg.input(self.vocal_file).output('-', af='silencedetect=noise=-10dB:d=0.4', f='null')
            _, out = ffmpeg.run(cmd, capture_stdout=True, capture_stderr=True)
            out = out.decode('utf-8')

            # Silence end means vocal start, so the 'silence_end" is at the first.
            pattern = re.compile(r'silence_end: (\d+\.\d+).*?silence_start: (\d+\.\d+)', re.DOTALL)
            matches = pattern.findall(out)
            for match in matches:
                start, end = map(float, match)
                tc_seconds.append(Timecode(start, end))

            self.logger.debug(tc_seconds)
            self.logger.info(f"Detected vocal event times with duration: {run_duration(start_time)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Detect vocal event times with exception: {ex.stderr.decode('utf-8')}")
            raise ex

        return tc_seconds

    def ocr_subtitle(self, tc_seconds: [Timecode]):
        start = time.perf_counter()
        self.logger.info(f"OCR subtitle from video: {self.video_file}")
        dummy_date = datetime(1900, 1, 1)

        ss_dir = os.path.join(self.output_dir, "screenshots")
        if os.path.exists(ss_dir):
            shutil.rmtree(ss_dir)
        os.makedirs(ss_dir)

        try:
            for ts in tc_seconds:
                step = 800
                s = int(ts.start * 1000)
                e = int(ts.end * 1000)
                ss = (dummy_date + timedelta(seconds=ts.start)).strftime("%H:%M:%S.%f")[:-3]
                # Crop one screenshot per 800 milliseconds
                for idx in range(s, e, step):
                    ms = idx + step
                    img_file = os.path.join(ss_dir, f"{idx:010}.jpg")
                    ocr_text = self.crop_and_ocr(ss, img_file)
                    if ocr_text == "":  # use step/2 to retry
                        ms = idx + step / 2
                        img_file = os.path.join(ss_dir, f"{ms:010}.jpg")
                        ns = (dummy_date + timedelta(milliseconds=ms)).strftime("%H:%M:%S.%f")[:-3]
                        ocr_text = self.crop_and_ocr(ns, img_file)

                    # Check if it has duplicated subtitle and if it needs to update the end time
                    ee = (dummy_date + timedelta(milliseconds=ms)).strftime("%H:%M:%S.%f")[:-3]
                    if ocr_text and ocr_text != "":
                        last_sub = self.subtitles[-1] if len(self.subtitles) > 0 else None
                        if ocr_text != last_sub:
                            self.timecodes.append(Timecode(ss, ee))
                            self.subtitles.append(ocr_text)
                        elif len(self.timecodes) > 0:
                            pre = self.timecodes[-1]
                            pre.end = ee
                    ss = ee

            with open(self.timecode_file, 'w') as file:
                for ts in self.timecodes:
                    file.write(f"{ts.start} --> {ts.end}\n")

            with open(self.subtitle_file, 'w') as file:
                for sub in self.subtitles:
                    file.write(f"{sub}\n")

            self.logger.debug(f"OCR subtitles text in file: {self.subtitle_file}")
            self.logger.debug(f"Updated event timecodes in file: {self.timecode_file}")
            self.logger.info(f"Process timecodes and subtitles with duration: {run_duration(start)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Crop video with error: {ex.stderr}")
            raise ex
        except Exception as ex:
            self.logger.error(f"OCR subtitle with error: {ex}")
            raise ex
        finally:
            pass

    def crop_and_ocr(self, start: str, img_file: str):
        ocr_text = ""
        (ffmpeg.input(self.video_file, ss=start)
         .output(img_file, vf='crop=700:80:10:860', vframes=1, loglevel='quiet').run(overwrite_output=True))

        ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False,
                        det_model_dir="models/PaddleOCR/ch_PP-OCRv4_det_server_infer",
                        rec_model_dir="models/PaddleOCR/ch_PP-OCRv4_rec_server_infer")
        result = ocr.ocr(img_file, cls=False)
        self.logger.debug(f"OCR Result: {result} from image: {img_file}, at time: {start}")
        if result and result != [None]:
            # [[[[[191.0, 10.0], [511.0, 10.0], [511.0, 49.0], [191.0, 49.0]], ('你好吗', 0.99381166696)]]]
            result = list(chain.from_iterable(result))
            result = list(chain.from_iterable(result))
            for idx, line in enumerate(result):
                if idx == 1:
                    ocr_text = line[0]
                    break
        return ocr_text

    def translate_subtitle(self):
        start = time.perf_counter()
        self.logger.info(f"Translating subtitle to different languages")
        langs = ["chinese"]
        try:
            for lang in langs:
                srt_name = os.path.join(self.output_dir, f"{lang}.vtt")
                with open(srt_name, 'w') as file:
                    file.write("WEBVTT\n")
                    file.write(WEBVTT_STYLE)
                    for idx, sub in enumerate(self.subtitles):
                        if len(self.timecodes) >= idx + 1:
                            ts = self.timecodes[idx]
                            file.write(f"{ts.start} --> {ts.end}\n")
                            file.write(f"{sub}\n\n")
            self.logger.info(f"Translation complete in directory: {self.output_dir}, with duration: {run_duration(start)}")
        except Exception as ex:
            self.logger.error(f"Translate subtitle with error: {ex}")
            raise ex

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
