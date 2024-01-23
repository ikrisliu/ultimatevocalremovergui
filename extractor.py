import os
import re
import time
import logging
import warnings
import shutil
import ffmpeg
import subprocess
import multiprocessing
from dataclasses import dataclass
from itertools import chain
from datetime import datetime, timedelta
from uvr_cli import UVR
from paddleocr import PaddleOCR
from webvtt_template import WEBVTT_STYLE


def run_duration(start: float):
    elapsed_time = time.perf_counter() - start
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (elapsed_time - int(elapsed_time)) * 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"


def format_time(seconds: float):
    dummy_date = datetime(1900, 1, 1)
    return (dummy_date + timedelta(seconds=seconds)).strftime("%H:%M:%S.%f")[:-3]


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
        self.screenshot_dir = os.path.join(self.output_dir, "screenshots")
        self.video_file = os.path.join(output_dir, "video.mp4")
        self.audio_file = os.path.join(output_dir, "audio.aac")
        self.vocal_file = os.path.join(output_dir, "1_audio_(Vocals).wav")
        self.subtitle_file = os.path.join(output_dir, "subtitles.txt")
        self.timecode_file = os.path.join(output_dir, "timecodes.txt")
        self.timecodes: [Timecode] = []
        self.subtitles: [str] = []
        self.log_level = log_level
        self.log_formatter = log_formatter
        self.logger = self.get_logger()

    def start(self):
        # self.merge_videos()
        # self.separate_audio()
        # self.separate_vocal()
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
            self.logger.info(f"Separated vocal file: {self.vocal_file}, with duration: {run_duration(start)}")
        except Exception as ex:
            self.logger.error(f"Separate vocal with error: {ex}")
            raise ex

    def detect_audio_timecode(self):
        start_time = time.perf_counter()
        self.logger.info(f"Detecting vocal timecodes from vocal file: {self.vocal_file}")
        tc_seconds: [Timecode] = []
        try:
            cmd = ffmpeg.input(self.vocal_file).output('-', af='silencedetect=noise=-15dB:d=0.4', f='null')
            _, out = ffmpeg.run(cmd, capture_stdout=True, capture_stderr=True)
            out = out.decode('utf-8')

            # Silence end means vocal start, so the 'silence_end" is at the first.
            pattern = re.compile(r'silence_end: (\d+\.\d+).*?silence_start: (\d+\.\d+)', re.DOTALL)
            matches = pattern.findall(out)
            for match in matches:
                start, end = map(float, match)
                tc_seconds.append(Timecode(start, end))

            with open(self.timecode_file, 'w') as file:
                for ts in tc_seconds:
                    file.write(f"{ts.start} --> {ts.end}\n")

            self.logger.debug(tc_seconds)
            self.logger.info(f"Detected vocal timecodes in file: {self.timecode_file}, "
                             f"with duration: {run_duration(start_time)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Detect vocal timecodes with exception: {ex.stderr.decode('utf-8')}")
            raise ex

        return tc_seconds

    def ocr_subtitle(self, tc_seconds: [Timecode]):
        start_time = time.perf_counter()
        self.logger.info(f"OCR subtitle from video: {self.video_file}")

        ss_dir = self.screenshot_dir
        if os.path.exists(ss_dir):
            shutil.rmtree(ss_dir)
        os.makedirs(ss_dir)

        try:
            st = time.perf_counter()
            chunk_size = 10
            chunks = [tc_seconds[i:i + chunk_size] for i in range(0, len(tc_seconds), chunk_size)]
            self.logger.info(f"Cropping images by batch ...")
            with multiprocessing.Pool() as pool:
                img_files = pool.map(self.crop_images, chunks)
            img_files = [file for chunk_file in img_files for file in chunk_file]
            self.logger.info(f"Cropped images by batch with duration: {run_duration(st)}")

            self.subtitles = self.do_ocr(img_files, tc_seconds)
            # for idx, text in enumerate(self.subtitles):


            # vocal_rate = 0.15  # Say one word in seconds
            # min_duration = 0.5  # Subtitle display minimum duration
            # for ts in tc_seconds:
            #     next_start = ts.start
            #     while next_start < ts.end:
            #         add_timecode = False
            #         loop_interval = 0.3  # Loop interval in seconds
            #         curr_start = next_start
            #         ocr_text = self.crop_and_ocr(format_time(curr_start))
            #
            #         if ocr_text and ocr_text != "":
            #             last_sub = self.subtitles[-1] if len(self.subtitles) > 0 else None
            #             if ocr_text != last_sub:
            #                 add_timecode = True
            #                 loop_interval = len(ocr_text) * vocal_rate
            #                 curr_end = curr_start + loop_interval
            #                 curr_end = curr_end if curr_end < ts.end else ts.end
            #
            #                 # Make sure subtitle display duration is not less than 0.5 seconds
            #                 d = curr_end - curr_start
            #                 ss = curr_start if d > min_duration else curr_start - (min_duration - d)
            #
            #                 self.subtitles.append(ocr_text)
            #                 self.timecodes.append(Timecode(format_time(ss), format_time(curr_end)))
            #
            #         if not add_timecode and len(self.timecodes) > 0:
            #             pre = self.timecodes[-1]
            #             if format_time(ts.start) < pre.end:
            #                 curr_end = curr_start + loop_interval
            #                 curr_end = curr_end if curr_end < ts.end else ts.end
            #                 pre.end = format_time(curr_end)
            #             else:
            #                 pass  # It means begin next timecode loop
            #
            #         next_start += loop_interval

            with open(self.subtitle_file, 'w') as file:
                for sub in self.subtitles:
                    file.write(f"{sub}\n")

            self.logger.debug(f"OCR subtitles text in file: {self.subtitle_file}")
            self.logger.debug(f"Updated event timecodes in file: {self.timecode_file}")
            self.logger.info(f"Process timecodes and subtitles with duration: {run_duration(start_time)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Crop screenshot from video with error: {ex.stderr}")
            raise ex
        except Exception as ex:
            self.logger.error(f"OCR subtitle with error: {ex}")
            raise ex
        finally:
            pass

    def output_filename(self, start: str):
        name = int(start * 1000)
        return os.path.join(self.screenshot_dir, f"{name:010}.jpg")

    def crop_images(self, timecodes: [Timecode]):
        output_files = []
        cmd = ["ffmpeg", "-i", self.video_file]
        for tc in timecodes:
            file = self.output_filename(tc.start)
            output_files.append(file)
            ss = f"{tc.start}"
            cmd.extend(["-ss", ss, "-vf", "crop=700:100:10:850", "-vframes", "1", "-loglevel", "quiet", file, "-y"])
        subprocess.run(cmd)
        return output_files

    def do_ocr(self, img_files: [str], timecodes: [Timecode]):
        ocr_texts = []
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False,
                        rec_model_dir="models/PaddleOCR/ch_PP-OCRv4_rec_server_infer",
                        det_model_dir="models/PaddleOCR/ch_PP-OCRv4_det_server_infer")
        for idx, file in enumerate(img_files):
            result = ocr.ocr(file, cls=False)
            self.logger.debug(f"OCR Result: {result} from image: {file}, at time: {timecodes[idx].start}")
            if result and result != [None]:
                # [[[[[191.0, 10.0], [511.0, 10.0], [511.0, 49.0], [191.0, 49.0]], ('你好吗', 0.99381166696)]]]
                result = list(chain.from_iterable(result))
                result = list(chain.from_iterable(result))
                for i, line in enumerate(result):
                    if i == 1:
                        ocr_texts.append(line[0])
                        break
            else:
                ocr_texts.append("")
        return ocr_texts

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
            self.logger.info(f"Translation complete in dir: {self.output_dir}, with duration: {run_duration(start)}")
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
