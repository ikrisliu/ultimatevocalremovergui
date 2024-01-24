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
        self.logger.info(f"Merging multiple video files: {self.video_clips}")
        list_file = "list.txt"

        start = time.perf_counter()
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
        self.logger.info(f"Separating audio from video")
        start = time.perf_counter()
        try:
            ffmpeg.input(self.video_file).output(self.audio_file, acodec='acc', c='copy').run(overwrite_output=True)
            self.logger.info(f"Separated audio file: {self.audio_file}, with duration: {run_duration(start)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Separate audio from video with error: {ex.stderr.decode('utf-8')}")
            raise ex

    def separate_vocal(self):
        self.logger.info(f"Separating vocal from audio file: {self.audio_file}")
        start = time.perf_counter()
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
        self.logger.info(f"Detecting vocal timecodes from vocal file: {self.vocal_file}")
        tc_seconds: [Timecode] = []

        start_time = time.perf_counter()
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
        self.logger.info(f"OCR subtitle from video: {self.video_file}")

        ss_dir = self.screenshot_dir
        if os.path.exists(ss_dir):
            shutil.rmtree(ss_dir)
        os.makedirs(ss_dir)

        start_time = time.perf_counter()
        try:
            # 1. Crop images according to timecodes(second) and do the OCR
            img_files = self.batch_crop_images(tc_seconds)
            ocr_texts = self.do_ocr(tc_seconds, img_files)

            # 2. Crop images according to segmented timecodes (segment timecodes by a constant interval)
            segment_times = []
            vocal_rate = 0.15       # Say one word in seconds
            loop_interval = 0.3     # Loop interval in seconds
            min_duration = 0.5      # Subtitle display minimum duration
            forward_time = 0.3      # Forward time in seconds as start timecode for cropping image
            invalid_interval = 3.0

            for idx, text in enumerate(ocr_texts):
                tc = tc_seconds[idx]
                start = tc.start
                duration = tc.end - tc.start

                if text != "":
                    if duration - len(text) * vocal_rate < min_duration:
                        continue
                    else:
                        start += len(text) * vocal_rate
                else:
                    if tc.end - tc.start >= invalid_interval:
                        continue

                    if tc.end - tc.start > min_duration:
                        start += forward_time
                    else:   # Make sure subtitle display duration is not less than `min_duration` seconds
                        start -= min_duration
                        start = start if idx > 0 and tc_seconds[idx-1].end < start else tc.start

                while start < tc.end:
                    end = start + loop_interval
                    segment_times.append(Timecode(start, end))
                    start = end

            self.batch_crop_images(segment_times)

            # 3. Further OCR text handling (Duplicated or empty subtitles)

            # for idx, text in enumerate(ocr_texts):
            #     tc = tc_seconds[idx]
            #
            #     if text != "":
            #         if idx > 0:
            #             pre_text = ocr_texts[idx - 1]
            #             if text == pre_text:
            #                 pre_tc = tc_seconds[idx - 1]
            #                 pre_tc.end = tc.end
            #                 indexes.append(idx)
            #         continue
            #
            #     start = tc.start
            #     if tc.end - tc.start > min_duration:
            #         start += forward_time
            #     else:   # Make sure subtitle display duration is not less than `min_duration` seconds
            #         start -= min_duration
            #         start = start if idx > 0 and tc_seconds[idx-1].end < start else tc.start
            #         tc.start = start
            #
            #     t = self.crop_and_ocr(start)
            #     if t != "":
            #         if idx == 0:
            #             ocr_texts[idx] = t
            #             continue
            #         pre_text = ocr_texts[idx - 1]
            #         if t != pre_text:
            #             ocr_texts[idx] = t
            #         else:
            #             pre_tc = tc_seconds[idx - 1]
            #             pre_tc.end = tc.end
            #             indexes.append(idx)
            #     else:
            #         indexes.append(idx)
            #
            # ocr_texts = [it for index, it in enumerate(ocr_texts) if index not in indexes]
            # tc_seconds = [it for index, it in enumerate(tc_seconds) if index not in indexes]
            #
            # # 3. Further OCR text handling (May have multiple subtitles between start and end timecode)
            # vocal_rate = 0.15  # Say one word in seconds
            # for idx, tc in enumerate(tc_seconds):
            #     loop_interval = 0.3  # Loop interval in seconds
            #     ocr_text = ocr_texts[idx]
            #     next_start = tc.start + len(ocr_text) * vocal_rate
            #     curr_end = next_start if next_start + loop_interval < tc.end else tc.end
            #     timecode = Timecode(format_time(tc.start), format_time(curr_end))
            #
            #     self.subtitles.append(ocr_text)
            #     self.timecodes.append(timecode)
            #
            #     if next_start + loop_interval >= tc.end:
            #         continue
            #
            #     while next_start < tc.end:
            #         add_timecode = False
            #         loop_interval = 0.3
            #         curr_start = next_start
            #         text = self.crop_and_ocr(curr_start)
            #
            #         if text != "":
            #             last_sub = self.subtitles[-1] if len(self.subtitles) > 0 else None
            #             if text != last_sub:
            #                 add_timecode = True
            #                 loop_interval = len(text) * vocal_rate
            #                 curr_end = curr_start + loop_interval
            #                 curr_end = curr_end if curr_end < tc.end else tc.end
            #
            #                 # Make sure subtitle display duration is not less than `min_duration` seconds
            #                 d = curr_end - curr_start
            #                 ss = curr_start if d > min_duration else curr_start - (min_duration - d)
            #                 timecode = Timecode(format_time(ss), format_time(curr_end))
            #                 self.subtitles.append(text)
            #                 self.timecodes.append(timecode)
            #
            #         if not add_timecode and len(self.timecodes) > 0:
            #             pre = self.timecodes[-1]
            #             curr_end = curr_start + loop_interval
            #             curr_end = curr_end if curr_end < tc.end else tc.end
            #             pre.end = format_time(curr_end)
            #
            #         next_start += loop_interval

            with open(self.subtitle_file, 'w') as file:
                for sub in self.subtitles:
                    file.write(f"{sub}\n")

            self.logger.debug(f"OCR subtitles text in file: {self.subtitle_file}")
            self.logger.info(f"Process timecodes and subtitles with duration: {run_duration(start_time)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Crop screenshot from video with error: {ex.stderr}")
            raise ex
        except Exception as ex:
            self.logger.error(f"OCR subtitle with error: {ex}")
            raise ex
        finally:
            pass

    def image_filename(self, start: str):
        name = int(start * 1000)
        return os.path.join(self.screenshot_dir, f"{name:010}.jpg")

    def crop_images(self, ss: [str]):
        output_files = []
        cmd = ["ffmpeg", "-hwaccel", "cuda", "-i", self.video_file]
        for s in ss:
            file = self.image_filename(s)
            output_files.append(file)
            cmd.extend(["-ss", f"{s}", "-vf", "crop=700:100:10:850", "-vframes", "1", "-loglevel", "quiet", file, "-y"])
        subprocess.run(cmd)

        return output_files

    def batch_crop_images(self, timecodes: [Timecode]):
        chunk_size = 20
        tt = list(map(lambda it: it.start + 0.1, timecodes))
        chunks = [tt[i:i + chunk_size] for i in range(0, len(tt), chunk_size)]

        start = time.perf_counter()
        self.logger.info(f"Cropping images base on timecodes by batch size {chunk_size} ...")
        with multiprocessing.Pool() as pool:
            img_files = pool.map(self.crop_images, chunks)
        img_files = [file for chunk_file in img_files for file in chunk_file]
        self.logger.info(f"Cropped images base on timecodes with duration: {run_duration(start)}")

        return img_files

    def do_ocr(self,  ss: [str], img_files: [str]):
        ocr_texts = []
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False,
                        rec_model_dir="models/PaddleOCR/ch_PP-OCRv4_rec_server_infer",
                        det_model_dir="models/PaddleOCR/ch_PP-OCRv4_det_server_infer")
        for idx, file in enumerate(img_files):
            result = ocr.ocr(file, cls=False)
            self.logger.debug(f"OCR Result: {result} from image: {file}, at time: {ss[idx]}")
            if result and result != [None]:
                # [[[[[191.0, 10.0], [511.0, 10.0], [511.0, 49.0], [191.0, 49.0]], ('你好吗', 0.99381166696)]]]
                result = list(chain.from_iterable(result))
                result = list(chain.from_iterable(result))
                for i, line in enumerate(result):
                    if i == 1:
                        ocr_texts.append(line[0].strip())
                        break
            else:
                ocr_texts.append("")
        return ocr_texts

    # def crop_and_ocr(self, start: str):
    #     img_files = self.crop_images([start])
    #     return self.do_ocr([start], img_files)[0]

    def translate_subtitle(self):
        self.logger.info(f"Translating subtitle to different languages")
        langs = ["chinese"]

        start = time.perf_counter()
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
