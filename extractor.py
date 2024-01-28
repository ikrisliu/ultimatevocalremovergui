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


def run_duration(start):
    elapsed_time = time.perf_counter() - start
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (elapsed_time - int(elapsed_time)) * 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"


def format_time(sec):
    dummy_date = datetime(1900, 1, 1)
    return (dummy_date + timedelta(seconds=sec)).strftime("%H:%M:%S.%f")[:-3]


@dataclass
class Timecode:
    start: str
    end: str


class Extractor:
    def __init__(
            self, video_dir: str,
            output_dir: str,
            subtitle_box: str,
            use_gpu: bool,
            log_level=logging.DEBUG,
            log_formatter=None
    ):
        self.video_dir = video_dir
        self.video_clips = sorted(os.listdir(video_dir))
        self.output_dir = output_dir
        self.subtitle_box = subtitle_box
        self.use_gpu = use_gpu
        self.log_level = log_level
        self.log_formatter = log_formatter
        self.logger = self.get_logger()

        self.screenshot_dir = os.path.join(self.output_dir, "screenshots")
        self.video_file = os.path.join(output_dir, "video.mp4")
        self.audio_file = os.path.join(output_dir, "audio.aac")
        self.vocal_file = os.path.join(output_dir, "1_audio_(Vocals).wav")
        self.subtitle_file = os.path.join(output_dir, "subtitles.txt")
        self.timecode_file = os.path.join(output_dir, "timecodes.txt")
        self.timecodes: [Timecode] = []
        self.subtitles: [str] = []

    def start(self):
        self.merge_videos()
        self.separate_audio()
        self.separate_vocal()
        tc = self.detect_audio_timecode()
        self.ocr_subtitle(tc)
        self.translate_subtitle()

    def merge_videos(self):
        self.logger.info(f"Merging multiple video files: {self.video_clips}")
        list_file = "list.txt"

        perf_start = time.perf_counter()
        try:
            with open(list_file, 'w') as file:
                for vc in self.video_clips:
                    file.write(f"file '{os.path.join(self.video_dir, vc)}'\n")
            (ffmpeg.input(list_file, f="concat", safe=0).output(self.video_file, c="copy", loglevel=self.log_level)
             .run(overwrite_output=True))
            self.logger.info(f"Merged video file: {self.video_file}, with duration: {run_duration(perf_start)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Merge video clips with error: {ex.stderr.decode('utf-8')}")
            raise ex
        finally:
            os.remove(list_file)

    def separate_audio(self):
        self.logger.info(f"Separating audio from video")
        perf_start = time.perf_counter()
        tmp_file = os.path.join(self.output_dir, "audio-tmp.aac")
        try:
            (ffmpeg.input(self.video_file).output(tmp_file, acodec="copy", loglevel=self.log_level)
             .run(overwrite_output=True))
            # Normalize the audio to ensure consistent levels
            (ffmpeg.input(tmp_file).output(self.audio_file, af="loudnorm=I=-11:LRA=10:TP=0", loglevel=self.log_level)
             .run(overwrite_output=True))
            self.logger.info(f"Separated audio file: {self.audio_file}, with duration: {run_duration(perf_start)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Separate audio from video with error: {ex.stderr.decode('utf-8')}")
            raise ex
        finally:
            os.remove(tmp_file)

    def separate_vocal(self):
        self.logger.info(f"Separating vocal from audio file: {self.audio_file}")
        perf_start = time.perf_counter()
        try:
            uvr = UVR(
                input_paths=[self.audio_file],
                export_path=self.output_dir,
            )
            uvr.process_start()
            self.logger.info(f"Separated vocal file: {self.vocal_file}, with duration: {run_duration(perf_start)}")
        except Exception as ex:
            self.logger.error(f"Separate vocal with error: {ex}")
            raise ex

    def detect_audio_timecode(self):
        self.logger.info(f"Detecting vocal timecodes from vocal file: {self.vocal_file}")
        tc_seconds: [Timecode] = []

        perf_start = time.perf_counter()
        try:
            cmd = ffmpeg.input(self.vocal_file).output("-", af="silencedetect=noise=-14.5dB:d=0.4", f="null")
            _, out = ffmpeg.run(cmd, capture_stdout=True, capture_stderr=True)
            out = out.decode('utf-8')
            self.logger.debug(f"Detected vocal timecodes: {out}")

            # Silence end means vocal start, so the 'silence_end" is at the first.
            pattern = re.compile(r"silence_end: (\d+\.\d+).*?silence_start: (\d+\.?\d*)", re.DOTALL)
            matches = pattern.findall(out)
            for match in matches:
                start, end = map(float, match)
                tc_seconds.append(Timecode(start, end))

            with open(self.timecode_file, 'w') as file:
                for ts in tc_seconds:
                    file.write(f"{ts.start} --> {ts.end}\n")

            self.logger.info(f"Detected vocal timecodes in file: {self.timecode_file}, "
                             f"with duration: {run_duration(perf_start)}")
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

        ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=self.use_gpu, show_log=False, use_space_char=True,
                        rec_model_dir="models/PaddleOCR/ch_PP-OCRv4_rec_server_infer",
                        det_model_dir="models/PaddleOCR/ch_PP-OCRv4_det_server_infer")

        start_time = time.perf_counter()
        try:
            # 1. Crop images according to timecodes(second) and do the OCR
            img_files = self.batch_crop_images(tc_seconds)

            self.logger.info(f"OCR images ...")
            perf_start = time.perf_counter()
            ocr_texts = self.do_ocr(ocr, img_files)
            self.logger.info(f"OCR images with duration: {run_duration(perf_start)}")

            # 2. Crop images according to segmented timecodes (segment timecodes by a constant interval)
            vocal_rate = 0.15  # Say one word in seconds
            loop_interval = 0.3  # Loop interval in seconds
            min_duration = 0.5  # Subtitle display minimum duration
            invalid_interval = 10.0

            all_texts: [str] = []
            all_times: [Timecode] = []
            segment_times: [Timecode] = []

            for idx, text in enumerate(ocr_texts):
                tc = tc_seconds[idx]
                start = tc.start
                end = tc.end
                duration = tc.end - tc.start
                vocal_dur = len(text) * vocal_rate

                all_texts.append(text)
                all_times.append(tc)

                if text != "":
                    if duration <= min_duration:
                        tc.start -= (min_duration - duration)
                        continue
                    if duration - vocal_dur <= vocal_rate:
                        continue
                    start += vocal_dur
                    tc.end = start
                else:
                    if duration >= invalid_interval:
                        continue

                    if duration > min_duration:
                        start += max(duration / 5.0, loop_interval)
                    else:  # Make sure subtitle display duration is not less than `min_duration` seconds
                        start -= min_duration
                        start = start if idx > 0 and tc_seconds[idx - 1].end < start else tc.start

                while start < end:
                    next_start = start + loop_interval
                    timecode = Timecode(start, next_start)
                    all_texts.append("")
                    all_times.append(timecode)
                    segment_times.append(timecode)
                    start = next_start

            self.batch_crop_images(segment_times)

            # 3. Further OCR text handling (Duplicated or empty subtitles)
            def upsert(s: str, t: Timecode):
                pre_t = self.subtitles[-1] if len(self.subtitles) > 0 else ""
                if s != pre_t:
                    self.subtitles.append(s)
                    fs = format_time(t.start)
                    fe = format_time(t.end)
                    self.timecodes.append(Timecode(fs, fe))
                elif len(self.timecodes) > 0:
                    pre = self.timecodes[-1]
                    pre.end = format_time(t.end)

            self.logger.info(f"Further OCR images ...")
            perf_start = time.perf_counter()

            for idx, text in enumerate(all_texts):
                tc = all_times[idx]

                if text != "":
                    upsert(text, tc)
                    continue

                file = self.image_filename(tc.start)
                ocr_tt = self.do_ocr(ocr, [file])
                ocr_t = ocr_tt[0]
                if len(ocr_tt) > 0 and ocr_t != "":
                    upsert(ocr_t, tc)

            self.logger.info(f"Further OCR images with duration: {run_duration(perf_start)}")

            # 4. Write subtitles to a text file
            with open(self.subtitle_file, 'w') as file:
                for sub in self.subtitles:
                    file.write(f"{sub}\n")

            self.logger.debug(f"Timecodes({len(self.timecodes)}): {self.timecodes}")
            self.logger.debug(f"Subtitles({len(self.subtitles)}): {self.subtitles}")
            self.logger.info(f"OCR subtitles text in file: {self.subtitle_file}")
            self.logger.info(f"Process timecodes and subtitles with duration: {run_duration(start_time)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Crop screenshot from video with error: {ex.stderr}")
            raise ex
        except Exception as ex:
            self.logger.error(f"OCR subtitle with error: {ex}")
            raise ex
        finally:
            pass

    @staticmethod
    def cropping_start_time(start):
        return start + 0.1  # Delay 0.1 seconds for getting correct cropped image

    def image_filename(self, start: str):
        name = int(self.cropping_start_time(start) * 1000)
        return os.path.join(self.screenshot_dir, f"{name:010}.jpg")

    def crop_images(self, ss: [str]):
        output_files = []
        cmd = ["ffmpeg"]
        if self.use_gpu:
            cmd.extend(["-hwaccel", "cuda"])

        for s in ss:
            file = self.image_filename(s)
            output_files.append(file)
            start = str(self.cropping_start_time(s))
            # Minimize the decoding and seeking operations by using the -ss (seek) option `before` the input file
            cmd.extend(["-ss", start, "-i", self.video_file])
            cmd.extend(["-vf", "crop=700:100:10:850", "-vframes", "1", "-loglevel", "quiet", file, "-y"])
        subprocess.run(cmd)

        return output_files

    def batch_crop_images(self, timecodes: [Timecode]):
        chunk_size = 1
        tt = list(map(lambda it: it.start, timecodes))
        chunks = [tt[i:i + chunk_size] for i in range(0, len(tt), chunk_size)]

        perf_start = time.perf_counter()
        self.logger.info(f"Cropping images base on timecodes by batch size {chunk_size} ...")
        with multiprocessing.Pool() as pool:
            img_files = pool.map(self.crop_images, chunks)
        img_files = [file for chunk_file in img_files for file in chunk_file]
        self.logger.info(f"Cropped images base on timecodes with duration: {run_duration(perf_start)}")

        return img_files

    def do_ocr(self, ocr: PaddleOCR, img_files: [str]):
        ocr_texts = []
        for idx, file in enumerate(img_files):
            try:
                result = ocr.ocr(file, cls=False)
                self.logger.debug(f"OCR Result: {result} from image: {file}")

                if result and result != [None]:
                    # [[
                    # [[[191.0, 10.0], [511.0, 10.0], [511.0, 49.0], [191.0, 49.0]], ('CRAB', 0.99381166696)],
                    # [[[586.0, 0.0], [665.0, 44.0], [650.0, 70.0], [572.0, 22.0]], ('你好吗', 0.9869911670684814)]
                    # ]]
                    result = list(chain.from_iterable(result))
                    max_len_ocr = max(result, key=lambda v: len(v[1][0]))
                    text = max_len_ocr[1][0].strip()
                    ocr_texts.append(text)
                else:
                    ocr_texts.append("")
            except Exception as ex:
                ocr_texts.append("")
                self.logger.error(ex)
        return ocr_texts

    def translate_subtitle(self):
        self.logger.info(f"Translating subtitle to different languages")
        langs = ["chinese"]

        perf_start = time.perf_counter()
        try:
            for lang in langs:
                srt_name = os.path.join(self.output_dir, f"{lang}.vtt")
                with open(srt_name, 'w') as file:
                    file.write("WEBVTT\n")
                    file.write(WEBVTT_STYLE)
                    for idx, sub in enumerate(self.subtitles):
                        tc = self.timecodes[idx]
                        file.write(f"{tc.start} --> {tc.end}\n")
                        file.write(f"{sub}\n\n")
            self.logger.info(
                f"Translation complete in dir: {self.output_dir}, with duration: {run_duration(perf_start)}")
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
