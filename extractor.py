import os
import re
import time
import json
import logging
import warnings
import shutil
import ffmpeg
import subprocess
import multiprocessing
from dataclasses import dataclass
from collections import Counter
from dotenv import load_dotenv
from itertools import chain
from datetime import datetime, timedelta
from uvr_cli import UVR
from paddleocr import PaddleOCR
from openai import OpenAI, RateLimitError
from constants import WEBVTT_TEMPLATE, OPENAI_MODEL, PADDLE_REC_MODEL_DIR, PADDLE_DET_MODEL_DIR, TARGET_LANGUAGES, \
    OPENAI_SYSTEM_MESSAGE, SOURCE_LANGUAGE


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


@dataclass
class Subtitle:
    language: str
    contents: [str]
    filename: str


@dataclass
class VideoMetadata:
    resolution: str
    frame_rate: str


VOCAL_RATE = 0.15           # Say one word in seconds
LOOP_INTERVAL = 0.5         # Loop interval in seconds
MIN_DURATION = 0.5          # Subtitle display minimum duration
INVALID_INTERVAL = 10.0     # Vocal duration exceeds 10s means invalid
DEFAULT_FRAME_RATE = "25"   # The video default frame rate


class Extractor:
    def __init__(
            self, video_dir: str,
            output_dir: str,
            subtitle_box: str,
            multi_lines_subtitle: bool,
            use_gpu: bool,
            reencode: bool,
            encode_res: str,
            encode_bitrate: str,
            preprocess: bool,
            sample_duration: float,
            gen_multi_langs: bool,
            log_level=logging.DEBUG,
            log_formatter=None
    ):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.subtitle_box = subtitle_box
        self.multi_lines_subtitle = multi_lines_subtitle
        self.use_gpu = use_gpu
        self.reencode = reencode
        self.encode_res = encode_res
        self.encode_bitrate = encode_bitrate
        self.preprocess = preprocess
        self.sample_duration = sample_duration
        self.gen_multi_langs = gen_multi_langs
        self.log_level = log_level
        # self.level_name = logging.getLevelName(log_level).lower()
        self.log_formatter = log_formatter
        self.logger = self.get_logger()

        self.screenshot_dir = os.path.join(self.output_dir, "screenshots")
        self.video_file = os.path.join(output_dir, "video.mp4")
        self.video_only_file = os.path.join(output_dir, "video-only.mp4")
        self.video_remix_file = os.path.join(output_dir, "video-remix.mp4")
        self.audio_file = os.path.join(output_dir, "audio.m4a")
        self.vocal_file = os.path.join(output_dir, "audio-vocal.wav")
        self.instrumental_file = os.path.join(output_dir, "audio-instrumental.wav")
        self.subtitle_file = os.path.join(output_dir, f"{SOURCE_LANGUAGE}.txt")
        self.timecode_file = os.path.join(output_dir, "timecodes.txt")
        self.timecodes: [Timecode] = []
        self.sub_texts: [str] = []

    def start(self):
        if self.gen_multi_langs:
            self.generate_subtitles(TARGET_LANGUAGES)
            return

        if self.preprocess:
            self.merge_videos()
            self.separate_audio()
            self.separate_vocal()
            self.remix_video()

        tc = self.detect_audio_timecode()
        self.ocr_subtitle(tc)
        self.generate_subtitles([SOURCE_LANGUAGE])

    def merge_videos(self):
        def get_video_clips(folder: str):
            clips = [v for v in os.listdir(folder) if not v.startswith(".")]
            clips = sorted(clips, key=lambda s: int(re.search(r'\d+', s).group()))
            return clips

        video_clips = get_video_clips(self.video_dir)
        self.logger.info(f"Merging multiple video files: {video_clips}")
        list_file = os.path.join(self.output_dir, "list.txt")

        perf_start = time.perf_counter()
        sample_file = os.path.join(self.output_dir, "video-sample.mp4")

        def get_video_metadata(file_path: str):
            c = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json', file_path]
            result = subprocess.run(c, capture_output=True, text=True)
            json_obj = json.loads(result.stdout)
            self.logger.debug(json_obj)

            width = json_obj['streams'][0]['width']
            height = json_obj['streams'][0]['height']
            fps = eval(json_obj['streams'][0]['r_frame_rate'])
            return VideoMetadata(resolution=f"{width}x{height}", frame_rate=fps)

        try:
            if len(video_clips) > 0:
                clips_metadata = []
                for vc in video_clips:
                    video_file = os.path.join(self.video_dir, vc)
                    meta = get_video_metadata(video_file)
                    clips_metadata.append(meta)
                    self.logger.info(meta)

                with open(list_file, 'w') as file:
                    for vc in video_clips:
                        video_file = os.path.join(self.video_dir, vc)
                        file.write(f"file '{video_file}'\n")
                        clips_metadata.append(get_video_metadata(video_file))

                check_result = True
                resolutions = list(map(lambda v: v.resolution, clips_metadata))
                frame_rates = list(map(lambda v: v.frame_rate, clips_metadata))
                if len(set(resolutions)) != 1:
                    check_result = False
                    self.logger.error("The video clips' resolution are not same.")
                if len(set(frame_rates)) != 1:
                    check_result = False
                    self.logger.error("The video clips' frame rate are not same.")

                if not self.reencode and not self.sample_duration:
                    if not check_result:
                        raise Exception

                # Make sure all your files have the same format, codec, frame rate for video, and sample rate for audio.
                cmd = ["ffmpeg"]
                if self.use_gpu:
                    cmd.extend(["-hwaccel", "cuda"])

                if self.reencode:
                    for vc in video_clips:
                        video_file = os.path.join(self.video_dir, vc)
                        cmd.extend(["-i", video_file])

                    filter_complex = ""
                    video_res = self.encode_res if self.encode_res else Counter(resolutions).most_common(1)[0][0]
                    video_res = video_res if video_res else "720:1280"

                    for i in range(len(video_clips)):
                        filter_complex += f"[{i}:v]setpts=PTS-STARTPTS[v{i}];"
                    for i in range(len(video_clips)):
                        filter_complex += f"[v{i}][{i}:a]"
                    filter_complex += f"concat=n={len(video_clips)}:v=1:a=1[v][a]"

                    cmd.extend(["-filter_complex", filter_complex])
                    cmd.extend(["-map", "[v]", "-map", "[a]"])
                    cmd.extend(["-vf", f"scale={video_res},fps={DEFAULT_FRAME_RATE}"])
                    cmd.extend(["-c:v", "h264", "-c:a", "aac", "-ar", "44100"])
                else:
                    cmd.extend(["-f", "concat", "-safe", "0", "-i", list_file])
                    cmd.extend(["-c", "copy"])
                cmd.extend(["-loglevel", "error", self.video_file])
                subprocess.run(cmd)

                if self.reencode:
                    try:
                        out_file = os.path.join(self.output_dir, "video-enc.mp4")
                        cmd = ["fr.handbrake.HandBrakeCLI", "-i", self.video_file, "-e", "x265", "-r", DEFAULT_FRAME_RATE]
                        if self.encode_bitrate:
                            cmd.extend(["-b", self.encode_bitrate])
                        cmd.extend(["-o", out_file])
                        subprocess.run(cmd)
                        if os.path.exists(out_file):
                            os.remove(self.video_file)
                            os.rename(out_file, self.video_file)
                    except Exception as ex:
                        self.logger.error(ex)

            if self.sample_duration:
                (ffmpeg.input(self.video_file).output(sample_file, t=self.sample_duration, c="copy", loglevel="error")
                 .run(overwrite_output=True))
                self.video_file = sample_file

            self.logger.info(f"Merged video file: {self.video_file}, with duration: {run_duration(perf_start)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Merge video clips with error: {ex.stderr.decode('utf-8')}")
            raise ex
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)

    def separate_audio(self):
        self.logger.info(f"Separating audio from video")
        perf_start = time.perf_counter()
        audio_tmp_file = os.path.join(self.output_dir, "audio-tmp.m4a")
        try:
            # Separate video and audio
            node = ffmpeg.input(self.video_file)
            node.output(self.video_only_file, an=None, vcodec='copy', loglevel="error").run(overwrite_output=True)
            node.output(audio_tmp_file, vn=None, acodec='copy', loglevel="error").run(overwrite_output=True)

            # Normalize the audio to ensure consistent levels
            (ffmpeg.input(audio_tmp_file).output(self.audio_file, af="loudnorm=I=-11:LRA=10:TP=0", loglevel="error")
             .run(overwrite_output=True))
            self.logger.info(f"Separated audio file: {self.audio_file}, with duration: {run_duration(perf_start)}")
        except ffmpeg.Error as ex:
            self.logger.error(f"Separate audio from video with error: {ex.stderr.decode('utf-8')}")
            raise ex
        finally:
            os.remove(audio_tmp_file)

    def separate_vocal(self):
        self.logger.info(f"Separating vocal from audio file: {self.audio_file}")
        perf_start = time.perf_counter()

        audio_vocal = os.path.join(self.output_dir, "1_audio_(Vocals).wav")
        audio_inst = os.path.join(self.output_dir, "1_audio_(Instrumental).wav")

        def get_audio_duration(file_path: str):
            try:
                command = ['ffprobe', '-i', file_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
                duration_str = subprocess.check_output(command).decode('utf-8').strip()
                return float(duration_str)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Get audio duration with error: {e}")
                return 0

        audio_duration = get_audio_duration(self.audio_file)
        max_dur = 7200  # in seconds
        segment = 100 * 60
        audio_clips = []
        vocal_files = []
        instrumental_files = []
        list_file = os.path.join(self.output_dir, "list.txt")

        def clip_audio_files():
            if audio_duration > max_dur:
                start = 0
                idx = 1
                while start <= audio_duration:
                    end = start + segment
                    output_file = os.path.join(self.output_dir, f"{idx}.m4a")
                    (ffmpeg.input(self.audio_file, ss=start, to=end).output(output_file, c="copy", loglevel="error")
                     .run(overwrite_output=True))
                    audio_clips.append(output_file)
                    start = end
                    idx += 1
            else:
                audio_clips.append(self.audio_file)

        def merge_audio_files(is_vocal: bool):
            with open(list_file, 'w') as lf:
                suffix = "Vocals" if is_vocal else "Instrumental"
                for idx, af in enumerate(audio_clips, start=1):
                    o_file = os.path.join(self.output_dir, f"{idx}_{idx}_({suffix}).wav")
                    if is_vocal:
                        vocal_files.append(o_file)
                    else:
                        instrumental_files.append(o_file)
                    lf.write(f"file '{o_file}'\n")
            output_file = audio_vocal if is_vocal else audio_inst
            (ffmpeg.input(list_file, f="concat", safe=0).output(output_file, c="copy", loglevel="error")
             .run(overwrite_output=True))

        try:
            clip_audio_files()

            uvr = UVR(input_paths=audio_clips, export_path=self.output_dir)
            uvr.process_start()

            if len(audio_clips) > 1:
                merge_audio_files(is_vocal=True)
                merge_audio_files(is_vocal=False)

            vocal_duration = get_audio_duration(audio_vocal)
            factor = 1 + (vocal_duration - audio_duration) / audio_duration
            self.logger.info(f"Change audio speed factor with: {factor}")

            (ffmpeg.input(audio_vocal).output(self.vocal_file, filter=f"atempo={factor}", loglevel="error")
             .run(overwrite_output=True))
            (ffmpeg.input(audio_inst).output(self.instrumental_file, filter=f"atempo={factor}", loglevel="error")
             .run(overwrite_output=True))
        except Exception as ex:
            self.logger.error(f"Separate vocal with error: {ex}")
            raise ex
        finally:
            os.remove(audio_vocal)
            os.remove(audio_inst)
            if len(audio_clips) > 1:
                for file in audio_clips:
                    os.remove(file)
                for file in vocal_files:
                    os.remove(file)
                for file in instrumental_files:
                    os.remove(file)
            if os.path.exists(list_file):
                os.remove(list_file)
            self.logger.info(f"Separated vocal file: {self.vocal_file}, with duration: {run_duration(perf_start)}")

    def remix_video(self):
        self.logger.info(f"Remixing video file: {self.video_only_file}, and vocal audio file: {self.vocal_file}")
        perf_start = time.perf_counter()

        try:
            cmd = ["ffmpeg"]
            if self.use_gpu:
                cmd.extend(["-hwaccel", "cuda"])
            cmd.extend(["-i", self.video_only_file, "-i", self.vocal_file, "-loglevel", "error"])
            cmd.extend(["-c:v", "copy", "-c:a", "aac", self.video_remix_file, "-y"])
            subprocess.run(cmd)
        except Exception as ex:
            self.logger.error(f"Remix video and vocal audio with error: {ex}")
            raise ex
        finally:
            os.remove(self.video_only_file)
            self.logger.info(f"Remixed video file: {self.video_remix_file}, with duration: {run_duration(perf_start)}")

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

            self.logger.info(f"Detected vocal timecodes with duration: {run_duration(perf_start)}")
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

        ocr = PaddleOCR(use_angle_cls=False, lang="ch", use_gpu=self.use_gpu, show_log=False, det_db_unclip_ratio=2,
                        rec_model_dir=PADDLE_REC_MODEL_DIR, det_model_dir=PADDLE_DET_MODEL_DIR)

        start_time = time.perf_counter()
        try:
            # 1. Crop images according to timecodes(second) and do the OCR
            img_files = self.crop_images(tc_seconds)

            self.logger.info(f"OCR images ...")
            perf_start = time.perf_counter()
            ocr_texts = self.do_ocr(ocr, img_files)
            self.logger.info(f"OCR images with duration: {run_duration(perf_start)}")

            # 2. Crop images according to segmented timecodes (segment timecodes by a constant interval)
            all_texts: [str] = []
            all_times: [Timecode] = []
            segment_times: [Timecode] = []

            for idx, text in enumerate(ocr_texts):
                tc = tc_seconds[idx]
                start = tc.start
                end = tc.end
                duration = tc.end - tc.start
                vocal_dur = len(text) * VOCAL_RATE

                if text != "":
                    all_texts.append(text)
                    all_times.append(tc)

                    if duration <= MIN_DURATION:
                        # Make sure subtitle display duration is not less than `MIN_DURATION` seconds
                        tc.start -= (MIN_DURATION - duration)
                        continue
                    if duration - vocal_dur <= VOCAL_RATE:
                        continue
                    start += vocal_dur
                    tc.end = start
                else:
                    if duration >= INVALID_INTERVAL:
                        continue
                    if duration > MIN_DURATION:
                        start += max(duration / 5.0, LOOP_INTERVAL)
                    else:
                        # Forward seconds to check if it has subtitle
                        start = tc.start - VOCAL_RATE

                while start < end:
                    next_start = start + LOOP_INTERVAL
                    timecode = Timecode(start, min(next_start, end))
                    all_texts.append("")
                    all_times.append(timecode)
                    segment_times.append(timecode)
                    start = next_start

            self.crop_images(segment_times)

            # 3. Further OCR text handling (Duplicated or empty subtitles)
            def upsert(s: str, t: Timecode):
                pre_t = self.sub_texts[-1] if len(self.sub_texts) > 0 else ""
                if s != pre_t:
                    self.sub_texts.append(s)
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

            # 4. Write timecodes and subtitles to text file
            with open(self.timecode_file, 'w') as file:
                for tc in self.timecodes:
                    file.write(f"{tc.start} --> {tc.end}\n")

            with open(self.subtitle_file, 'w', encoding="utf-8") as file:
                for text in self.sub_texts:
                    file.write(f"{text}\n")

            self.logger.debug(f"Timecodes({len(self.timecodes)}): {self.timecodes}")
            self.logger.debug(f"Subtitles({len(self.sub_texts)}): {self.sub_texts}")
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
        return start + VOCAL_RATE  # Delay in seconds for getting correct cropped image

    def image_filename(self, start: str):
        name = int(self.cropping_start_time(start) * 1000)
        return os.path.join(self.screenshot_dir, f"{name:010}.jpg")

    def crop_images(self, timecodes: [Timecode]):
        chunk_size = 1
        tt = [tc.start for tc in timecodes]
        chunks = [tt[i:i + chunk_size] for i in range(0, len(tt), chunk_size)]

        perf_start = time.perf_counter()
        self.logger.info(f"Cropping images base on timecodes by batch size {chunk_size} ...")
        with multiprocessing.Pool() as pool:
            img_files = pool.map(self.crop_image_by_batch, chunks)
        img_files = [file for chunk_file in img_files for file in chunk_file]
        self.logger.info(f"Cropped images base on timecodes with duration: {run_duration(perf_start)}")

        return img_files

    def crop_image_by_batch(self, ss: [str]):
        out_files = []
        cmd = ["ffmpeg"]
        if self.use_gpu:
            cmd.extend(["-hwaccel", "cuda"])

        for s in ss:
            file = self.image_filename(s)
            out_files.append(file)
            start = str(self.cropping_start_time(s))
            # Minimize the decoding and seeking operations by using the -ss (seek) option `before` the input file
            cmd.extend(["-ss", start, "-i", self.video_file])
            cmd.extend(["-vf", f"crop={self.subtitle_box}", "-vframes", "1", "-loglevel", "error", file, "-y"])
        subprocess.run(cmd)

        return out_files

    def do_ocr(self, ocr: PaddleOCR, img_files: [str]):
        ocr_texts = []
        for idx, file in enumerate(img_files):
            try:
                result = ocr.ocr(file, cls=False)
                self.logger.debug(f"OCR Result: {result} from image: {file}")

                if result and result != [None]:
                    # [[
                    # [[[191.0, 10.0], [511.0, 10.0], [511.0, 49.0], [191.0, 49.0]], ('CRAB', 0.99381166696)],
                    # [[[586.0, 0.0], [665.0, 44.0], [650.0, 70.0], [572.0, 22.0]], ('在外面转悠', 0.9869911670684814)]
                    # ]]
                    result = list(chain.from_iterable(result))
                    # Only keep Chinese text
                    filtered = [v for v in result if re.sub(r'[^\u4e00-\u9fa5]', '', v[1][0])]
                    result = filtered if len(filtered) > 0 else result

                    if self.multi_lines_subtitle:
                        text = "".join(map(lambda v: v[1][0], result))
                    else:
                        max_len_ocr = max(result, key=lambda v: len(v[1][0]))
                        text = max_len_ocr[1][0].strip(" ·，；：．。")
                    ocr_texts.append(text)
                else:
                    ocr_texts.append("")
            except Exception as ex:
                self.logger.error(f"OCR image {file}, with error: {ex}")
        return ocr_texts

    def translate_subtitle(self):
        self.logger.info(f"Translating subtitle to {TARGET_LANGUAGES}")

        subtitles = [Subtitle(lang, [], os.path.join(self.output_dir, f"{lang}.vtt")) for lang in TARGET_LANGUAGES]

        def text_to_dict(s: str):
            rst_dict = {}
            sections = s.split('\n\n')
            for section in sections:
                lines = section.split('\n')
                language = lines[0].strip(':')
                rst_dict[language] = lines[1:]
            return rst_dict

        perf_start = time.perf_counter()
        chunk_size = 50
        chunks = [self.sub_texts[i:i + chunk_size] for i in range(0, len(self.sub_texts), chunk_size)]

        try:
            load_dotenv()
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            messages = [OPENAI_SYSTEM_MESSAGE]
            assistant_msg = None
            start = time.perf_counter()

            for idx, chuck in enumerate(chunks):
                if idx > 0 and assistant_msg:
                    messages.append(assistant_msg)

                messages.append({
                    "role": "user",
                    "content": "\n".join(chuck)
                })
                self.logger.debug(f"Request messages with: {messages}")

                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=3000,
                )
                assist_msg = response.choices[0].message
                self.logger.debug(f"Chat Response: {response}")
                self.logger.debug(f"Token usage: {response.usage}")

                for sub in subtitles:
                    sub.contents = text_to_dict(assist_msg.content)[sub.language]

                # To avoid exceed context window maximum tokens (16,385), keep the first system message and
                # the last two messages(user and assistant) as overlap for keeping chat context.
                if (idx + 1) % 5 == 0:
                    messages = [messages[0], messages[-2]]
                    self.logger.debug(f"Reset messages with: {messages}")

                # Make sure don't exceed the rate limits (60,000 TPM) and delay some duration to next around requests.
                if (idx + 1) % 20 == 0:
                    dur = time.perf_counter() - start
                    delay = max(60 - int(dur), 0)
                    self.logger.info(f"Delay with {delay} seconds ...")
                    time.sleep(delay)
                    start = time.perf_counter()

            # Write subtitles into VTT files
            for sub in subtitles:
                self.write_vtt_file(sub.contents, sub.filename)

            self.logger.info(
                f"Translation complete with VTT files in dir: {self.output_dir}, with duration: {run_duration(perf_start)}")
        except RateLimitError as ex:
            self.logger.error(f"OpenAI API rate limit with: {ex}")
        except Exception as ex:
            self.logger.error(f"Translate subtitle with error: {ex}")
            raise ex

    def generate_subtitles(self, languages: [str]):
        self.logger.info(f"Generating subtitles with WebVTT format")
        perf_start = time.perf_counter()
        subtitles = [Subtitle(lang, [], os.path.join(self.output_dir, f"{lang}.vtt")) for lang in languages]

        for sub in subtitles:
            txt_file = os.path.join(self.output_dir, f"{sub.language}.txt")
            with open(txt_file, 'r') as file:
                sub.contents = [line.strip("\n") for line in file.readlines()]
            self.write_vtt_file(sub.contents, sub.filename)

        self.logger.info(f"Generated subtitles in dir: {self.output_dir}, with duration: {run_duration(perf_start)}")

    def write_vtt_file(self, contents: [str], filename: str):
        if len(self.timecodes) == 0:
            with open(self.timecode_file, 'r') as file:
                for line in file.readlines():
                    start, end = line.strip("\n").split(" --> ")
                    self.timecodes.append(Timecode(start, end))

        with open(filename, 'w', encoding="utf-8") as file:
            file.write(WEBVTT_TEMPLATE)
            for idx, line in enumerate(contents):
                tc = self.timecodes[idx]
                file.write(f"{tc.start} --> {tc.end}\n")
                file.write(f"{line}\n\n")

    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)
        log_formatter = self.log_formatter

        log_handler = logging.StreamHandler()

        if log_formatter is None:
            log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")

        log_handler.setFormatter(log_formatter)

        if not logger.hasHandlers():
            logger.addHandler(log_handler)

        # Filter out noisy warnings from PyTorch for users who don't care about them
        if self.log_level > logging.DEBUG:
            warnings.filterwarnings("ignore")
        return logger
