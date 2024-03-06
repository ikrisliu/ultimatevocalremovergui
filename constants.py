PADDLE_DET_MODEL_DIR = "models/PaddleOCR/ch_PP-OCRv4_det_server_infer"
PADDLE_REC_MODEL_DIR = "models/PaddleOCR/ch_PP-OCRv4_rec_server_infer"

WEBVTT_TEMPLATE = """WEBVTT

STYLE
::cue {
  background-image: linear-gradient(to bottom, dimgray, lightgray);
  color: silver;
  font-size: 36px;
}

"""

SOURCE_LANGUAGE = "Chinese"
TARGET_LANGUAGES = ["English"]

OPENAI_MODEL = "gpt-3.5-turbo-1106"
OPENAI_SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are a subtitle translator. Translate Chinese into Japanese, Korean, Indonesian, Thai, " +
               "and Vietnamese. Translate line by line."
}
