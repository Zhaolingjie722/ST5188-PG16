import os
import re
import uuid
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

load_dotenv('password.env')
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

client = ElevenLabs(
    api_key=ELEVENLABS_API_KEY,
)


def remove_linebreaks(text):
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_lines(s: str) -> list:
    return [line.strip() for line in s.splitlines() if line.strip()]


def split_text_by_period(text, min_words=10):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    segments = []
    current_segment = ""
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_segment == "":

            if sentence_word_count >= min_words:
                segments.append(sentence + ".")
            else:
                current_segment = sentence
                current_word_count = sentence_word_count
        else:

            tentative_segment = current_segment + ". " + sentence
            tentative_word_count = current_word_count + sentence_word_count
            if tentative_word_count < min_words:

                current_segment = tentative_segment
                current_word_count = tentative_word_count
            else:

                segments.append(tentative_segment + ".")
                current_segment = ""
                current_word_count = 0

    if current_segment:
        segments.append(current_segment + ".")
    return segments


def text_to_speech_file(text: str, voice: str, save_dir: str = "audio_outputs") -> str:
    response = client.text_to_speech.convert(
        voice_id=voice,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    os.makedirs(save_dir, exist_ok=True)

    filename = f"{uuid.uuid4()}.mp3"
    save_file_path = os.path.join(save_dir, filename)

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: 音频保存成功！")
    return save_file_path


if __name__ == '__main__':
    # 1. 读取 TXT 文件
    # 请确保 article.txt 文件与此脚本在同一目录下，或修改为正确的路径
    with open("speech.txt", "r", encoding="iso-8859-1") as f:
        article = f.read()

    # 2. 去掉所有换行符并合并空格
    cleaned_text = remove_linebreaks(article)

    # 3. 按句号分割并合并句子，保证每个段落至少 20 个词
    segments = split_text_by_period(cleaned_text, min_words=20)

    # 4. 读取reader id
    with open("reader_ids.txt", "r", encoding="iso-8859-1") as f:
        reader_ids = f.read()

    ids = clean_lines(reader_ids)
    print(ids)
    for a in segments:
        for b in ids:
            text_to_speech_file(a, b)
