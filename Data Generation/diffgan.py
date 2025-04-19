import re


def remove_linebreaks(text):
    """
    去除所有换行符，并将多个空格合并成一个空格
    """
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def split_text_by_period(text, min_words=10):
    """
    按句号分割文本，再将相邻句子合并，
    使得每个段落至少包含 min_words 个词，且尽可能短，保持句子完整。
    """
    # 按句号分割，并过滤掉空字符串
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    segments = []
    current_segment = ""
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        # 如果当前没有累积句子
        if current_segment == "":
            # 如果单个句子本身就满足要求，直接作为一段（加上句号）
            if sentence_word_count >= min_words:
                segments.append(sentence + ".")
            else:
                current_segment = sentence
                current_word_count = sentence_word_count
        else:
            # 尝试合并当前句子
            tentative_segment = current_segment + ". " + sentence
            tentative_word_count = current_word_count + sentence_word_count
            if tentative_word_count < min_words:
                # 还未达到阈值，继续合并
                current_segment = tentative_segment
                current_word_count = tentative_word_count
            else:
                # 合并后达到或超过阈值，则作为一个段落输出
                segments.append(tentative_segment + ".")
                current_segment = ""
                current_word_count = 0

    # 如果最后还有剩余句子，则也作为一段输出
    if current_segment:
        segments.append(current_segment + ".")
    return segments


def generate_commands(segments):
    """
    对每个段落生成对应的命令，
    将段落中的文本替换到命令的 --text 参数中
    """
    commands = []
    for seg in segments:
        # 转义双引号，防止命令中引号冲突
        escaped_seg = seg.replace('"', r'\"')
        command = f'python3 synthesize.py --text "{escaped_seg}" --model naive --restore_step 200000 --mode single --dataset LJSpeech'
        commands.append(command)
    return commands


if __name__ == '__main__':
    # 1. 读取 TXT 文件
    # 请确保 article.txt 文件与此脚本在同一目录下，或修改为正确的路径
    with open("text.txt", "r", encoding="utf-8") as f:
        article = f.read()

    # 2. 去掉所有换行符并合并空格
    cleaned_text = remove_linebreaks(article)

    # 3. 按句号分割并合并句子，保证每个段落至少 20 个词
    segments = split_text_by_period(cleaned_text, min_words=10)

    # 4. 为每个段落生成对应的命令
    commands = generate_commands(segments)

    # 将生成的命令写入 commands.sh 文件中
    with open("commands.sh", "w", encoding="utf-8") as f:
        for cmd in commands:
            f.write(cmd + "\n")

    print("已生成", len(commands), "条命令，保存在 commands.sh 文件中。")
