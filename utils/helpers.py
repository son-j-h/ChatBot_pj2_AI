def load_text(filepath: str) -> str:
    """주어진 파일 경로의 텍스트를 읽어 문자열로 반환합니다."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def load_few_shot_examples(filepath: str) -> str:
    """주어진 파일 경로의 few-shot 예제 텍스트를 읽어 문자열로 반환합니다."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
