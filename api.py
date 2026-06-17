"""
SpeakEasy API 集成模块
======================
集成了以下 API 功能：
  1. Free Dictionary API —— 查询单词的音标、释义、标准发音音频
  2. 有道智云语音评测 API —— 对用户发音进行评分
  3. DeepSeek API —— 结合评分结果给出改进建议
"""

import sys
import uuid
import time
import json
import base64
import hashlib
import io
import wave

import requests
import numpy as np

# ============================================================
# 第一部分：Free Dictionary API —— 单词查询
# ============================================================

def get_word_info(word):
    """
    调用 Free Dictionary API 获取单词的释义、音标和发音音频链接

    参数:
        word (str): 要查询的英文单词

    返回:
        dict: 包含音标、释义、音频链接等信息的字典，失败时返回 None
    """
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            word_data = data[0]

            phonetics = word_data.get('phonetics', [])
            phonetic_text = None
            audio_url = None

            for p in phonetics:
                if p.get('audio'):
                    audio_url = p['audio']
                    if p.get('text'):
                        phonetic_text = p['text']
                    break

            if not phonetic_text and phonetics:
                phonetic_text = phonetics[0].get('text', 'N/A')

            meanings = word_data.get('meanings', [])
            definitions = []
            for meaning in meanings[:3]:
                part_of_speech = meaning.get('partOfSpeech', '')
                first_def = meaning.get('definitions', [{}])[0].get('definition', '')
                if first_def:
                    definitions.append(f"{part_of_speech}: {first_def}")

            return {
                'word': word,
                'phonetic': phonetic_text or 'N/A',
                'audio_url': audio_url or 'N/A',
                'definitions': definitions,
            }
        else:
            return None

    except requests.exceptions.RequestException as e:
        print(f"网络请求出错: {e}")
        return None


# ============================================================
# 第二部分：有道智云语音评测 API —— 发音评分
# ============================================================

YOUDAO_URL = 'https://openapi.youdao.com/iseapi'
APP_KEY = '4d04588dd0fe5ead'
APP_SECRET = 'zgCQhigaegsYRRzpcxSkEFeF9VMdg6ts'


def _truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def _encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """将 numpy 音频数组转换为 WAV 格式的 bytes"""
    audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    buf.seek(0)
    return buf.read()


def score_pronunciation_youdao(audio: np.ndarray, text: str,
                               sample_rate: int = 16000,
                               lang_type: str = 'en') -> dict:
    """
    调用有道智云语音评测 API 对用户发音进行评分

    参数:
        audio:    用户录音的 numpy 数组 (float32, 范围 [-1.0, 1.0])
        text:     目标文本（例如 "apple" 或 "A"）
        sample_rate: 采样率（默认 16000）
        lang_type:   语言类型 'en' (英文) 或 'zh-CHS' (中文)

    返回:
        dict: {
            "errorCode": str,        # "0" 表示成功
            "overall": float,        # 综合评分 0-100
            "pronunciation": float,  # 发音准确度 0-100
            "fluency": float,        # 流利度 0-100
            "speed": float,          # 语速 (词/分钟)
            "words": list,           # 词级别详情
            "suggestions": list,     # 基于分数的改进建议
        }
    """
    wav_bytes = _numpy_to_wav_bytes(audio, sample_rate)
    q = base64.b64encode(wav_bytes).decode('utf-8')

    data = {
        'text': text,
        'q': q,
        'langType': lang_type,
        'rate': sample_rate,
        'format': 'wav',
        'channel': 1,
        'type': 1,  # 1 = 英文评测
    }

    curtime = str(int(time.time()))
    salt = str(uuid.uuid1())
    data['curtime'] = curtime
    data['salt'] = salt
    data['appKey'] = APP_KEY
    data['sign'] = _encrypt(APP_KEY + _truncate(q) + salt + curtime + APP_SECRET)
    data['signType'] = "v2"

    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(YOUDAO_URL, data=data, headers=headers, timeout=15)

    result = json.loads(response.content.decode("utf-8"))

    suggestions = []
    if result.get("errorCode") == "0":
        for word_item in result.get("words", []):
            for ph in word_item.get("phonemes", []):
                score = ph["pronunciation"]
                phoneme = ph["phoneme"]
                if score < 60:
                    suggestions.append(
                        f'音素 "{phoneme}" 得分 {score:.0f}，需要重点练习')
                elif score < 75:
                    suggestions.append(
                        f'音素 "{phoneme}" 得分 {score:.0f}，还需加强')

        if result.get("pronunciation", 100) < 60:
            suggestions.append("整体发音偏差较大，建议多听标准发音后跟读")
        elif result.get("pronunciation", 100) < 75:
            suggestions.append("发音基本正确，可以更清晰地发每个音节")

        if result.get("fluency", 100) < 70:
            suggestions.append("朗读不够流利，多练习连贯朗读")
        if result.get("speed", 120) > 180:
            suggestions.append("语速偏快，试着放慢一些")
        elif result.get("speed", 120) < 60:
            suggestions.append("语速偏慢，可以更自然一些")

        if not suggestions:
            suggestions.append("✨ 发音很好，继续保持！")

    result["suggestions"] = suggestions

    overall = result.get("overall", 0)
    pronunciation = result.get("pronunciation", 0)
    fluency = result.get("fluency", 0)
    speed = result.get("speed", 0)

    if isinstance(overall, (int, float)):
        result["overall"] = float(overall)
    if isinstance(pronunciation, (int, float)):
        result["pronunciation"] = float(pronunciation)
    if isinstance(fluency, (int, float)):
        result["fluency"] = float(fluency)
    if isinstance(speed, (int, float)):
        result["speed"] = float(speed)

    return result


# ============================================================
# 第三部分：DeepSeek API —— 智能改进建议
# ============================================================

DEEPSEEK_API_KEY = 'sk-a241a3c7a1204d06984bf35e78d4ba32'
DEEPSEEK_API_URL = 'https://api.deepseek.com/chat/completions'


def get_deepseek_suggestions(word: str, phonetic: str, definitions: list,
                             youdao_result: dict) -> str:
    """
    将 Youdao 评分结果上传给 DeepSeek，获取智能改进建议

    参数:
        word:           目标单词
        phonetic:       音标
        definitions:    释义列表
        youdao_result:  有道评测结果字典

    返回:
        str: DeepSeek 返回的改进建议文本（失败时返回空字符串）
    """
    # 构建评分摘要
    error_code = youdao_result.get("errorCode", "-1")
    if error_code != "0":
        return ""

    overall = youdao_result.get("overall", 0)
    pronunciation = youdao_result.get("pronunciation", 0)
    fluency = youdao_result.get("fluency", 0)
    speed = youdao_result.get("speed", 0)

    # 收集低分音素
    weak_phonemes = []
    for word_item in youdao_result.get("words", []):
        for ph in word_item.get("phonemes", []):
            score = ph.get("pronunciation", 0)
            phoneme = ph.get("phoneme", "")
            if score < 75:
                weak_phonemes.append(
                    {"phoneme": phoneme, "score": round(float(score), 1)})

    weak_phonemes_str = json.dumps(weak_phonemes, ensure_ascii=False) if weak_phonemes else "无"

    prompt = f"""你是一位专业的英语发音教练。请根据以下信息，用中文给出具体的、可操作的发音改进建议。

**目标单词**: {word}
**音标**: {phonetic}
**释义**: {', '.join(definitions) if definitions else 'N/A'}

**用户发音评分（有道智云评测）**:
- 综合评分: {overall:.0f}/100
- 发音准确度: {pronunciation:.0f}/100
- 流利度: {fluency:.0f}/100
- 语速: {speed:.0f} 词/分钟

**低分音素（需改进）**: {weak_phonemes_str}

请给出：
1. 总体评价（1-2句）
2. 针对低分音素的具体练习方法（如果存在）
3. 整体发音改进技巧（嘴型、舌位、重音、连读等）
4. 推荐的练习步骤

请用中文回复，语言要友好、鼓励性，不要其他格式的文本，尽量最简单的txt（不要有加粗）返回，控制在200字以内。"""

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
    }

    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system',
             'content': '你是一位专业的英语发音教练，擅长用中文给出具体的发音改进建议。'},
            {'role': 'user', 'content': prompt},
        ],
        'max_tokens': 500,
        'temperature': 0.7,
    }

    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers,
                             json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            print(f"DeepSeek API 请求失败: {resp.status_code} {resp.text}")
            return ""
    except Exception as e:
        print(f"DeepSeek API 异常: {e}")
        return ""


# ============================================================
# 第四部分：综合评分接口（融合有道 + DeepSeek）
# ============================================================

def comprehensive_score(audio: np.ndarray, word: str,
                        sample_rate: int = 16000) -> dict:
    """
    一站式综合评分：查词 → 有道评分 → DeepSeek 建议

    参数:
        audio:       用户录音 numpy 数组
        word:        目标单词
        sample_rate: 采样率

    返回:
        dict: {
            "word_info": {...},        # 单词信息（音标、释义等）
            "youdao_score": {...},     # 有道评分结果
            "deepseek_advice": str,    # DeepSeek 改进建议
        }
    """
    word_info = get_word_info(word)
    youdao_result = score_pronunciation_youdao(audio, word, sample_rate)

    phonetic = word_info.get('phonetic', 'N/A') if word_info else 'N/A'
    definitions = word_info.get('definitions', []) if word_info else []

    deepseek_advice = ""
    if youdao_result.get("errorCode") == "0":
        deepseek_advice = get_deepseek_suggestions(
            word, phonetic, definitions, youdao_result)

    return {
        "word_info": word_info,
        "youdao_score": youdao_result,
        "deepseek_advice": deepseek_advice,
    }


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    # 测试单词查询
    word = "different"
    result = get_word_info(word)

    if result:
        print(f"📖 单词: {result['word']}")
        print(f"🔊 音标: {result['phonetic']}")
        print(f"🎵 发音音频链接: {result['audio_url']}")
        print("📝 释义:")
        for idx, definition in enumerate(result['definitions'], 1):
            print(f"   {idx}. {definition}")
    else:
        print(f"未找到单词 '{word}' 的释义。")

    # 测试有道评分（需要实际音频文件）
    print("\n--- 有道评分测试 ---")
    test_audio_path = 'data/templates/audio/b.wav'
    try:
        audio_test, sr = _load_wav_for_test(test_audio_path)
        youdao_r = score_pronunciation_youdao(audio_test, 'B', sr)
        print(f"综合评分: {youdao_r.get('overall', 'N/A')}/100")
        print(f"发音准确度: {youdao_r.get('pronunciation', 'N/A')}/100")
        print(f"流利度: {youdao_r.get('fluency', 'N/A')}/100")
        print(f"语速: {youdao_r.get('speed', 'N/A')} 词/分钟")
        print("📝 改进建议：")
        for s in youdao_r.get('suggestions', []):
            print(f"  - {s}")
    except Exception as e:
        print(f"有道评分测试跳过（文件不存在或网络问题）: {e}")


def _load_wav_for_test(path: str):
    """辅助函数：加载 WAV 文件用于测试"""
    import soundfile as sf
    audio, sr = sf.read(path)
    return audio.astype(np.float32), sr
