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
# 第一部分：Free Dictionary API —— 单词查询 + 中文释义
# ============================================================

# 内置常用词中文词典
_CN_DICT = {
    "hello": "你好；打招呼用语",
    "world": "世界；地球",
    "apple": "苹果；苹果公司",
    "beautiful": "美丽的；漂亮的",
    "different": "不同的；有差异的",
    "important": "重要的；有重大影响的",
    "student": "学生；学习者",
    "teacher": "教师；老师",
    "family": "家庭；家族",
    "friend": "朋友；友人",
    "music": "音乐；乐曲",
    "computer": "计算机；电脑",
    "language": "语言；语言文字",
    "breakfast": "早餐；早饭",
    "adventure": "冒险；奇遇",
    "chocolate": "巧克力；巧克力糖",
    "elephant": "大象",
    "guitar": "吉他；六弦琴",
    "hospital": "医院",
    "kitchen": "厨房",
    "library": "图书馆；藏书室",
    "mountain": "山；山脉",
    "ocean": "海洋；大洋",
    "piano": "钢琴",
    "rainbow": "彩虹",
    "sunshine": "阳光；日光",
    "telephone": "电话；电话机",
    "umbrella": "雨伞；保护伞",
    "village": "村庄；乡村",
    "weather": "天气；气象",
    "yesterday": "昨天",
    "animal": "动物；兽",
    "basketball": "篮球；篮球运动",
    "camera": "照相机；摄影机",
    "diamond": "钻石；金刚石",
    "english": "英语；英文",
    "flower": "花；花卉",
    "garden": "花园；菜园",
    "holiday": "假日；节日",
    "internet": "互联网；因特网",
    "journey": "旅行；旅程",
    "knowledge": "知识；学识",
    "morning": "早晨；上午",
    "notebook": "笔记本；笔记本电脑",
    "orange": "橙子；橙色",
    "picture": "图片；照片",
    "question": "问题；疑问",
    "restaurant": "餐厅；饭店",
    "sandwich": "三明治",
    "tomorrow": "明天",
    "university": "大学；综合性大学",
    "vacation": "假期；休假",
    "window": "窗户；窗口",
    "afternoon": "下午；午后",
    "birthday": "生日；诞辰",
    "dictionary": "词典；字典",
    "exercise": "练习；锻炼",
    "favorite": "最喜欢的；特别喜爱的",
    "goodbye": "再见",
    "homework": "家庭作业",
    "island": "岛屿",
    "jacket": "夹克；短上衣",
    "cat": "猫",
    "dog": "狗",
    "book": "书；书籍",
    "egg": "鸡蛋；蛋",
    "fish": "鱼；鱼类",
    "goat": "山羊",
    "hat": "帽子",
    "ice": "冰；冰块",
    "jam": "果酱",
    "key": "钥匙；关键",
    "lion": "狮子",
    "map": "地图",
    "net": "网；网络",
    "owl": "猫头鹰",
    "pen": "钢笔；笔",
    "queen": "女王；王后",
    "rat": "老鼠；大鼠",
    "sun": "太阳；阳光",
    "tree": "树；树木",
    "water": "水；水域",
    "time": "时间；时代",
    "love": "爱；热爱",
    "happy": "快乐的；幸福的",
    "sad": "悲伤的；难过的",
    "big": "大的；重要的",
    "small": "小的；微小的",
    "good": "好的；优良的",
    "bad": "坏的；不好的",
    "new": "新的；新鲜的",
    "old": "老的；旧的",
    "hot": "热的；辣的",
    "cold": "冷的；寒冷的",
    "fast": "快的；迅速的",
    "slow": "慢的；缓慢的",
    "easy": "容易的；简单的",
    "difficult": "困难的；艰难的",
    "happy": "快乐的；幸福的",
    "angry": "生气的；愤怒的",
    "tired": "疲劳的；累的",
    "hungry": "饥饿的",
    "thirsty": "口渴的",
    "color": "颜色；色彩",
    "number": "数字；号码",
    "people": "人们；人民",
    "country": "国家；乡村",
    "city": "城市；都市",
    "school": "学校；学院",
    "doctor": "医生；博士",
    "police": "警察；警方",
    "money": "钱；货币",
    "market": "市场；集市",
    "airport": "机场；航空站",
    "hotel": "酒店；旅馆",
    "movie": "电影；影片",
    "party": "聚会；政党",
    "game": "游戏；比赛",
    "sport": "运动；体育",
    "science": "科学；自然科学",
    "history": "历史；历史学",
    "art": "艺术；美术",
    "peace": "和平；安宁",
    "war": "战争；斗争",
    "dream": "梦；梦想",
    "success": "成功；成就",
    "problem": "问题；难题",
    "answer": "答案；回答",
    "help": "帮助；援助",
    "work": "工作；劳动",
    "play": "玩；玩耍",
    "read": "阅读；朗读",
    "write": "写；写作",
    "speak": "说话；讲话",
    "listen": "听；倾听",
    "think": "思考；认为",
    "learn": "学习；学会",
    "teach": "教；教导",
    "run": "跑；运行",
    "walk": "走；步行",
    "eat": "吃；吃饭",
    "drink": "喝；饮",
    "sleep": "睡觉；睡眠",
    "sing": "唱歌；演唱",
    "dance": "跳舞；舞蹈",
    "swim": "游泳",
    "fly": "飞；飞行",
    "drive": "驾驶；开车",
    "buy": "买；购买",
    "sell": "卖；出售",
    "open": "打开；开放的",
    "close": "关闭；接近的",
    "begin": "开始；着手",
    "finish": "完成；结束",
    "remember": "记住；记得",
    "forget": "忘记；遗忘",
    "understand": "理解；明白",
    "believe": "相信；认为",
    "hope": "希望；期望",
    "wish": "希望；祝愿",
    "thank": "感谢；谢谢",
    "sorry": "抱歉的；遗憾的",
    "please": "请；使高兴",
    "welcome": "欢迎；受欢迎的",
    "together": "一起；共同",
}


def _translate_to_chinese(text: str) -> str:
    """使用 MyMemory 免费翻译 API 将英文翻译为中文"""
    try:
        url = "https://api.mymemory.translated.net/get"
        resp = requests.get(url, params={"q": text, "langpair": "en|zh-CN"}, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            translated = data.get("responseData", {}).get("translatedText", "")
            if translated and translated != text:
                return translated
    except Exception:
        pass
    return ""


def _get_chinese_meaning(word: str, definitions: list) -> str:
    """获取单词的中文释义，优先本地词典，其次在线翻译"""
    word_lower = word.lower()
    # 1. 优先本地词典
    if word_lower in _CN_DICT:
        return _CN_DICT[word_lower]
    # 2. 在线翻译英文释义（只翻译取第一条）
    if definitions:
        first_def = definitions[0]
        # 去掉词性前缀 "noun: " → "definition"
        if ": " in first_def:
            first_def = first_def.split(": ", 1)[1]
        translated = _translate_to_chinese(first_def)
        if translated:
            return translated
    return "暂无中文释义"


def get_word_info(word):
    """
    查询单词的英文音标、中文释义和发音音频链接

    参数:
        word (str): 要查询的英文单词

    返回:
        dict: 包含音标、中文释义、英文释义、音频链接等信息的字典，失败时返回 None
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

            # 获取中文释义
            cn_meaning = _get_chinese_meaning(word, definitions)

            return {
                'word': word,
                'phonetic': phonetic_text or 'N/A',
                'audio_url': audio_url or 'N/A',
                'definitions': definitions,      # 英文释义
                'definitions_cn': cn_meaning,    # 中文释义
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