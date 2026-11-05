import regex as re
from typing import Dict

from textnorm._common import remove_paralinguistic_tags

def normalize(text: str) -> str:
    """
    埃及阿拉伯语文本规范化函数
    
    功能：
    1. 标准化alef的各种变体
    2. 统一处理hamza符号
    3. 将西方数字转换为阿拉伯数字
    4. 清理多余空格和标点符号
    5. 其他埃及方言特有的规范化处理
    
    参数：
    text: 输入的埃及阿拉伯语文本
        
    返回：
    规范化后的埃及阿拉伯语文本
    
    示例：
    >>> normalize("إزيك يا جماعة ده 123")
    "ازيك يا جماعة دي ١٢٣"
    """
    if not text.strip():
        return text

    # 移除副语言标签和填充词
    text = remove_paralinguistic_tags(text)

    # 移除 code-switch 的拉丁字符（英文词）
    text = re.sub(r'[a-zA-Z]+', ' ', text)

    # 移除所有标点符号（Unicode 类别 P 和 S）
    text = re.sub(r'[\p{P}\p{S}]', '', text)

    # 移除波斯语/非阿拉伯字母
    text = re.sub(r'[پچڤگژڜی]', '', text)

    # 第一步：标准化阿拉伯字母变体
    
    # Alef的各种变体映射为标准alef
    ALEF_VARIANTS = {
        'أ': 'ا',  # 带hamza的alef
        'إ': 'ا',  # 下方带hamza的alef
        'آ': 'ا',  # alef上面带madda
        'ٱ': 'ا',  # 波浪形alef
    }
    
    # Hamza的各种变体标准化
    HAMZA_VARIANTS = {
        'ء': 'ئ',  # 独立hamza转为字母上的hamza
        'ؤ': 'و',  # waw上的hamza转为普通waw
    }
    
    # 应用alef标准化
    for variant, standard in ALEF_VARIANTS.items():
        text = text.replace(variant, standard)
    
    # 应用hamza标准化
    for variant, standard in HAMZA_VARIANTS.items():
        text = text.replace(variant, standard)

    # 第二步：东方阿拉伯数字 → 西方阿拉伯数字
    eastern_to_western = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    for e_num, w_num in eastern_to_western.items():
        text = text.replace(e_num, w_num)
    
    # 第三步：埃及方言特定处理
    
    # 常见埃及方言词汇标准化
    egyptian_specific = {
        'ده': 'دي',     # 常见指示代词标准化
        'انت': 'انتِ',  # 阳性形式修正
        'انتا': 'انتَ', # 阴性形式修正
        'ايه': 'اي',    # 常见疑问词简化
        'مش': 'موش',    # 否定词标准化
        'عايز': 'عاوز', # 想要的不同说法标准化
        'عوز': 'عاوز',  # 
    }
    
    # 应用埃及方言词汇替换
    words = text.split()
    normalized_words = []
    for word in words:
        # 检查是否为埃及方言词汇
        lowered = word.lower()
        if lowered in egyptian_specific:
            normalized_words.append(egyptian_specific[lowered])
        else:
            normalized_words.append(word)
    
    text = ' '.join(normalized_words)
    
    # 第四步：清理文本
    
    # 移除所有变音符号（除shadda外）
    text = re.sub(r'[\u064B-\u065F]', '', text)  # Unicode范围涵盖阿拉伯语变音符号
    
    # 标准化空格和处理特殊空白字符
    text = re.sub(r'[ ]+', ' ', text)  # 多个空格替换为单个
    text = re.sub(r'[\u00A0\u1680\u2000-\u200F\u2028-\u202F\u205F\u3000\uFEFF]', ' ', text)  # 各种特殊空格处理
    # 最终清理：只保留阿拉伯字符、数字、空格
    text = re.sub(r'[^؀-ۿ\s0-9]', '', text)

    text = text.strip()
    
    return text


def get_normalizer(language_code: str):
    """
    获取规范化函数的工厂方法
    
    参数：
    language_code: ISO 639-3语言代码(如'ARE'表示埃及阿拉伯语)
        
    返回：
    对应语言的规范化函数
        
    异常：
    ValueError: 当传入不支持的语言代码时抛出
    """
    if language_code.upper() == 'EGY':
        return normalize
    else:
        raise ValueError(f"不支持的语言代码: {language_code}. 目前仅支持'EGY'(埃及语)")
