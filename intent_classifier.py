# intent_classifier.py
import os
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv


class IntentType(Enum):
    """意图类型枚举"""
    IMAGE_GENERATION = "image_generation"  # 生成图像
    OUTPAINTING = "outpainting"  # 图像扩展/外绘
    OTHER = "other"  # 其他意图


class IntentClassifier:
    """
    意图分类器，用于判断用户输入是否是想生成图像或进行扩图操作
    """

    def __init__(self, api_key=None):
        """
        初始化意图分类器

        参数:
        api_key (str): DeepSeek API密钥。如果为None，则从环境变量中读取
        """
        self.api_key = api_key or self._get_api_key()
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def _get_api_key(self):
        """从环境变量中获取API密钥"""
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置，请在.env文件中设置或直接传入api_key参数")
        return api_key

    def classify_intent(self, prompt: str) -> IntentType:
        """
        分类用户输入的意图

        参数:
        prompt (str): 用户输入的文本

        返回:
        IntentType: 意图类型枚举值
        """
        system_prompt = """
        你是一个意图分类助手，请判断用户输入是以下哪种意图：
        1. 生成新图像 (IMAGE_GENERATION)
        2. 对现有图像进行扩展/外绘 (OUTPAINTING)
        3. 其他意图 (OTHER)

        请只回答以下三种之一: IMAGE_GENERATION, OUTPAINTING, OTHER

        判断标准:
        - 如果用户要求创建新图像、画图、生成图片等，返回 IMAGE_GENERATION
        - 如果用户要求扩展图像、扩大画布、外绘、增加背景等，返回 OUTPAINTING
        - 其他情况返回 OTHER
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        try:
            result = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=10,
                temperature=0
            )
            reply = result.choices[0].message.content.strip().upper()

            # 映射回复到枚举类型
            if "IMAGE_GENERATION" in reply:
                return IntentType.IMAGE_GENERATION
            elif "OUTPAINTING" in reply:
                return IntentType.OUTPAINTING
            else:
                return IntentType.OTHER

        except Exception as e:
            print(f"意图识别失败: {e}")
            # 如果识别失败，默认返回OTHER
            return IntentType.OTHER

    def is_image_prompt(self, prompt: str) -> bool:
        """
        判断用户输入是否是想生成图像（兼容旧版本）

        参数:
        prompt (str): 用户输入的文本

        返回:
        bool: 如果是图像提示(生成或扩图)返回True，否则返回False
        """
        intent = self.classify_intent(prompt)
        return intent in [IntentType.IMAGE_GENERATION, IntentType.OUTPAINTING]

    def is_outpainting_prompt(self, prompt: str) -> bool:
        """
        判断用户输入是否是想进行扩图操作

        参数:
        prompt (str): 用户输入的文本

        返回:
        bool: 如果是扩图提示返回True，否则返回False
        """
        intent = self.classify_intent(prompt)
        return intent == IntentType.OUTPAINTING


if __name__ == "__main__":
    import sys

    # 创建分类器实例
    classifier = IntentClassifier(api_key="your_api_key_here")

    # 从命令行参数获取问题或使用默认问题
    test_prompts = [
        "画一张美丽的风景画",
        "扩展这张图像的右侧",
        "帮我把这张照片的背景扩大一些",
        "今天的天气真好",
        "给这张图片增加更多的天空部分"
    ]

    if len(sys.argv) > 1:
        test_prompts = [sys.argv[1]]

    for prompt in test_prompts:
        print(f"问题: '{prompt}'")
        intent = classifier.classify_intent(prompt)
        print(f"意图类型: {intent}")
        print(f"是图像提示: {classifier.is_image_prompt(prompt)}")
        print(f"是扩图提示: {classifier.is_outpainting_prompt(prompt)}")
        print("---")