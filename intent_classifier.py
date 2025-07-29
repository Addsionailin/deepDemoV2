# intent_classifier.py
import os
from openai import OpenAI
from dotenv import load_dotenv


class IntentClassifier:
    """
    意图分类器，用于判断用户输入是否是想生成图像
    """

    def __init__(self, api_key=None):
        """
        初始化意图分类器

        参数:
        api_key (str): DeepSeek API密钥。如果为None，则从环境变量中读取
        """
        # self.api_key = api_key or self._get_api_key()
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def _get_api_key(self):
        """从环境变量中获取API密钥"""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置，请在.env文件中设置或直接传入api_key参数")
        return api_key

    def is_image_prompt(self, prompt: str) -> bool:
        """
        判断用户输入是否是想生成图像

        参数:
        prompt (str): 用户输入的文本

        返回:
        bool: 如果是图像提示返回True，否则返回False
        """
        messages = [
            {"role": "system", "content": "你是一个分类助手，请判断用户输入是否是想要生成图像。只回答 '是' 或 '否'。"},
            {"role": "user", "content": prompt}
        ]
        try:
            result = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=5,
                temperature=0
            )
            reply = result.choices[0].message.content.strip().lower()
            return reply == "是"
        except Exception as e:
            print(f"意图识别失败: {e}")
            # 如果识别失败，默认返回False
            return False


if __name__ == "__main__":
    import sys

    # 创建分类器实例
    classifier = IntentClassifier(api_key="api_key")

    # 从命令行参数获取问题或使用默认问题
    q = sys.argv[1] if len(sys.argv) > 1 else "画一张美丽的风景画"
    print(f"问题: '{q}'")
    result = classifier.is_image_prompt(q)
    print("是图像提示" if result else "不是图像提示")