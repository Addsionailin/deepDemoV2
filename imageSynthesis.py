# aliyun_image_generator.py
import os
from http import HTTPStatus
from urllib.parse import urlparse, unquote
from pathlib import PurePosixPath
import requests
from dashscope import ImageSynthesis

from dotenv import load_dotenv


class AliyunImageGenerator:
    def __init__(self, api_key=None):
        """
        初始化阿里云图像生成器

        参数:
        api_key (str): 阿里云API密钥。如果为None，则从环境变量中读取
        """
        # load_dotenv()  # 加载.env文件中的环境变量

        # self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("未提供API密钥且环境变量DASHSCOPE_API_KEY未设置")

    def generate_image(self, prompt, model="wanx2.1-t2i-turbo", size="1024*1024",
                       n=1, save_dir=".", file_prefix="generated", verbose=False):
        """
        生成图像并保存到本地

        参数:
        prompt (str): 图像描述文本
        model (str): 使用的模型，默认为"wanx2.1-t2i-turbo"
        size (str): 图像尺寸，格式为"宽*高"，默认为"1024*1024"
        n (int): 生成图像数量，默认为1
        save_dir (str): 保存图像的目录，默认为当前目录
        file_prefix (str): 文件名前缀，默认为"generated"
        verbose (bool): 是否打印详细输出，默认为False

        返回:
        list: 保存的文件路径列表
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        if verbose:
            print(f"----正在生成图像: '{prompt}'----")
            print(f"----使用模型：'{model}'")
        # 调用阿里云API
        rsp = ImageSynthesis.call(
            api_key=self.api_key,
            model=model,
            prompt=prompt,
            n=n,
            size=size
        )

        saved_files = []

        if rsp.status_code == HTTPStatus.OK:
            # 处理每个生成的图像
            for i, result in enumerate(rsp.output.results):
                # 从URL中提取文件名
                file_name = PurePosixPath(unquote(urlparse(result.url).path)).parts[-1]
                # 添加前缀并确保唯一
                unique_name = f"{file_prefix}_{i}_{file_name}"
                save_path = os.path.join(save_dir, unique_name)

                # 下载并保存图像
                with open(save_path, 'wb+') as f:
                    f.write(requests.get(result.url).content)

                saved_files.append(save_path)

                if verbose:
                    print(f"已保存图像: {save_path}")
        else:
            error_msg = f'生成失败, 状态码: {rsp.status_code}, 错误码: {rsp.code}, 消息: {rsp.message}'
            if verbose:
                print(error_msg)
            raise RuntimeError(error_msg)

        return saved_files


# 示例使用
if __name__ == "__main__":
    # 创建生成器实例
    generator = AliyunImageGenerator()

    # 生成图像
    prompt = "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"
    saved_files = generator.generate_image(
        prompt,
        model="wanx2.1-t2i-turbo",
        save_dir="../generated_images",
        file_prefix="flower_shop",
        verbose=True
    )

    print(f"成功生成 {len(saved_files)} 张图像")