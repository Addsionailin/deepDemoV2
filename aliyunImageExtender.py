import dashscope
import uuid
from pathlib import Path
from PIL import Image
import base64
import io
import os


class AliyunImageExtender:
    def __init__(self, api_key: str):
        """
        初始化阿里云图像扩展

        参数:
        api_key (str): 阿里云API密钥。如果为None，则从环境变量中读取
        """
        dashscope.api_key = api_key

    def _image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def extend_image(
        self,
        image_path: str,
        prompt: str,
        save_dir: str = "./extended",
        model: str = "wanx2.1-inpainting",  # 图像扩图模型
        size: str = "1024x1024",
        outpaint_direction: str = "all",  # 可选：left/right/top/bottom/all
    ):
        os.makedirs(save_dir, exist_ok=True)
        image_b64 = self._image_to_base64(image_path)

        response = dashscope.ImageSynthesis.call(
            model=model,
            prompt=prompt,
            image=image_b64,
            parameters={
                "n": 1,
                "size": size,
                "style": "photographic",
                "outpaint": {
                    "direction": outpaint_direction
                }
            }
        )

        if response and hasattr(response, "output") and response.output.get("images"):
            img_url = response.output["images"][0]
            filename = Path(save_dir) / f"extended_{uuid.uuid4().hex[:8]}.png"
            dashscope.utils.download_file(img_url, str(filename))
            return str(filename)
        else:
            raise RuntimeError(f"扩图失败: {response}")
