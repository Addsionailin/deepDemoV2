import base64
import uuid
from pathlib import Path
from typing import Union, Tuple
from io import BytesIO
from PIL import Image


class Base64ImageProcessor:
    """Base64 图片处理工具类"""

    # 支持的图片格式及其对应的MIME类型
    SUPPORTED_FORMATS = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff'
    }

    @classmethod
    def image_to_base64(cls, image_input: Union[str, Path, BytesIO, Image.Image],
                        output_format: str = None) -> str:
        """
        将图片转换为Base64编码的字符串

        参数:
            image_input: 图片输入，可以是文件路径、BytesIO对象或PIL Image对象
            output_format: 输出格式（如'png', 'jpeg'），默认为原格式或PNG

        返回:
            Base64编码的图片字符串
        """
        # 处理不同类型的输入
        if isinstance(image_input, (str, Path)):
            # 从文件路径加载
            image_path = Path(image_input)
            if not image_path.exists():
                raise FileNotFoundError(f"图片文件不存在: {image_path}")

            # 确定MIME类型
            ext = image_path.suffix.lower()
            if ext not in cls.SUPPORTED_FORMATS:
                raise ValueError(f"不支持的图片格式: {ext}。支持格式: {list(cls.SUPPORTED_FORMATS.keys())}")

            mime_type = cls.SUPPORTED_FORMATS[ext]

            # 读取文件并编码
            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

        elif isinstance(image_input, BytesIO):
            # 从BytesIO对象处理
            pil_image = Image.open(image_input)
            return cls.pil_to_base64(pil_image, output_format)

        elif isinstance(image_input, Image.Image):
            # 从PIL Image对象处理
            return cls.pil_to_base64(image_input, output_format)

        else:
            raise TypeError("不支持的输入类型，支持类型: 文件路径、BytesIO或PIL Image")

        return f"data:{mime_type};base64,{base64_data}"

    @classmethod
    def pil_to_base64(cls, pil_image: Image.Image, output_format: str = None) -> str:
        """
        将PIL Image对象转换为Base64字符串

        参数:
            pil_image: PIL Image对象
            output_format: 输出格式（如'png', 'jpeg'），默认为原格式或PNG

        返回:
            Base64编码的图片字符串
        """
        if output_format is None:
            # 尝试获取原格式，如果没有则使用PNG
            output_format = pil_image.format.lower() if pil_image.format else 'png'

        # 确保格式有效
        if output_format not in [ext.lstrip('.') for ext in cls.SUPPORTED_FORMATS.keys()]:
            output_format = 'png'

        # 将PIL图像保存到BytesIO缓冲区
        buffer = BytesIO()
        pil_image.save(buffer, format=output_format)

        # 获取Base64编码
        base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        mime_type = cls.SUPPORTED_FORMATS.get(f'.{output_format}', 'image/png')

        return f"data:{mime_type};base64,{base64_data}"

    @classmethod
    def base64_to_image(cls, base64_str: str) -> Image.Image:
        """
        将Base64字符串转换为PIL Image对象

        参数:
            base64_str: Base64编码的图片字符串

        返回:
            PIL Image对象
        """
        # 检查并去除可能的数据URI前缀
        if base64_str.startswith('data:'):
            base64_str = base64_str.split(',', 1)[1]

        # 解码Base64字符串
        image_data = base64.b64decode(base64_str)

        # 转换为PIL Image
        return Image.open(BytesIO(image_data))

    @classmethod
    def base64_to_file(cls, base64_str: str, output_path: Union[str, Path] = None,
                       file_format: str = None) -> str:
        """
        将Base64字符串保存为图片文件

        参数:
            base64_str: Base64编码的图片字符串
            output_path: 输出文件路径，如果为None则自动生成
            file_format: 文件格式（如'png', 'jpeg'），默认为从Base64字符串推断或PNG

        返回:
            保存的文件路径
        """
        # 获取PIL Image对象
        image = cls.base64_to_image(base64_str)

        # 确定输出路径
        if output_path is None:
            # 自动生成文件名
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"image_{uuid.uuid4().hex[:8]}"

            # 添加文件扩展名
            if file_format:
                output_path = output_path.with_suffix(f'.{file_format}')
            else:
                # 尝试从Base64字符串获取格式
                if base64_str.startswith('data:'):
                    mime_type = base64_str.split(';')[0].split(':')[1]
                    for ext, mime in cls.SUPPORTED_FORMATS.items():
                        if mime == mime_type:
                            output_path = output_path.with_suffix(ext)
                            break
                # 默认使用PNG
                if not output_path.suffix:
                    output_path = output_path.with_suffix('.png')
        else:
            # 如果提供了output_path，确保它没有重复的后缀
            output_path = Path(output_path)
            if file_format:
                # 如果指定了文件格式，移除可能存在的后缀并添加新后缀
                output_path = output_path.with_suffix(f'.{file_format}')

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 确定保存格式
        if file_format:
            save_format = file_format
        else:
            # 从文件扩展名推断格式
            ext = output_path.suffix.lower().lstrip('.')
            save_format = ext if ext in [fmt.lstrip('.') for fmt in cls.SUPPORTED_FORMATS.keys()] else 'png'

        # 保存图片
        image.save(output_path, format=save_format)

        return str(output_path)

    @classmethod
    def get_image_info(cls, base64_str: str) -> tuple:
        """
        获取Base64图片的基本信息

        参数:
            base64_str: Base64编码的图片字符串

        返回:
            (格式, 文件大小(字节))
        """
        # 检查并去除可能的数据URI前缀
        if base64_str.startswith('data:'):
            # 提取MIME类型
            mime_type = base64_str.split(';')[0].split(':')[1]
            # 查找对应的扩展名
            for ext, mime in cls.SUPPORTED_FORMATS.items():
                if mime == mime_type:
                    format_name = ext.lstrip('.')
                    break
            else:
                format_name = 'unknown'

            # 获取Base64数据部分
            base64_data = base64_str.split(',', 1)[1]
        else:
            format_name = 'unknown'
            base64_data = base64_str

        # 计算文件大小 (Base64编码的数据比原始数据大约大33%)
        base64_size = len(base64_data)
        # 近似计算原始文件大小
        original_size = int(base64_size * 0.75)

        return format_name, original_size


# 使用示例
if __name__ == "__main__":
    image_path = "E:/PyCharm/py/deepDemoV2/tmp/cat.png"

    # 示例1: 将图片文件转换为Base64
    base64_str = Base64ImageProcessor.image_to_base64(image_path)
    print(f"Base64字符串长度: {len(base64_str)}")

    # 示例2: 获取图片信息
    format_name, size = Base64ImageProcessor.get_image_info(base64_str)
    print(f"图片格式: {format_name}, 大小: {size} 字节")

    # 示例3: 将Base64保存为文件（修复重复后缀问题）
    output_path = Base64ImageProcessor.base64_to_file(base64_str, "../outpaint_results/image_test.png", "png")
    print(f"图片已保存到: {output_path}")  # 应该输出: ../outpaint_results/image_test.jpg

    # 示例4: 使用PIL Image对象
    from PIL import Image

    pil_image = Image.open(image_path)
    base64_from_pil = Base64ImageProcessor.pil_to_base64(pil_image, "png")
    print(f"从PIL对象生成的Base64长度: {len(base64_from_pil)}")