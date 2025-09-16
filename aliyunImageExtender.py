import uuid
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Union, Dict, Any
from utils.base64ImageProcessor import Base64ImageProcessor
import requests


class AliyunImageExtender:
    def __init__(self, api_key=None):
        BASE_DIR = Path(__file__).parent
        env_path = BASE_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请先设置 DASHSCOPE_API_KEY")

        # 初始化Base64处理器
        self.image_processor = Base64ImageProcessor()

    def image_to_base64(self, file_path: Union[str, Path]) -> str:
        """使用Base64ImageProcessor将图片转换为Base64"""
        return self.image_processor.image_to_base64(file_path)

    def submit_extend_task(self, image_b64: str, prompt: str, **kwargs) -> str:
        """
        提交扩图任务到阿里云

        参数:
            image_b64: Base64编码的图片
            prompt: 扩图提示词
            **kwargs: 其他参数，包括:
                - model: 使用的模型，默认为"wanx2.1-imageedit"
                - top_scale: 顶部扩图比例，默认为1.2
                - bottom_scale: 底部扩图比例，默认为1.2
                - left_scale: 左侧扩图比例，默认为1.2
                - right_scale: 右侧扩图比例，默认为1.2
                - n: 生成图片数量，默认为1
                - seed: 随机种子，用于重现结果

        返回:
            任务ID
        """
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis"
        headers = {
            "X-DashScope-Async": "enable",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 构建请求数据
        data = {
            "model": kwargs.get("model", "wanx2.1-imageedit"),
            "input": {
                "function": "expand",
                "prompt": prompt,
                "base_image_url": image_b64
            },
            "parameters": {
                "top_scale": kwargs.get("top_scale", 1.2),
                "bottom_scale": kwargs.get("bottom_scale", 1.2),
                "left_scale": kwargs.get("left_scale", 1.2),
                "right_scale": kwargs.get("right_scale", 1.2),
                "n": kwargs.get("n", 1),
            },
        }

        # 添加可选参数
        if "seed" in kwargs:
            data["parameters"]["seed"] = kwargs["seed"]

        try:
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            resp.raise_for_status()

            result = resp.json()
            if "output" not in result or "task_id" not in result["output"]:
                raise RuntimeError(f"响应中缺少task_id: {result}")

            task_id = result["output"]["task_id"]
            print(f"任务提交成功，任务ID: {task_id}")
            return task_id

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"网络请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"响应解析失败: {str(e)}, 原始响应: {resp.text}")

    def get_task_result(self, task_id: str, save_dir: str = "extended",
                        timeout: int = 300, interval: int = 3, max_retries: int = 3) -> Dict[str, Any]:
        """
        获取任务结果

        参数:
            task_id: 任务ID
            save_dir: 保存目录
            timeout: 超时时间(秒)
            interval: 查询间隔(秒)
            max_retries: 最大重试次数

        返回:
            包含结果信息的字典
        """
        url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        os.makedirs(save_dir, exist_ok=True)
        start = time.time()
        retries = 0

        while time.time() - start < timeout:
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                result = resp.json()

                status = result.get("output", {}).get("task_status")
                print(f"任务状态: {status} (已等待 {int(time.time() - start)} 秒)")

                if status == "SUCCEEDED":
                    result_url = result["output"]["results"][0]["url"]

                    # 生成有意义的文件名
                    original_name = Path(result_url).name.split('?')[0]  # 去除URL参数
                    filename = Path(save_dir) / f"{original_name or uuid.uuid4().hex[:8]}"

                    # 下载图片
                    for i in range(max_retries):
                        try:
                            img_resp = requests.get(result_url, timeout=30)
                            img_resp.raise_for_status()

                            with open(filename, "wb") as f:
                                f.write(img_resp.content)

                            print(f"扩图完成，保存到: {filename}")

                            # 返回结果信息
                            return {
                                "local_file": str(filename),
                                "result_url": result_url,
                                "task_info": result,
                                "file_size": os.path.getsize(filename)
                            }

                        except requests.exceptions.RequestException as e:
                            if i < max_retries - 1:
                                print(f"下载失败，第 {i + 1} 次重试: {str(e)}")
                                time.sleep(2)
                            else:
                                raise RuntimeError(f"图片下载失败: {str(e)}")

                if status == "FAILED":
                    error_msg = result["output"].get("message", "未知错误")
                    raise RuntimeError(f"扩图失败: {error_msg}")

                time.sleep(interval)

            except requests.exceptions.RequestException as e:
                retries += 1
                if retries > max_retries:
                    raise RuntimeError(f"查询任务状态失败次数过多: {str(e)}")

                print(f"查询任务状态时出错: {str(e)}，将在 {interval} 秒后重试 (第 {retries} 次)")
                time.sleep(interval)

        raise TimeoutError(f"扩图超时，已等待 {timeout} 秒")

    def extend_image(self, file_path: Union[str, Path], prompt: str, **kwargs) -> Dict[str, Any]:
        """
        扩图主函数

        参数:
            file_path: 图片文件路径
            prompt: 扩图提示词
            **kwargs: 其他参数，包括:
                - save_dir: 保存目录，默认为"extended"
                - 其他submit_extend_task支持的参数

        返回:
            扩图结果信息
        """
        # 验证文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"图片文件不存在: {file_path}")

        # 转换图片为Base64
        img_b64 = self.image_to_base64(file_path)

        # 提交扩图任务
        task_id = self.submit_extend_task(img_b64, prompt, **kwargs)

        # 获取任务结果
        save_dir = kwargs.get("save_dir", "extended")
        return self.get_task_result(task_id, save_dir=save_dir)

    def batch_extend_images(self, file_paths: list, prompt: str, **kwargs) -> list:
        """
        批量扩图

        参数:
            file_paths: 图片文件路径列表
            prompt: 扩图提示词
            **kwargs: 其他参数

        返回:
            扩图结果列表
        """
        results = []
        for i, file_path in enumerate(file_paths):
            print(f"处理第 {i + 1}/{len(file_paths)} 张图片: {file_path}")
            try:
                result = self.extend_image(file_path, prompt, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"处理图片 {file_path} 时出错: {str(e)}")
                results.append({"file": file_path, "error": str(e)})

        return results


if __name__ == "__main__":
    try:
        extender = AliyunImageExtender()

        # 测试图片路径
        test_images = [
            "E:/PyCharm/py/deepDemoV2/tmp/cat.png",
        ]

        # 只处理存在的图片
        existing_images = [img for img in test_images if os.path.exists(img)]

        if not existing_images:
            print("错误: 没有找到测试图片")
            print("请创建 tmp 目录并放入测试图片")
            exit(1)

        prompt = "延展图片，保持自然过渡，扩展背景"

        # 批量扩图示例（如果有多个图片）
        if len(existing_images) > 1:
            print("\n开始批量扩图...")
            results = extender.batch_extend_images(
                existing_images,
                prompt,
                save_dir="batch_outpaint_results"
            )
            for i, result in enumerate(results):
                print(f"\n结果 {i + 1}:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # 单张图片扩图示例
            print("开始单张图片扩图...")
            result = extender.extend_image(
                existing_images[0],
                prompt,
                save_dir="outpaint_results",
                top_scale=1.2,
                bottom_scale=1.2,
                left_scale=1.5,  # 左侧扩展更多
                right_scale=1.5,  # 右侧扩展更多
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback

        traceback.print_exc()