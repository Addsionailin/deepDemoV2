# 上传远程临时文件(暂未使用)
import os
import sys
import json
import oss2
from pathlib import Path
import requests
from dotenv import load_dotenv


class AliyunUploader:
    def __init__(self, api_key=None):
        if getattr(sys, 'frozen', False):
            BASE_DIR = Path(sys.executable).parent
        else:
            BASE_DIR = Path(__file__).parent

        env_path = BASE_DIR / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"已从 {env_path} 加载环境变量")
        else:
            print(f"未找到 .env 文件在 {env_path}，将使用系统环境变量")
            load_dotenv()

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量或传入的 api_key")
        self.upload_token_url = "https://dashscope.aliyuncs.com/api/v1/uploads"

    def upload_file(self, file_path: str, model: str = "wanx-v1", verbose: bool = False) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 1. 获取上传凭证
        resp = requests.get(
            self.upload_token_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            params={"model": model},
        )
        resp.raise_for_status()
        result = resp.json()
        data = result.get("data")
        if not data:
            raise ValueError(f"返回数据异常: {json.dumps(result, indent=2, ensure_ascii=False)}")

        upload_host = data["upload_host"]
        upload_dir = data["upload_dir"]
        policy = data["policy"]
        signature = data["signature"]
        oss_access_key_id = data["oss_access_key_id"]

        # 2. 拼接 object_name
        file_name = os.path.basename(file_path)
        object_name = f"{upload_dir}/{file_name}"

        # 3. 表单直传
        with open(file_path, "rb") as f:
            files = {"file": f}
            form_data = {
                "key": object_name,
                "policy": policy,
                "OSSAccessKeyId": oss_access_key_id,
                "success_action_status": "200",
                "signature": signature,
                "x-oss-object-acl": data.get("x_oss_object_acl", "public-read"),
                "x-oss-forbid-overwrite": data.get("x_oss_forbid_overwrite", "false"),
            }
            r = requests.post(upload_host, data=form_data, files=files)
            if r.status_code != 200:
                raise RuntimeError(f"上传失败: {r.status_code} {r.text}")

        # 4. 生成访问 URL（注意 ACL 默认 public-read，可能不能直接公网访问）
        url = f"{upload_host}/{object_name}"

        return {
            "oss_url": f"oss://{upload_host.split('//')[1].split('.')[0]}/{object_name}",
            "url": url,
            "raw": data,
        }


if __name__ == "__main__":
    uploader = AliyunUploader()
    test_file = r"E:\PyCharm\py\deepDemoV2\tmp\cat.png"
    if os.path.exists(test_file):
        try:
            result = uploader.upload_file(test_file, model="wanx-v1", verbose=True)
            print("✅ 上传成功，返回所有信息:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # 初始化OSS客户端
            auth = oss2.Auth("your-access-key-id", "your-access-key-secret")
            bucket = oss2.Bucket(auth, "https://oss-cn-beijing.aliyuncs.com", "your-bucket-name")
            print(result['oss_url'])
            url = bucket.sign_url("GET", result['oss_url'], 3600)  # 有效期为3600秒
            print("临时访问链接：", url)
        except Exception as e:
            print("❌ 上传失败:", e)
    else:
        print(f"❌ 测试文件不存在: {test_file}")

