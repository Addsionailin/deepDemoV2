import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime
from openai import OpenAI
from langchain_openai import ChatOpenAI
import threading
import sys
from pathlib import Path
import os
from PIL import Image, ImageTk

from imageSynthesis import AliyunImageGenerator
from intent_classifier import IntentClassifier, IntentType
from aliyunImageExtender import AliyunImageExtender

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

generated_images_dir = BASE_DIR / "generated_images"
os.makedirs(generated_images_dir, exist_ok=True)

api_key_path = BASE_DIR / ".apikey"

# 全局API Key变量
openai_key = ""
dashscope_key = ""


# 应用状态管理类
class AppState:
    """应用状态管理类"""

    def __init__(self):
        self.last_generated_image = None
        self.extender = None
        self.generator = None

    def init_services(self, dashscope_key):
        """初始化图像服务"""
        try:
            self.generator = AliyunImageGenerator(api_key=dashscope_key)
            self.extender = AliyunImageExtender(api_key=dashscope_key)
        except Exception as e:
            print(f"服务初始化失败: {e}")
            self.generator = None
            self.extender = None


# 创建全局应用状态实例
app_state = AppState()


def load_api_keys():
    global openai_key, dashscope_key
    try:
        with open(api_key_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
            openai_key = lines[0] if len(lines) > 0 else ""
            dashscope_key = lines[1] if len(lines) > 1 else ""
    except FileNotFoundError:
        openai_key = ""
        dashscope_key = ""


def save_api_keys(openai_val: str, dashscope_val: str):
    with open(api_key_path, "w", encoding="utf-8") as f:
        f.write(f"{openai_val.strip()}\n{dashscope_val.strip()}")
    load_api_keys()  # 保存后立即重新加载
    init_services()  # 重新初始化服务


def init_services():
    """初始化图像服务"""
    global app_state
    app_state.init_services(dashscope_key)


load_api_keys()

# 确保生成器实例延迟初始化
llm = None


def init_llm():
    global llm
    if not openai_key.strip():
        print("未设置 OpenAI Key，跳过 LLM 初始化")
        llm = None
        return
    try:
        llm = ChatOpenAI(
            base_url='https://api.deepseek.com/v1',
            model='deepseek-chat',
            openai_api_key=openai_key
        )
    except Exception as e:
        messagebox.showerror("错误", f"LLM 初始化失败: {e}")
        llm = None


init_llm()
init_services()  # 初始化图像服务


def save_to_file(file, content, is_question=False):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if is_question:
        file.write(f"\n[{timestamp}] Question:\n{content}\n\n[{timestamp}] Answer:\n")
    else:
        file.write(content)


def send_query_to_openai(query, conversation_history, file, chat_history):
    client = OpenAI(api_key=openai_key, base_url="https://api.deepseek.com")
    conversation_history.append({"role": "user", "content": query})
    save_to_file(file, query, is_question=True)
    try:
        response1 = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "你是一个助手"}] + conversation_history,
            max_tokens=1024,
            temperature=0.7,
            stream=True
        )
        answer = ""
        for chunk in response1:
            content = chunk.choices[0].delta.content
            if content:
                print(content)
                answer += content
                chat_history.insert(tk.END, f"{content}")
                chat_history.yview(tk.END)
                chat_history.update()
        save_to_file(file, answer)
        conversation_history.append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        try:
            error_json = e.response.json()
            message = error_json.get("error", {}).get("message", str(e))
        except AttributeError:
            message = str(e)
        error_msg = f"请求错误: {message}\n"

        chat_history.insert(tk.END, message)
        chat_history.yview(tk.END)
        chat_history.update()

        print(error_msg)
        save_to_file(file, error_msg)
        return error_msg


def open_image(prompt, chat_history, model_name):
    try:
        if app_state.generator is None:
            chat_history.insert(tk.END, "图像生成功能未初始化，请检查API密钥设置\n\n")
            chat_history.yview(tk.END)
            return None

        saved_files = app_state.generator.generate_image(
            prompt,
            model=model_name,
            save_dir=str(generated_images_dir),
            file_prefix="image",
            verbose=True
        )
        if saved_files:
            chat_history.insert(tk.END, f"\n\n已生成图像: {prompt}\n")
            show_image_thumbnail(chat_history, saved_files[0])

            # 更新最后生成的图像路径
            app_state.last_generated_image = saved_files[0]

            def open_image_callback():
                try:
                    img = Image.open(saved_files[0])
                    img.show()
                except Exception as e:
                    messagebox.showerror("错误", f"无法打开图像: {str(e)}")

            open_btn = tk.Button(chat_history, text="打开完整图像", command=open_image_callback, bg="#4CAF50",
                                 fg="white")
            chat_history.window_create(tk.END, window=open_btn)
            chat_history.insert(tk.END, "\n\n")
            chat_history.yview(tk.END)
            return saved_files[0]
        else:
            chat_history.insert(tk.END, "\n\n图像生成失败\n\n")
            chat_history.yview(tk.END)
            return None
    except Exception as e:
        error_msg = f"图像生成过程中发生错误: {str(e)}\n"
        chat_history.insert(tk.END, error_msg)
        return None


def extend_image(prompt, image_path, chat_history):
    try:
        if app_state.extender is None:
            chat_history.insert(tk.END, "图像扩展功能未初始化，请检查API密钥设置\n\n")
            chat_history.yview(tk.END)
            return

        # 解析扩图方向
        direction = parse_outpainting_direction(prompt)

        # 执行扩图 - 修正参数名
        saved_files = app_state.extender.extend_image(
            file_path=image_path,  # 将 image_path 改为 file_path
            prompt=prompt,
            direction=direction,
            save_dir=str(generated_images_dir),
            file_prefix="extended"
        )

        if saved_files:
            chat_history.insert(tk.END, f"\n\n已扩展图像: {prompt}\n")
            show_image_thumbnail(chat_history, saved_files[0])

            # 更新最后生成的图像路径
            app_state.last_generated_image = saved_files[0]

            def open_image_callback():
                try:
                    img = Image.open(saved_files[0])
                    img.show()
                except Exception as e:
                    messagebox.showerror("错误", f"无法打开图像: {str(e)}")

            open_btn = tk.Button(chat_history, text="打开完整图像", command=open_image_callback, bg="#4CAF50",
                                 fg="white")
            chat_history.window_create(tk.END, window=open_btn)
            chat_history.insert(tk.END, "\n\n")
            chat_history.yview(tk.END)
        else:
            chat_history.insert(tk.END, "\n\n图像扩展失败\n\n")
            chat_history.yview(tk.END)
    except Exception as e:
        error_msg = f"图像扩展过程中发生错误: {str(e)}\n"
        chat_history.insert(tk.END, error_msg)
        chat_history.yview(tk.END)


def parse_outpainting_direction(prompt):
    """从提示中解析扩图方向"""
    prompt_lower = prompt.lower()

    if "左" in prompt_lower or "left" in prompt_lower:
        return "left"
    elif "右" in prompt_lower or "right" in prompt_lower:
        return "right"
    elif "上" in prompt_lower or "top" in prompt_lower:
        return "top"
    elif "下" in prompt_lower or "bottom" in prompt_lower:
        return "bottom"
    elif "四" in prompt_lower or "all" in prompt_lower or "周围" in prompt_lower:
        return "all"
    else:
        # 默认扩展所有方向
        return "all"


def show_image_thumbnail(chat_history, image_path):
    try:
        img = Image.open(image_path)
        img.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(chat_history, image=photo)
        img_label.image = photo
        chat_history.window_create(tk.END, window=img_label)
        chat_history.insert(tk.END, "\n\n")
    except Exception as e:
        error_msg = f"无法显示图像缩略图: {str(e)}\n"
        chat_history.insert(tk.END, error_msg)


def create_gui():
    root = tk.Tk()
    root.title("deepseek 查询助手")
    root.geometry("745x600")
    root.configure(bg="#f0f0f0")

    default_font = ("Microsoft YaHei", 10)
    title_font = ("Microsoft YaHei", 12, "bold")

    main_frame = tk.Frame(root, bg="#f0f0f0")
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    image_model_var = tk.StringVar(value="wanx2.1-t2i-turbo")

    model_frame = tk.Frame(main_frame, bg="#f0f0f0")
    model_frame.pack(anchor="w", pady=(0, 5))
    tk.Label(model_frame, text="图像模型:", font=default_font, bg="#f0f0f0").pack(side=tk.LEFT)
    tk.OptionMenu(model_frame, image_model_var, "wanx2.1-t2i-turbo", "wanx2.1-t2i-plus", "wanx2.0-t2i-turbo").pack(
        side=tk.LEFT)

    chat_label = tk.Label(main_frame, text="对话历史", font=title_font, bg="#f0f0f0")
    chat_label.pack(anchor="w", pady=(0, 5))

    chat_history = scrolledtext.ScrolledText(main_frame, width=80, height=20, wrap=tk.WORD, font=default_font,
                                             bg="white", relief=tk.SUNKEN)
    chat_history.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    input_frame = tk.Frame(main_frame, bg="#f0f0f0")
    input_frame.pack(fill=tk.X, pady=(5, 0))

    user_label = tk.Label(input_frame, text="问题输入:", font=default_font, bg="#f0f0f0")
    user_label.grid(row=0, column=0, sticky="w", padx=(0, 5))

    user_input = tk.Entry(input_frame, width=70, font=default_font)
    user_input.grid(row=0, column=1, sticky="ew", padx=(0, 5))

    send_button = tk.Button(input_frame, text="发送", width=10, font=default_font, bg="#4CAF50", fg="white")
    send_button.grid(row=0, column=2)

    def open_setting_window():
        setting_win = tk.Toplevel()
        setting_win.title("设置 API Key")
        setting_win.geometry("400x200")
        tk.Label(setting_win, text="DeepSeek API Key:").pack(pady=5)
        openai_entry = tk.Entry(setting_win, width=50)
        openai_entry.insert(0, openai_key)
        openai_entry.pack()
        tk.Label(setting_win, text="DashScope API Key:").pack(pady=5)
        dashscope_entry = tk.Entry(setting_win, width=50)
        dashscope_entry.insert(0, dashscope_key)
        dashscope_entry.pack()

        def save_key():
            save_api_keys(openai_entry.get(), dashscope_entry.get())
            init_llm()
            messagebox.showinfo("保存成功", "API Key 已保存并应用")
            setting_win.destroy()

        tk.Button(setting_win, text="保存", command=save_key).pack(pady=10)

    setting_button = tk.Button(main_frame, text="设置", width=10, font=default_font, bg="#FF9800", fg="white",
                               command=open_setting_window)
    setting_button.pack(pady=(5, 10))

    with open("docling.txt", "a", encoding="utf-8") as file:
        conversation_history = []

        def on_send_button_click():
            query = user_input.get().strip()
            if query:
                chat_history.insert(tk.END, f"\n您: {query}\n\n")
                chat_history.yview(tk.END)
                user_input.delete(0, tk.END)
                save_to_file(file, query, is_question=True)
                # 判断意图
                intent = IntentClassifier(api_key=openai_key)
                intentEnum = intent.classify_intent(query)

                if intentEnum == IntentType.IMAGE_GENERATION:
                    chat_history.insert(tk.END, "助手: 正在为您生成图像...\n\n")
                    chat_history.yview(tk.END)
                    chat_history.update()
                    model_name = image_model_var.get()  # 获取下拉框选中的模型名
                    threading.Thread(target=lambda: open_image(query, chat_history, model_name)).start()

                elif intentEnum == IntentType.OUTPAINTING:
                    chat_history.insert(tk.END, "助手: 正在扩展图片...\n\n")
                    chat_history.yview(tk.END)
                    chat_history.update()

                    # 检查是否有最近生成的图像
                    if app_state.last_generated_image is None or not os.path.exists(app_state.last_generated_image):
                        chat_history.insert(tk.END, "助手: 未找到可扩展的图像，请先生成一张图像。\n\n")
                        chat_history.yview(tk.END)
                        return

                    # 使用线程执行扩图操作
                    threading.Thread(target=lambda: extend_image(
                        query, app_state.last_generated_image, chat_history
                    )).start()
                    # threading.Thread(target=lambda: extend_image(
                    #     query, "E:/PyCharm/py/deepDemoV2/tmp/cat.png", chat_history
                    # )).start()

                elif intentEnum == IntentType.OTHER:
                    chat_history.insert(tk.END, "助手: \n\n")
                    chat_history.yview(tk.END)
                    chat_history.update()
                    send_query_to_openai(query, conversation_history, file, chat_history)

        send_button.config(command=on_send_button_click)
        user_input.bind("<Return>", lambda event: on_send_button_click())

        chat_history.insert(tk.END, "欢迎使用 deepseek 查询助手！\n您可以输入问题、生成图像或设置 API Key。\n\n")
        chat_history.yview(tk.END)

        root.mainloop()


if __name__ == "__main__":
    create_gui()