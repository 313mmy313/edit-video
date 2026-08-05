import os
import json
import tempfile
import subprocess
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from pathlib import Path
import threading
import time

# کتابخانه‌های اصلی
from faster_whisper import WhisperModel
import ffmpeg
import ollama

# =============== بخش تبدیل گفتار به متن با Whisper ===============
def transcribe_video(video_path, model_size="large-v3", device="cpu"):
    """
    تبدیل صوت ویدئو به متن با تایم‌کدهای کلمه (فارسی)
    """
    model = WhisperModel(model_size, device=device, compute_type="int8" if device=="cpu" else "float16")
    segments, info = model.transcribe(
        video_path,
        language="fa",
        beam_size=5,
        word_timestamps=True,
        vad_filter=True
    )
    words_data = []
    for seg in segments:
        for word in seg.words:
            words_data.append({
                "text": word.word,
                "start": round(word.start, 2),
                "end": round(word.end, 2)
            })
    return words_data

# =============== ارتباط با LLM برای اصلاح و تولید EDL ===============
def get_edl_from_llm(transcriptions, model_name="partai/dorna-llama3:8b-instruct-q5_0"):
    """
    ارسال متن‌ها به مدل محلی Ollama و دریافت JSON تدوین
    """
    prompt = f"""
    من چندین فایل مصاحبه به زبان فارسی دارم که متن و تایم‌کد آن‌ها را با Whisper استخراج کرده‌ام.
    لطفاً این کارها را انجام بده:

    ۱. خطاهای املایی و نگارشی متن‌های فارسی را اصلاح کن (مثلاً «میرم» را به «می‌روم» تبدیل کن).
    ۲. همهٔ مصاحبه‌ها را از نظر محتوایی بررسی کن و یک «مضمون مشترک» بین آن‌ها پیدا کن.
    ۳. برای آن مضمون، گویاترین و تأثیرگذارترین جملات را از بین همهٔ مصاحبه‌ها انتخاب کن. اگر جمله‌ای تکراری بود، فقط بهترین نسخه‌اش را نگه دار.
    ۴. در نهایت، یک خروجی JSON با فرمت زیر به من بده:

    {{
      "clips": [
        {{"source_file": "نام_فایل_اصلی.mp4", "start_time": ۱۰.۵, "end_time": ۲۵.۳}},
        {{"source_file": "نام_فایل_دیگر.mp4", "start_time": ۴۲.۰, "end_time": ۶۰.۸}}
      ]
    }}

    توجه: source_file باید دقیقاً همان نام فایلی باشد که در ورودی داده‌ام.
    حالا متن‌های خروجی Whisper به همراه نام فایل مربوطه:

    {json.dumps(transcriptions, ensure_ascii=False, indent=2)}

    فقط JSON را برگردان و هیچ توضیح دیگری نده.
    """
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": "تو یک ویراستار متخصص فارسی و تدوینگر حرفه‌ای هستی. فقط خروجی JSON بده."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # استخراج JSON از پاسخ مدل
    response_text = response['message']['content']
    # پاکسازی متن در صورت وجود توضیحات اضافی
    try:
        # پیدا کردن JSON بین { و }
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        json_str = response_text[start:end]
        return json.loads(json_str)
    except:
        # اگر مدل JSON خالص برگرداند
        return json.loads(response_text)

# =============== اجرای تدوین با FFmpeg ===============
def perform_editing(edl_data, output_path):
    """
    برش و ترکیب ویدئوها بر اساس JSON
    """
    clips = []
    for item in edl_data["clips"]:
        source = Path(item["source_file"])
        if not source.exists():
            raise FileNotFoundError(f"فایل {source} یافت نشد!")
        start = float(item["start_time"])
        end = float(item["end_time"])
        # برش
        trimmed = ffmpeg.input(str(source), ss=start, to=end)
        clips.append(trimmed)
    
    if not clips:
        raise ValueError("هیچ کلیپی برای ترکیب وجود ندارد.")
    
    # اتصال همه کلیپ‌ها
    joined = ffmpeg.concat(*clips, v=1, a=1).output(output_path)
    joined.run(overwrite_output=True)

# =============== کلاس رابط کاربری ===============
class EditingApp:
    def __init__(self, root):
        self.root = root
        root.title("تدوین خودکار مصاحبه‌های فارسی (با Ollama)")
        root.geometry("750x650")
        root.configure(bg='#f0f0f0')
        
        # متغیرها
        self.video_files = []
        # مدل Whisper
        self.whisper_model_var = tk.StringVar(value="large-v3")
        self.device_var = tk.StringVar(value="cpu")
        # مدل Ollama (جداگانه)
        self.ollama_model_var = tk.StringVar(value="partai/dorna-llama3:8b-instruct-q5_0")
        self.log_text = None
        
        self._build_ui()
    
    def _build_ui(self):
        # ====== بخش انتخاب فایل ======
        top_frame = tk.Frame(self.root, bg='#f0f0f0')
        top_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Button(top_frame, text="انتخاب فایل‌های ویدئویی", command=self.select_files,
                  bg='#4CAF50', fg='white', font=('Tahoma', 10)).pack(side='left', padx=5)
        
        self.file_label = tk.Label(top_frame, text="هیچ فایلی انتخاب نشده", bg='#f0f0f0', font=('Tahoma', 9))
        self.file_label.pack(side='left', padx=10, fill='x', expand=True)
        
        # ====== تنظیمات مدل‌ها ======
        settings_frame = tk.Frame(self.root, bg='#f0f0f0')
        settings_frame.pack(pady=5, padx=10, fill='x')
        
        # Whisper
        tk.Label(settings_frame, text="مدل Whisper:", bg='#f0f0f0').pack(side='left')
        whisper_options = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
        tk.OptionMenu(settings_frame, self.whisper_model_var, *whisper_options).pack(side='left', padx=5)
        
        tk.Label(settings_frame, text="دستگاه:", bg='#f0f0f0').pack(side='left', padx=(20,0))
        tk.OptionMenu(settings_frame, self.device_var, "cpu", "cuda").pack(side='left', padx=5)
        
        # Ollama
        tk.Label(settings_frame, text="مدل Ollama:", bg='#f0f0f0').pack(side='left', padx=(20,0))
        ollama_options = [
            "partai/dorna-llama3:8b-instruct-q5_0",
            "mshojaei77/gemma3persian",
            "llama3.2:3b",
            "mistral:7b",
            "zharfap/zharfa-open:7b"
        ]
        tk.OptionMenu(settings_frame, self.ollama_model_var, *ollama_options).pack(side='left', padx=5)
        
        # ====== دکمه شروع ======
        self.start_btn = tk.Button(self.root, text="شروع تدوین", command=self.start_editing,
                                   bg='#2196F3', fg='white', font=('Tahoma', 12), height=2)
        self.start_btn.pack(pady=10)
        
        # ====== لاگ ======
        self.log_text = scrolledtext.ScrolledText(self.root, height=20, font=('Tahoma', 9))
        self.log_text.pack(padx=10, pady=5, fill='both', expand=True)
        self.log_text.insert(tk.END, "👉 منتظر شروع عملیات...\n")
        self.log_text.config(state='disabled')
        
        # ====== نوار پیشرفت ======
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(pady=5)
    
    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="انتخاب فایل‌های ویدئویی",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if files:
            self.video_files = list(files)
            self.file_label.config(text=f"{len(files)} فایل انتخاب شد")
            self.log(f"✅ {len(files)} فایل انتخاب شد.")
            for f in files:
                self.log(f"   - {Path(f).name}")
        else:
            self.log("⚠️ هیچ فایلی انتخاب نشد.")
    
    def start_editing(self):
        if not self.video_files:
            messagebox.showwarning("خطا", "لطفاً ابتدا فایل‌های ویدئویی را انتخاب کنید.")
            return
        
        self.start_btn.config(state='disabled')
        self.progress['value'] = 0
        self.log("🚀 شروع فرآیند تدوین...")
        
        # اجرا در یک ترد جداگانه تا GUI قفل نشود
        threading.Thread(target=self._process, daemon=True).start()
    
    def _process(self):
        try:
            # ===== مرحله ۱: Whisper =====
            self.log("📥 مرحله ۱: تبدیل گفتار به متن با Whisper...")
            all_transcriptions = []
            total = len(self.video_files)
            for idx, video_path in enumerate(self.video_files, 1):
                self.log(f"   پردازش فایل {idx}/{total}: {Path(video_path).name}")
                words = transcribe_video(
                    video_path,
                    model_size=self.whisper_model_var.get(),
                    device=self.device_var.get()
                )
                all_transcriptions.append({
                    "file": Path(video_path).name,
                    "full_path": video_path,
                    "words": words
                })
                self.progress['value'] = (idx / total) * 30
                self.root.update_idletasks()
            self.log("✅ تبدیل به متن با موفقیت انجام شد.")
            
            # ===== مرحله ۲: Ollama =====
            self.log("🧠 مرحله ۲: ارسال به هوش مصنوعی محلی (Ollama) و دریافت برنامه تدوین...")
            edl_json = get_edl_from_ollama(
                all_transcriptions,
                model_name=self.ollama_model_var.get()
            )
            self.log(f"✅ برنامه تدوین دریافت شد: {len(edl_json['clips'])} کلیپ انتخاب شده.")
            self.progress['value'] = 70
            self.root.update_idletasks()
            
            # ===== مرحله ۳: انتخاب مسیر خروجی =====
            output_file = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")],
                title="ذخیره ویدئوی نهایی"
            )
            if not output_file:
                self.log("❌ عملیات لغو شد (خروجی ذخیره نشد).")
                self._finish()
                return
            
            # ===== مرحله ۴: اجرای FFmpeg =====
            self.log("✂️ مرحله ۳: اجرای تدوین با FFmpeg...")
            # نیاز به تصحیح مسیر فایل‌های مبدأ در EDL (چون LLM فقط نام فایل را می‌دهد، باید مسیر کامل را بیابیم)
            file_map = {Path(f).name: f for f in self.video_files}
            for clip in edl_json['clips']:
                if clip['source_file'] not in file_map:
                    raise ValueError(f"فایل {clip['source_file']} در ورودی یافت نشد.")
                clip['source_file'] = file_map[clip['source_file']]
            
            perform_editing(edl_json, output_file)
            self.log(f"✅ ویدئوی نهایی با موفقیت ساخته شد: {output_file}")
            self.progress['value'] = 100
            messagebox.showinfo("پایان", "تدوین با موفقیت انجام شد!")
        except Exception as e:
            self.log(f"❌ خطا: {str(e)}")
            messagebox.showerror("خطا", str(e))
        finally:
            self._finish()
    
    def _finish(self):
        self.start_btn.config(state='normal')
        self.progress['value'] = 0

# =============== اجرای برنامه ===============
if __name__ == "__main__":
    root = tk.Tk()
    app = EditingApp(root)
    root.mainloop()