import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from pathlib import Path
import time
import cv2
from PIL import Image, ImageTk

# --- کتابخانه‌های اصلی ---
from faster_whisper import WhisperModel
import ffmpeg
import ollama

try:
    import arabic_reshaper
    ARABIC_RESHAPER_AVAILABLE = True
except ImportError:
    ARABIC_RESHAPER_AVAILABLE = False

try:
    from bidi.algorithm import get_display
    BIDI_AVAILABLE = True
except ImportError:
    BIDI_AVAILABLE = False

def reshape_persian_text(text):
    """اتصال حروف و تنظیم ترتیب نمایش برای متن فارسی"""
    if not text:
        return text
    if ARABIC_RESHAPER_AVAILABLE:
        text = arabic_reshaper.reshape(text)
    if BIDI_AVAILABLE:
        text = get_display(text)
    return text

# ================================================================
#  بخش ۱: تبدیل گفتار به متن (Whisper)
# ================================================================

def transcribe_video(video_path, model_size="large-v3", device="cpu"):
    """
    Convert video audio to text with word-level timestamps (Persian)
    """
    model = WhisperModel(model_size, device=device, compute_type="int8" if device == "cpu" else "float16")
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


# ================================================================
#  بخش ۲: ارتباط با Ollama (محلی) برای تولید EDL
# ================================================================

def get_edl_from_ollama(transcriptions, model_name="partai/dorna-llama3:8b-instruct-q5_0"):
    """
    Send transcribed texts + timestamps to local Ollama model and get editing decision list (JSON)
    """
    prompt = f"""
    I have several Persian interview video files with their Whisper transcriptions and timestamps.
    Please do the following:

    1. Correct spelling and grammar errors in the Persian text (e.g., change "میرم" to "می‌روم").
    2. Analyze all interviews to find a common theme.
    3. For that theme, select the most eloquent and impactful sentences from all interviews. Remove repetitions.
    4. Finally, output a JSON with the following format:

    {{
      "clips": [
        {{"source_file": "original_filename.mp4", "start_time": 10.5, "end_time": 25.3}},
        {{"source_file": "another_file.mp4", "start_time": 42.0, "end_time": 60.8}}
      ]
    }}

    Important: "source_file" must exactly match the filenames I provide.

    Here are the Whisper outputs with filenames:

    {json.dumps(transcriptions, ensure_ascii=False, indent=2)}

    Return ONLY the JSON, no extra text.
    """
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a professional Persian editor and video editor. Output only JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    
    raw_text = response['message']['content']
    # extract JSON from possible extra text
    try:
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        if start != -1 and end != 0:
            json_str = raw_text[start:end]
            return json.loads(json_str)
        else:
            return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {raw_text}") from e


# ================================================================
#  بخش ۳: اجرای تدوین با FFmpeg
# ================================================================

def perform_editing(edl_data, output_path):
    """
    Cut and concatenate videos based on EDL JSON
    """
    clips = []
    for item in edl_data["clips"]:
        source = Path(item["source_file"])
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")
        start = float(item["start_time"])
        end = float(item["end_time"])
        trimmed = ffmpeg.input(str(source), ss=start, to=end)
        clips.append(trimmed)
    
    if not clips:
        raise ValueError("No clips to concatenate.")
    
    joined = ffmpeg.concat(*clips, v=1, a=1).output(output_path)
    joined.run(overwrite_output=True)


# ================================================================
#  بخش ۴: رابط کاربری اصلی (بدون VLC)
# ================================================================

class VideoEditorApp:
    def __init__(self, root):
        self.root = root
        root.title("Persian Interview Editor (Whisper + Ollama + FFmpeg)")
        root.geometry("1100x750")
        root.configure(bg='#f0f0f0')
        
        # متغیرها
        self.video_files = []
        self.current_video_index = 0
        self.transcriptions = []
        self.corrected_texts = []
        self.edl_json = None
        
        # متغیرهای مدل (افزوده‌شده)
        self.whisper_model_var = tk.StringVar(value="large-v3")
        self.device_var = tk.StringVar(value="cpu")
        self.ollama_model_var = tk.StringVar(value="partai/dorna-llama3:8b-instruct-q5_0")
        
        # OpenCV video capture
        self.cap = None
        self.is_playing = False
        self.current_time = 0  # milliseconds
        self.total_duration = 0
        
        # ساخت UI
        self._build_ui()
        self.text_widget.tag_configure("highlight", background="yellow")
    
    def _build_ui(self):
        # ===== نوار ابزار بالا =====
        toolbar = tk.Frame(self.root, bg='#e0e0e0')
        toolbar.pack(side='top', fill='x', padx=5, pady=5)
        
        # دکمه‌های اصلی
        tk.Button(toolbar, text="Select Video Files", command=self.select_files,
                  bg='#4CAF50', fg='white').pack(side='left', padx=2)
        
        self.transcribe_btn = tk.Button(toolbar, text="Transcribe All", command=self.transcribe_all,
                                bg='#2196F3', fg='white')
        self.transcribe_btn.pack(side='left', padx=2)
        
        tk.Button(toolbar, text="Previous Video", command=self.prev_video,
                  bg='#FF9800', fg='white').pack(side='left', padx=2)
        
        tk.Button(toolbar, text="Next Video", command=self.next_video,
                  bg='#FF9800', fg='white').pack(side='left', padx=2)
        
        tk.Button(toolbar, text="Save Corrected Text", command=self.save_corrected_text,
                  bg='#9C27B0', fg='white').pack(side='left', padx=2)
        
        tk.Button(toolbar, text="Load Corrected Text", command=self.load_corrected_text,
                  bg='#9C27B0', fg='white').pack(side='left', padx=2)
        
        tk.Button(toolbar, text="Send to Ollama & Edit", command=self.send_to_ollama,
                  bg='#F44336', fg='white').pack(side='left', padx=2)
        
        tk.Button(toolbar, text="Export Final Video", command=self.export_final,
                  bg='#4CAF50', fg='white').pack(side='left', padx=2)
        
        # ===== انتخاب مدل‌ها (افزوده‌شده) =====
        # Whisper model
        tk.Label(toolbar, text="Whisper:", bg='#e0e0e0').pack(side='left', padx=(10,0))
        whisper_combo = ttk.Combobox(toolbar, textvariable=self.whisper_model_var,
                                     values=["tiny", "base", "small", "medium", "large-v3", "large-v2"],
                                     width=10, state='readonly')
        whisper_combo.pack(side='left', padx=2)
        
        # Device
        tk.Label(toolbar, text="Device:", bg='#e0e0e0').pack(side='left', padx=(5,0))
        device_combo = ttk.Combobox(toolbar, textvariable=self.device_var,
                                    values=["cpu", "cuda"], width=6, state='readonly')
        device_combo.pack(side='left', padx=2)
        
        # Ollama model
        tk.Label(toolbar, text="Ollama model:", bg='#e0e0e0').pack(side='left', padx=(10,0))
        ollama_entry = tk.Entry(toolbar, textvariable=self.ollama_model_var, width=20)
        ollama_entry.pack(side='left', padx=2)
        
        # اطلاعات فایل فعلی
        self.file_info = tk.Label(toolbar, text="No file loaded", bg='#e0e0e0')
        self.file_info.pack(side='right', padx=10)
        
        # ===== پنل ویدئو (چپ) با Canvas =====
        video_frame = tk.Frame(self.root, bg='black', width=640, height=360)
        video_frame.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        video_frame.pack_propagate(False)
        self.video_canvas = tk.Canvas(video_frame, bg='black', width=640, height=360)
        self.video_canvas.pack(fill='both', expand=True)
        # بایند برای تغییر اندازه (افزوده‌شده)
        self.video_canvas.bind('<Configure>', self.on_canvas_resize)
        
        # ===== پنل متن (راست) =====
        text_frame = tk.Frame(self.root, bg='#f0f0f0')
        text_frame.pack(side='right', padx=5, pady=5, fill='both', expand=True)

        self.text_widget = tk.Text(text_frame, wrap='word', font=('Tahoma', 11), height=20)
        self.text_widget.pack(side='top', fill='both', expand=True)

        # تنظیم جهت RTL (در نسخه‌های جدید Tk)
        try:
            self.text_widget.configure(direction='rtl')
        except:
            pass  # اگر پشتیبانی نشد، از تگ rtl استفاده می‌کنیم

        # تگ برای راست‌چین کردن محتوا
        self.text_widget.tag_configure("rtl", justify='right')

        scroll = tk.Scrollbar(text_frame, command=self.text_widget.yview)
        scroll.pack(side='right', fill='y')
        self.text_widget.config(yscrollcommand=scroll.set)
        
        # ===== کنترل‌های پخش =====
        controls = tk.Frame(self.root, bg='#e0e0e0')
        controls.pack(side='bottom', fill='x', padx=5, pady=5)
        
        self.play_btn = tk.Button(controls, text="Play", command=self.play_pause, width=8)
        self.play_btn.pack(side='left', padx=2)
        self.stop_btn = tk.Button(controls, text="Stop", command=self.stop, width=8)
        self.stop_btn.pack(side='left', padx=2)
        
        self.time_slider = tk.Scale(controls, from_=0, to=100, orient='horizontal', length=400,
                                    command=self.slider_changed)
        self.time_slider.pack(side='left', padx=10, fill='x', expand=True)
        
        self.time_label = tk.Label(controls, text="00:00 / 00:00", bg='#e0e0e0')
        self.time_label.pack(side='left', padx=5)
        
        # ===== نوار پیشرفت کلی =====
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress.pack(side='bottom', fill='x', padx=5, pady=2)
        
        # ===== لاگ =====
        self.log_text = scrolledtext.ScrolledText(self.root, height=5, font=('Tahoma', 8))
        self.log_text.pack(side='bottom', fill='x', padx=5, pady=5)
        self.log_text.insert(tk.END, "Ready.\n")
        self.log_text.config(state='disabled')
        
        # تایمر برای به‌روزرسانی
        self.update_timer = None
    
    # ----------------------------------------------------------------
    #  توابع لاگ
    # ----------------------------------------------------------------
    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()
    
    # ----------------------------------------------------------------
    #  انتخاب فایل‌ها
    # ----------------------------------------------------------------
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select video files",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if files:
            self.video_files = list(files)
            self.current_video_index = 0
            self.transcriptions = []
            self.corrected_texts = []
            self.file_info.config(text=f"{len(files)} files loaded")
            self.log(f"✅ {len(files)} video files selected.")
            for f in files:
                self.log(f"   - {Path(f).name}")
            self.load_video(0)
        else:
            self.log("⚠️ No files selected.")
    
    # ----------------------------------------------------------------
    #  بارگذاری ویدئو با OpenCV
    # ----------------------------------------------------------------
    def load_video(self, index):
        if not self.video_files or index < 0 or index >= len(self.video_files):
            return
        self.current_video_index = index
        video_path = self.video_files[index]
        self.file_info.config(text=f"File {index+1}/{len(self.video_files)}: {Path(video_path).name}")
        
        # بستن قبلی
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.total_duration = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS) * 1000)
        self.current_time = 0
        self.is_playing = False
        self.play_btn.config(text="Play")
        
        # نمایش اولین فریم
        self.show_frame_at(0)
        self.start_update_timer()
        
        # نمایش متن متناظر (اگر موجود باشد)
        if self.transcriptions and index < len(self.transcriptions):
            self.display_transcription(index)
        else:
            self.text_widget.delete('1.0', tk.END)
            self.text_widget.insert(tk.END, "No transcription yet. Click 'Transcribe All'.")
    
    def show_frame_at(self, ms):
        """Show frame at given millisecond with proper scaling (افزوده‌شده)"""
        if self.cap is None:
            return
        # تبدیل میلی‌ثانیه به فریم
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(ms / 1000 * fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
    
    def display_frame(self, frame):
        """Display frame scaled to fit canvas while preserving aspect ratio (افزوده‌شده)"""
        # دریافت ابعاد فعلی canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 640
            canvas_height = 360
        
        h, w = frame.shape[:2]
        # محاسبه نسبت
        scale_w = canvas_width / w
        scale_h = canvas_height / h
        scale = min(scale_w, scale_h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # تغییر اندازه
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # تبدیل به RGB
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(resized_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # پاک کردن canvas و نمایش در مرکز
        self.video_canvas.delete("all")
        x_center = (canvas_width - new_w) // 2
        y_center = (canvas_height - new_h) // 2
        self.video_canvas.create_image(x_center, y_center, anchor='nw', image=imgtk)
        self.video_canvas.image = imgtk  # keep reference
    
    def on_canvas_resize(self, event):
        """هنگام تغییر اندازه canvas، فریم فعلی را دوباره نمایش بده (افزوده‌شده)"""
        if self.cap is not None:
            self.show_frame_at(self.current_time)
    
    # ----------------------------------------------------------------
    #  کنترل‌های پخش
    # ----------------------------------------------------------------
    def play_pause(self):
        if self.cap is None:
            return
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="Play")
        else:
            self.is_playing = True
            self.play_btn.config(text="Pause")
            self.play_loop()
    
    def play_loop(self):
        if not self.is_playing or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
            # به‌روزرسانی زمان
            self.current_time += 30  # حدود 30 میلی‌ثانیه
            self.update_slider_and_label()
            self.root.after(30, self.play_loop)
        else:
            # پایان ویدئو
            self.is_playing = False
            self.play_btn.config(text="Play")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def stop(self):
        self.is_playing = False
        self.play_btn.config(text="Play")
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_time = 0
            self.show_frame_at(0)
            self.update_slider_and_label()
    
    def slider_changed(self, value):
        if self.cap:
            ms = int(float(value) * self.total_duration / 100)
            self.current_time = ms
            self.show_frame_at(ms)
            self.update_slider_and_label()
    
    def update_slider_and_label(self):
        if self.total_duration > 0:
            self.time_slider.set(100 * self.current_time / self.total_duration)
            cur_str = self._format_time(self.current_time)
            total_str = self._format_time(self.total_duration)
            self.time_label.config(text=f"{cur_str} / {total_str}")
    
    def start_update_timer(self):
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_ui()
    
    def update_ui(self):
        if self.cap is not None:
            self.update_slider_and_label()
        self.update_timer = self.root.after(100, self.update_ui)
    
    def _format_time(self, ms):
        s = ms // 1000
        m = s // 60
        s = s % 60
        return f"{m:02d}:{s:02d}"
    
    # ----------------------------------------------------------------
    #  نمایش متن
    # ----------------------------------------------------------------
    def display_transcription(self, index):
        self.text_widget.delete('1.0', tk.END)
        if index >= len(self.transcriptions):
            return
        words = self.transcriptions[index]
        
        # اگر متن تصحیح‌شده موجود است
        if self.corrected_texts and index < len(self.corrected_texts) and self.corrected_texts[index]:
            text = self.corrected_texts[index]
            text = reshape_persian_text(text)
            self.text_widget.insert(tk.END, text, ('rtl',))
            return

        # ساخت لیستی از کلمات پردازش‌شده
        reshaped_words = []
        for w in words:
            word_text = w['text']
            # reshape هر کلمه به‌طور جداگانه برای اتصال حروف
            if ARABIC_RESHAPER_AVAILABLE:
                word_text = arabic_reshaper.reshape(word_text)
            if BIDI_AVAILABLE:
                word_text = get_display(word_text)
            reshaped_words.append((word_text, w['start']))
        
        # معکوس کردن لیست برای نمایش راست‌به‌چپ
        for word_text, start in reversed(reshaped_words):
            tag_name = f"word_{start}"
            # درج کلمه با یک فاصله بعد از آن
            self.text_widget.insert(tk.END, word_text + " ", (tag_name, 'rtl'))
            self.text_widget.tag_config(tag_name, foreground="blue", underline=True)
            self.text_widget.tag_bind(tag_name, "<Button-1>",
                                    lambda e, s=start: self.seek_and_play(s))
        self.log(f"Displayed {len(words)} words for file {index+1}")
    
    def seek_and_play(self, seconds):
        ms = int(seconds * 1000)
        self.current_time = ms
        self.show_frame_at(ms)
        self.update_slider_and_label()
        if not self.is_playing:
            self.is_playing = True
            self.play_btn.config(text="Pause")
            self.play_loop()
    
    # ----------------------------------------------------------------
    #  توابع ناوبری بین فایل‌ها
    # ----------------------------------------------------------------
    def prev_video(self):
        if self.current_video_index > 0:
            self.load_video(self.current_video_index - 1)
        else:
            self.log("Already at first video.")
    
    def next_video(self):
        if self.current_video_index < len(self.video_files) - 1:
            self.load_video(self.current_video_index + 1)
        else:
            self.log("Already at last video.")
    
    # ----------------------------------------------------------------
    #  متدهای ویرایش متن و ذخیره/بارگذاری
    # ----------------------------------------------------------------
    def save_corrected_text(self):
        if self.current_video_index >= len(self.corrected_texts):
            return
        text = self.text_widget.get('1.0', tk.END).strip()
        self.corrected_texts[self.current_video_index] = text
        self.log(f"✅ Corrected text saved for file {self.current_video_index+1}")
        messagebox.showinfo("Success", f"Text saved for file {self.current_video_index+1}")
    
    def load_corrected_text(self):
        file_path = filedialog.askopenfilename(
            title="Load corrected text",
            filetypes=[("Text files", "*.txt"), ("JSON files", "*.json")]
        )
        if not file_path:
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) == len(self.video_files):
                self.corrected_texts = data
                self.log(f"✅ Loaded corrected texts from {file_path}")
                if self.current_video_index < len(self.corrected_texts):
                    self.text_widget.delete('1.0', tk.END)
                    text = self.corrected_texts[self.current_video_index]
                    text = reshape_persian_text(text)
                    self.text_widget.insert(tk.END, text, ('rtl',))
            else:
                messagebox.showerror("Error", "Invalid file format.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    # ----------------------------------------------------------------
    #  تبدیل متن با Whisper (با نمایش پیشرفت)
    # ----------------------------------------------------------------
    def transcribe_all(self):
        if not self.video_files:
            messagebox.showwarning("Warning", "Please select video files first.")
            return
        
        self.transcribe_btn.config(state='disabled')
        self.progress['value'] = 0
        self.log("🚀 Starting transcription with Whisper...")
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def _transcribe_worker(self):
        """Worker thread for transcription"""
        try:
            total = len(self.video_files)
            self.transcriptions = []
            self.corrected_texts = []
            
            for idx, video_path in enumerate(self.video_files, 1):
                self.log(f"🔄 Transcribing file {idx}/{total}: {Path(video_path).name}")
                self.root.after(0, lambda: self.progress.config(value=(idx-1)/total*50))
                
                words = transcribe_video(
                    video_path,
                    model_size=self.whisper_model_var.get(),
                    device=self.device_var.get()
                )
                self.transcriptions.append(words)
                self.corrected_texts.append("")
                
                # به‌روزرسانی نوار پیشرفت
                progress_val = (idx / total) * 50
                self.root.after(0, lambda v=progress_val: self.progress.config(value=v))
                self.log(f"✅ File {idx}/{total} done: {len(words)} words extracted.")
            
            self.log("✅ All transcriptions complete.")
            self.root.after(0, lambda: self.progress.config(value=50))
            if self.transcriptions and self.video_files:
                self.root.after(0, lambda: self.load_video(self.current_video_index))
                self.root.after(0, lambda: self.display_transcription(self.current_video_index))
        except Exception as e:
            self.log(f"❌ Error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.root.after(0, lambda: self.transcribe_btn.config(state='normal'))
    
    # ----------------------------------------------------------------
    #  ارسال به Ollama + نمایش و ویرایش EDL (افزوده‌شده)
    # ----------------------------------------------------------------
    def send_to_ollama(self):
        if not self.transcriptions or not self.video_files:
            messagebox.showwarning("Warning", "No transcriptions found. Please transcribe first.")
            return
        
        final_transcriptions = []
        for idx, words in enumerate(self.transcriptions):
            if self.corrected_texts and idx < len(self.corrected_texts) and self.corrected_texts[idx]:
                final_transcriptions.append({
                    "file": Path(self.video_files[idx]).name,
                    "full_path": self.video_files[idx],
                    "words": words,
                    "corrected_text": self.corrected_texts[idx]
                })
            else:
                final_transcriptions.append({
                    "file": Path(self.video_files[idx]).name,
                    "full_path": self.video_files[idx],
                    "words": words
                })
        
        self.log("🧠 Sending to Ollama for editing decision...")
        self.progress['value'] = 60
        try:
            edl_json = get_edl_from_ollama(
                final_transcriptions,
                model_name=self.ollama_model_var.get()
            )
            self.log(f"✅ EDL received: {len(edl_json['clips'])} clips selected.")
            self.progress['value'] = 80
            # نمایش پنجره ویرایش (افزوده‌شده)
            self.show_edl_editor(edl_json)
        except Exception as e:
            self.log(f"❌ Error from Ollama: {e}")
            messagebox.showerror("Error", str(e))
    
    def show_edl_editor(self, edl_data):
        """پنجره ویرایش EDL (افزوده‌شده)"""
        editor_window = tk.Toplevel(self.root)
        editor_window.title("ویرایش EDL (لیست تصمیم تدوین)")
        editor_window.geometry("800x600")
        
        # ویجت متن برای نمایش و ویرایش JSON
        text_area = scrolledtext.ScrolledText(editor_window, wrap='word', font=('Tahoma', 10))
        text_area.pack(side='top', fill='both', expand=True, padx=5, pady=5)
        # نمایش JSON زیبا
        pretty_json = json.dumps(edl_data, ensure_ascii=False, indent=2)
        text_area.insert('1.0', pretty_json)
        
        # دکمه‌ها
        btn_frame = tk.Frame(editor_window)
        btn_frame.pack(side='bottom', fill='x', pady=5)
        
        def apply_edl():
            try:
                edited_text = text_area.get('1.0', tk.END).strip()
                new_edl = json.loads(edited_text)
                # بررسی ساختار
                if not isinstance(new_edl, dict) or 'clips' not in new_edl:
                    raise ValueError("JSON must contain 'clips' key.")
                self.edl_json = new_edl
                self.log("✅ EDL edited and applied.")
                self.progress['value'] = 80
                editor_window.destroy()
            except Exception as e:
                messagebox.showerror("خطا", f"JSON نامعتبر: {e}")
        
        tk.Button(btn_frame, text="تأیید و ادامه", command=apply_edl,
                  bg='#4CAF50', fg='white', padx=10).pack(side='right', padx=5)
        tk.Button(btn_frame, text="لغو", command=editor_window.destroy,
                  bg='#f44336', fg='white', padx=10).pack(side='right', padx=5)
    
    # ----------------------------------------------------------------
    #  ساخت ویدئوی نهایی
    # ----------------------------------------------------------------
    def export_final(self):
        if not self.edl_json or not self.edl_json.get('clips'):
            messagebox.showwarning("Warning", "No EDL data. Run 'Send to Ollama & Edit' first.")
            return
        
        output_file = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")],
            title="Save Final Video"
        )
        if not output_file:
            self.log("❌ Export cancelled.")
            return
        
        self.log("✂️ Running FFmpeg to create final video...")
        self.progress['value'] = 80
        try:
            file_map = {Path(f).name: f for f in self.video_files}
            for clip in self.edl_json['clips']:
                if clip['source_file'] not in file_map:
                    raise ValueError(f"File '{clip['source_file']}' not found in input.")
                clip['source_file'] = file_map[clip['source_file']]
            
            perform_editing(self.edl_json, output_file)
            self.log(f"✅ Final video created: {output_file}")
            self.progress['value'] = 100
            messagebox.showinfo("Success", f"Video saved to:\n{output_file}")
        except Exception as e:
            self.log(f"❌ FFmpeg error: {e}")
            messagebox.showerror("Error", str(e))


# ================================================================
#  اجرا
# ================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEditorApp(root)
    root.mainloop()