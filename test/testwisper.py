#!/usr/bin/env python3
"""
Simple test script for faster-whisper (Persian)
Usage: python test_whisper.py <video_file_path>
"""

import sys
import time
from pathlib import Path
from faster_whisper import WhisperModel

def transcribe_test(video_path, model_size="large-v3",device="cpu"):
    """
    Test transcription with verbose output
    """
    print(f"🚀 Loading Whisper model '{model_size}' on {device}...")
    start_time = time.time()
    
    try:
        # بارگذاری مدل
        model = WhisperModel(model_size, device=device, compute_type="int8" if device == "cpu" else "float16")
        print(f"✅ Model loaded in {time.time() - start_time:.2f} seconds.")
        
        print(f"📥 Transcribing: {video_path}")
        start_time = time.time()
        
        # اجرای تبدیل
        segments, info = model.transcribe(
            video_path,
            language="fa",          # فارسی
            beam_size=5,
            word_timestamps=True,
            vad_filter=True
        )
        
        print(f"🔄 Transcription took {time.time() - start_time:.2f} seconds.")
        print(f"📊 Detected language: {info.language} (probability: {info.language_probability:.2f})")
        print("\n" + "="*60)
        print("TRANSCRIPTION RESULT (with word timestamps):")
        print("="*60)
        
        word_count = 0
        for segment in segments:
            print(f"\n[ {segment.start:.2f} -> {segment.end:.2f} ] {segment.text}")
            # نمایش کلمات با تایم‌کد دقیق
            for word in segment.words:
                print(f"  {word.start:.2f} -> {word.end:.2f} : {word.word}")
                word_count += 1
        
        print("\n" + "="*60)
        print(f"✅ Total words extracted: {word_count}")
        print("✅ Transcription completed successfully!")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_whisper.py <video_file_path>")
        print("Example: python test_whisper.py sample.mp4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    if not Path(video_file).exists():
        print(f"❌ File not found: {video_file}")
        sys.exit(1)
    
    # قابل تنظیم: مدل و دستگاه
    model_size = "large-v3"   # می‌توانید به "base", "medium", "large-v3" تغییر دهید
    device = "cpu"         # یا "cuda" اگر کارت گرافیک دارید
    
    transcribe_test(video_file, model_size, device)
