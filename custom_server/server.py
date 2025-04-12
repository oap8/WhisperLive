import pyaudio
import numpy as np
from faster_whisper import WhisperModel

# Audio Configuration (Matches your working setup)
CHUNK_SIZE = 16000 * 5  # 5-second chunks
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Whisper Model Setup (CPU-optimized)
model_size = "tiny.en"
model = WhisperModel(
    model_size,
    device="cpu",
    compute_type="int8",  # Quantized for CPU
    cpu_threads=4,        # Matches your i5-7300HQ (4 cores)
    num_workers=2         # Balanced parallelism
)

def transcribe_audio():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    try:
        while True:
            data = stream.read(CHUNK_SIZE)
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            segments, _ = model.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                vad_filter=True  # Voice Activity Detection
            )

            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

    except KeyboardInterrupt:
        print("\nStopping transcription...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    transcribe_audio()