import asyncio
import websockets
import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8",
    cpu_threads=4,
    num_workers=2
)

async def handle_client(websocket):
    try:
        async for audio_bytes in websocket:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            segments, _ = model.transcribe(
                audio_np,
                beam_size=5,
                vad_filter=True,
                language="en"
            )
            
            for seg in segments:
                await websocket.send(f"[{seg.start:.2f}-{seg.end:.2f}s] {seg.text}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await websocket.close()

async def start_server():
    async with websockets.serve(handle_client, "localhost", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(start_server())