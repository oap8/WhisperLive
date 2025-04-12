# custom_client/ws_client.py (updated)
import asyncio
import websockets
import pyaudio
import numpy as np

CHUNK = 16000 * 3
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

async def record_and_stream():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    try:
        async with websockets.connect("ws://localhost:8765") as websocket:
            print("Recording... Press Ctrl+C to stop")
            while True:
                try:
                    data = stream.read(CHUNK)
                    await websocket.send(data)
                    response = await websocket.recv()
                    print(response)
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server")
                    break
                    
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio resources released")

if __name__ == "__main__":
    asyncio.run(record_and_stream())