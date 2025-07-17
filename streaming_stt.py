import os
import pyaudio
import psutil
import time
from six.moves import queue
from google.cloud import speech
import json
from datetime import datetime
from collections import deque

# --- Config ---
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks
TRANSCRIPT_FILE = "live_transcript.txt"
METRICS_FILE = "resource_log.txt"

# --- Streaming Setup ---
class MicrophoneStream:
    def __init__(self, rate, chunk):
        self.rate = rate
        self.chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self.audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get_nowait()
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

# --- Processing Loop ---
def listen_print_loop(responses, output_file=TRANSCRIPT_FILE):
    process = psutil.Process()
    process.cpu_percent(interval=None)  # init CPU tracking

    timestamps = deque(maxlen=100)  # send timestamp buffer

    cpu_snapshots = []
    ram_snapshots = []
    latency_samples = []

    peak_memory = 0
    total_transcribed_time = 0
    session_start = time.time()

    print(f"\n📝 Writing transcript to: {output_file}")
    with open(output_file, "w") as f:
        try:
            for response in responses:
                cpu = process.cpu_percent(interval=None)
                ram = process.memory_percent()
                memory = process.memory_info().rss / (1024 * 1024)  # MB

                cpu_snapshots.append(cpu)
                ram_snapshots.append(ram)
                peak_memory = max(peak_memory, memory)

                now = time.time()

                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                if result.is_final:
                    duration = result.result_end_time.total_seconds()
                    latency = now - session_start - duration
                    latency_samples.append(latency)
                    total_transcribed_time += duration

                    print(f"✅ Final: {transcript}")
                    f.write(transcript + "\n")
                    f.flush()
                else:
                    print(f"⏳ Interim: {transcript}", end="\r")

        except KeyboardInterrupt:
            print("\n⛔ Transcription stopped by user.")

        finally:
            elapsed_time = time.time() - session_start
            avg_latency = sum(latency_samples) / len(latency_samples) if latency_samples else 0
            avg_cpu = sum(cpu_snapshots) / len(cpu_snapshots) if cpu_snapshots else 0
            avg_ram = sum(ram_snapshots) / len(ram_snapshots) if ram_snapshots else 0
            rtf = elapsed_time / total_transcribed_time if total_transcribed_time else 0

            print("\n🧠 Resource Usage Summary:")
            print(f"⏱️ Total Processing Time: {elapsed_time:.2f}s")
            print(f"🧪 Real-Time Factor (RTF): {rtf:.2f}")
            print(f"⏳ Avg Latency per Final Result: {avg_latency:.2f}s")
            print(f"📈 Peak Memory Usage: {peak_memory:.2f} MB")
            print(f"⚙️ Average CPU Usage: {avg_cpu:.2f}%")
            print(f"🧮 Average RAM Usage: {avg_ram:.2f}%")

            with open(METRICS_FILE, "w") as log:
                summary = {
                    "processing_time": round(elapsed_time, 2),
                    "real_time_factor": round(rtf, 2),
                    "avg_latency": round(avg_latency, 2),
                    "peak_memory_mb": round(peak_memory, 2),
                    "avg_cpu_percent": round(avg_cpu, 2),
                    "avg_ram_percent": round(avg_ram, 2),
                }
                log.write(json.dumps(summary, indent=2))

# --- Main Execution ---
def main():
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses)

if __name__ == "__main__":
    main()
