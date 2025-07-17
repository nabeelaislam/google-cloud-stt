import os
import pyaudio
import psutil
import time
from six.moves import queue
from google.cloud import speech

RATE = 16000
CHUNK = int(RATE / 10)

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

def listen_print_loop(responses, output_file="live_transcript.txt"):
    process = psutil.Process()
    peak_memory = 0
    cpu_snapshots = []
    final_latencies = []
    total_transcribed_time = 0

    print(f"\n📝 Writing transcript to: {output_file}")
    with open(output_file, "w") as f:
        try:
            start_time = time.time()
            for response in responses:
                memory = process.memory_info().rss / (1024 * 1024)
                cpu = process.cpu_percent(interval=0.1)
                cpu_snapshots.append(cpu)
                peak_memory = max(peak_memory, memory)

                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                if result.is_final:
                    latency = time.time() - start_time
                    final_latencies.append(latency)
                    start_time = time.time()  # ← reset for next result
                    print(f"✅ Final: {transcript}")
                    f.write(transcript + "\n")
                    f.flush()
                else:
                    print(f"⏳ Interim: {transcript}", end="\r")
        except KeyboardInterrupt:
            print("\n⛔ Transcription stopped by user.")
        finally:
            elapsed_time = sum(final_latencies) if final_latencies else 1
            avg_latency = sum(final_latencies) / len(final_latencies) if final_latencies else 0
            avg_cpu = sum(cpu_snapshots) / len(cpu_snapshots) if cpu_snapshots else 0
            real_time_factor = elapsed_time / total_transcribed_time if total_transcribed_time else 0

            print("\n🧠 Resource Usage Summary:")
            print(f"⏱️ Total Processing Time: {elapsed_time:.2f}s")
            print(f"🧪 Real-Time Factor (RTF): {real_time_factor:.2f}")
            print(f"⏳ Avg Latency per Final Result: {avg_latency:.2f}s")
            print(f"📈 Peak Memory Usage: {peak_memory:.2f} MB")
            print(f"⚙️ Average CPU Usage: {avg_cpu:.2f}%")

            with open("resource_log.txt", "w") as log:
                log.write(f"Total Processing Time: {elapsed_time:.2f}s\n")
                log.write(f"Real-Time Factor: {real_time_factor:.2f}\n")
                log.write(f"Avg Latency per Result: {avg_latency:.2f}s\n")
                log.write(f"Peak Memory: {peak_memory:.2f} MB\n")
                log.write(f"Average CPU: {avg_cpu:.2f}%\n")


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
        listen_print_loop(responses, output_file="live_transcript.txt")

if __name__ == "__main__":
    main()
