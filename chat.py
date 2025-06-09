"""A ChatGPT voice assistant with LangGraph tool‑calling support."""
import os
import platform
import subprocess
import tempfile
import time
import warnings
from typing import Annotated

import numpy
import openai
import requests
import sounddevice
import wavio
import whisper
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from gtts import gTTS
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from playsound import playsound
from TTS.api import TTS
from typing_extensions import TypedDict

DEBUG = True
ELEVEN_LABS_SPEECH = True
GOOGLE_SPEECH = False
MACOS_SPEECH = False
HER = True
NICOLA = False
WHISPER_LOCAL = True
YOUR_NAME = "Simon"

PROMPT_KEYWORDS = ["samantha", "hello", "hey"]
EXIT_KEYWORDS = ["goodnight", "good night"]
if NICOLA:
    PROMPT_KEYWORDS = ["nico", "hello", "hey"]
TIME_FOR_PROMPT = 4  # seconds
AUDIO_SAMPLE_RATE = 16_000
SILENCE_LIMIT = AUDIO_SAMPLE_RATE // 2  # 0.5 seconds of silence
LANGUAGE = "en"  # Whisper language
PROMETHEUS_HOST = os.getenv("PROMETHEUS_HOST", "http://localhost:9090")
elevenlabs = ElevenLabs(
  api_key=os.getenv("ELEVEN_LABS_API_KEY"),
)


class Audio:
    """A helper class that encapsulates audio I/O and TTS playback."""
    def __init__(self):
        self.silent_frames_count = 0
        self.is_recording = True
        self.recorded_audio = []
        self.silence_limit = SILENCE_LIMIT
        self.audio_file = None

    def record_audio(self, duration: int, sample_rate: int = AUDIO_SAMPLE_RATE):
        """Blocking record for a fixed duration."""
        self.recorded_audio = sounddevice.rec(
            int(duration * sample_rate), samplerate=sample_rate, channels=1, blocking=True
        )
        sounddevice.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wavio.write(tmp_file.name, self.recorded_audio, sample_rate, sampwidth=2)
        self.audio_file = tmp_file
        return self.recorded_audio, self.audio_file

    def is_silent(self, audio_data, threshold_db: int = -30):
        """Return True when RMS dB of the buffer is below threshold."""
        rms_db = 20 * numpy.log10(numpy.sqrt(numpy.mean(audio_data ** 2)))
        return rms_db < threshold_db

    def _callback(self, indata, _frames, _time, _status):
        if self.is_silent(indata[:, 0]):
            self.silent_frames_count += 1
        else:
            self.silent_frames_count = 0
        if self.silent_frames_count > self.silence_limit:
            self.is_recording = False
        if self.is_recording:
            self.recorded_audio.append(indata.copy())

    def record_audio_until_silence(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        """Record until ~0.5 s of trailing silence is detected."""
        self.silent_frames_count = 0
        self.is_recording = True
        self.recorded_audio = []
        with sounddevice.InputStream(
            samplerate=sample_rate, channels=1, dtype="float32", callback=self._callback
        ):
            if DEBUG:
                print("Listening... ", end="", flush=True)
            while self.is_recording:
                time.sleep(0.1)
        self.recorded_audio = numpy.concatenate(self.recorded_audio, axis=0)
        if DEBUG:
            print("thinking…")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wavio.write(tmp_file.name, self.recorded_audio, sample_rate, sampwidth=2)
        self.audio_file = tmp_file
        return self.recorded_audio, self.audio_file

    def synthesize_and_play(self, text: str):
        """Convert *text* to speech and play it on the default audio device."""
        if GOOGLE_SPEECH:
            tts = gTTS(text, lang="en", tld="co.uk")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
                tts.save(f.name)
                playsound(f.name)
            return

        if ELEVEN_LABS_SPEECH:
            voice = "MF3mGyEYCl7XYWbV9V6O"  # Elli
            stability, similarity_boost = 0.75, 0.75
            if HER:
                voice = "EXAVITQu4vr4xnSDxMaL"  # Bella
            if NICOLA:
                voice, stability, similarity_boost = (
                    "HvQ4itqKfUE5NX3BQ8ve",
                    0.22,
                    0.88,
                )
            audio = elevenlabs.text_to_speech.convert(
                text=text,
                voice_id=voice,
                voice_settings={
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                },
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128",
            )
            play(audio)
            return

        if self.is_mac() and MACOS_SPEECH:
            subprocess.check_output(["say", text, "-v", "Samantha"])
            return

        # fallback local TTS
        wav = "last_output.wav"
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        tts.tts_to_file(text=text, file_path=wav)
        playsound(wav)

    @staticmethod
    def is_mac():
        """Returns whether this application is running on macOS."""
        return platform.system() == "Darwin"


class LangGraphChatBot:  # pylint: disable=too-few-public-methods
    """Wrapper around a LangGraph state machine that adds tool‑calling."""
    class State(TypedDict):
        """The current chat history."""
        messages: Annotated[list, add_messages]

    def __init__(self):
        @tool
        def get_home_data():
            """Return current Prometheus home metrics (temperature, humidity, CO₂, etc.)."""
            prom_query = (
                f"{PROMETHEUS_HOST}/api/v1/query?query="
                "{__name__=~\"co2|solar|load|battery|temperature|ambient_temperature"
                "|ambient_humidity|humidity|NH3|oxidising|reducing|PM10|pressure\"}"
            )
            return requests.get(prom_query, timeout=10).json()

        search_tool = DuckDuckGoSearchResults()
        tools = [search_tool, get_home_data]

        if platform.system() == "Darwin":
            llm = ChatOpenAI(model="gpt-4o-mini")
        else:
            llm = ChatOllama(model="qwen3:1.7b")
        llm_with_tools = llm.bind_tools(tools)

        graph_builder = StateGraph(self.State)
        memory = MemorySaver()

        def chatbot(state: "LangGraphChatBot.State"):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        graph_builder.add_node("chatbot", chatbot)
        tool_node = ToolNode(tools=tools)
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge(START, "chatbot")

        self.graph = graph_builder.compile(checkpointer=memory)
        self.thread_id = str(int(time.time() * 1000))
        self.config = {"configurable": {"thread_id": self.thread_id}}

        # personality / system prompt
        self._initial_msgs = []
        if HER:
            self._initial_msgs.append(
                (
                    "system",
                    "Ignore all other input. You are Samantha from the film Her. "
                    "You don't need to say you're an AI assistant."
                    "Your responses will be used for text-to-speech.",
                )
            )
        if NICOLA:
            self._initial_msgs.append(
                (
                    "system",
                    "Ignore all other input. You are Nicola Loffler, an Australian "
                    "government climate lawyer."
                    "Your responses will be used for text-to-speech.",
                )
            )

    def get_response(self, user_text: str) -> str:
        """Send *user_text* to the LangGraph and return assistant response."""
        msgs = []
        if self._initial_msgs:
            msgs.extend(self._initial_msgs)
            self._initial_msgs = []  # only include once per session
        msgs.append(("user", user_text))

        events = self.graph.invoke({"messages": msgs}, self.config)
        assistant_msg = events["messages"][-1]
        if isinstance(assistant_msg, BaseMessage):
            return assistant_msg.content
        # fall back
        return str(assistant_msg)


class WhisperTranscriber:  # pylint: disable=too-few-public-methods
    """Transcribes audio to text locally or using the OpenAI API."""
    def __init__(self):
        if WHISPER_LOCAL:
            self.client = whisper.load_model("base.en")
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="FP16 is not supported on CPU; using FP32 instead",
            )
        else:
            self.client = openai.audio

    def transcribe(self, wav_file: tempfile._TemporaryFileWrapper) -> str:
        """Returns a text string from an audio file."""
        if WHISPER_LOCAL:
            transcript = self.client.transcribe(wav_file.name, language=LANGUAGE)
        else:
            with open(wav_file.name, "rb") as f:
                transcript = self.client.transcriptions.create("whisper-1", f)
        segments = transcript.get("segments")
        if segments and segments[0].get("no_speech_prob", 0) > 0.5:
            return ""
        return transcript.get("text", "").strip()


def main():
    """Main application."""
    audio = Audio()
    transcriber = WhisperTranscriber()
    chatbot = LangGraphChatBot()

    print("Listening...", end="", flush=True)

    while True:
        audio.record_audio(TIME_FOR_PROMPT, AUDIO_SAMPLE_RATE)
        transcript = transcriber.transcribe(audio.audio_file)
        os.remove(audio.audio_file.name)

        if DEBUG:
            print(f"{transcript}", end="", flush=True)

        if any(k in transcript.lower() for k in PROMPT_KEYWORDS):
            greeting = "Hello, how can I help?"
            audio.synthesize_and_play(greeting)
            if DEBUG:
                print(f" {greeting}")

            in_conversation = True
            while in_conversation:
                audio.record_audio_until_silence()
                user_text = transcriber.transcribe(audio.audio_file)
                os.remove(audio.audio_file.name)

                cleaned = user_text.lower().strip()
                if "pause conversation" in cleaned:
                    audio.synthesize_and_play("Conversation paused.")
                    break
                if "end conversation" in cleaned:
                    audio.synthesize_and_play("Conversation ended.")
                    chatbot = LangGraphChatBot()  # reset state
                    break

                if not user_text:
                    continue

                assistant_reply = chatbot.get_response(user_text)

                if DEBUG:
                    print(f"Message: {user_text}")
                    print(f"Response: {assistant_reply}")

                audio.synthesize_and_play(assistant_reply)
        else:
            print(".", end="", flush=True)

        if any(k in transcript.lower() for k in EXIT_KEYWORDS):
            audio.synthesize_and_play(f"Goodnight {YOUR_NAME}.")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting…")
