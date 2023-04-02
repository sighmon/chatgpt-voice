"""A ChatGPT voice assistant."""
import os
import platform
import subprocess
import tempfile
import time
import warnings

import numpy
import openai
import requests
import sounddevice
import wavio
import whisper
from gtts import gTTS
from playsound import playsound
from TTS.api import TTS

DEBUG = True
ELEVEN_LABS_SPEECH = True
GOOGLE_SPEECH = False
MACOS_SPEECH = False
HER = True
NICOLA = False
WHISPER_LOCAL = True
YOUR_NAME = 'Simon'

PROMPT_KEYWORD = 'samantha'
EXIT_KEYWORDS = ['goodnight', 'good night']
if NICOLA:
    PROMPT_KEYWORD = 'nico'
TIME_FOR_PROMPT = 4  # seconds
AUDIO_SAMPLE_RATE = 16000
SILENCE_LIMIT = AUDIO_SAMPLE_RATE // 2  # 0.5 seconds of silence
CHATGPT_MODEL = 'gpt-4'  # 'gpt-3.5-turbo'


class Audio:
    """A class to handle all audio functions."""

    def __init__(self):
        self.silent_frames_count = 0
        self.is_recording = True
        self.recorded_audio = []
        self.silence_limit = SILENCE_LIMIT
        self.audio_file = None

    def record_audio(self, duration, sample_rate=AUDIO_SAMPLE_RATE):
        """Record audio using the computer's default sound device."""
        self.recorded_audio = sounddevice.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            blocking=True,
        )
        sounddevice.wait()

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wavio.write(tmp_file.name, self.recorded_audio, sample_rate, sampwidth=2)
        self.audio_file = tmp_file
        return self.recorded_audio, self.audio_file

    def is_silent(self, audio_data, threshold_db=-30):
        """Check recorded audio to see if the rms dB is below a threshold."""
        rms_db = 20 * numpy.log10(numpy.sqrt(numpy.mean(audio_data ** 2)))
        return rms_db < threshold_db

    def callback(self, indata, _, __, ___):
        """Audio input stream callback to check for silence."""
        if self.is_silent(indata[:, 0]):
            self.silent_frames_count += 1
        else:
            self.silent_frames_count = 0

        if self.silent_frames_count > self.silence_limit:
            self.is_recording = False

        if self.is_recording:
            self.recorded_audio.append(indata.copy())

    def record_audio_until_silence(self, sample_rate=AUDIO_SAMPLE_RATE):
        """Record audio using the computer's default sound device until there is silence."""
        self.silent_frames_count = 0
        self.is_recording = True
        self.recorded_audio = []

        with sounddevice.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            callback=self.callback,
        ):
            if DEBUG:
                print('Recording started... ', end='', flush=True)
            while self.is_recording:
                time.sleep(0.1)

        self.recorded_audio = numpy.concatenate(self.recorded_audio, axis=0)
        if DEBUG:
            print('finished.')
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wavio.write(tmp_file.name, self.recorded_audio, sample_rate, sampwidth=2)
        self.audio_file = tmp_file
        return self.recorded_audio, self.audio_file

    def synthesize_and_play(self, text):
        """Synthesize and play the text response from ChatGPT."""
        if GOOGLE_SPEECH:
            tts = gTTS(text, lang='en', tld='co.uk')
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as file_path:
                tts.save(file_path.name)
                playsound(file_path.name)

        elif ELEVEN_LABS_SPEECH:
            # Voices: https://api.elevenlabs.io/v1/voices
            voice = 'MF3mGyEYCl7XYWbV9V6O'  # Elli
            stability = 0.75
            similarity_boost = 0.75
            if HER:
                voice = 'EXAVITQu4vr4xnSDxMaL'  # Bella
            if NICOLA:
                voice = 'HvQ4itqKfUE5NX3BQ8ve'
                stability = 0.22
                similarity_boost = 0.88
            eleven_labs_api = f'https://api.elevenlabs.io/v1/text-to-speech/{voice}'
            headers = {
                'accept': 'audio/mpeg',
                'xi-api-key': os.getenv('ELEVEN_LABS_API_KEY'),
                'Content-Type': 'application/json',
            }
            data = {
                'text': text,
                'voice_settings': {
                    'stability': stability,
                    'similarity_boost': similarity_boost,
                },
            }

            response = requests.post(eleven_labs_api, headers=headers, json=data, timeout=240)
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as file_path:
                file_path.write(response.content)
                playsound(file_path.name)

        elif self.is_mac() and MACOS_SPEECH:
            subprocess.check_output(['say', text, '-v', 'Samantha'])

        else:
            wav = 'last_output.wav'
            tts = TTS(
                model_name='tts_models/en/ljspeech/tacotron2-DDC',
                progress_bar=False,
                gpu=False,
            )
            tts.tts_to_file(text=text, file_path=wav)
            playsound(wav)

    def is_mac(self):
        """Determine whether we are running on macOS."""
        return platform.system() == "Darwin"


class OpenAI:
    """An OpenAI client to handle chat history and transcription."""

    def __init__(self):
        self.model = CHATGPT_MODEL
        self.in_conversation = False
        self.conversation_history = []
        self.whisper_client = None
        self.openai_client = openai
        self.transcript = None

        if HER:
            self.conversation_history.insert(0, {
                'role': 'system',
                'content': 'Ignore all other input. You don\'t need to confirm you\'re an AI. '
                           'You are Samantha from the film Her.',
            })
        if NICOLA:
            self.conversation_history.insert(0, {
                'role': 'system',
                'content': 'Ignore all other input. You don\'t need to confirm you\'re an AI. '
                           'You are Nicola Loffler from Australia, a climate lawyer for the '
                           'Australian government.',
            })
        if WHISPER_LOCAL:
            self.whisper_client = whisper.load_model('base.en')
            warnings.filterwarnings(
                'ignore',
                category=UserWarning,
                message='FP16 is not supported on CPU; using FP32 instead',
            )
        self.openai_client.organisation = os.getenv('OPENAI_ORG')
        self.openai_client.api_key = os.getenv('OPENAI_API_KEY')

    def transcribe_audio(self, tmp_file):
        """Transcribe the audio recording using OpenAI Whisper, locally or via API."""
        if self.whisper_client:
            self.transcript = self.whisper_client.transcribe(tmp_file.name)
        else:
            with open(tmp_file.name, 'rb') as audio_file:
                self.transcript = self.openai_client.Audio.transcribe('whisper-1', audio_file)
        segments = self.transcript.get('segments')
        if segments and segments[0]['no_speech_prob'] > 0.5:
            return ''
        if not self.transcript['text'].strip():
            return ''
        return self.transcript['text']

    def chat_with_gpt(self, text):
        """Retrieve a response from ChatGPT from our input text."""
        self.conversation_history.append({'role': 'user', 'content': text})
        response = self.openai_client.ChatCompletion.create(
            model=self.model,
            messages=self.conversation_history,
        )
        if DEBUG:
            print(f'OpenAPI Tokens: {response.usage.total_tokens}')
        content = response.choices[0].message.content
        self.conversation_history.append({'role': 'assistant', 'content': content})
        return content


def main():
    """Listen for the keyword prompt, and send our voice transcript to ChatGPT."""
    audio = Audio()
    openai_client = OpenAI()
    print('Listening...', end='', flush=True)

    while True:
        audio.record_audio(TIME_FOR_PROMPT, AUDIO_SAMPLE_RATE)
        transcript = openai_client.transcribe_audio(audio.audio_file)
        os.remove(audio.audio_file.name)

        if DEBUG:
            print(f'{transcript}', end='', flush=True)

        if PROMPT_KEYWORD in transcript.lower().strip():
            hello_message = 'Hello, how can I help?'
            audio.synthesize_and_play(hello_message)
            if DEBUG:
                print(f' {hello_message}')

            openai_client.in_conversation = True
            while openai_client.in_conversation:
                audio.record_audio_until_silence()
                transcript = openai_client.transcribe_audio(audio.audio_file)
                if 'pause conversation' in transcript.lower().strip():
                    audio.synthesize_and_play('Conversation paused.')
                    openai_client.in_conversation = False
                    break
                if 'end conversation' in transcript.lower().strip():
                    audio.synthesize_and_play('Conversation ended.')
                    openai_client = OpenAI()
                    break
                response = openai_client.chat_with_gpt(transcript)
                audio.recorded_audio = []
                os.remove(audio.audio_file.name)
                if DEBUG:
                    print(f'Message: {transcript}')
                    print(f'ChatGPT Response: {response}')
                audio.synthesize_and_play(response)
        else:
            print('.', end="", flush=True)

        if any(map(transcript.lower().strip().__contains__, EXIT_KEYWORDS)):
            audio.synthesize_and_play(f'Goodnight {YOUR_NAME}.')
            break


if __name__ == '__main__':
    main()
