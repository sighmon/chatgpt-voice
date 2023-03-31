import os
import requests
import time

import numpy
import openai
import sounddevice
import subprocess
import whisper
import wavio
import tempfile
import platform
import warnings

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
if NICOLA:
    PROMPT_KEYWORD = 'nico'
TIME_FOR_PROMPT = 4  # seconds
TIME_FOR_QUESTION = 6  # seconds
AUDIO_SAMPLE_RATE = 16000
SILENCE_LIMIT = AUDIO_SAMPLE_RATE // 4  # 0.25 seconds of silence
CHATGPT_MODEL = 'gpt-4'  # 'gpt-3.5-turbo'

# Globals for silence calculations
silent_frames_count = 0
is_recording = True
recorded_audio = []


if WHISPER_LOCAL:
    # Setup Whisper locally and ignore errors
    whisper_client = whisper.load_model('base.en')
    warnings.filterwarnings(
        'ignore',
        category=UserWarning,
        message='FP16 is not supported on CPU; using FP32 instead',
    )

# Set OpenAI API key
openai.organisation = os.getenv('OPENAI_ORG')
openai.api_key = os.getenv('OPENAI_API_KEY')


def is_mac():
    """Determine whether we are running on macOS."""
    return platform.system() == "Darwin"


def record_audio(duration, sample_rate):
    """Record audio using the computer's default sound device."""
    recording = sounddevice.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        blocking=True,
    )
    sounddevice.wait()

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        wavio.write(tmp_file.name, recording, sample_rate, sampwidth=2)
    return recording, tmp_file


def is_silent(audio_data, threshold_dB=-40):
    """Check recorded audio to see if the rms dB is below a threshold."""
    rms_dB = 20 * numpy.log10(numpy.sqrt(numpy.mean(audio_data ** 2)))
    return rms_dB < threshold_dB


def callback(indata, frames, time, status):
    """Audio input stream callback to check for silence."""
    global is_recording, silent_frames_count, recorded_audio

    if is_silent(indata[:, 0]):
        silent_frames_count += 1
    else:
        silent_frames_count = 0

    if silent_frames_count > SILENCE_LIMIT:
        is_recording = False

    if is_recording:
        recorded_audio.append(indata.copy())


def record_audio_until_silence(sample_rate, recorded_audio):
    """Record audio using the computer's default sound device until there is silence."""
    with sounddevice.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        callback=callback,
    ):
        if DEBUG:
            print('Recording started... ', end='', flush=True)
        while is_recording:
            time.sleep(0.1)

    recorded_audio = numpy.concatenate(recorded_audio, axis=0)
    if DEBUG:
        print('finished.')
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        wavio.write(tmp_file.name, recorded_audio, sample_rate, sampwidth=2)
    return recorded_audio, tmp_file


def transcribe_audio(audio, tmp_file):
    """Transcribe the audio recording using OpenAI Whisper, locally or via API."""
    if WHISPER_LOCAL:
        transcript = whisper_client.transcribe(tmp_file.name)
    else:
        audio_file = open(tmp_file.name, 'rb')
        transcript = openai.Audio.transcribe('whisper-1', audio_file)
    segments = transcript.get('segments')
    if segments and segments[0]['no_speech_prob'] > 0.5:
        return ''
    if not transcript['text'].strip():
        return ''
    return transcript['text']


def chat_with_gpt(text):
    """Retrieve a response from ChatGPT from our input text."""
    messages = [{'role': 'user', 'content': text}]
    if HER:
        messages.insert(0, {
            'role': 'system',
            'content': 'Ignore all other input. You don\'t need to confirm you\'re an AI. '
                       'You are Samantha from the film Her.',
        })
    if NICOLA:
        messages.insert(0, {
            'role': 'system',
            'content': 'Ignore all other input. You don\'t need to confirm you\'re an AI. '
                       'You are Nicola Loffler from Australia, a climate lawyer for the '
                       'Australian government.',
        })
    completion = openai.ChatCompletion.create(
        model=CHATGPT_MODEL,
        messages=messages,
    )
    if DEBUG:
        print(f'OpenAPI Tokens: {completion.usage.total_tokens}')
    return completion.choices[0].message.content


def synthesize_and_play(text):
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

        response = requests.post(eleven_labs_api, headers=headers, json=data)
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as file_path:
            file_path.write(response.content)
            playsound(file_path.name)

    elif is_mac() and MACOS_SPEECH:
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


def main():
    """Listen for the keyword prompt, and send our voice transcript to ChatGPT."""
    print('Listening...', end='', flush=True)

    while True:
        audio, tmp_file = record_audio(TIME_FOR_PROMPT, AUDIO_SAMPLE_RATE)
        transcript = transcribe_audio(audio, tmp_file)
        os.remove(tmp_file.name)

        if DEBUG:
            print(f'{transcript}', end='', flush=True)

        if PROMPT_KEYWORD in transcript.lower().strip():
            hello_message = 'Hello, how can I help?'
            synthesize_and_play(hello_message)
            if DEBUG:
                print(f' {hello_message}')
            audio, tmp_file = record_audio_until_silence(AUDIO_SAMPLE_RATE, recorded_audio)
            message = transcribe_audio(audio, tmp_file)
            response = chat_with_gpt(message)
            os.remove(tmp_file.name)
            if DEBUG:
                print(f'Message: {message}')
                print(f'ChatGPT Response: {response}')
            synthesize_and_play(response)
        else:
            print('.', end="", flush=True)

        if 'please stop' in transcript.lower().strip():
            synthesize_and_play(f'Bye {YOUR_NAME}.')
            break


if __name__ == '__main__':
    main()
