import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import wavio

from chat import Audio, OpenAI


@pytest.fixture
def audio():
    return Audio()


@pytest.fixture
def openai_client():
    return OpenAI()


def test_is_silent(audio):
    loud_audio_data = np.array([1.0] * 100)
    silent_audio_data = np.array([0.0] * 100)

    assert not audio.is_silent(loud_audio_data)
    assert audio.is_silent(silent_audio_data)


def test_is_mac(audio):
    import platform

    if platform.system() == "Darwin":
        assert audio.is_mac()
    else:
        assert not audio.is_mac()


def test_record_audio(audio):
    audio.record_audio = MagicMock(return_value=(None, None))
    audio.record_audio(1)
    audio.record_audio.assert_called_once_with(1)


def test_record_audio_until_silence(audio):
    audio.record_audio_until_silence = MagicMock(return_value=(None, None))
    audio.record_audio_until_silence()
    audio.record_audio_until_silence.assert_called_once()


@patch('chat.whisper.transcribe', return_value={'segments': [{'no_speech_prob': 0.9}]})
def test_transcribe_audio(mock_transcribe, openai_client):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
        audio.recorded_audio = np.random.rand(16000)
        wavio.write(tmp_file.name, audio.recorded_audio, 16000, sampwidth=2)
        assert not openai_client.transcribe_audio(tmp_file)


def test_chat_with_gpt(openai_client):
    openai_client.openai_client.ChatCompletion.create = MagicMock()
    response_mock = MagicMock()
    response_mock.choices[0].message.content = 'test response'
    response_mock.usage.total_tokens = 42
    openai_client.openai_client.ChatCompletion.create.return_value = response_mock

    response = openai_client.chat_with_gpt('Hello, how are you?')
    assert response == 'test response'


@patch('chat.GOOGLE_SPEECH', True)
def test_synthesize_and_play_google_speech(audio):
    with patch('chat.gTTS') as mock_gtts, patch('chat.playsound') as mock_playsound:
        audio.synthesize_and_play('Hello')
        mock_gtts.assert_called_once_with('Hello', lang='en', tld='co.uk')
        mock_playsound.assert_called_once()


def test_synthesize_and_play_eleven_labs_speech(audio):
    with patch('chat.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.content = b'mocked_content'
        mock_post.return_value = mock_response

        with patch('chat.playsound') as mock_playsound:
            audio.synthesize_and_play('Hello')
            mock_post.assert_called_once()
            mock_playsound.assert_called_once()


@patch('chat.GOOGLE_SPEECH', False)
@patch('chat.MACOS_SPEECH', True)
@patch('chat.ELEVEN_LABS_SPEECH', False)
def test_synthesize_and_play_macos_speech(audio):
    with patch('chat.platform.system', return_value="Darwin"), patch('chat.subprocess.check_output') as mock_check_output:
        audio.synthesize_and_play('Hello')
        mock_check_output.assert_called_once_with(['say', 'Hello', '-v', 'Samantha'])


@patch('chat.GOOGLE_SPEECH', False)
@patch('chat.MACOS_SPEECH', False)
@patch('chat.ELEVEN_LABS_SPEECH', False)
def test_synthesize_and_play_tts(audio):
    with patch('chat.TTS') as mock_tts, patch('chat.playsound') as mock_playsound:
        audio.synthesize_and_play('Hello')
        mock_tts.assert_called_once_with(
            model_name='tts_models/en/ljspeech/tacotron2-DDC',
            progress_bar=False,
            gpu=False,
        )
        mock_playsound.assert_called_once()
