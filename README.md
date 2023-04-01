# ChatGPT voice assistant

A voice assistant using [OpenAI Whisper](https://openai.com/research/whisper), [ChatGPT completion API](https://platform.openai.com/docs/api-reference/chat/create), and text-to-speech.

By default it emulates Samantha from the movie [Her](https://www.imdb.com/title/tt1798709/).

## How it works

* OpenAI Whisper running locally listens for a keyword in every 4 seconds of audio
* When it hears that keyword it listens for your question
* That question is sent to ChatGPT
* ChatGPT's response is synthesised into speech, and saved into a conversation history
* You can end a conversation with the keywords `end conversation` or pause it with `pause conversation`
* You can then start fresh or resume your conversation by saying the prompt keyword again

## Installation

* Install virtualenv: `pip install virtualenv`
* Create a virtual environment: `virtualenv venv`
* Activate your environment: `source venv/bin/activate`
* Install the Python dependencies: `pip install -r requirements.txt`
* Create a `.envrc` file: `cp template.envrc .envrc`
* Fill in your API keys
* Allow the environment variables to be loaded `direnv allow`

## Create API Keys

* Create an OpenAI API Key: https://platform.openai.com/account
* Eleven Labs voice API Key: https://beta.elevenlabs.io

## Running

* After activating your environment, run: `python chat.py`
* First say the keyword, which by default is `samantha`
* Then ask your question

### Options

* Set `ELEVEN_LABS_SPEECH=True` to use [Eleven Labs text-to-speech voices](https://api.elevenlabs.io/docs#/text-to-speech) (default)
* Set `GOOGLE_SPEECH=True` to use [Google text-to-speech](https://gtts.readthedocs.io/en/latest/)
* Set `MACOS_SPEECH=True` to use built in [macOS say text-to-speech](https://ss64.com/osx/say.html)
* Set all of the above `False` to use [TTS text-to-speech](https://github.com/coqui-ai/TTS)
* Set `HER=False` to use ChatGPT defaults, and not pretend to be Samantha from Her

### Notes

There are some delays in response, these are currently:

* Whisper returning the transcription of your audio
* ChatGPT returning the response to your question
* Synthesising the text-to-speech voice audio

## TODO

* Train ChatGPT on supplied documents/papers/embeddings/plugins
* Train Eleven Labs voices on `.wav` recordings
* Use [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) for full off-line functionality
