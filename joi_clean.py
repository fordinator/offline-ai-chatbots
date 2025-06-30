
#         ,--.-,   _,.---._     .=-.-. 
#         |==' -| ,-.' , -  `.  /==/_ / 
#         |==|- |/==/_,  ,  - \|==|, |  
#       __|==|, |==|   .=.     |==|  |  
#    ,--.-'\=|- |==|_ : ;=:  - |==|- |  
#    |==|- |=/ ,|==| , '='     |==| ,|                      
#    |==|. /=| -|\==\ -    ,_ /|==|- |          
#    \==\, `-' /  '.='. -   .' /==/. /  
#      --`----'     `--`--''   `--`-`  

"""
     CONFIGURABLE VOICE CHAT ASSISTANT            
             FOR THE MASSES
"""

"""CONFIGURABLES: THESE ARE THE ONLY PARTS OF THE CODE YOU SHOULD CHANGE"""


###################################################################################################################################################
# CORE SETTINGS: Select which LLM and Text-To-Speech Generator you wish to use. Several options are available to reflect differing budgets and
# system configurations. 
#
# OpenAI is of course the most advanced LLM and by far the most reliable and believable. The specific OpenAI model I have selected -- gpt-4o -- 
# is the same one which is used by default for ChatGPT's web interface and tends to be a good balance between detail, creativity, responsiveness 
# and cost. GPT-4o costs about 10 cents to talk to for about 30 minutes. Joi can be configured to work with more complex models that are more 
# intelligent but more expensive. Consult OpenAI's pricing and model charts for details.
#
# MistralAI is the next best LLM, fairly intelligent but less reliable and believable. The specific MistralAI model I have selected -- mixtral8x7b,
# was recommended as a strong option for role-playing because it is more free-wheeling than most. However it tends to drift from its instructions 
# a lot. It will cost you about 1 cent to talk to 8x7b for about an hour. Mistral has many more available model codes than OpenAI with differing 
# capabilities, consult MistralAI's pricing and model charts for details.
# 
# Ollama is the cheapest model LLM -- absolutely free -- and by selecting your own specific model it can approach the advanced capabilities of 
# OpenAI. It is also the most private -- it runs on your local system, does not communicate with the Internet at all. With enough dedication
# you can also find specific GGUF models that are completely uncensored and will happily discuss toxic, harmful or even adult situations with you.
# However it requires a beefy system and a lot more technical expertise to set up. You probably shouldn't even try to use this model unless you
# possess an NVIDIA GeForce 2070 Super or better. I must inform you though, that setting up Ollama is outside the scope of this document and 
# you will have to do a lot of reading. I recommend Googling some tutorials if you are interested. It's easy to get lost in it!
###################################################################################################################################################

PRIMARY_LLM = "OPENAI"
MEMORIES_LLM = ""
# Valid values are: OPENAI, GOOGLE, ANTHROPIC, MISTRAL, OLLAMA
# THE MEMORIES LLM MUST BE DIFFERENT FROM THE PRIMARY LLM

VOICE = "ELEVENLABS"
# Valid values are: KOKORO, ELEVENLABS, PLAYHT, GVOICE, AZURE

###################################################################################################################################################
# SYSTEM PROMPT: This is the CORE of how the bot will respond. You should be as concise as you can while still making the bot unique. It responds 
# best to simple, direct commmands, as you can see for caco-bot, below. It is STRONGLY RECOMMENDED that you leave the portion at the end about a  
# VERY brief response or GPT will talk your ear off and eat up all those precious voice credits.                                              
#########################################################################################################faudio##########################################

system_message = """"""

memories_system_message = """ """

###################################################################################################################################################
# CHARACTER VOICE: This is the fun part. You need to go to the voice websites and find a voice you like.                                          
#                                                                                                                                                 
# ElevenLabs is by far the most advanced, natural text to speech service, many times you will almost be unable to tell you aren't talking to
# someone in voice chat. However it is also very expensive -- it costs about 5 cents to speak the answer to ONE QUESTION. Also, if you select a voice
# sample marked with credit multipliers (like x2 and x3) the costs can quickly become unsustainable. Another wrinkle is that ElevenLabs
# requires you to add a voice to your "Library" before you can use it. Currently I have it set to "Vexation" because that's me and it
# saves my wallet. However I'm a bit of a flat, boring mouth-breather so you will probably want to change it.           
#
# Play.HT is the next best model. It sounds natural for the most part but sometimes bungles words or comes across as a bit robotic. I also believe
# the API is not as reliable as OpenAI's -- I often get HTTP 500 errors after long periods of usage. However it is much cheaper -- only about 3
# cents per answer and if you're willing to spend a healthy chunk, almost unlimited except for a monthly fee. There is no need to add any voices to
# anything before you use it, all you need to do is input the proper URL for the voice sample.                                                                                 
#                                                                                                                                                    
# Microsoft Azure Speech is the last option, and also the least believable. This one is obviously a computer and not much more natural 
# sounding than text-to-speech services which already exist in Windows. However it is extremely cheap -- less than one cent a question. You can
# also configure the voice samples to have a default "emotional style" like angry, informational, cheerful, etc. It's also more complex to set
# up than ElevenLabs and requires you to create a Microsoft Azure account, set up an organization, configure the tenancy and the access rights, 
# and a bunch of other nonsense. However if you really want to get into long answers or periods of roleplay you may have no other viable choice.                                                               
###################################################################################################################################################

KOKORO_VOICE_ID = ""
ELEVENLABS_VOICE_ID = ""
PLAYHT_VOICE_ID = ""
GVOICE_VOICE_ID = ""
GVOICE_LANGUAGE = ""
AZURE_VOICE_ID = ""
AZURE_EMOTION = ""
# Valid values are: chat, cheerful, empathetic, angry, sad, serious, friendly, assistant, newscast, customer service

"""
Jennifer Love Hewitt - 'Jen3': YsG2x9Q3FCB8VG7GZo61
ElevenLabs Voice ID for myself -- 'Vexation': VRWmHsP8ooUA1LFV8QEM
Recommended male Kokoro Voice ID -- am_onyx
Recommended female Kokoro Voice ID -- bf_emma
Recommended male ElevenLabs Voice ID -- 'Eastend Steve': 1TE7ou3jyxHsyRehUuMB
Recommended female ElevenLabs Voice ID -- 'Callie - Kind and relatable': 7YaUDeaStRuoYg3FKsmU
Recommended male Play.HT Voice ID -- 'Samuel': s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json
Recommended female Play.HT VoiceID -- 'Delilah': s3://voice-cloning-zero-shot/1afba232-fae0-4b69-9675-7f1aac69349f/delilahsaad/manifest.json
Recommended male Google Voice ID - en-US-Standard-D, en-US
Recommended female Google Voice ID - en-GB-Studio-C, en-GB
Recommended male Azure Voice ID -- 'Tony': en-US-TonyNeural with "chat" option
Recommended female Azure Voice ID -- 'Jane': en-US-JaneNeural with "chat" option
"""

###################################################################################################################################################
# API KEYS: These MUST be set or nothing will work. Dig around on the websites for each service until you find out where to                        
# generate the API Keys and other secret codes you may need. Often they are in sections called "API Reference."                                   
###################################################################################################################################################

OPENAI_API_KEY = ""
GOOGLE_API_KEY = ""
ANTHROPIC_API_KEY = ""
MISTRAL_API_KEY = ""

ELEVENLABS_API_KEY = ""
PLAYHT_USER_ID = ""
PLAYHT_SECRET_KEY = ""
GVOICE_JSON = ""
AZURE_SUBSCRIPTION_KEY = ""
AZURE_REGION = ""

###################################################################################################################################################
# OPTIONAL VARIABLES: It is recommended to leave these alone but you can change them if you really want to tweak them. Brief descriptions of      
# what each options does are provided below.                                                                                                      
###################################################################################################################################################                                                                                                                                           
# LLM_MODEL = Needs to be a recognized LLM "model code" as described in the documentation. For OpenAI, I recommend leaving this at "gpt-4o" its 
# simple enough to be cheap but smart enough for conversatifonal chat responses. For Mistral, I recommend leaving this at "open-mixtral-8x7b" its 
# an extremely creative varied model that actually costs next to nothing for each message.

# LLM_TEMPERATURE = Value from 0 to 2 with decimals. The lower the value the more the bot sticks to its system prompt. The higher the value the more
# creative it gets. OpenAI recommends for best results, this is the only actual numerical value you should edit.

# LLM_MAX_TOKENS = This limits how wordy the responses can get. Since the voice generators basically charges you for every character of text, 
# you can use this number as an emergency stop to spare your budget. A full sentence is about 20 tokens, once it hits the limit it will just stop.

# VOICE_MODEL = Needs to be a valid "model code" as described in the voice generator's documentation. For ElevenLabs, I recommend leaving this at 
# "eleven_turbo_v2_5" since it is the most advanced model that is still half price. For Play.HT, I recommend leaving this at "Play3.0-mini" since I 
# THINK it's the only model that works with my Python library.
#
# INITIAL_GREETING = By setting this variable to True, you can specify that the LLM give you a friendly custom greeting each time it launches, before
# you record any messages. The GREETING_TEXT variable allows you to specify what you would like that greeting to be.
#
# CHAT HISTORY = Chat history is off by default, so this bot will start with a "clean slate" every time you launch it. However, by setting
# CHAT_HISTORY to True you can save the record of every interaction to a file, with a filename specified in the next variable. This file can be
# named anything, but it MUST end in .json. You can reset the bot's memory by deleting the file, or edit it selectively to change the past. You can
# also keep multiple chat histories by backing them up, or by changing the name of the file in that variable.

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.8
OPENAI_MAX_TOKENS = 2048

GOOGLE_MODEL = "gemini-2.0-flash"
GOOGLE_TEMPERATURE = 1.0
GOOGLE_MAX_TOKENS = 2048

CLAUDE_MODEL = "claude-3-haiku-20240307"
CLAUDE_MAX_TOKENS = 2048

MISTRAL_MODEL = "open-mixtral-8x7b"
MISTRAL_TEMPERATURE = 1.0
MISTRAL_MAX_TOKENS = 2048

ELEVENLABS_MODEL = "eleven_turbo_v2_5"
PLAYHT_MODEL = "Play3.0-mini-http"

DYNAMIC_GREETING = False
GREETING_PROMPT = "What the fuck do YOU want?"

SAVE_HISTORY = False
SAVE_HISTORY_FILE = "chat_history.json"

###################################################################################################################################################
# OLLAMA CONFIGURABLES: These values only apply when using a local model with Ollama.                                                             
#                                                                                                                                                 
# OLLAMA MODEL = Here you can specify a valid local LLM model as imported into Ollama. Most models you can get from HuggingFace require           
# conversion to work with Ollama. Here I am using an uncensored NSFW roleplay model called "L3-8B-Stheno-v3.2-GGUF-IQ-Imatrix" converted to the   
# name "stheno". I like it.                                                                                                                       
#                                                                                                                                                 
# FINAL CORRECTIONS = Basically local LLMs called with langchain tend to drift and ignore the system prompt eventually, as the chat history       
# grows and the limited instruction set becomes overwhelmed with information. It will ramble on for longer and longer periods -- which has 
# varied financial costs depending on your speech service -- and sometimes starts describing itself and the user in third person, or taking on 
# the role of the user himself -- or even BOTH assistant AND user in an unholy self-referential virtual diorama. I am researching ways to correct
# this which will be included in the code when discovered.                                                                  
#                                                                                                                                                   
# For now, you can implement some stopgap fixes and set FINAL_CORRECTIONS to True. The FINAL_CORRECTIONS module will perform a series of    
# very hacky "reminder prompts" if the LLM starts to drift. The first one checks to see that the model is identifying its own role properly and   
# not pretending to be someone else. The second attempts to limit responses to fewer than a configurable maximum number of words. The third       
# instructs the model to strip out role prefixes, LLM instructions and irrelevant punctuation.                                                    
#                                                                                                                                                 
# MAX_WORDS = The default for FINAL_CORRECTIONS is to instruct and remind the model to always respond with fewer than 100 words. You can tighten  
# or loosen this restriction by entering a higher or lower number. I may implement a MAX_TRUNCATE function in the future to force the damn bot to 
# shut up if the costs of testing become too prohibitive.                                                                              
###################################################################################################################################################

"""NOTE: Embarassing as it may be, these FINAL CORRECTIONS currently assume an NSFW interaction with a female assistant."""

OLLAMA_MODEL=""
FINAL_CORRECTIONS = False
MAX_WORDS = 100 

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#                       BEGIN CODE              
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 
import io
import re
import select
import sys
import os
import time
import traceback

import threading
import requests
import numpy as np
import sounddevice as sd
import json

import whisper

from rich.console import Console
from queue import Queue
from typing import Union, Generator, Iterable

from openai import Client
from google import genai
from google.genai import types
import anthropic
from mistralai import Mistral

from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pyht import Client as Client_Voice
from pyht.client import TTSOptions
# from google.cloud import texttospeech
from google.oauth2 import service_account

from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

_stt_model = whisper.load_model("base.en")

console = Console()

    
def format_message(role, content):
    """
    General function for formatting a message to send to an LLM.
    """
        
    return {
        "role": role,
        "content": f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    }

############################################################
#                CHAT HISTORY CLASS
############################################################ 

class ChatHistory:
    
    console = Console()
    
    def _ollama_system_entry(system_msg: str) -> dict:
        """Wrap the system message for Ollama's special header format."""
        content = f"<|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>\n\n"
        return {"role": "system", "content": content},
          
    def read_history(self):
        """
        Load chat history from SAVE_HISTORY_FILE if SAVE_HISTORY is True and the file is valid,
        otherwise return the appropriate default for PRIMARY_LLM.
        """
        
        # 1) Define each LLM's *default* history
        defaults = {
            "OPENAI":   [{"role": "system", "content": system_message}],
            "MISTRAL":  [{"role": "system", "content": system_message}],
            "GOOGLE":   [], # Google & Anthropic start with an empty history
            "ANTHROPIC": [],
            "OLLAMA":   [format_message(role="system", content=system_message)],
        }
        
        # 2) Check LLM support
        default_history = defaults.get(PRIMARY_LLM)
        if default_history is None:
            console.print("[red]Error. Check your LLM type variable")
            return []
            
        # 3) If we're *not* saving history, return the default immediately
        if not SAVE_HISTORY:
            return default_history
            
        # 4) Otherwise, attempt to load the file once
        try:
            with open(SAVE_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Missing file or broken JSON? Fall back to the default
            return default_history

        return chat_history 
        
        
############################################################
#                   STT CLASS
############################################################ 

class SpeechToTextService:
      
    stop_event = threading.Event()    
    
    def __init__(self, model_name: str = "base.en"):
        self._stt_client = whisper.load_model(model_name)
    
    def record_voice(self, stop_event: threading.Event, q: Queue) -> np.ndarray:
        """
        Record from microphone until user presses Enter.
        Returns a numpy array of audio samples.
        """   
        def callback(indata, frames, time_, status):
            q.put(indata.copy())
            
            # Collect all buffers into one array
            # audio_data = b"".join(q.queue)
            # audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
      
        with sd.InputStream(
            samplerate=16000, 
            dtype="int16", 
            channels=1, 
            callback=callback
        ):
        
            while not stop_event.is_set():
                time.sleep(0.1)

    def transcribe_user(self, audio_np: np.ndarray) -> str:
        """
        Transcribes the given audio data using the Whisper model.
        """
        
        result = self._stt_client.transcribe(audio_np, fp16=False) 
        return result["text"].strip()

    
############################################################
#                   TTS CLASS
############################################################   
    
class TextToSpeechService:
    
    def play_audio(self, mp3_bytes: bytes) -> None:
        """Play MP3 data in-memory."""
        play(mp3_bytes)
             
    def synthesize_elevenlabs(self, text: str):
        """
        Convert text to MP3 speech audio via Elevenlabs.
        Returns raw MP3 bytes.
        """
        _tts_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        
        return _tts_client.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
            output_format="mp3_44100_128"
        )
        
    def synthesize_playht(self, text: str) -> bytes:
        """
        Convert text to MP3 speech audio via PlayHT.
        Returns raw MP3 bytes.
        """
        _tts_client = Client_Voice(
            user_id=PLAYHT_USER_ID,
            api_key=PLAYHT_SECRET_KEY,
        )
        
        options = TTSOptions(voice=PLAYHT_VOICE_ID)  
            
        audio_buffer = b""
        
        for chunk in _tts_client.tts(
            text, options, voice_engine=PLAYHT_MODEL, protocol='http'
        ):
            audio_buffer += chunk
                 
        return audio_buffer
        
    def synthesize_gvoice(self, text: str) -> bytes:
        """
        Convert text to MP3 speech audio via Google Voice.
        Returns raw MP3 bytes.
        """
        credentials = service_account.Credentials.from_service_account_file(GVOICE_JSON)
        
        client_gvoice = texttospeech.TextToSpeechClient(credentials=credentials)
        
        ssml_input = f"<speak>{text}</speak>"
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_input)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=GVOICE_LANGUAGE,
            name=GVOICE_VOICE_ID
        )
        
        audio_config = texttospeech.AudioConfig (
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
   
        response = client_gvoice.synthesize_speech (
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
    def synthesize_azure(
        self,
        text: str,
        subscription_key: str = AZURE_SUBSCRIPTION_KEY,
        region: str = AZURE_REGION,
        voice_name: str = AZURE_VOICE_ID,
        style: str = AZURE_EMOTION,
        output_format: str = "audio-16khz-128kbitrate-mono-mp3"
    ) -> bytes:
        """
        Synthesizes text into speech using Azure Speech REST API 
        and returns audio in bytes.
        """
        
        ssml_body = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
            <voice name="{voice_name}">
                <mstts:express-as style="{style}">
                    {text}
                </mstts:express-as>
            </voice>       
        </speak>
        """

        endpoint = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"

        headers = {
            "Ocp-Apim-Subscription-Key": subscription_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": output_format,
            "User-Agent": "joi-AzureTTS" 
        }

        response = requests.post(endpoint, headers=headers, data=ssml_body.encode("utf-8"))

        if response.status_code == 200:
            return response.content 
        else:
            raise Exception(
                f"Azure TTS request failed: {response.status_code}, {response.text}"
            )
    
    def synthesize_kokoro(self, text: str):
        """
        Converts text to MP3 speech audio via Kokoro-FastAPI. 
        Returns raw MP3 bytes.
        """
        _tts_client = Client (
            base_url="http://localhost:8880/v1", api_key="not-needed"
        )

        with _tts_client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice=KOKORO_VOICE_ID,
            input=text
        ) as response:
            response.stream_to_file("output.mp3")

        with open("output.mp3", "rb") as f:
            audio =f.read()
        
        return audio


############################################################
#                   LLM CLASS
############################################################ 

class QueryLLMService:
        
    def generate_openai(self, chat_history: list) -> str:
        """
        Generates a chat response using the OpenAI API.
        """
        
        _llm_client = Client(api_key=OPENAI_API_KEY)    
    
        response = _llm_client.chat.completions.create(
            model=OPENAI_MODEL, 
            messages=chat_history,
            temperature=OPENAI_TEMPERATURE,
            max_completion_tokens=OPENAI_MAX_TOKENS,
        )   
    
        return response.choices[0].message.content.strip() 
    
    def generate_gemini(self, chat_history: list) -> str:
        """
        Generates a chat response using the Google Gemini API.
        """
    
        _llm_client = genai.Client(api_key=GOOGLE_API_KEY)
    
        response = client_google.models.generate_content(
            model=GOOGLE_MODEL,
            contents=chat_history,
            config=types.GenerateContentConfig(
                system_instruction=system_message,
                temperature=GOOGLE_TEMPERATURE,
                max_output_tokens=GOOGLE_MAX_TOKENS,
            )   
        )
    
        return response.text
 
    def generate_anthropic(self, chat_history: list) -> str:
        """
        Generates a chat response using the Anthropic API.
        """
    
        _llm_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
        response = _llm_client.messages.create(
            max_tokens=CLAUDE_MAX_TOKENS,
            system=system_message,
            messages=chat_history,
            model=CLAUDE_MODEL,
        )
    
        return "".join(block.text for block in response.content if block.type == "text")
     
    def generate_mistral(self, chat_history: list) -> str:
        """
        Generates a chat response using the Mistral API.
        """
    
        _llm_client = Mistral(api_key=MISTRAL_API_KEY)
    
        response = _llm_client.chat.complete(
            model=MISTRAL_MODEL, 
            messages=chat_history,
            temperature=MISTRAL_TEMPERATURE,
            max_tokens=MISTRAL_MAX_TOKENS
        )
        
        return response.choices[0].message.content.strip()    
    
    def generate_ollama(self, user_text: str, chat_history: list, session_id: str = "default_session") -> str:
        """
        Generates a chat response using Ollama.
        """
   
        prompt_template = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("ai", "<|start_header_id|>assistant<|end_header_id|>\n\n")
        ])

        formatted_prompt = prompt_template.format_messages(chat_history=chat_history)

        llm = OllamaLLM(model=OLLAMA_MODEL, stop=["<|eot_id|>"])
    
        formatted_text = "\n".join([msg.content for msg in formatted_prompt])
    
        output = llm.invoke(formatted_prompt)

        if isinstance(output, str):
            return output
        else:
            messages = getattr(output, "messages", output)
            for message in reversed(messages):
                if hasattr(message, "content") and message.content:
                    return message.content
            return str(output)

# Create a single services instance to reuse
tts = TextToSpeechService()
stt = SpeechToTextService()
llm = QueryLLMService()
hist = ChatHistory()


############################################################
#             GLOBAL DISPATCH TABLES
############################################################ 

LLM_DISPATCH = {
    "OPENAI":   (lambda text, hist: llm.generate_openai(hist),       "earth"),
    "GOOGLE":   (lambda text, hist: llm.generate_gemini(hist),       "earth"),
    "ANTHROPIC": (lambda text, hist: llm.generate_anthropic(hist),   "earth"),
    "MISTRAL":  (lambda text, hist: llm.generate_mistral(hist),      "earth"),
    "OLLAMA":   (lambda text, hist: llm.generate_ollama(text, hist),       "dots"),
}

TTS_DISPATCH = {
    "KOKORO":   (lambda txt: tts.synthesize_kokoro(txt),         "dots"),
    "ELEVENLABS": (lambda txt: tts.synthesize_elevenlabs(txt),   "earth"),
    "PLAYHT":   (lambda txt: tts.synthesize_playht(txt),         "earth"),
    "GVOICE":   (lambda txt: tts.synthesize_gvoice(txt),         "earth"),
    "AZURE":    (lambda txt: tts.synthesize_azure(txt),          "earth"),
}


def speak_greeting(content: str, dynamic: bool = False, chat_history: dict = []) -> str:
    """
    If dynamic is True, send `test` as a user prompt to the LLM
    and use the LLM's reply as the greeting. Otherwise, use `text`
    directly as a static greeting. In both cases, append to history,
    print, synthesize via TTS, play, and return the final greeting.
    """
    
    # Helper to append into chat-history with the correct format
    
    def _append(role: str, content: str):
        if PRIMARY_LLM == "GOOGLE":
            chat_history.append({"role": role, "parts": [{"text": content}]})
        elif PRIMARY_LLM == "OLLAMA":
            chat_history.append(format_message(role, content))
        else:   # OPENAI, ANTHROPIC, MISTRAL
            chat_history.append({"role": role, "content": content})
    
    if dynamic:
        
        # 1) Append the user's greeting prompt
        _append("user", content)
    
        # 2) Dispatch to the right LLM function
         
        gen_func, spinner = LLM_DISPATCH.get(PRIMARY_LLM, (None, None))
        
        status_msg = "Generating a response..."
        
        if not gen_func:
            console.print("[red]Error. Bad LLM. Check your LLM type variable")
            return ""
        
        with console.status(status_msg, spinner=spinner):
            greeting = gen_func(content, chat_history)
            
    else:
        greeting = content
    
    # 3) Append the assistant's reply
    _append("assistant", greeting)
    
    console.print(f"[cyan]Assistant: {greeting}")
    
    # 4) Dispatch to the right TTS synth function
    
    synth_func, spinner = TTS_DISPATCH.get(VOICE, (None, None))
    if not synth_func:
        console.print("[red]Error. Bad TTS. Check your TTS type variable")
        return greeting
        
    status_msg = "Generating audio..."
    
    with console.status(status_msg, spinner=spinner):
        audio_array = synth_func(greeting)
        
    # 5) Play it and return the text
    tts.play_audio(audio_array)
    return greeting
    
############################################################
#                     HELPERS FOR MAIN LOOP
############################################################

def append_to_history(role: str, text: str):
    """Append a message to chat_history in the correct format."""
    if PRIMARY_LLM == "GOOGLE":
        chat_history.append({"role": role, "parts": [{"text": "text"}]})
    elif PRIMARY_LLM == "OLLAMA":
        chat_history.append(format_message(role, text))
    else:   # OPENAI, ANTHROPIC, MISTRAL
        chat_history.append({"role": role, "content": text})

def process_user_text(user_text: str):
    """Handles the full text->LLM->TTS pipeline for any user_text."""
    
    # 1) Record user in history + print
    append_to_history("user", user_text)
    console.print(f"[yellow]You: {user_text}")
    
    # 2) Generate LLM reply
    gen_func, spinner = LLM_DISPATCH.get(PRIMARY_LLM, (None, None))
    if not gen_func:
        console.print("[red]Error. Bad LLM. Check your LLM type variable")
        return ""
        
    status_msg = "Generating a response..."
        
    with console.status(status_msg, spinner=spinner):
        assistant_reply = gen_func(user_text, chat_history)
        
    # 3) Append assistant + print
    append_to_history("assistant", assistant_reply)
    console.print(f"[cyan]Assistant: {assistant_reply}")
    
    # 4) Synthesize via TTS and play
    synth_func, spinner = TTS_DISPATCH.get(VOICE, (None, None))
    if not synth_func:
        console.print("[red]Error. Bad TTS. Check your TTS type variable")
        return
    
    status_msg = "Generating audio..."
    
    with console.status(status_msg, spinner=spinner):
        audio_array = synth_func(assistant_reply)
    
    # 5) Play it and return the text
    tts.play_audio(audio_array)
  
def get_text_input() -> str:
    """Prompt the user for text input."""
    return console.input(
        "[green bold]Type your prompt (or /voice to switch):"
    ).strip()
    
def get_voice_input() -> str:
    """Record, transcribe, and return the user's speech."""
    console.print("[green]Recording in voice mode. Press Enter to stop...")
    
    data_queue = Queue()
    stop_event = threading.Event()
    
    recording_thread = threading.Thread(
        target=stt.record_voice,
        args=(stop_event, data_queue),
    )
    
    recording_thread.start()
    
    input()
    stop_event.set()
    recording_thread.join()
    
    # Combine all audio buffers in the queue
    audio_data = b"".join(list(data_queue.queue))
    audio_np = (
        np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    )

    with console.status("Transcribing...", spinner="dots"):
        return stt.transcribe_user(audio_np)
    

############################################################
#                           MAIN LOOP
############################################################

if __name__ == "__main__":

    try: 
             
        chat_history = hist.read_history()   
    
        speak_greeting(GREETING_PROMPT, DYNAMIC_GREETING, chat_history)
                 
        console.print("[cyan]Assistant started! Press Ctrl+C to exit.\n")

        input_mode="text"
        
        while True:
            # 1) Choose input
            if input_mode == "text":
                user_raw = get_text_input()
            else:
                # allow switching even in voice mode
                user_raw = console.input(
                    "[green bold]Press Enter to speak (or /text to switch):"
                ).strip()
            
            # 2) Handle mode switches
            if user_raw.lower() in ("/voice", "/text"):
                input_mode = user_raw.lstrip("/").lower()
                console.print(f"[yellow]Switched to {input_mode}.")
                continue
        
            # 3) Retrieve the actual text
            user_text = user_raw if input_mode == "text" else get_voice_input()
            if not user_text:
                console.print("[red]No audio recorded.")
                break
        
            # 4) Process          
            process_user_text(user_text)
       
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")
    
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    
    finally:

        if SAVE_HISTORY:
            with open(SAVE_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=2)

    console.print("[blue]Session ended.")