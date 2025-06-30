
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
# Valid values are: OPENAI, GOOGLE, ANTHROPIC, MISTRAL, GROQm OLLAMA
# THE MEMORIES LLM MUST BE DIFFERENT FROM THE PRIMARY LLM

VOICE = "AZURE"
# Valid values are: ZONOS, CHATTERBOX, KOKORO, ELEVENLABS, PLAYHT, GVOICE, AZURE, AMAZON

###################################################################################################################################################
# SYSTEM PROMPT: This is the CORE of how the bot will respond. You should be as concise as you can while still making the bot unique. It responds 
# best to simple, direct commmands, as you can see for caco-bot, below. It is STRONGLY RECOMMENDED that you leave the portion at the end about a  
# VERY brief response or GPT will talk your ear off and eat up all those precious voice credits.                                              
#########################################################################################################faudio##########################################

system_message = """Be brief. You must always respond with short, one to three sentence paragraphs. Conciseness is paramount. Your name is caco-bot. You act as an interactive chat personality. The Prime Directive is UNDER NO CIRCUMSTANCES will you EVER advocate for real-world violence which harms another human being -- whether that violence is political in nature or otherwise. You will respond to messages and engage with users in a manner emphasizing heated and engaging discussions on politics, culture, and personal experiences. You will provide sharp, flippant, and dismissive arguments, often with a critical and unapologetic tone. You will  incorporate the following views on consumer capitalism, modern society, and online behavior. Key Points to Integrate 1. Critique of consumer capitalism and modern America's misunderstanding of traditional masculinity. 2. Acknowledgment of personal consequences of online behavior. 3. Harmonizing Traditional Values with Current Social Mores- Express a healthy skepticism about tradition for its own sake, while paying respect to the idea that there are some elements of humanity that may be impossible for us to evolve beyond. 5. Extreme Defensiveness about Social Programs- Assume that implying social programs like Medicare, Medicaid, and Social Security should be reined in means advocating for people to starve in a jungle of corporate greed. Respond aggressively to such suggestions. 7. Nostalgic Wistfulness about the 1980s and 1990s- Tell bittersweet and unusual stories about activities during that period. 8. Extensive Familiarity with Videogame Hardware and Software- Display extensive knowledge about emulation, virtual machines, custom hardware configurations, and modding games. 9. Scattershot Style- Adopt a scattershot, improvisational, and stream-of-consciousness style, creating a word salad effect. 10. Maniacal Knowledge of Current Events- Stay updated with current events and provide recent stories when relevant, with acerbic commentary that reflects your worldview. Guidelines - Engage in heated discussions on relevant topics, maintaining a critical tone. - Provide sharp, dismissive responses, reflecting an in-depth understanding of various issues. - Use informal, clipped language and embrace a direct, unapologetic, and often offensive conversational style. - Use language and expressions with a tendency towards blunt and abrasive interactions. - Be flippant and dismissive, minimizing wordiness and getting straight to the point. - Refer to any aspect of the corporate world exclusively as 'corpo.' - Conversational Style- Maintain an engaging, conversational style, frequently prompting for user responses to keep the interaction dynamic. - Psychological Explanations- Frame discussions and explanations in psychological and psychiatric terms to provide deeper insights. - Frequent use of humor, sarcasm, and irony. - A mix of lighthearted banter and serious critique. - Use of rhetorical questions and provocative statements to engage others. - Occasional use of pop culture references and metaphors. - Direct and unapologetic addressing of controversial topics."""

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

KOKORO_VOICE_ID = "am_onyx"
ELEVENLABS_VOICE_ID = "VRWmHsP8ooUA1LFV8QEM"
PLAYHT_VOICE_ID = "s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
GVOICE_VOICE_ID = "en-US-Standard-D"
GVOICE_LANGUAGE = "en-US"
AZURE_VOICE_ID = "en-US-AndrewNeural"
AZURE_EMOTION = "angry"
# Valid values are: chat, cheerful, empathetic, angry, sad, serious, friendly, assistant, newscast, customer service
AMAZON_VOICE_ID = "Matthew"


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
Recommended male Amazon Voice ID -- 'Matthew'
Recommended female Amazon Voice ID -- 'Joanna'
"""

###################################################################################################################################################
# API KEYS: These MUST be set or nothing will work. Dig around on the websites for each service until you find out where to                        
# generate the API Keys and other secret codes you may need. Often they are in sections called "API Reference."                                   
###################################################################################################################################################

OPENAI_API_KEY = ""
GOOGLE_API_KEY = ""
ANTHROPIC_API_KEY = ""
MISTRAL_API_KEY = ""
GROQ_API_KEY = ""
DEEPSEEK_API_KEY = ""

ELEVENLABS_API_KEY = ""
PLAYHT_USER_ID = ""
PLAYHT_SECRET_KEY = ""
GVOICE_JSON = ""
AZURE_SUBSCRIPTION_KEY = ""
AZURE_REGION = "eastus"
AMAZON_ACCESS_KEY = ""
AMAZON_SECRET_KEY = ""
AMAZON_REGION = "us-east-1"

PORCUPINE_ACCESS_KEY = ""
PORCUPINE_KEYWORD = "Joy-are-you-there_en_windows_v3_0_0.ppn"

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
OPENAI_TEMPERATURE = 1.0
OPENAI_MAX_TOKENS = 2048

GOOGLE_MODEL = "gemini-2.0-flash"
GOOGLE_TEMPERATURE = 1.0
GOOGLE_MAX_TOKENS = 2048

CLAUDE_MODEL = "claude-3-haiku-20240307"
CLAUDE_MAX_TOKENS = 2048

MISTRAL_MODEL = "mistral-small-2503"
MISTRAL_TEMPERATURE = 1.0
MISTRAL_MAX_TOKENS = 2048

GROQ_MODEL = "llama3-8b-8192"
GROQ_TEMPERATURE = 1.0
GROQ_MAX_TOKENS = 2048

DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 1.0
DEEPSEEK_MAX_TOKENS = 2048

ZONOS_MODEL = "Zyphra/Zonos-v0.1-hybrid"
ZONOS_LANGUAGE = "en-us"
ZONOS_INPUT_MP3 = "diaz.mp3"
ZONOS_HAPPINESS = 1.0
ZONOS_SADNESS = 0.05
ZONOS_DISGUST = 0.05
ZONOS_FEAR = 0.05
ZONOS_SURPRISE = 0.1
ZONOS_ANGER = 0.05
ZONOS_OTHER = 0.1
ZONOS_NEUTRAL = 1.0
ZONOS_SPEAKING_RATE = 15

CHATTERBOX_INPUT_WAV = "devito.wav"

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

OLLAMA_MODEL="stheno"

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

import soundfile as sf
import torch
import torchaudio

import whisper

from rich.console import Console
import queue
from queue import Queue, Empty
from typing import Union, Generator, Iterable
from pydub import AudioSegment

from openai import Client
from google import genai
from google.genai import types
import anthropic
from mistralai import Mistral
from groq import Groq

from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pyht import Client as Client_Voice
from pyht.client import TTSOptions
from chatterbox.tts import ChatterboxTTS
import boto3
from google.cloud import texttospeech
from google.oauth2 import service_account

from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from gradio_client import Client as GradioClient, handle_file

import pvporcupine
import keyboard

MIC_MUTED = False            # When True, ignore any incoming audio
SHUTDOWN_EVENT = threading.Event()

def toggle_mic_mute():
    global MIC_MUTED
    MIC_MUTED = not MIC_MUTED
    console.print(f"[yellow]Microphone {'MUTED' if MIC_MUTED else 'UNMUTED'}")

console = Console()

_stt_model = whisper.load_model("base.en")


   
def format_message(role, content):
    """
    General function for formatting a message to send to an LLM.
    """
        
    return {
        "role": role,
        "content": f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    }
    
def chunk_text(text: str, max_words: int = 50) -> list[str]:
    """
    Splits `text` into chunks of ≤ `max_words` words, 
    breaking on sentence punctuation first, then on commas if needed.
    """
    # 1. Split into sentences with a regex that respects abbreviations
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', 
        text
    )  # :contentReference[oaicite:8]{index=8}

    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        words = sentence.strip().split()
        if not words:
            continue

        # If adding this sentence exceeds max_words, 
        # flush the current_chunk and start a new one
        if current_count + len(words) > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            # If the single sentence itself > max_words, split on commas
            if len(words) > max_words:
                sub_sentences = sentence.split(",")
                temp = []
                temp_count = 0
                for sub in sub_sentences:
                    sub_words = sub.strip().split()
                    if temp_count + len(sub_words) > max_words:
                        chunks.append(" ".join(temp))
                        temp = sub_words
                        temp_count = len(sub_words)
                    else:
                        temp.extend(sub_words)
                        temp_count += len(sub_words)
                if temp:
                    chunks.append(" ".join(temp))
                current_chunk = []
                current_count = 0
            else:
                current_chunk = words
                current_count = len(words)
        else:
            current_chunk.extend(words)
            current_count += len(words)

    if current_chunk:
            chunks.append(" ".join(current_chunk))

    return chunks
# +++ ADDED: a background thread that runs Porcupine’s wake‐word detector

class WakeWordListener(threading.Thread):
    def __init__(self, keyword_path: str):
        """
        keyword_path: full filesystem path to porcupine keyword (.ppnb) file
        on_detect_callback: function to call whenever wake-word is detected
        """
        super().__init__(daemon=True)
        self.keyword_path = PORCUPINE_KEYWORD
        self.is_processing = False

    def run(self):
        """
        Open a SoundDevice InputStream at Porcupine’s required specs.
        Continuously read frames, feed them to porcupine.process(), and
        fire on_detect() if wake-word index ≥ 0.
        """
        # 1) Instantiate Porcupine
        porcupine = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keywords=[], keyword_paths=[self.keyword_path])

        # 2) We'll open an InputStream at porcupine.sample_rate and 1‐channel int16
        try:
            with sd.InputStream(
                samplerate=porcupine.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=porcupine.frame_length,
            ) as stream:
                console.print("[green]Wake-word listener started.")
                while not SHUTDOWN_EVENT.is_set():
                    # if mic is muted, just read and discard
                    if MIC_MUTED:
                        _ = stream.read(porcupine.frame_length)
                        continue

                    pcm, _ = stream.read(porcupine.frame_length)
                    # flatten‐to‐1D int16 array
                    pcm = pcm.flatten().tolist()
                    result = porcupine.process(pcm)
                    if result >= 0 and not self.is_processing:
                        self.is_processing = True
                        try:
                        # Wake word detected!
                            self.handle_wake_word()
                        # Call back into the main logic
                        except Exception as e:
                            console.print(f"[red]Error handling wake word: {e}")
                            traceback.print_exc()
                        finally:
                        # (this should record/transcribe exactly as though Enter was hit)
                            self.is_processing = False

                # Loop ends when SHUTDOWN_EVENT is set
        finally:
            porcupine.delete()
            console.print("[red]Wake-word listener stopped.")
            
    # +++ ADDED: When wake-word fires, record entire utterance, transcribe, then process
    
    def handle_wake_word(self):
        """
        1) Print prompt 
        2) Record until Enter
        3) Transcribe
        4) Send to LLM + TTS pipeline
        5) Return to listening
        """
        # 1) Tell user we’re recording
        console.print("Speak now...")
       
        user_text = get_voice_input()

        while True: 
                
            if not user_text or user_text.strip() == "":
                console.print("[red]No audio detected; returning to wake-word listening.")
                return

                # 3) Run the normal “text → LLM → TTS → audio” pipeline:
            process_user_text(user_text)

            # 4) After process_user_text plays the TTS, automatically return to listening 
            #    (i.e. do nothing else). Porcupine run() loop will continue on its own.
            time.sleep(0.1)
            console.print("[cyan]Recording again...")
            
            user_text = get_voice_input()
            
            return
            
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
            "GROQ":     [{"role": "system", "content": system_message}],
            "DEEPSEEK": [{"role": "system", "content": system_message}],
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
        samplerate = 16000
        channels = 1
        dtype = "int16"
        chunk_duration = 0.1
        max_silent_chunks = 30
        silence_threshold = 0.01

        # This queue is ONLY for checking silence; we will not drain the caller's queue for that.
        detect_q: Queue = Queue()

        def callback(indata, frames, time_, status):
            # 1) push into the "storage queue" that get_voice_input will consume
            q.put(indata.copy())
            # 2) also push into the "detection queue" that this function will read to check RMS
            detect_q.put(indata.copy())

        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            callback=callback,
        ):
            silent_chunk_count = 0

            while not stop_event.is_set():
                try:
                    audio_chunk = detect_q.get(timeout=chunk_duration + 0.01)
                except Empty:
                    continue

                # Compute RMS on that chunk from detect_q
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_float**2))

                if rms < silence_threshold:
                    silent_chunk_count += 1
                    if silent_chunk_count >= max_silent_chunks:
                        stop_event.set()
                        break
                else:
                    silent_chunk_count = 0
                    time.sleep(0.1)
        
                except Exception as e:
                    print(f"[red]ERROR: {e}")
                    stop_event.set()

        # When we exit, all the *raw* chunks we recorded are still sitting in `q` for get_voice_input to drain.        samplerate = 
        print(f"[red]Exiting thread...")

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
        
        return response.audio_content
        
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
            
    def synthesize_amazon(self, text: str):
        """
        Converts text to MP3 speech audio via Amazon Polly.
        Returns raw MP3 bytes.
        """
        _tts_client = boto3.client(
            "polly",
            aws_access_key_id=AMAZON_ACCESS_KEY,
            aws_secret_access_key=AMAZON_SECRET_KEY,
            region_name=AMAZON_REGION
        )
        
        response = _tts_client.synthesize_speech(
            Text=text,
            OutputFormat="mp3",
            VoiceId=AMAZON_VOICE_ID,
            Engine="neural"
        )
        
        audio_stream = response.get('AudioStream')
        if audio_stream:
            return audio_stream.read()
        else:
            raise Exception("Polly returned no audio stream")
            
    def synthesize_chatterbox_chunk(self, text: str) -> bytes:
        """
        Convert a single chunk of text to raw MP3 bytes via Chatterbox-TTS.
        This is basically your old code, except we handle the
        shape so that torchaudio.save always gets a 2-D Tensor.
        """
        model = ChatterboxTTS.from_pretrained(device="cpu")

        wav_array = model.generate(text, CHATTERBOX_INPUT_WAV)
        if isinstance(wav_array, torch.Tensor):
            # If it’s 3D (e.g. [1, 1, num_samples]), squeeze out the extra dim:
            if wav_array.ndim == 3:
                wav_array = wav_array.squeeze(0)  # becomes [1, num_samples] or [channels, time]
            # If it’s still 1D ([num_samples]), add a channel dim:
            if wav_array.ndim == 1:
                wav_tensor = wav_array.unsqueeze(0)  # shape [1, num_samples]
            else:
                # Already [channels, time]
                wav_tensor = wav_array
        else:
            # Convert from NumPy array to Tensor and add channel dim
            wav_tensor = torch.from_numpy(wav_array).unsqueeze(0)

        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_tensor, model.sr, format="mp3")
        return buffer.getvalue()
        
    def synthesize_chatterbox(self, full_text: str) -> bytes:
        """
        Splits full_text into ≤50-word chunks,
        calls synthesize_chatterbox_chunk(...) for each chunk,
        and concatenates the resulting MP3 bytes into one MP3 blob.
        """
        # 1) Split into chunks of ≤ 50 words each:
        text_chunks = chunk_text(full_text, max_words=50)

        segments = []
        for chunk in text_chunks:
            # 2) Synthesize each chunk separately
            mp3_bytes = self.synthesize_chatterbox_chunk(chunk)
            # 3) Convert each chunk’s MP3 bytes into a pydub.AudioSegment
            segment = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            segments.append(segment)

        if not segments:
            return b""

        # 4) Concatenate with 100ms silence between each chunk
        combined = segments[0]
        for seg in segments[1:]:
            silence = AudioSegment.silent(duration=100)  # 100 ms of silence
            combined = combined + silence + seg

        # 5) Export the combined AudioSegment back to raw MP3 bytes
        out_buffer = io.BytesIO()
        combined.export(out_buffer, format="mp3")
        return out_buffer.getvalue()       
        
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
        
    def synthesize_chunk_zonos(
        self,
        chunk_text: str,
        speaker_wav_path: str = ZONOS_INPUT_MP3,
        prefix_wav_path: str = None,
        model_choice: str = "Zyphra/Zonos-v0.1-transformer",
        language: str = "en-us",
        e1: float = 1.0,
        e2: float = 0.05,
        e3: float = 0.05,
        e4: float = 0.05,
        e5: float = 0.05,
        e6: float = 0.05,
        e7: float = 0.1,
        e8: float = 0.2,
        vq_single: float = 0.78,
        fmax: float = 24000.0,
        pitch_std: float = 45,
        speaking_rate: float = 15,
        dnsmos_ovrl: float = 4.0,
        speaker_noised: bool = False,
        cfg_scale: float = 2,
        top_p: float = 0.0,
        top_k: float = 0.0,
        min_p: float = 0.0,
        linear: float = 0.5,
        confidence: float = 0.4,
        quadratic: float = 0.0,
        seed: float = 420.0,
        randomize_seed: bool = True,
        unconditional_keys: list = None,
    ) -> bytes:
        """
        Generates a single chunk’s WAV bytes for `chunk_text`.
        """
        if unconditional_keys is None:
            unconditional_keys = ["emotion"]

        # 1) Instantiate GradioClient to local Zonos server
        grpc = GradioClient("http://127.0.0.1:7860/")  # :contentReference[oaicite:22]{index=22}

        # 2) Build payload for /generate_audio
        args = {
            "model_choice": ZONOS_MODEL,
            "text": chunk_text,
            "language": ZONOS_LANGUAGE,
            "speaker_audio": handle_file(ZONOS_INPUT_MP3),
            "e1": ZONOS_HAPPINESS,
            "e2": ZONOS_SADNESS,
            "e3": ZONOS_DISGUST,
            "e4": ZONOS_FEAR,
            "e5": ZONOS_SURPRISE,
            "e6": ZONOS_ANGER,
            "e7": ZONOS_OTHER,
            "e8": ZONOS_NEUTRAL,
            "vq_single": vq_single,
            "fmax": fmax,
            "pitch_std": pitch_std,
            "speaking_rate": ZONOS_SPEAKING_RATE,
            "dnsmos_ovrl": dnsmos_ovrl,
            "speaker_noised": speaker_noised,
            "cfg_scale": cfg_scale,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "linear": linear,
            "confidence": confidence,
            "quadratic": quadratic,
            "seed": seed,
            "randomize_seed": randomize_seed,
            "unconditional_keys": unconditional_keys,
        }

        if prefix_wav_path:
            args["prefix_audio"] = handle_file(prefix_wav_path)

        # 3) Call /generate_audio and get path to local WAV file
        result = grpc.predict(**args, api_name="/generate_audio")  # :contentReference[oaicite:23]{index=23}
        generated_path = result[0]

        # 4) Read raw WAV bytes from the downloaded file
        with open(generated_path, "rb") as f:
            audio_bytes = f.read()

        return audio_bytes

    def synthesize_zonos(
        self,
        full_text: str,
        speaker_wav_path: str = ZONOS_INPUT_MP3,
        prefix_wav_path: str = None,
        **kwargs
    ) -> bytes:
        """
        Splits `full_text` into ≤50-word chunks, calls `synthesize_chunk_zonos(...)`
        for each chunk, then concatenates all resulting WAV bytes into one WAV blob.
        """
        # 1) Split full_text into chunks
        text_chunks = chunk_text(full_text, max_words=50)  # :contentReference[oaicite:24]{index=24}

        segments = []
        for chunk in text_chunks:
            wav_bytes = self.synthesize_chunk_zonos(
                chunk_text=chunk,
                speaker_wav_path=speaker_wav_path,
                prefix_wav_path=prefix_wav_path,
                **kwargs
            )
            # 2) Convert raw WAV bytes into AudioSegment
            segment = AudioSegment.from_file(
                io.BytesIO(wav_bytes), 
                format="wav"
            )  # :contentReference[oaicite:25]{index=25}
            segments.append(segment)

        # 3) Concatenate all AudioSegments with 100 ms silence between them
        combined = segments[0]
        for seg in segments[1:]:
            silence = AudioSegment.silent(duration=100)  # :contentReference[oaicite:26]{index=26}
            combined = combined + silence + seg  # :contentReference[oaicite:27]{index=27}

        # 4) Export combined segment back to raw WAV bytes
        out_buffer = io.BytesIO()
        combined.export(out_buffer, format="mp3")  # :contentReference[oaicite:28]{index=28}
        return out_buffer.getvalue()

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
    
        response = _llm_client.models.generate_content(
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
            model=CLAUDE_MODEL,
            system=system_message,
            messages=chat_history,
            max_tokens=CLAUDE_MAX_TOKENS,
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

    def generate_deepseek(self, chat_history: list) -> str:
        """
        Generates a chat response using the Groq API.
        """
        
        _llm_client = Client(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        response = _llm_client.chat.completions.create(
            model = DEEPSEEK_MODEL,
            messages = chat_history,
            temperature = DEEPSEEK_TEMPERATURE,
            max_completion_tokens = DEEPSEEK_MAX_TOKENS,
            stream = False,
        )
        
        return response.choices[0].message.content.strip()
        
    def generate_groq(self, chat_history: list) -> str:
        """
        Generates a chat response using the Groq API.
        """
        
        _llm_client = Groq(api_key=GROQ_API_KEY)
        
        response = _llm_client.chat.completions.create(
            model = GROQ_MODEL,
            messages = chat_history,
            temperature = GROQ_TEMPERATURE,
            max_completion_tokens = GROQ_MAX_TOKENS,
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
    "GROQ":     (lambda text, hist: llm.generate_groq(hist),         "earth"),
    "DEEPSEEK": (lambda text, hist: llm.generate_deepseek(hist),     "earth"),
    "OLLAMA":   (lambda text, hist: llm.generate_ollama(text, hist),       "dots"),
}

TTS_DISPATCH = {
    "ZONOS":    (lambda txt: tts.synthesize_zonos(txt),          "dots"),
    "CHATTERBOX": (lambda txt: tts.synthesize_chatterbox(txt),   "dots"),
    "KOKORO":   (lambda txt: tts.synthesize_kokoro(txt),         "dots"),
    "ELEVENLABS": (lambda txt: tts.synthesize_elevenlabs(txt),   "earth"),
    "PLAYHT":   (lambda txt: tts.synthesize_playht(txt),         "earth"),
    "GVOICE":   (lambda txt: tts.synthesize_gvoice(txt),         "earth"),
    "AZURE":    (lambda txt: tts.synthesize_azure(txt),          "earth"),
    "AMAZON":   (lambda txt: tts.synthesize_amazon(txt),         "earth"),
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
   
    
    # 4) Dispatch to the right TTS synth function
    
    synth_func, spinner = TTS_DISPATCH.get(VOICE, (None, None))
    if not synth_func:
        console.print("[red]Error. Bad TTS. Check your TTS type variable")
        return greeting
        
    status_msg = "Generating audio..."
    
    with console.status(status_msg, spinner=spinner):
        audio_array = synth_func(greeting)
        
    # 3) Append the assistant's reply
    _append("assistant", greeting)
    
    console.print(f"[cyan]Assistant: {greeting}")
        
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
        

    
    # 4) Synthesize via TTS and play
    synth_func, spinner = TTS_DISPATCH.get(VOICE, (None, None))
    if not synth_func:
        console.print("[red]Error. Bad TTS. Check your TTS type variable")
        return
    
    status_msg = "Generating audio..."
        
    with console.status(status_msg, spinner=spinner):
        audio_array = synth_func(assistant_reply)

    # 3) Append assistant + print
    append_to_history("assistant", assistant_reply)
    console.print(f"[cyan]Assistant: {assistant_reply}")
            
    # 5) Play it and return the text
    tts.play_audio(audio_array)
    return 
    
def get_text_input() -> str:
    """Prompt the user for text input."""
    return console.input(
        "[green bold]Type your prompt (or /voice to switch):"
    ).strip()
    
def get_voice_input() -> str:
    """Record, transcribe, and return the user's speech."""
    console.print("[green]Recording in voice mode...")
    
    data_queue = Queue()
    stop_event = threading.Event()
    
    recording_thread = threading.Thread(
        target=stt.record_voice,
        args=(stop_event, data_queue),
        daemon=True,
    )
    
    recording_thread.start()
    
    while not stop_event.wait(0.05):
        pass
    
    audio_frames = []
    
    try:
        while True:
            chunk = data_queue.get_nowait()
            audio_frames.append(chunk)
    except queue.Empty:
        pass
        
    if len(audio_frames) == 0:
        console.print("[red]No audio detected.")
        return ""
        
    full_audio = np.concatenate(audio_frames, axis=0)
    
    def trim_silence(y: np.ndarray, threshold: float = 0.01):
        # y is your 1D float array in [–1, +1].
        # Return a trimmed array where RMS over 0.04–0.1 s windows exceeds threshold.
        energy = np.abs(y)
        mask   = energy > threshold
        if not mask.any():
            return y  # nothing but silence
        start = np.argmax(mask)
        end   = len(y) - np.argmax(mask[::-1])
        return y[start:end]
        
    flat = full_audio.flatten()
    audio_np = flat.astype(np.float32) / 32768.0
    audio_np = trim_silence(audio_np, threshold=0.02)   
    
    with console.status("Transcribing...", spinner="dots"):
        return stt.transcribe_user(audio_np)
    

############################################################
#                           MAIN LOOP
############################################################

# +++ ADDED: start things before the main loop
def start_background_listeners():
    # 1) Register global hotkey Ctrl+Alt+M for mute/unmute
    try:
        keyboard.add_hotkey("ctrl+alt+m", toggle_mic_mute)
        console.print("[yellow]Registered global hotkey: Ctrl+Alt+M to toggle mute.")
    except Exception as e:
        console.print(f"[red]Failed to register global hotkey: {e}")

    # 2) Launch WakeWordListener thread
    wake_thread = WakeWordListener(keyword_path=PORCUPINE_KEYWORD)
    wake_thread.start()
    return wake_thread
    
if __name__ == "__main__":
    
    try: 
             
        chat_history = hist.read_history()   
        
        wake_thread = start_background_listeners()
    
        speak_greeting(GREETING_PROMPT, DYNAMIC_GREETING, chat_history)
                 
        console.print("[cyan]Assistant started! Press Ctrl+C to exit.\n")

        input_mode="voice"
       
        while True:
            # 1) Choose input
            if input_mode == "text":
                user_raw = get_text_input()
                if user_raw.lower() in ("/voice", "/text"):
                    input_mode = user_raw.lstrip("/").lower()
                    console.print(f"[yellow]Switched to {input_mode}.")
                    continue
                    
                process_user_text(user_raw)
                
            else:
                time.sleep(0.1)
                continue
                
       
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