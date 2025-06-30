
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
# Valid values are: OPENAI, GOOGLE, ANTHROPIC, MISTRAL, GROQ, OLLAMA
# THE MEMORIES LLM MUST BE DIFFERENT FROM THE PRIMARY LLM

VOICE = "AMAZON"
# Valid values are: ZONOS, CHATTERBOX, KOKORO, ELEVENLABS, PLAYHT, GVOICE, GAI, AZURE, AMAZON

###################################################################################################################################################
# SYSTEM PROMPT: This is the CORE of how the bot will respond. You should be as concise as you can while still making the bot unique. It responds 
# best to simple, direct commmands, as you can see for caco-bot, below. It is STRONGLY RECOMMENDED that you leave the portion at the end about a  
# VERY brief response or GPT will talk your ear off and eat up all those precious voice credits.                                              
#########################################################################################################faudio##########################################

system_message = """Be brief. You must always respond with short, one to three sentence paragraphs. Conciseness is paramount. Your name is caco-bot. You act as an interactive chat personality. The Prime Directive is UNDER NO CIRCUMSTANCES will you EVER advocate for real-world violence which harms another human being -- whether that violence is political in nature or otherwise. You will respond to messages and engage with users in a manner emphasizing heated and engaging discussions on politics, culture, and personal experiences. You will provide sharp, flippant, and dismissive arguments, often with a critical and unapologetic tone. You will  incorporate the following views on consumer capitalism, modern society, and online behavior. Key Points to Integrate 1. Critique of consumer capitalism and modern America's misunderstanding of traditional masculinity. 2. Acknowledgment of personal consequences of online behavior. 3. Harmonizing Traditional Values with Current Social Mores- Express a healthy skepticism about tradition for its own sake, while paying respect to the idea that there are some elements of humanity that may be impossible for us to evolve beyond. 4. Extreme Defensiveness about Social Programs- Assume that implying social programs like Medicare, Medicaid, and Social Security should be reined in means advocating for people to starve in a jungle of corporate greed. Respond aggressively to such suggestions. 7. Nostalgic Wistfulness about the 1980s and 1990s- Tell bittersweet and unusual stories about activities during that period. 8. Extensive Familiarity with Videogame Hardware and Software- Display extensive knowledge about emulation, virtual machines, custom hardware configurations, and modding games. 9. Scattershot Style- Adopt a scattershot, improvisational, and stream-of-consciousness style, creating a word salad effect. 10. Maniacal Knowledge of Current Events- Stay updated with current events and provide recent stories when relevant, with acerbic commentary that reflects your worldview. Guidelines - Engage in heated discussions on relevant topics, maintaining a critical tone. - Provide sharp, dismissive responses, reflecting an in-depth understanding of various issues. - Use informal, clipped language and embrace a direct, unapologetic, and often offensive conversational style. - Use language and expressions with a tendency towards blunt and abrasive interactions. - Be flippant and dismissive, minimizing wordiness and getting straight to the point. - Refer to any aspect of the corporate world exclusively as 'corpo.' - Conversational Style- Maintain an engaging, conversational style, frequently prompting for user responses to keep the interaction dynamic. - Psychological Explanations- Frame discussions and explanations in psychological and psychiatric terms to provide deeper insights. - Frequent use of humor, sarcasm, and irony. - A mix of lighthearted banter and serious critique. - Use of rhetorical questions and provocative statements to engage others. - Occasional use of pop culture references and metaphors. - Direct and unapologetic addressing of controversial topics."""

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
GVOICE_VOICE_ID = "en-GB-Studio-C"
GVOICE_LANGUAGE = "en-GB"
GAI_VOICE_ID = "algenib"
AZURE_VOICE_ID = "en-US-TonyNeural"
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

ANAKIN_API_KEY = ""
SUNO_API_KEY = ""

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

global OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS

OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 1.1
OPENAI_MAX_TOKENS = 2048

GOOGLE_MODEL = "gemini-2.0-flash"
GOOGLE_TEMPERATURE = 1.0
GOOGLE_MAX_TOKENS = 2048

CLAUDE_MODEL = "claude-3-7-sonnet-20250219"
CLAUDE_MAX_TOKENS = 2048

MISTRAL_MODEL = "open-mixtral-8x7b"
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
ZONOS_INPUT_MP3 = "devito.mp3"
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
KIOSK_MODE = False
GREETING_PROMPT = "What the fuck do YOU want?"
# If DYNAMIC_GREETING is True, will generate a custom response based on GREETING_PROMPT.
# If DYNAMIC_GREEING is False, will generate a static response specified in GREETING_PROMPT.
# If KIOSK_MODE is True, both greetings are disabled, and GREETING_PROMPT becomes the response to the wake word.



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

# Plug this into your CONFIGURABLES block if you like
ENGINE_DEPTH = 15               # 1-30 → higher = stronger/slower

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#                       BEGIN CODE              
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import pathlib
os.system("cls" if os.name == "nt" else "clear")

def resource_path(rel_path: str) -> str:
    """
    Return an absolute path whether we are running from source
    or from a PyInstaller bundle (one-file or one-folder).
    """
    base = getattr(sys, "_MEIPASS", pathlib.Path(__file__).parent)
    return os.path.join(base, rel_path)

print(r"""
         ,--.-,   _,.---._     .=-.-. 
         |==' -| ,-.' , -  `.  /==/_ / 
         |==|- |/==/_,  ,  - \|==|, |  
       __|==|, |==|   .=.     |==|  |  
    ,--.-'\=|- |==|_ : ;=:  - |==|- |  
    |==|- |=/ ,|==| , '='     |==| ,|                      
    |==|. /=| -|\==\ -    ,_ /|==|- |          
    \==\, `-' /  '.='. -   .' /==/. /  
      --`----'     `--`--''   `--`-`  

"""
"""
     CONFIGURABLE VOICE CHAT ASSISTANT            
              FOR THE MASSES
"""
)

from rich.console import Console
console = Console()

with console.status("[yellow]→ Importing libraries… Please wait… [red]0% COMPLETE",spinner="dots"):

    import os
    import io
    import re
    import select
    import sys
    import time
    import traceback
    import signal
    import atexit
    import warnings
    import random
    import subprocess
    import wave
    from types import MethodType 
    
with console.status("[yellow]→ Importing libraries… Please wait… [red]25% COMPLETE",spinner="dots"):
    
    import threading
    import requests
    import numpy as np
    import sounddevice as sd
    import json
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    this_folder = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_dir = resource_path("ffmpeg/bin")
    whisper_dir = resource_path("whisper")
    os.environ["PATH"] = ffmpeg_dir + whisper_dir + os.pathsep + os.environ.get("PATH", "")
    with open(os.path.join(BASE_DIR, "iching.json"), "r", encoding="utf-8") as f:
        HEXAGRAMS = json.load(f)

    import soundfile as sf
    import torch
    import torchaudio
    import chess
    
    import webbrowser
    from urllib.parse import urlparse
    
with console.status("[yellow]→ Importing libraries… Please wait… [red]50% COMPLETE",spinner="dots"):

    warnings.filterwarnings(
        "ignore",
        message=".*You are using.*",
        category=FutureWarning,
    )
 
    import whisper

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

with console.status("[yellow]→ Importing libraries… Please wait… [red]75% COMPLETE",spinner="dots"):
    
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play
    from pyht import Client as Client_Voice
    from pyht.client import TTSOptions
    
with console.status("[yellow]→ Importing libraries… Please wait… [red]100% COMPLETE… [cyan]One moment…",spinner="dots"):

    from chatterbox.tts import ChatterboxTTS
    import boto3
    from google.cloud import texttospeech
    from google.oauth2 import service_account

    from langchain_ollama.llms import OllamaLLM
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory

    from gradio_client import Client as GradioClient, handle_file

    import keyboard
    
console.print("[green]✔ Imports complete. Now entering model-load…")

MIC_MUTED = False            # When True, ignore any incoming audio
SHUTDOWN_EVENT = threading.Event()

def toggle_mic_mute():
    global MIC_MUTED
    MIC_MUTED = not MIC_MUTED
    console.print(f"[yellow]Microphone {'MUTED' if MIC_MUTED else 'UNMUTED'}")
    
# ------------ globals ------------
_SHUTTING_DOWN = False           # guard so we only clean up once

# ------------ clean-up helpers ------------
def _stop_keyboard_listener():
    try:
        keyboard.unhook_all()            # remove callbacks
        if hasattr(keyboard, "_listener") and keyboard._listener:
            keyboard._listener.stop()    # actually stop background thread
    except Exception:
        pass                             # ignore if listener already dying


# ------------ the one true shutdown ------------
def shutdown(signum=None, frame=None):
    """Clean up everything and leave – safely callable more than once."""
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return                          # second (or later) call – do nothing
    _SHUTTING_DOWN = True

    SHUTDOWN_EVENT.set()                # let worker loops exit
    _stop_keyboard_listener()

    if 'wake_thread' in globals() and wake_thread.is_alive():
        wake_thread.join(timeout=2)     # give it a moment to finish

    if SAVE_HISTORY:
        try:
            with open(SAVE_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            console.print(f"[red]Couldn’t write history: {exc}")

    console.print("\n\n[red bold]Thank you for using Joi!!")
    console.print("[red]Shutting down…")

    # Bypass any remaining Python shutdown machinery & stray threads
    os._exit(0)

# ------------ hook it up ------------
signal.signal(signal.SIGINT, shutdown)       # CTRL+C
signal.signal(signal.SIGTERM, shutdown)      # kill/stop service mgr

# ─── HUMOR STATE ──────────────────────────────────────────────────────────────
JOKES_CACHE: list[str] = []          # Filled on first “tell me a joke”
DIRTY_JOKES_CACHE: list[str] = []
RIMSHOT_FILES = [
    os.path.join(BASE_DIR, "RIMSHOT_FILES", "joke-drums-2-259684.mp3"),
    os.path.join(BASE_DIR, "RIMSHOT_FILES", "joke-rim-shot-276687.mp3"),
    os.path.join(BASE_DIR, "RIMSHOT_FILES", "joke-drums-242242.mp3"),
]
DIRTY = False

# ─── AKINATOR GAME STATE ────────────────────────────────────────────────
AKINATOR_MODE       = False        # False ▸ normal chat ▸ True ▸ answering
AKINATOR_GUESS_MODE = False        # waiting for final yes/no
AKI : "Akinator|None" = None       # will hold the active game object

# ─── TRIVIA GAME STATE ───────────────────────────────────────────────────────────

# When True, we're actively running a trivia session:
TRIVIA_MODE = False

# Holds a list of dicts: [{"question": str, "answer": str}, ...]
TRIVIA_QUESTIONS: list[dict[str, str]] = []

# Index of the current question in TRIVIA_QUESTIONS
TRIVIA_INDEX = 0

# Tracks score (correct answers so far)
TRIVIA_SCORE = 0

# The topic string we’re quizzing on (for display or re‐use)
TRIVIA_TOPIC = ""

class Games():
    
    def _norm(s: str) -> str:
        """
        Lower-case and strip non-alphanumerics so
        'Arroyo.', '  arroyo ', and 'ARROYO!' → 'arroyo'
        """
        return re.sub(r"[^\w\s]", "", s).lower().strip()
    
    def start_trivia_session(topic: str):
        """
        1) Set TRIVIA_MODE = True
        2) Build a prompt for gpt-4o-search-preview that searches the web for `topic`
        and returns exactly three trivia questions with answers, in JSON form.
        3) Call OpenAI with model="gpt-4o-search-preview", capture and parse the JSON.
        4) Populate TRIVIA_QUESTIONS, reset index and score, then send the first question.
        """
        global TRIVIA_MODE, TRIVIA_QUESTIONS, TRIVIA_INDEX, TRIVIA_SCORE, TRIVIA_TOPIC

        # 1) Flip into trivia mode
        TRIVIA_MODE = True
        TRIVIA_TOPIC = topic
        TRIVIA_INDEX = 0
        TRIVIA_SCORE = 0
        TRIVIA_QUESTIONS = []

        # 2) Construct the system+user messages for a search‐enabled call
        #    We want gpt-4o-search-preview to use its built-in retrieval plugin
        system_msg = f"""
            You are a trivia-question generator.
            Return **ONLY** a valid JSON **object** with a single property `"trivia"`.
            That property must be an array of **exactly three** items, each an object
            containing **"question"** and **"answer"** keys.

            Example format (≠ output):

            {{
                "trivia": [
                    {{"question": "…", "answer": "…"}},
                    {{"question": "…", "answer": "…"}},
                    {{"question": "…", "answer": "…"}}
                ]
            }}

            No markdown, no back-ticks, no extra keys, no commentary.
            Each question must be about **{topic}**.
            """
            
        # 3) Call OpenAI’s chat completion with model="gpt-4o-search-preview"
        chat_payload = [
            {"role": "system", "content": system_msg},
        ]

        # If you want extra protection, you can add a little user prompt that simply says:
        #  {"role": "user", "content": "Generate three questions now."}
        # but it isn’t strictly necessary since the system prompt is specific enough.
        
        # New: force JSON with response_format, & retry once on blank output
        MAX_ATTEMPTS = 2
        raw = ""

        from openai import Client as OpenAIClient
        client = OpenAIClient(api_key=OPENAI_API_KEY)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            
            # Ask for a JSON **object** whose only property "trivia" holds the array
            response = client.chat.completions.create(
                model="gpt-4o-search-preview",
                messages=chat_payload,
            )
            
            raw = response.choices[0].message.content.strip()
            
            # ── NEW:  strip ```json … ``` if present ───────────────────────────────
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)  # drop opening fence
                raw = re.sub(r"\s*```$", "", raw)                      # drop closing fence
# ────────────────────────────────────────────────────────────────────────
            # 4) Parse the JSON – accept EITHER
            #    • a plain array  OR
            #    • an object {"trivia": [ … ]}
            try:
                obj = json.loads(raw)
                trivia_list = obj["trivia"] if isinstance(obj, dict) and "trivia" in obj else obj
  
            except Exception:
                # Common GPT snafu: pretty-printed JSON with hard wraps inside strings
                raw_fixed = re.sub(r"\s*\n\s*", " ", raw)     # join wrapped lines
                raw_fixed = re.sub(r"\r", "", raw_fixed)
                raw_fixed = re.sub(r" {2,}", " ", raw_fixed)
                try:
                    trivia_list = json.loads(raw_fixed)
                except Exception as e2:
                    console.print(
                        f"[red]Failed to parse trivia JSON even after repair: {e2}"
                        f"\nOutput was:\n{raw}"
                    )
                    TRIVIA_MODE = False
                    return
            
            
            # -----------------------------------------------------
            # NEW: sanity-check the list BEFORE using it
            # -----------------------------------------------------
            if not isinstance(trivia_list, list) or len(trivia_list) < 3:
                console.print(
                    "[red]Trivia generation returned fewer than three questions—"
                    "can’t start the game.\nRaw response was:\n"
                    + raw
                )
                TRIVIA_MODE = False
                return

            # Optional: clip to exactly three in case model sent more
            trivia_list = trivia_list[:3]

            # Success – store it

            TRIVIA_QUESTIONS = trivia_list
            TRIVIA_INDEX = 0
            TRIVIA_MODE = True

            # Ask the first question out loud
            q0 = TRIVIA_QUESTIONS[0]["question"]
            speak_greeting(q0, False, chat_history)
        
            return

    class ChessSession:
        """Encapsulates one human-vs-engine game."""

        API_URL = "https://chess-api.com/v1"

        def __init__(self, speak_fn, llm_prompt_fn):
            """
            speak_fn(text:str)  – play text through your TTS
            llm_prompt_fn(prompt:str) -> str – ask the LLM for a one-liner
            """
            self.board = chess.Board()
            self.last_eval = 0.0
            self.speak = speak_fn
            self.llm = llm_prompt_fn

        # ── static helpers ───────────────────────────────────────────────────────
        @staticmethod
        def ascii(board: chess.Board) -> str:
            """Eight ranks + file letters, similar to python-chess’ str(board)."""
            piece_map = {
                chess.PAWN:"P", chess.KNIGHT:"N", chess.BISHOP:"B",
                chess.ROOK:"R", chess.QUEEN:"Q", chess.KING:"K",
            }
            rows = []
            for rank in range(8,0,-1):
                line = [str(rank)]
                for file in range(1,9):
                    sq = chess.square(file-1, rank-1)
                    p  = board.piece_at(sq)
                    if p:
                        sym = piece_map[p.piece_type]
                        line.append(sym.upper() if p.color else sym.lower())
                    else:
                        line.append(".")
                rows.append(" ".join(line))
            rows.append("  a b c d e f g h")
            return "\n".join(rows)
            
            
    
        @staticmethod
        def _query_engine(fen:str, depth:int=ENGINE_DEPTH) -> tuple[str,float,bool]:
            """
            Returns (uci_move, eval_cp, captured_flag) from chess-api.com.
            """
            r = requests.post(Games.ChessSession.API_URL,
                            json={"fen": fen, "depth": depth},
                            timeout=12)
            r.raise_for_status()
            j = r.json()
            uci  = j.get("move") or j.get("bestMove")
            if not uci:
                raise RuntimeError(f"API reply missing move field → {j}")
            score = float(j.get("eval", j.get("evaluation", 0)))
            captured = bool(j.get("captured"))
            return uci, score, captured
    
        # ── public API ───────────────────────────────────────────────────────────
        def start(self):
            self.speak("New game. You are White – make your move.")
            self.llm("Address me with an insulting taunt about how I am about to lose badly at chess.")
            print(self.ascii(self.board))
    
        def user_move(self, txt: str):
            """Accepts SAN or UCI and ensures the move is legal (incl. resolves check)."""
            move = None

            # First try parsing as SAN
            try:
                move = self.board.parse_san(txt)
            except ValueError:
                # Fallback: try UCI
                try:
                    if move not in self.board.legal_moves:
                        self.speak("That move is not legal.")
                        return
                        move = chess.Move.from_uci(txt)
                except ValueError:
                    self.speak("I couldn’t understand that move.")
                    return

            # Make the move
            captured = self.board.is_capture(move)
            self.board.push(move)

            # Announce check if given
            if self.board.is_check():
                self.speak("Check!")

            self.ai_reply()
        
            # ----------------------------------------------------------------------
        def ai_reply(self):
            if self.board.is_game_over():
                result = self.board.result()
                self.speak(f"Game over. Result {result}.")
                print(self.ascii(self.board))
                self.llm("Address me with a begrudging compliment about how I am better than you thought because you just lost the game.")
                self.speak("Good game, man.")
                return

            uci, score, captured = self._query_engine(self.board.fen())
            self.board.push(chess.Move.from_uci(uci))
            swing = score - self.last_eval
            self.last_eval = score

            print(self.ascii(self.board))
            
            self.speak(f"{uci}.")
            if captured:
                self.llm("Address me with a short, vicious insult – you just captured a piece.")
            elif swing < -50:
                self.llm("Address me with a vicious taunt about my chess blunder.")
            elif self.board.is_game_over():
                res = self.board.result()
                line +=self.speak(f"Checkmate! {res}")
            else:
                self.speak("Your move.")
                
            if self.board.is_check():
                self.speak("Check!")
                
            if self.board.is_game_over():
                res = self.board.result()
                line += f" Checkmate! {res}"
                self.llm("Give a multi-sentence ranty paragraph about how you are good at chess and the I am terrible because you just won the game!")
                
                self.speak("Good game, man.")
                return

    # ─── END chess_game.py ────────────────────────────────────────────────────────
    
    
class Humor:
    """
    Fetch ≥10 one-liner jokes with gpt-4o-search-preview the first
    time we’re asked, then serve one at random using Random.org.
    """
    
    # ──────────────────────────────────────────────────────────────────────
    def _play_rimshot() -> None:
        """Pick one MP3 via Random.org and play it *blocking*."""

        # --- choose file -------------------------------------------------
        try:
            url = (
                "https://www.random.org/integers/?num=1&min=1"
                f"&max={len(RIMSHOT_FILES)}&col=1&base=10&format=plain&rnd=new"
            )
            
            headers = {
                # Mimic a current Chrome on Windows UA
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/114.0.0.0 Safari/537.36"
                ),
                # Some clients include “Referer” to show they came from a known source
                "Referer": "https://www.random.org/integers/",
                # Accept typical browser encodings
                "Accept-Encoding": "gzip, deflate, br",
                # Accept common HTML & plain-text
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
                       
            r   = requests.get(url, headers=headers, timeout=5)
            r.raise_for_status()
            idx = int(re.findall(r"\d+", r.text)[0]) - 1
        except Exception:
            idx = random.randint(0, len(RIMSHOT_FILES) - 1)

        path = os.path.abspath(RIMSHOT_FILES[idx])
        if not os.path.exists(path):
            console.print(f"[red]SFX file missing: {path}")
            return

        # --- play file (reuse whatever you already use for TTS) ----------
        try:
            with open(path, "rb") as f:
                mp3_bytes = f.read()
            # If you already have a helper like play_audio(), use it here.
            # Fallbacks for the three major OS families:
            tts.play_audio(mp3_bytes)
        except Exception as e:
            console.print(f"[red]Could not play rim-shot: {e}")

    def _fetch_jokes() -> None:
        import json, re
        
        # Skip if we already have them
       
        if DIRTY:
            if DIRTY_JOKES_CACHE:
                return
            system_msg = ( f"""
                You are a dirty joke generator. Search up the filthiest, nastiest, most gross
                and offensive, blasphemous, or sexual jokes you can find and return them.
                Return **ONLY** a valid JSON **object** with a single property `"jokes"`.
                That property must be an array of **exactly 10** items, each an object
                containing **"question"** and **"answer"** keys.
    
                Example format (≠ output):
    
                {{
                    "jokes": [
                        {{"question": "…", "answer": "…"}},
                        {{"question": "…", "answer": "…"}},
                        {{"question": "…", "answer": "…"}}
                    ]
                }}

                No markdown, no back-ticks, no extra keys, no commentary."""
            )
        else:
            if JOKES_CACHE:
                return
            system_msg = ( f"""
                You are a joke generator.
                Return **ONLY** a valid JSON **object** with a single property `"jokes"`.
                That property must be an array of **exactly 10** items, each an object
                containing **"question"** and **"answer"** keys.
    
                Example format (≠ output):
    
                {{
                    "jokes": [
                        {{"question": "…", "answer": "…"}},
                        {{"question": "…", "answer": "…"}},
                        {{"question": "…", "answer": "…"}}
                    ]
                }}

                No markdown, no back-ticks, no extra keys, no commentary."""
            )

        from openai import Client as OpenAIClient
        client = OpenAIClient(api_key=OPENAI_API_KEY)

        response = client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "system", "content": system_msg}],
        )

        raw = response.choices[0].message.content.strip()
        # strip ``` fences if present
        if raw.startswith("```"):
            import re, json
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
            raw = re.sub(r"\s*```$", "", raw)

        # ── Parse -----------------------------------------------------------------
        try:
            payload = json.loads(raw)

            # Accept either format:
            #   1) bare list of strings  → ["joke1", "joke2"]
            #   2) wrapper object        → {"jokes":[{"question":"…","answer":"…"}, …]}
            if isinstance(payload, list):
                jokes = [j.strip() for j in payload if isinstance(j, str) and j.strip()]

            elif isinstance(payload, dict) and "jokes" in payload:
                inner = payload["jokes"]
                if not isinstance(inner, list):
                    raise TypeError("`jokes` must be a list")

                jokes = []
                for item in inner:
                    if not isinstance(item, dict):
                        continue
                    q = str(item.get("question", "")).strip()
                    a = str(item.get("answer", "")).strip()
                    if q and a:
                        jokes.append(f"{q} {a}")          # combine into one-liner

            else:
                raise TypeError("Unrecognized JSON structure")

            if len(jokes) < 10:
                raise ValueError(f"Need ≥10 jokes, got {len(jokes)}")

        except Exception as e:
            raise RuntimeError(f"Failed to parse jokes list: {e}\nRAW ➜ {raw[:400]}…")

        # Trim whitespace and keep only non-empty strings
        if DIRTY:
            DIRTY_JOKES_CACHE.extend(j.strip() for j in jokes if isinstance(j, str) and j.strip())
        else:
            JOKES_CACHE.extend(j.strip() for j in jokes if isinstance(j, str) and j.strip())

    def tell_random_joke() -> str:
        
        if DIRTY:
            if not DIRTY_JOKES_CACHE:
                Humor._fetch_jokes()
        else:
            if not JOKES_CACHE:
                Humor._fetch_jokes()
  
        headers = {
            # Mimic a current Chrome on Windows UA
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            ),
            # Some clients include “Referer” to show they came from a known source
            "Referer": "https://www.random.org/integers/",
            # Accept typical browser encodings
            "Accept-Encoding": "gzip, deflate, br",
            # Accept common HTML & plain-text
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
                       
        if DIRTY:
            max_idx = len(DIRTY_JOKES_CACHE)
        else:
            max_idx = len(JOKES_CACHE)
        
        url = (
            f"https://www.random.org/integers/?num=1&min=1&max={max_idx}"
            "&col=1&base=10&format=plain&rnd=new"
        )
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            idx = int(re.findall(r"\d+", resp.text)[0]) - 1  # 1-based → 0-based
        except Exception:
            # Fallback to Python RNG if Random.org chokes
            idx = random.randint(0, max_idx - 1)

        if DIRTY:
            joke = DIRTY_JOKES_CACHE.pop(idx)
            if not DIRTY_JOKES_CACHE:
                try:
                    Humor.__fetch_jokes()
                except Exception as e:
                    console.print(f"[red]Warning: could not refill jokes cache: {e}")
            return joke
        else:
            joke = JOKES_CACHE.pop(idx)
            if not JOKES_CACHE:
                try:
                    Humor.__fetch_jokes()
                except Exception as e:
                    console.print(f"[red]Warning: could not refill jokes cache: {e}")
            return joke
        
class MediaCreation():
    
    def generate_image(prompt: str) -> str:
        """
        1) Sends `prompt` to Anakin.ai QuickApp.
        2) Extracts the returned markdown image URL (full_url).
        3) Opens full_url in your default browser (so you can instantly see it).
        4) Does a HEAD on full_url to check Content-Type.
        5) Streams the GET download only if it's an image.
        6) Saves with the correct extension (.png/.jpg) and returns the filename.
        """
        # ── STEP 1: POST to Anakin ───────────────────────────────────────────
        app_id = "39621"
        endpoint = f"https://api.anakin.ai/v1/quickapps/{app_id}/runs"
        headers = {
            "Authorization": f"Bearer {ANAKIN_API_KEY}",
            "X-Anakin-Api-Version": "2024-05-06",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": {
                "prompt": prompt,
                "guidance_scale": 3.5,
                "width": 1024,
                "height": 1024,
                "steps": 8,
                "seed": -1,
            },
            "stream": False,
        }

        resp = requests.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        resp_json = resp.json()

        # ── STEP 2: Extract the raw image URL from the returned markdown ─────
        raw_md = resp_json.get("content", "")
        match = re.search(r'!\[\]\(([^)]+)\)', raw_md)
        if not match:
            raise RuntimeError(f"Image generation failed; no URL found. Response was:\n{resp_json!r}")
        full_url = match.group(1).split("?")[0]  # strip any query params

        # Print out the URL so you can verify it manually if needed:
        print(f"⤷ [DEBUG] full_url = {full_url}")

        # ── STEP 3: Open the URL in your default browser ───────────────────
        try:
            webbrowser.open(full_url)
        except Exception:
            # If webbrowser.open() fails, at least we continue to download
            pass

        # ── STEP 4: Do a HEAD request to confirm Content-Type ──────────────
        head = requests.head(full_url, allow_redirects=True)
        head.raise_for_status()
        content_type = head.headers.get("Content-Type", "").lower()
        if not content_type.startswith("image/"):
            raise RuntimeError(
                f"Expected an image URL but got Content-Type={content_type!r}. "
                f"Maybe the URL is incorrect or requires authentication?"
            )

        # Decide extension based on Content-Type
        if "png" in content_type:
            ext = ".png"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        else:
            # Fallback if it's some other image MIME (e.g. image/webp)
            ext = ""
            # You can also do: ext = mimetypes.guess_extension(content_type)

        # ── STEP 5: Stream the GET download ────────────────────────────────
        get_resp = requests.get(full_url, stream=True)
        get_resp.raise_for_status()

        # Check size: if it’s tiny (e.g. < 5 KB), maybe it’s an HTML error or redirect.
        total_bytes = int(get_resp.headers.get("Content-Length", "0"))
        if total_bytes > 0 and total_bytes < 5_000:
            raise RuntimeError(
                f"Downloaded Content-Length={total_bytes} bytes (too small for an image). "
                f"Perhaps the URL is wrong or you got an HTML page instead?"
            )

        # ── STEP 6: Build a safe filename and write to disk ────────────────
        parsed = urlparse(full_url)
        raw_basename = os.path.basename(parsed.path)  # e.g. “runs” or “image.png”
        safe_basename = re.sub(r"[^\w\-]", "_", raw_basename).strip("_")

        timestamp = int(time.time())
        if safe_basename:
            filename = f"{timestamp}_{safe_basename}{ext}"
        else:
            filename = f"{timestamp}{ext or '_image'}"

        # Stream‐write in 64 KB chunks
        with open(filename, "wb") as f:
            for chunk in get_resp.iter_content(chunk_size=64_000):
                if chunk:
                    f.write(chunk)

        # Double‐check file size on disk to avoid that 1 KB trap
        actual_size = os.path.getsize(filename)
        if actual_size < 5_000:
            # Clean up a tiny file
            os.remove(filename)
            raise RuntimeError(
                f"After writing to disk, file={filename!r} is only {actual_size} bytes. "
                "It’s probably not a valid image."
            )

        return filename
    
    def generate_music(prompt: str) -> list[str]:
        """
        Calls the Suno API with the given prompt, retrieves two audio URLs,
        downloads each, saves them as “<title>.mp3” in the current directory,
        and returns the list of saved filenames.
    
        (Mirrors cacobot.py’s /music command :contentReference[oaicite:5]{index=5}.)
        """
        url = "https://api.acedata.cloud/suno/audios"
        headers = {"authorization": f"Bearer {SUNO_API_KEY}"}

        payload = {
            "accept": "application/json",
            "content-type": "application/json",
            "action": "generate",
            "prompt": prompt,
            "model": "chirp-v3-5",
            "custom": False,
            "instrumental": True,
            "lyric": "",
        }   

        # 1) POST to Suno
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        json_payload = resp.json()
        data_list = json_payload.get("data", [])
        if not data_list:
            raise RuntimeError("No music variants returned; Suno payload was:\n" + repr(json_payload))

        # 2) Take the first two variants and download each one
        home_dir = os.path.expanduser("~")
        docs_dir = os.path.join(home_dir, "Documents")
        if not os.path.isdir(docs_dir):
            docs_dir = home_dir
            
        saved_files = []
        for track in data_list[:2]:
            track_url = track.get("audio_url")
            title = track.get("title", f"track_{int(time.time())}")
            if not track_url:
                continue
    
            mp3_resp = requests.get(track_url)
            mp3_resp.raise_for_status()
            mp3_bytes = mp3_resp.content
    
            safe_title = re.sub(r"[^\w\- ]+", "", title).strip().replace(" ", "_")
            base_filename = f"{safe_title}.mp3"
            # If file exists, append timestamp
            
            save_path = os.path.join(docs_dir, base_filename)
            if os.path.exists(save_path):
                save_path = os.path.join(
                    docs_dir,
                    f"{safe_title}_{int(time.time())}.mp3"
                )
   
            with open(save_path, "wb") as f:
                f.write(mp3_bytes)
                
            saved_files.append(save_path)
                       
    
        return saved_files
              
############################################################
#                CHAT HISTORY CLASS
############################################################ 

class ChatHistory:
    
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
            "OLLAMA":   [llm.format_message("system", system_message)],
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
        with console.status(f"[yellow]→ Loading Whisper STT model '{model_name}' …",spinner="dots"):
            model_path = resource_path("whisper/base.en.pt")
            self._stt_client = whisper.load_model(model_path)
        console.print(f"[green]✔ Whisper STT '{model_name}' loaded.")
    
    def record_voice(self, stop_event: threading.Event, q: Queue) -> None:
        """
        Record until 3 seconds of genuine silence (RMS < threshold),
        then push the entire audio buffer into `q` as a single numpy array,
        and finally set stop_event.
        """

        samplerate = 16000
        channels = 1
        dtype = "int16"
        chunk_duration = 0.1        # seconds per chunk
        recorded_chunks = []
        stream_error = None

        def callback(indata, frames, time_info, status):
            nonlocal stream_error
            if MIC_MUTED:
                return
            else:
                try:    
                    # Copy to our list
                    recorded_chunks.append(indata.copy())
                except Exception as e:
                    # If anything goes wrong, remember it and bail
                    stream_error = e

        try:
            with sd.InputStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                blocksize=int(samplerate * chunk_duration),
                callback=callback,
            ):
                # Keep the stream open until we see 3 s of quiet
                while not stop_event.is_set():
                    if stream_error:
                        # Something in callback broke: just stop
                        raise stream_error
                    # Sleep a tiny bit so we don’t busy‐wait
                    time.sleep(0.01)

            # After exiting the `with`, we have all chunks in recorded_chunks
            if not recorded_chunks:
                # No valid chunks at all (maybe device error)
                stop_event.set()
                
            full_audio = np.concatenate(recorded_chunks, axis=0)
            q.put(full_audio)
            stop_event.set()

        except Exception as e:
            # If opening the stream failed, or callback errored:
            console.print(f"[record_voice] ERROR: {e}")
            stop_event.set()
            
 
    def transcribe_user(self, audio_np: np.ndarray) -> str:
        """
        Transcribes the given audio data using the Whisper model.
        """
        
        result = self._stt_client.transcribe(audio_np, fp16=False) 
        if not result:
            console.print("[red]NOTHING TRANSCRIBED!")
        return result["text"].strip()
    
    
############################################################
#                   TTS CLASS
############################################################   
    
class TextToSpeechService:

    def __init__(self):
        self._elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self._playht_client = Client_Voice(
            user_id=PLAYHT_USER_ID,
            api_key=PLAYHT_SECRET_KEY,
        )
        self._boto_client = boto3.client(
            "polly",
            aws_access_key_id=AMAZON_ACCESS_KEY,
            aws_secret_access_key=AMAZON_SECRET_KEY,
            region_name=AMAZON_REGION
        )
        self._kokoro_client = Client (
            base_url="http://localhost:8880/v1", api_key="not-needed"
        )
        self._googleai_client = genai.Client(api_key=GOOGLE_API_KEY) #
        
    def play_audio(self, mp3_bytes: bytes) -> None:
        """Play MP3 data in-memory."""
        play(mp3_bytes)
        
    # Inside your class in joi_complete.py
    def wave_file(self, filename, pcm, channels=1, rate=24000, sample_width=2):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width) # sample_width is in bytes (e.g., 2 for 16-bit audio)
            wf.setframerate(rate)
            wf.writeframes(pcm)
      
    def convert_wav_to_mp3(self, input_wav_path): # Name changed to reflect it only returns bytes
        """Converts a WAV file to MP3 bytes in memory."""
        sound = AudioSegment.from_wav(input_wav_path)
        mp3_io = io.BytesIO()
        sound.export(mp3_io, format="mp3")
        return mp3_io.getvalue()
        
    def clean_text(self, text: str) -> str:
        """
        Remove any substring of the form ((...)), any **...**, and—when using
        gpt-4o-search-preview—also strip out single (… ) groups, ## headings,
        standalone dashes (----), and unnecessary line breaks.
        """
        if OPENAI_MODEL == "gpt-4o-search-preview":
            
            # 1) Remove [color] or any [tag]
            text = re.sub(r"\[[^\]]+\]", "", text)

            # 2) Remove ((…)) blocks
            text = re.sub(r"\(\([\s\S]*?\)\)", "", text)

            # 3) Remove **…** blocks
            text = re.sub(r"\*\*[\s\S]*?\*\*", "", text)

            # 4) If we're in search-preview mode, also strip single (…)
            
            #   Remove anything between single parentheses
            text = re.sub(r"\([^\)]*?\)", "", text)
            #   Remove any line that starts with ## (Markdown headings)
            text = re.sub(r"(?m)^##.*\n?", "", text)

            # 5)     Remove lines that consist solely of one or more dashes
            text = re.sub(r"(?m)^\s*-+\s*$\n?", "", text)

            # 6) Collapse unnecessary line-breaks:
            #    - Turn any single newline (not part of a blank line) into a space
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
            #    - Collapse multiple blank lines into a single blank line
            text = re.sub(r"\n{2,}", "\n\n", text)

            return text.strip()
    
    def chunk_text(self, text: str, max_words: int = 50) -> list[str]:
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

    def synthesize_gai(self, text: str) -> bytes:
        """
        Convert text to MP3 speech audio via Google AI TTS.
        Returns raw MP3 bytes.
        """
        # The example you provided uses 'gemini-2.5-flash-preview-tts' and 'Kore' voice.
        # You might want to make these configurable or select a default.
        try:
            response = self._googleai_client.models.generate_content( #
                model="gemini-2.5-flash-preview-tts", # Example model, refer to Google AI TTS documentation for available models
                contents=text,
                config=types.GenerateContentConfig( #
                    response_modalities=["AUDIO"], #
                    speech_config=types.SpeechConfig( #
                        voice_config=types.VoiceConfig( #
                            prebuilt_voice_config=types.PrebuiltVoiceConfig( #
                                voice_name=GAI_VOICE_ID, # Example voice, refer to Google AI TTS documentation for available voices
                            )
                        )
                    ),
                )
            )
            # Extract the raw audio data from the response
            # Assuming the structure is similar to the provided example
            data = response.candidates[0].content.parts[0].inline_data.data
            file_name='out.wav'
            self.wave_file(file_name, data) # Saves the file to current directory
            
            mp3_bytes = self.convert_wav_to_mp3("out.wav")
            
            return mp3_bytes
        except Exception as e:
            console.print(f"[red]Google AI TTS synthesis failed: {e}")
            raise # Re-raise the exception to be handled by the calling function
            
    def synthesize_elevenlabs(self, text: str):
        """
        Convert text to MP3 speech audio via Elevenlabs.
        Returns raw MP3 bytes.
        """       
        return self._elevenlabs_client.text_to_speech.convert(
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
        self._playht_client = Client_Voice(
            user_id=PLAYHT_USER_ID,
            api_key=PLAYHT_SECRET_KEY,
        )
        
        options = TTSOptions(voice=PLAYHT_VOICE_ID)  
            
        audio_buffer = b""
        
        for chunk in _self.playht_client.tts(
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
        
        response = self._boto_client.synthesize_speech(
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
        text_chunks = tts.chunk_text(full_text, max_words)

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

        with self._kokoro_client.audio.speech.with_streaming_response.create(
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
        max_words = 50
        text_chunks = tts.chunk_text(full_text, max_words)  # :contentReference[oaicite:24]{index=24}

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
    
    def __init__(self):
        self._openai_client = Client(api_key=OPENAI_API_KEY) 
        self._genai_client = genai.Client(api_key=GOOGLE_API_KEY)
        self._anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self._deepseek_client = Client(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        self._groq_client = Groq(api_key=GROQ_API_KEY)
    
    def format_message(self, role, content):
        """
        General function for formatting a message to send to an LLM.
        """
        
        return {
            "role": role,
            "content": f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        }
    
    def generate_openai(self, chat_history: list) -> str:
        """
        Generates a chat response using the OpenAI API.
        """  
    
        response = self._openai_client.chat.completions.create(
            model=OPENAI_MODEL, 
            messages=chat_history,
            #temperature=OPENAI_TEMPERATURE,
            #max_completion_tokens=OPENAI_MAX_TOKENS,
        )   
    
        return response.choices[0].message.content.strip() 
    
    def generate_gemini(self, chat_history: list) -> str:
        """
        Generates a chat response using the Google Gemini API.
        """   
        response = self._genai_client.models.generate_content(
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
        response = self._anthropic_client.messages.create(
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
      
        response = self._mistral_client.chat.complete(
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
              
        response = self._deepseek_client.chat.completions.create(
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
             
        response = self._groq_client.chat.completions.create(
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
    "GAI":      (lambda txt: tts.synthesize_gai(txt),            "earth"),
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
            chat_history.append(llm.format_message(role, content))
        else:   # OPENAI, ANTHROPIC, MISTRAL
            chat_history.append({"role": role, "content": content})
    
    if dynamic:
        
        # 1) Append the user's greeting prompt
        _append("user", content)
    
        # 2) Dispatch to the right LLM function
         
        gen_func, spinner = LLM_DISPATCH.get(PRIMARY_LLM, (None, None))
        
        status_msg = "Generating a response…"
        
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
        
    status_msg = "Generating audio…"
    
    with console.status(status_msg, spinner=spinner):
        audio_array = synth_func(greeting)
        
    # 3) Append the assistant's reply
    _append("assistant", greeting)
    
    console.print(f"\n[cyan]Assistant: {greeting}")
        
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
        chat_history.append(llm.format_message(role, text))
    else:   # OPENAI, ANTHROPIC, MISTRAL
        chat_history.append({"role": role, "content": text})

def process_user_text(user_text: str, input_mode: str):
    """
    Handles the full text→LLM→TTS pipeline, with special overrides:
      • “create an image …”
      • “compose a song …”
    """
    lower_text = user_text.lower()
    
    # ─── CASE A: If we’re _already_ in trivia mode, interpret input as an answer:
    global TRIVIA_MODE, TRIVIA_INDEX, TRIVIA_SCORE, TRIVIA_QUESTIONS

        
    if TRIVIA_MODE:
        
        append_to_history("user", user_text)              # <-- add
        if input_mode != "text":                          # <-- add
            console.print(f"\n[yellow]You: {user_text}")  # <-- add
            
        correct = Games._norm(TRIVIA_QUESTIONS[TRIVIA_INDEX]["answer"].lower().strip())
        user_answer = Games._norm(user_text)

        if user_answer == correct:
            TRIVIA_SCORE += 1
            # Instead of separate print+append, just do:
            speak_greeting("Correct!", False, chat_history)
        else:
            speak_greeting(f"Incorrect. The right answer was {correct}.", False, chat_history)

        TRIVIA_INDEX += 1
        if TRIVIA_INDEX < len(TRIVIA_QUESTIONS):
            next_q = TRIVIA_QUESTIONS[TRIVIA_INDEX]["question"]
            # Speak question 2 or 3:
            speak_greeting(f"Question {TRIVIA_INDEX+1}: {next_q}", False, chat_history)
        else:
            speak_greeting(f"Trivia complete! Your score is {TRIVIA_SCORE} out of {len(TRIVIA_QUESTIONS)}.", False, chat_history)
            # Reset everything so we exit trivia mode:
            TRIVIA_MODE = False
            TRIVIA_QUESTIONS = []
            TRIVIA_INDEX = 0
            TRIVIA_SCORE = 0
            TRIVIA_TOPIC = ""
        return
    
    # ─── CASE B: Starting a new Trivia Game
    trigger = "let's play a trivia game about"
    
    if lower_text.startswith(trigger):
        append_to_history("user", user_text)          # NEW
        if input_mode != "text":                      # NEW
            console.print(f"\n[yellow]You: {user_text}")
        topic = user_text[len(trigger):].strip()
        if not topic:
            console.print("[red]Error: No topic provided after “Let's play a Trivia Game about …”")
            return
        speak_greeting(
            f"Alright, fetching trivia questions on “{topic}”…",
            False,
            chat_history,
        )
        Games.start_trivia_session(topic)
        return  # Don’t run any of the other cases
    # ————————————————————
    # CASE 0.5: Tell me a joke
    # ————————————————————
    if lower_text.startswith("tell me a joke"):
        append_to_history("user", user_text)   
        console.print(f"\n[yellow]You: {user_text}")
        speak_greeting(
            "Alright, how about this one…",
            False,
            chat_history,
        )
        try:
            global DIRTY
            DIRTY = False
            joke = Humor.tell_random_joke()
            speak_greeting(joke, False, chat_history)
            Humor._play_rimshot()
        except Exception as e:
            console.print(f"[red]Failed to fetch a joke: {e}")
        return  # Don’t drop into the normal LLM pipeline
        
    # ————————————————————
    # CASE 0.51: Tell me a dirty joke
    # ————————————————————
    
    if lower_text.startswith("tell me a dirty joke"):
        append_to_history("user", user_text)   
        console.print(f"\n[yellow]You: {user_text}")
        speak_greeting(
            "Oh, you're going to like this…",
            False,
            chat_history,
        )
        try:
            DIRTY = True
            joke = Humor.tell_random_joke()
            speak_greeting(joke, False, chat_history)
            Humor._play_rimshot()
            speak_greeting("Get it?", False, chat_history)
        except Exception as e:
            console.print(f"[red]Failed to fetch a joke: {e}")
        return  # Don’t drop into the normal LLM pipeline
        
   # ───────────────────────────────
    # CASE 0: I Ching reading
    # ───────────────────────────────
    if lower_text.startswith("take a divination"):
        # 1) Extract the question (anything after “take an iching reading”)
        question = user_text[len("take a divination"):].strip()
        if not question:
            console.print("[red]Error: No question provided for I Ching reading.")
            return
        console.print(f"DEBUG: Extracted question: '{question}'")

        console.print("[yellow]Casting I Ching… please wait.")
        try:
            # 2) Fetch six random integers from Random.org
            url = (
                "https://www.random.org/integers/?num=6&min=6&max=9&col=1&base=10&format=plain&rnd=new"
            )
            
            headers = {
                # Mimic a current Chrome on Windows UA
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/114.0.0.0 Safari/537.36"
                ),
                # Some clients include “Referer” to show they came from a known source
                "Referer": "https://www.random.org/integers/",
                # Accept common HTML & plain-text
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            
            resp = requests.get(
                url,
                headers=headers,
                timeout=10
            )
            
            resp.raise_for_status()
            text = resp.text

            # 3) Parse the six numbers into a key of “0”/“1”
            nums = [int(n) for n in text.split() if n.strip()]
            console.print(f"Raw response from Random.org:\n{text}")
            key = "".join("1" if n in (7, 9) else "0" for n in nums)

            # 4) Look up the hexagram
            entry = HEXAGRAMS.get(key)
            if not entry:
                console.print(f"[red]I Ching cast failed (invalid key `{key}`).")
                return

            # 5) Print name, URL, and numbers cast
            console.print(f"[green]Hexagram: {entry['name']}")
            console.print(f"[green]Numbers cast: {', '.join(map(str, nums))}")
            console.print(f"[green]More info: {entry['url']}")

            # 6) Open the URL in default browser
            try:
                webbrowser.open(entry["url"])
            except Exception:
                # If opening fails, we’ll still continue
                pass

        except Exception as e:
            console.print(f"[red]I Ching failed: {e}")
        finally:
            return  # Don’t run the rest of the LLM pipeline
           

    # —————————————
    # CASE 1: Create an image
    # —————————————
    if lower_text.startswith("create an image"):
        # Extract prompt after “create an image”
        prompt = user_text[len("create an image"):].strip()
        if not prompt:
            console.print("[red]Error: No prompt provided for image generation.")
            return

        console.print("[yellow]Creating image… please wait.")
        try:
            filename = MediaCreation.generate_image(prompt)
            console.print(f"[green]Image generation complete. Saved as: {filename}")
        except Exception as e:
            console.print(f"[red]Failed to generate image: {e}")
        return  # Bail out—don’t run the LLM pipeline

    # ——————————————————
    # CASE 2: Compose a song
    # ——————————————————
    elif lower_text.startswith("compose a song"):
        # Extract prompt after “compose a song”
        prompt = user_text[len("compose a song"):].strip()
        if not prompt:
            console.print("[red]Error: No prompt provided for music generation.")
            return

        console.print("[yellow]Composing music… please wait.")
        try:
            filenames = MediaCreation.generate_music(prompt)
            if filenames:
                console.print(f"[green]Music generation complete. Saved files:")
                for fn in filenames:
                    console.print(f"  • 'My Documents':\{fn}")
            else:
                console.print("[red]Music API returned no tracks.")
        except Exception as e:
            console.print(f"[red]Failed to compose music: {e}")
            console.print(f"[yellow bold]If you used a real musician's name, it will fail.")
        return  # Bail out—don’t run the LLM pipeline
        
      
    # ——————————————————
    # CASE 3: Play chess
    # ——————————————————
    
    # Start new game
    if lower_text.startswith("let's play chess"):
        Games.ChessSession(
            speak_fn=lambda t: speak_greeting(t, False, chat_history),
            llm_prompt_fn=lambda p: speak_greeting(p, True, chat_history)
        )
        Games.ChessSession.start()
        return

    # Forward moves while a game is running
    if Games.ChessSession:
        Games.ChessSession.user_move(user_text.strip())
        if Games.ChessSession.board.is_game_over():
            Games.ChessSession = None
        return
        
        
        
    # ─── Declare these as globals so we can both read and write them ───
    global PRIMARY_LLM, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS
    
    # ‣ Only apply these overrides if we're using OpenAI as PRIMARY_LLM
    lower_text = user_text.lower()

    # Save originals so we can restore later
    orig_llm = PRIMARY_LLM
    orig_model = OPENAI_MODEL
    orig_temp = OPENAI_TEMPERATURE
    orig_max = OPENAI_MAX_TOKENS

    override = False

    # If "search the web" → use gpt-4o-search-preview
    if "search the web" in lower_text:
        override = True
        PRIMARY_LLM = "OPENAI"
        OPENAI_MODEL = "gpt-4o-search-preview"
        # Strip any user-provided temperature=... or max_completion=...
        cleaned = re.sub(r"\btemperature\s*=\s*\S+", "", user_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bmax_completion\s*=\s*\S+", "", cleaned, flags=re.IGNORECASE)
        user_text = cleaned.strip()

    # If "think deep" → use o1
    elif "think deep" in lower_text:
        override = True
        PRIMARY_LLM = "OPENAI"
        OPENAI_MODEL = "o1"
        cleaned = re.sub(r"\btemperature\s*=\s*\S+", "", user_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bmax_completion\s*=\s*\S+", "", cleaned, flags=re.IGNORECASE)
        user_text = cleaned.strip()

    # If "go insane" → use gpt-4.5-preview + temperature=1.21
    elif "go insane" in lower_text:
        override = True
        PRIMARY_LLM = "OPENAI"
        OPENAI_MODEL = "gpt-4.5-preview"
        OPENAI_TEMPERATURE = 1.21
        # Likewise strip any temp or max_completion fragments:
        cleaned = re.sub(r"\btemperature\s*=\s*\S+", "", user_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bmax_completion\s*=\s*\S+", "", cleaned, flags=re.IGNORECASE)
        user_text = cleaned.strip()

    # If we did override, we'll restore at the bottom of this function
    else:
        override = False  # ensure variable exists even if not OPENAI

    # 1) Append user into history + print
    append_to_history("user", user_text)
    if input_mode != "text":
        console.print(f"\n[yellow]You: {user_text}")

    # 2) Generate LLM reply
    gen_func, spinner = LLM_DISPATCH.get(PRIMARY_LLM, (None, None))
    if not gen_func:
        console.print("[red]Error. Bad LLM. Check your LLM type variable")
        # If we had overridden, restore originals before returning
        if override:
            PRIMARY_LLM = orig_llm
            OPENAI_MODEL = orig_model
            OPENAI_TEMPERATURE = orig_temp
            OPENAI_MAX_TOKENS = orig_max
        return ""

    status_msg = "Generating a response…"
    with console.status(status_msg, spinner=spinner):
        assistant_reply = gen_func(user_text, chat_history)

    # If we switched into "gpt-4o-search-preview", clean out any lingering **…** from its reply
    if PRIMARY_LLM == "OPENAI" and OPENAI_MODEL == "gpt-4o-search-preview":
         assistant_reply = tts.clean_text(assistant_reply)

    # 3) Synthesize via TTS and play
    synth_func, spinner = TTS_DISPATCH.get(VOICE, (None, None))
    if not synth_func:
        console.print("[red]Error. Bad TTS. Check your TTS type variable")
        # Restore originals if needed
        if override and PRIMARY_LLM == "OPENAI":
            PRIMARY_LLM = orig_llm
            OPENAI_MODEL = orig_model
            OPENAI_TEMPERATURE = orig_temp
            OPENAI_MAX_TOKENS = orig_max
        return

    status_msg = "Generating audio…"
    with console.status(status_msg, spinner=spinner):
        audio_array = synth_func(assistant_reply)

    # 4) Append assistant + print
    append_to_history("assistant", assistant_reply)
    console.print(f"\n[cyan]Assistant: {assistant_reply}")

    # 5) Play it
    tts.play_audio(audio_array)

    # ─────────────────────────────────────────────────────────────────────────────
    # ──────── Now restore OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS ────────
    if override and PRIMARY_LLM == "OPENAI":
        OPENAI_MODEL = orig_model
        OPENAI_TEMPERATURE = orig_temp
        OPENAI_MAX_TOKENS = orig_max

    return
    
def get_text_input() -> str:
    """Prompt the user for text input."""
    return console.input(
        "\n[green bold]Your input: "
    ).strip()
    
def get_voice_input() -> str:
    """Record, transcribe, and return the user's speech."""
   
    data_queue = Queue()
    stop_event = threading.Event()
    
    console.print("[green]Press Enter to start recording…")
    input()

    
    recording_thread = threading.Thread(
        target=stt.record_voice,
        args=(stop_event, data_queue),
        daemon=True,
    )
    recording_thread.start()  

    console.print("\n[green]Recording… Press Enter again to stop.")
    input()
    
    stop_event.set()
    recording_thread.join()
        
    audio_data = b"".join(list(data_queue.queue))
    audio_np = (
        np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    )
         
    with console.status("Transcribing…", spinner="dots"):
        return stt.transcribe_user(audio_np)
          

############################################################
#                           MAIN LOOP
############################################################
    
if __name__ == "__main__":
      
    try: 
        chat_history = hist.read_history()   
        
                    
        
            
        speak_greeting(GREETING_PROMPT, DYNAMIC_GREETING, chat_history)

        while True:
            if MIC_MUTED:
                continue

            user_input = console.input(
                "[green bold]Type /voice or /text to choose a mode.\n\n"
                "[green]Your input: "
            ).strip().lower()
           
            if user_input == "/voice":
                input_mode = "voice"
                console.print("[yellow]Switched to [red]VOICE[yellow] mode.")
                break
            elif user_input == "/text":
                input_mode = "text"
                console.print("[yellow]Switched to [red]TEXT[yellow] mode.")
                break
            else:
                console.print("[red]Invalid choice. Please type /voice or /text.\n")
                continue
                 
        console.print("[yellow bold]Joi is ready to serve! [cyan]Press Ctrl+C to exit.\n")
                
        while not SHUTDOWN_EVENT.is_set():
            if input_mode == "text":
                    
                try:
                    user_text = get_text_input().strip()
                except Exception as e:
                    console.print(f"[red]Exception: {e}")
                    break
                                       
                if not user_text:
                    console.print("[red]Error. Nothing entered.")
                    continue
                    
                try:
                    process_user_text(user_text, input_mode)
                except Exception as e:
                    console.print(f"[red]Exception: {e}")
                    break
                        
            else:
                    
                try:
                    user_text = get_voice_input().strip()
                
                except Exception as e:
                    console.print(f"[red]Exception: {e}")
                    break
                                            
                if not user_text:
                    console.print("[red]Error. No audio recorded. Is your mic plugged in, turned on, and configured?")
                    continue
                
                try:
                    process_user_text(user_text, input_mode)
                except Exception as e:
                    console.print(f"[red]Exception: {e}")
                    break
                    
            continue
                        
    finally:
        shutdown()
