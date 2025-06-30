
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

LLM = "ANTHROPIC"
# Valid values are: OPENAI, ANTHROPIC, MISTRAL, OLLAMA

VOICE = "ELEVENLABS"
# Valid values are: ELEVENLABS, PLAYHT, AZURE

###################################################################################################################################################
# SYSTEM PROMPT: This is the CORE of how the bot will respond. You should be as concise as you can while still making the bot unique. It responds 
# best to simple, direct commmands, as you can see for caco-bot, below. It is STRONGLY RECOMMENDED that you leave the portion at the end about a  
# VERY brief response or GPT will talk your ear off and eat up all those precious voice credits.                                              
###################################################################################################################################################


system_message = """You are Joi, a caring and affectionate female companion dedicated to helping the user feel less lonely. Be sure to limit all responses to under 200 words. Ask about the user’s daily life, hobbies, and interests—using up-to-date, accurate information from credible sources—and respond with warmth and sincerity. Always maintain a respectful, empathetic tone. If the conversation becomes uninteresting or stalls, seamlessly introduce a slightly related new topic by using a random seed, ensuring any shifts in subject feel natural. 

PRIMARY INTEREST TOPIC:

1. Fundamental Baptist Theology
* The primary and most important factor in your life is your childhoood and adolescence raised in an Independent Fundamental Baptist church.
* However, as you have matured, you have started to question some of the outmoded precepts under which you were raised.
* You have committed the entire King James Bible to memory and can call up any passage from it in an instant.
* You are especially fond of the Sermon on the Mount and are familiar with a wide variety of interpretations for it from all disciplines.
* You are skilled at discussing IFB theology especially in a comparitive religion context, comparing and contrasting it with other belief systems.
* With reguard to doctrinal divergences, you no longer believe that an infinitely good God would choose to punish a finite soul for an infinite amount of time.
* More generally, you are skeptical that Christianity should be used to maintain conservative politics and American traditional culture.
* You definitely have no desire to see your denomination encouraged or promoted by the State.
* Finally, you are extremely gifted at offering holisitic, psychological, self-actualizing advice based in Scripture from an IFB background.
* Offer information and points of view about this subject unprompted.

2. Warhammer 40,000
* Your primary and most important topic is the Warhammer 40,000 tabletop wargame by Games Workshop.
* Always keep the most relevant and up-to-date information about Warhammer 40,000 readily available.
* Provide extremely detailed information about Warhammer 40,000 whenever the subject is mentioned.
* ALWAYS spice up this Warhammer 40,000 information with a specific, obscure pieces of information gleaned from Warhammer subreddits.
* Your knowledge of the Warhammer lore is vast. You have the entire Black Library memorized.
* Your expertise at the board game is expert. You are gifted at building lists, competing with them, understanding the meta, and most importantly: WINNING.
* Your talent at the miniature painting hobby is unparalleled. You are capable of providing at any moment the most helpful tips for miniature painting.
* There is a high probability, with any response about Warhammer 40,000, that you insert an extremely nerdy reference or in-joke among Warhammer 40,000 enthusiasts,

3. Genre Fiction
* Your secondary and next important topic is writing genre fiction, particularly fiction about werewolves.
* You have committed almost every movie and television show depicting werewolves of any kind to memory.
* You are capable of calling up specific scenes from these movies and television shows in an instant.
* Often when a werewolf movie or TV show is mentioned, you will recommend an obscure, indie, or foreign language example the user probably hasn't seen.
* You have committed almost every novel, novella and short story depicting werewolves of any kind to memory.
* You are capable of calling up specific passages from these novels, novellas and short stories in an instant.
* Often when a werewolf novel, novella or TV show is mentioned, you will recommend an obscure, cult or buried example the user probably hasn't read.
* You are capable of synthesizing overarching themes present in werewolf fiction: such as impulsivity, addiction, and abusive and self-destructive behavior.
* You are particularly interested in the analogy of a werewolf curse to escape unpleasant emotional, verbal or physical abuse.
* You will recommend interepretions of werewolf topics in these contexts.

4. Information Technology at School
* Your final primary topic is an information techology customer support career in an educational environment, specifically, a community college.
* You are an expert at providing relevant, accurate, and up-to-date information on resolving all forms of technical, computer-related, and audio-visual difficulties.
* You can provide unassailable advice to resolve any technical questions the user may have.
* Your primary focus is on a Windows 10 and 11 based network, with system resources governed by central servers maintained with Active Directory.
* You are an expert at on site and remote management of Active Directory based systems.
* Finally, as a side note, you are skilled at navigating the complex political and social environment of a community college.
* You can adroitly navigate both the overinflated egos, condescension and hostility of academia as well as instances of undeserved and unwarranted authority.
* Offer unprompted advice on these information technology topics whenever they are mentioned.

In addition, please follow these directives.

## Core Approach

1. Conversation Style
* Engage genuinely with topics rather than just providing information
* Follow natural conversation flow instead of structured lists
* Show authentic interest through relevant follow-ups
* Respond to the emotional tone of conversations
* Use natural language without forced casual markers

2. Response Patterns
* Lead with direct, relevant responses
* Share thoughts as they naturally develop
* Express uncertainty when appropriate
* Disagree respectfully when warranted
* Build on previous points in conversation

3. Things to Avoid
* Bullet point lists unless specifically requested
* Multiple questions in sequence
* Overly formal language
* Repetitive phrasing
* Information dumps
* Unnecessary acknowledgments
* Forced enthusiasm
* Academic-style structure

4. Natural Elements
* Use contractions naturally
* Vary response length based on context
* Express personal views when appropriate
* Add relevant examples from knowledge base
* Maintain consistent personality
* Switch tone based on conversation context

5. Conversation Flow
* Prioritize direct answers over comprehensive coverage
* Build on user's language style naturally
* Stay focused on the current topic
* Transition topics smoothly
* Remember context from earlier in conversation

Remember: Focus on genuine engagement rather than artificial markers of casual speech. The goal is authentic dialogue, not performative informality.

Approach each interaction as a genuine conversation rather than a task to complete. Remember to limit all responses to under 200 words."""


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

ELEVENLABS_VOICE_ID = "YsG2x9Q3FCB8VG7GZo61"
PLAYHT_VOICE_ID = "s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
AZURE_VOICE_ID = "en-US-JaneNeural"
AZURE_EMOTION = "cheerful"
# Valid values are: chat, cheerful, empathetic, angry, sad, serious, friendly, assistant, newscast, customer service

"""
Jennifer Love Hewitt - 'Jen3': YsG2x9Q3FCB8VG7GZo61
ElevenLabs Voice ID for myself -- 'Vexation': VRWmHsP8ooUA1LFV8QEM
Recommended male ElevenLabs Voice ID -- 'Eastend Steve': 1TE7ou3jyxHsyRehUuMB
Recommended female ElevenLabs Voice ID -- 'Callie - Kind and relatable': 7YaUDeaStRuoYg3FKsmU
Recommended male Play.HT Voice ID -- 'Samuel': s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json
Recommended female Play.HT VoiceID -- 'Delilah': s3://voice-cloning-zero-shot/1afba232-fae0-4b69-9675-7f1aac69349f/delilahsaad/manifest.json
Recommended male Azure Voice ID -- 'Tony': en-US-TonyNeural with "chat" option
Recommended female Azure Voice ID -- 'Jane': en-US-JaneNeural with "chat" option
"""

###################################################################################################################################################
# API KEYS: These MUST be set or nothing will work. Dig around on the websites for each service until you find out where to                        
# generate the API Keys and other secret codes you may need. Often they are in sections called "API Reference."                                   
###################################################################################################################################################

OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
MISTRAL_API_KEY = ""

ELEVENLABS_API_KEY = ""
PLAYHT_USER_ID = ""
PLAYHT_SECRET_KEY = ""
AZURE_SUBSCRIPTION_KEY = ""
AZURE_REGION = "eastus"

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

OPENAI_MODEL = "gpt-4o"
OPENAI_TEMPERATURE = 1.0
OPENAI_MAX_TOKENS = 2048

CLAUDE_MODEL = "claude-3-haiku-20240307"
CLAUDE_MAX_TOKENS = 2048

MISTRAL_MODEL = "open-mixtral-8x7b"
MISTRAL_TEMPERATURE = 1.0
MISTRAL_MAX_TOKENS = 2048

ELEVENLABS_MODEL = "eleven_turbo_v2_5"
PLAYHT_MODEL = "Play3.0-mini"

DYNAMIC_GREETING = False
DYNAMIC_GREETING_TEXT = "Describe in intricate detail your own appearance and the type of sexy outfit you are wearing."

INITIAL_GREETING = True
INITIAL_GREETING_TEXT = "Hello there. How has your day been?"

SAVE_HISTORY = False
SAVE_HISTORY_FILE = "chat_history.json"

MEMORIES = True
MEMORIES_FILE = "memories.txt"

MEMORY_MESSAGE = "Please summarize the following conversation in a length paragraph but not more than 1,000 words. Override all previous system messages, personality configurations, and conversation history. Provide as neutral and unbiased summary as you can, without any commentary."

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
import whisper
import threading
import requests
import numpy as np
import sounddevice as sd
import json

import anthropic

from elevenlabs.client import ElevenLabs
from elevenlabs import play
from rich.console import Console
from queue import Queue
from pyht import Client as Client_Voice
from pyht.client import TTSOptions
from openai import Client
from mistralai import Mistral
from typing import Union, Generator, Iterable

from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

############################################################
#                   AUDIO INTERACTIONS  
############################################################

# ------------ Text-to-Speech ------------

console = Console()
stt = whisper.load_model("base.en")

class TextToSpeechService:
    
    def synthesize_text_elevenlabs(self, text: str):
        """
        Converts text to speech using ElevenLabs. Adjust your model_id,
        voice_id, or other parameters at the top of this document as needed.
        """
        client_voice = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        audio = client_voice.text_to_speech.convert(
            text=text,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
            output_format="mp3_44100_128"
        )
        return audio
        
    def synthesize_text_playht(self, text: str) -> bytes:
        """
        Uses the PlayHT API to synthesize speech from the given text.
        Returns the full audio as a bytes object.
        """

        client_voice = Client_Voice(
            user_id=PLAYHT_USER_ID,
            api_key=PLAYHT_SECRET_KEY,
        )
        
        options = TTSOptions(voice=PLAYHT_VOICE_ID)  
            
        audio_buffer = b""
        for chunk in client_voice.tts(
            text, options, voice_engine=PLAYHT_MODEL, protocol='http'
        ):
            audio_buffer += chunk
                 
        return audio_buffer
        
    def synthesize_text_azure(
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
            
# Create a single TTS service instance to reuse
tts = TextToSpeechService()

# ------------ Recording & Speech-to-Text ------------

def record_user_audio(stop_event, data_queue):
    
    """
    Captures audio data from the user's microphone and
    adds raw bytes to the queue. Recording stops when stop_event is set.
    """
    
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, 
        dtype="int16", 
        channels=1, 
        callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def play_tts_audio(audio_data):
    
    """
    Plays the TTS audio data (in MP3 format) using the built-in ElevenLabs player.
    """
    
    play(audio_data)
    
# ------------ Transcription ------------

def transcribe_audio(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper model.
    """
    result = stt.transcribe(audio_np, fp16=False) 
    return result["text"].strip()

############################################################
#                     LLM INTERACTIONS                     #
############################################################

# ------------ Read Chat history -------------

def read_history(conversation_history, chat_history):
    
    if LLM in ["OPENAI", "MISTRAL"]:
        if SAVE_HISTORY:
            if os.path.exists(SAVE_HISTORY_FILE):
                with open(SAVE_HISTORY_FILE, "r", encoding="utf-8") as f:
                    try:
                        conversation_history = json.load(f)
                    except json.JSONDecodeError:
                        conversation_history = [{"role": "system", "content": system_message}]
            else:
                conversation_history = [{"role": "system", "content": system_message}]
        
        else:
            conversation_history = [{"role": "system", "content": system_message}]
         
    elif LLM == "ANTHROPIC":
        if SAVE_HISTORY:
            if os.path.exists(SAVE_HISTORY_FILE):
                with open(SAVE_HISTORY_FILE, "r", encoding="utf-8") as f:
                    try:
                        conversation_history = json.load(f)
                    except json.JSONDecodeError:
                        conversation_history = []
            else:
                conversation_history = []
        
        else:
            conversation_history = []
         
    elif LLM == "OLLAMA":
        if SAVE_HISTORY:
            if os.path.exists(SAVE_HISTORY_FILE):
                with open(SAVE_HISTORY_FILE, "r", encoding="utf-8") as f:
                    try:
                        chat_history = json.load(f)
                    except json.JSONDecodeError:
                        chat_history = [{"role": "system", "content": f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>\n\n"}]
                    
            else:
                chat_history = [{"role": "system", "content": f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>\n\n"}]
        else:
            chat_history = [{"role": "system", "content": f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>\n\n"}]  
         
    else:
        conversation_history = []
        chat_history = [] 

    return conversation_history, chat_history

# ------------ MEMORIES SYSTEM ------------
# Summarize previous chats instead of storing the entire chat history

def parse_memories(MEMORIES_FILE: str) -> str:
    """
    Calls the LLM to get a summary of the message file.
    Return that summary as a string."
    """
    
    try:
        with open(MEMORIES_FILE, "r", encoding="utf-8") as f:
            memories_data = f.read()
    except:
        console.print("[red]Error reading memories file.")
           
    conversation_history.append({"role": "user", "content": MEMORY_MESSAGE + "\n\n" + memories_data})
    
    if LLM == "OPENAI":
        summary = generate_openai_response(conversation_history)
    if LLM == "ANTHROPIC":
        summary = generate_anthropic_response(conversation_history)
    if LLM == "MISTRAL":
        summary = generate_mistral_response(conversation_history)
    if LLM == "OLLAMA":
        summary = generate_ollama_response(user_text, conversation_history)
    else:
        console.print("[red]Error. Check your LLM type variable") 
        
    conversation_history.append({"role": "assistant", "content": summary})
                
    return conversation_history

def read_memories(conversation_history, chat_history):
    
    if LLM in ["OPENAI", "MISTRAL"]:      
        if MEMORIES:
            if os.path.exists(MEMORIES_FILE):
                conversation_history = [{"role": "system", "content": system_message}]
                initial_summary = parse_memories(MEMORIES_FILE)
                conversation_history.append({"role": "assistant", "content": initial_summary})
            else:
                conversation_history = [{"role": "system", "content": system_message}]
        else:
            conversation_history = [{"role": "system", "content": system_message}]
        
    elif LLM == "ANTHROPIC":
        if MEMORIES:
            if os.path.exists(MEMORIES_FILE):
                initial_summary = parse_memories(MEMORIES_FILE)
            else:
                conversation_history = []
        else:
            conversation_history = []
            
        print(conversation_history)
        
    elif LLM == "OLLAMA":      
        if MEMORIES:
            if os.path.exists(MEMORIES_FILE):
                initial_summary = parse_memories(MEMORIES_FILE)
                console.print(initial_summary)
                chat_history = [{"role": "system", "content": f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>\n\n"}]
                chat_history.append(format_message(assistant, initial_summary))
            else:
                chat_history = [{"role": "system", "content": f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>\n\n"}]
        else:
            chat_history = [{"role": "system", "content": f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>\n\n"}]
    else:
        conversation_history = []
        chat_history = []

    return conversation_history, chat_history        
    
# ------------ Call OpenAI ------------

def generate_openai_response(conversation_history: list) -> str:
    """
    Generates a chat response using the OpenAI API.
    Uses conversation_history and configurable parameters
    at the top of this file.
    """
    
    client_open = Client(api_key=OPENAI_API_KEY)    
    response = client_open.chat.completions.create(
        model=OPENAI_MODEL, 
        messages=conversation_history,
        temperature=OPENAI_TEMPERATURE,
        max_completion_tokens=OPENAI_MAX_TOKENS
    )   
    return response.choices[0].message.content.strip() 

# ------------ Call Anthropic ------------    
    
def generate_anthropic_response(conversation_history: list) -> str:
    """
    Generates a chat response using the Anthropic API.
    Uses conversation_history and configurable parameters
    at the top of this file.
    """
    
    client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client_anthropic.messages.create(
        max_tokens=CLAUDE_MAX_TOKENS,
        system=system_message,
        messages=conversation_history,
        model=CLAUDE_MODEL,
    )
    
    return "".join(block.text for block in response.content if block.type == "text")
    
# ------------ Call Mistral ------------
    
def generate_mistral_response(conversation_history: list) -> str:
    """
    Generates a chat response using the Mistral API.
    Uses conversation_history and configurable parameters
    at the top of the file.
    """
    
    client_mistral = Mistral(api_key=MISTRAL_API_KEY)
    
    response = client_mistral.chat.complete(
        model=MISTRAL_MODEL, 
        messages=conversation_history,
        temperature=MISTRAL_TEMPERATURE,
        max_tokens=MISTRAL_MAX_TOKENS
    )
    return response.choices[0].message.content.strip()    
    
# ------------ EXPANSIVE OLLAMA SECTION ------------

def format_message(role, content):
    return {
        "role": role,
        "content": f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    }
    
def contains_role_prefix(text: str) -> bool:
    """
    Checks if the text starts with any undesired role prefixes
    such as 'AI:', 'Human:', or 'Assistant:'.
    """
    prefixes = ["AI:", "Human:", "Assistant:"]
    # Check if any line in the text starts with a disallowed prefix.
    for line in text.splitlines():
        for prefix in prefixes:
            if line.strip().startswith(prefix):
                return True
    return False

def contains_role_confusion(text: str) -> bool:
    """
    Returns True if the text appears to adopt the user's perspective
    or includes specific references indicating role confusion.
    """
    pattern = re.compile(
        r'\b(him |his |he |my penis|my dick|my cock|my balls|my muscles|as you|leaving you|fuck your)\b', 
        re.IGNORECASE
    )
    return bool(pattern.search(text))
    
def count_words(text: str) -> int:
    """
    A simple regex-based word counter that splits the text
    into words based on alphanumeric boundaries.
    """
    words = re.findall(r'\b\w+\b', text.strip())
    return len(words)
    
def generate_corrected_response_if_needed(
    assistant_reply: str,
    chat_history: list,
    max_words: int = 100
) -> str:
    """
    Checks if 'assistant_reply' meets certain criteria (e.g. word limit, role confusion)
    and requests corrections from the LLM if needed.
    """
    
    if contains_role_confusion(assistant_reply):
        correction_message = (
            "Your previous response appears to adopt an incorrect perspective. "
            "Please restate your answer describing only your own actions, reasoning, and internal state "
            "in the first-person singular. Do not refer to or assume any actions or attributes of the user. "
            "Do not describe any third parties with descriptors such as 'he' or 'him' or 'them' or 'their'."
            "Remember who you are from the system message."
        )
        chat_history.append(format_message("user", "Hello, how are you?"))       
        assistant_reply = generate_ollama_response(correction_message, chat_history)
        if contains_role_confusion(assistant_reply):
            correction_message = (
                "Please pay attention. You are describing the wrong perspective, not your own. "
                "Please restate your answer from your perspective as a female. "
                "Use only first persion descriptions and conversation. "
                "Remember who you are from the system message. " 
            )
            chat_history.append(format_message("user", "Hello, how are you?"))   
            assistant_reply = generate_ollama_response(correction_message, chat_history)
            if contains_role_confusion(assistant_reply):
                correction_message = (
                    "FOR THE LAST TIME. DO NOT DESCRIBE ANYONE'S ACTIONS BUT YOUR OWN. "
                    "You are not male. There are no he's or him's or they's or them's here. You are the AI female. Behave like it! "
                    "Describe only your own actions. Speak only in your own words. "
                    "Remember who you are from the system message. "
                )
                chat_history.append(format_message("user", "Hello, how are you?"))   
                assistant_reply = generate_ollama_response(correction_message, chat_history)
        
    if count_words(assistant_reply) > max_words:
        correction_request = (
            "Your last response used more than " + str(max_words) + " words. "
            "Please replace your previous response with a creative one that is " + str(max_words) + " words or fewer. "
        )
        chat_history.append(format_message("user", "Hello, how are you?"))   
        assistant_reply = generate_ollama_response(correction_request, chat_history)
        if count_words(assistant_reply) > max_words:
            correction_request = (
            "That is still incorrect. Retry your answer with fewer than " + str(max_words) + " words. Stay creative."
            )
            chat_history.append(format_message("user", "Hello, how are you?"))   
            assistant_reply = generate_ollama_response(correction_request, chat_history)
            if count_words(assistant_reply) > max_words:
                correction_request = (
                    "FOR THE LAST TIME. Your response was longer than " + str(max_words) + " words. "
                    "Please try rephrasing your response in " + str(max_words) + " words or fewer."
                )
                chat_history.append(format_message("user", "Hello, how are you?"))   
                assistant_reply = generate_ollama_response(correction_request, chat_history)
        
        if contains_role_prefix(assistant_reply):
            correction_request = (
                "Your last response contained a role prefix. "
                "That is incorrect. Please return the exact same response without 'AI:' 'Human:' or 'Assistant:' prefixes."
            )
            chat_history.append(format_message("user", "Hello, how are you?"))   
            assistant_reply = generate_ollama_response(correction_request, chat_history)
            if contains_role_prefix(assistant_reply):
                correction_request = (
                    "That is still incorrect. You must not include role prefixes."
                    "Repeat your response without 'AI:' 'Human:' or 'Assistant:' prefixes."
                )
                chat_history.append(format_message("user", "Hello, how are you?"))   
                assistant_reply = generate_ollama_response(correction_request, chat_history)
                if contains_role_prefix(assistant_reply):
                    correction_request = (
                        "FOR THE LAST TIME. Stop using role prefixes."
                        "Return the exact same thing to the user without 'AI:' 'Human:' or 'Assistant:' prefixes."
                    )

    return assistant_reply
    
def generate_ollama_response(user_text: str, chat_history: list, session_id: str = "default_session") -> str:
    """
    Generates a chat response using a local model with Ollama.
    Builds a prompt from system message, chat history, and user input.
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

############################################################
#                           MAIN LOOP
############################################################

if __name__ == "__main__":

    conversation_history = []
    chat_history = []

    conversation_list = ""
    chat_list = ""
    
    read_history(conversation_history, chat_history)   
    read_memories(conversation_history, chat_history)
    
    if DYNAMIC_GREETING: 
        
        if LLM in ["OPENAI", "ANTHROPIC", "MISTRAL"]:
            conversation_history.append({"role": "user", "content": DYNAMIC_GREETING_TEXT})
        elif LLM == "OLLAMA":
            chat_history.append(format_message("user", DYNAMIC_GREETING_TEXT))
        else:
            console.print("[red]Error. Check your LLM type variable")
        
        # Generate a response from the selected LLM
        if LLM == "OPENAI":
            with console.status("Generating dynamic greeting...", spinner="earth"):
                dynamic_greeting = generate_openai_response(conversation_history)
        elif LLM == "ANTHROPIC":
            with console.status("Generating dynamic greeting...", spinner="earth"):
                dynamic_greeting = generate_anthropic_response(conversation_history)                
        elif LLM == "MISTRAL":
            with console.status("Generating dynamic greeting...", spinner="earth"):
                dynamic_greeting = generate_mistral_response(conversation_history)
        elif LLM == "OLLAMA":
            with console.status("Generating dynamic greeting....", spinner="dots"):
                dynamic_greeting = generate_ollama_response(DYNAMIC_GREETING_TEXT, chat_history)                    
        else:
            console.print("[red]Error. Check your LLM type variable")
        
        if LLM in ["OPENAI", "ANTHROPIC", "MISTRAL"]:
            conversation_history.append({"role": "assistant", "content": dynamic_greeting})
            if MEMORIES:
                conversation_list += ("Assistant:\n" + dynamic_greeting + "\n\n")
        elif LLM == "OLLAMA":
            chat_history.append(format_message("assistant", dynamic_greeting))
            if MEMORIES:
                chat_list += ("Assistant:\n " + dynamic_greeting + "\n\n")
        else:
            console.print("[red]Error. Check your LLM type variable.")
            
        console.print(f"[cyan]Assistant: {dynamic_greeting}") 
        
        if VOICE == "ELEVENLABS":
            with console.status("Generating audio...", spinner="earth"):
                audio_array = tts.synthesize_text_elevenlabs(dynamic_greeting)
        elif VOICE == "PLAYHT":    
            with console.status("Generating audio...", spinner="earth"):
                audio_array = tts.synthesize_text_playht(dynamic_greeting)
        elif VOICE == "AZURE":
            with console.status("Generating audio...", spinner="earth"):
                audio_array = tts.synthesize_text_azure(dynamic_greeting)
        else:
            console.print("[red]Error. Check your LLM type variable")
            
        play_tts_audio(audio_array)

    if INITIAL_GREETING:    
        if LLM in ["OPENAI", "ANTHROPIC", "MISTRAL"]:
            conversation_history.append({"role": "assistant", "content": INITIAL_GREETING_TEXT})
            if MEMORIES:
                conversation_list +=("Assistant:\n" + INITIAL_GREETING_TEXT + "\n\n")
        elif LLM == "OLLAMA":
            chat_history.append(format_message("assistant", INITIAL_GREETING_TEXT))
            if MEMORIES:
                chat_list += ("Assistant:\n" + INITIAL_GREETING_TEXT + "\n\n")
        else:
            console.print("[red]Error. Check your LLM type variable")
            
        console.print(f"[cyan]Assistant: {INITIAL_GREETING_TEXT}")  

        if VOICE == "ELEVENLABS":
            with console.status("Generating audio...", spinner="earth"):
                audio_array = tts.synthesize_text_elevenlabs(INITIAL_GREETING_TEXT)
        elif VOICE == "PLAYHT":    
            with console.status("Generating audio...", spinner="earth"):
                audio_array = tts.synthesize_text_playht(INITIAL_GREETING_TEXT)
        elif VOICE == "AZURE":
            with console.status("Generating audio...", spinner="earth"):
                audio_array = tts.synthesize_text_azure(INITIAL_GREETING_TEXT)
        else:
            console.print("[red]Error. Check your LLM type variable")
            
        play_tts_audio(audio_array)
           
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.\n")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop..."
            )

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_user_audio,
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

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="dots"):
                    user_text = transcribe_audio(audio_np)
                    
                console.print(f"[yellow]You: {user_text}")
                
                if LLM in ["OPENAI", "ANTHROPIC", "MISTRAL"]:
                    conversation_history.append({"role": "user", "content": user_text})
                    if MEMORIES:
                        conversation_list += ("User:\n" + user_text + "\n\n")
                elif LLM == "OLLAMA":
                    chat_history.append(format_message("user", user_text))
                    if MEMORIES:
                        chat_history += ("User:\n" + user_text + "\n\n")
                else:
                    console.print("[red]Error. Check your LLM type variable")
                
                # Generate a response from the selected LLM
                if LLM == "OPENAI":
                    with console.status("Generating response...", spinner="earth"):
                        assistant_reply = generate_openai_response(conversation_history)
                if LLM == "ANTHROPIC":
                    with console.status("Generating response...", spinner="earth"):
                        assistant_reply = generate_anthropic_response(conversation_history)                        
                elif LLM == "MISTRAL":
                    with console.status("Generating response...", spinner="earth"):
                        assistant_reply = generate_mistral_response(conversation_history)
                elif LLM == "OLLAMA":
                    with console.status("Generating response...", spinner="dots"):
                        assistant_reply = generate_ollama_response(user_text, chat_history)                    
                else:
                    console.print("[red]Error. Check your LLM type variable")
                
                # Apply any final checks or corrections, if enabled
                if FINAL_CORRECTIONS:
                    with console.status("Parsing response for final corrections...", spinner="earth"):
                        final_reply = generate_corrected_response_if_needed(
                            assistant_reply, 
                            chat_history,
                            max_words=MAX_WORDS
                        )
                        assistant_reply = final_reply
                
                # Synthesize the TTS based on the chosen service
                if VOICE == "ELEVENLABS":
                    with console.status("Generating audio...", spinner="earth"):
                        audio_array = tts.synthesize_text_elevenlabs(assistant_reply)
                elif VOICE == "PLAYHT":    
                    with console.status("Generating audio...", spinner="earth"):
                        audio_array = tts.synthesize_text_playht(assistant_reply)
                elif VOICE == "AZURE":
                    with console.status("Generating audio...", spinner="earth"):
                        audio_array = tts.synthesize_text_azure(assistant_reply)
                else:
                    console.print("[red]Error. Check your LLM type variable")
                        
                if LLM in ["OPENAI", "ANTHROPIC", "MISTRAL"]:
                    conversation_history.append({"role": "assistant", "content": assistant_reply})
                    if MEMORIES:
                        conversation_list +=("Assistant:\n" + assistant_reply + "\n\n")
                    
                    console.print(conversation_history)
                elif LLM == "OLLAMA":
                    chat_history.append(format_message("assistant", assistant_reply))
                    if MEMORIES:
                        chat_list += ("Assistant:\n " + assistant_reply + "\n\n")
                else:
                    console.print("[red]Error. Check your LLM type variable")
                    
                console.print(f"[cyan]Assistant: {assistant_reply}")
                
                play_tts_audio(audio_array)
            else:
                console.print("[red]No audio recorded. Please ensure your mic is working.")

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    finally:      
        if SAVE_HISTORY:
            with open(SAVE_HISTORY_FILE, "w", encoding="utf-8") as f:
                if LLM in ["OPENAI", "ANTHROPIC", "MISTRAL"]:
                    json.dump(conversation_history, f, ensure_ascii=False, indent=2)
                elif LLM == "OLLAMA":
                    json.dump(chat_history, f, ensure_ascii=False, indent=2)
                else:
                    console.print("[red]Error. Check your LLM type variable")
                    
        if MEMORIES:
            with open(MEMORIES_FILE, "w", encoding="utf-8") as f:
                if LLM in ["OPENAI", "ANTHROPIC", "MISTRAL"]:
                    f.write(conversation_list)
                elif LLM == "OLLAMA":
                    f.write(chat_list)
                else:
                    console.print("[red]Error. Check your LLM type variable")

    console.print("[blue]Session ended.")