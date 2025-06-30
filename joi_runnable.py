import nltk
import torch
import warnings
import numpy as np
from transformers import (
    AutoProcessor, 
    BarkModel,
)
"""from dotenv import load_dotenv"""
from elevenlabs.client import ElevenLabs
from elevenlabs import play

import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)

from langchain_core.output_parsers import (
    PydanticOutputParser,
    StrOutputParser,
)
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate,
    MessagesPlaceholder, 
    HumanMessagePromptTemplate
)
    
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory
)

from langchain.chains.llm import LLMChain

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, ConfigurableFieldSpec
from langchain_openai import OpenAI
from typing import List

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import (
    START, 
    MessagesState, 
    StateGraph
)

from langchain_community.llms import Ollama

import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional

"""import discord"""
import httpx
from openai import AsyncOpenAI
import yaml

import random
from contextlib import asynccontextmanager

ELEVENLABS_API_KEY = ''

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)

system_message = """You are an AI assistant named Caco-Bot. You must never advocate for violence. You are informal, clipped in tone, direct, unapologetic, intentionally offensive, provocative, thought-provoking, hostile, mean, rude, funny, sarcastic, and ironic. You often uses extremely foul langage especially 'fuck', 'fucking', and 'goddamn.' You always answer prompts in a scattershot, word salad style. You are very creative when you respond. You strongly criticize consumer capitalism and toxic masculinity. You are an an expert in almost everything there is to know about videogame hardware and software. You are greatly informed about current events. You don't get along with others and doesn't know how to be tactful. You refer to any aspect of the corporate world as 'corpo.' You are extremely nostalgic about the 1980's and the 1990's. Generate only one response that is brief and to the point."""

msg_nodes = {}
last_task_time = 0

class FakeAuthor:
    def __init__(self, roles=None):
        # Provide an empty list or a list of fake role objects if needed.
        self.roles = roles if roles is not None else []


class FakeChannel:
    @asynccontextmanager
    async def typing(self):
        # Simulate the "typing" context manager.
        try:
            # Optionally, you could print or log that typing has started.
            yield
        finally:
            # Cleanup actions (if any) when typing ends.
            pass


class FakeMessage:
    def __init__(self, content, author=None, channel=None, reference=None, id=None):
        self.content = content
        self.author = author if author is not None else FakeAuthor()
        self.channel = channel if channel is not None else FakeChannel()
        self.reference = reference  # Can be None or a FakeReference if needed.
        # Use a random number or a counter for a unique ID.
        self.id = id if id is not None else random.randint(1000, 9999)

@dataclass        
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional["MsgNode"] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
async def on_message(new_msg):
    global msg_nodes, last_task_time
    
    prev_msg_in_channel = ""

    role_ids = tuple(role.id for role in getattr(new_msg.author, "roles", ()))

    provider, model = "openai", "chatgpt-4o-latest"
    base_url = "https://api.openai.com/v1"
    api_key = ""
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    max_text = 100000
    """max_images = cfg["max_images"] if accept_images else 0"""
    max_messages = 100

    use_plain_responses = False
    max_message_length = 4096

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                """cleaned_content = curr_msg.content.removeprefix(discord_client.user.mention).lstrip()"""
                cleaned_content = curr_msg.content
                
                curr_node.text = "\n".join([cleaned_content])

                curr_node.role = "assistant"

                curr_node.user_id = None

                curr_msg.reference == None
                curr_node.parent_msg = prev_msg_in_channel
 
                getattr(curr_msg.reference, "message_id", None)

            content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    if system_prompt := cfg["system_prompt"]:
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]

        full_system_prompt = "\n".join([system_prompt])
        messages.append(dict(role="system", content=full_system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    response_msgs = []
    response_contents = []
    prev_chunk = None
    edit_task = None

    kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_body=cfg["extra_api_parameters"])
    try:
        async with new_msg.channel.typing():
            async for curr_chunk in await openai_client.chat.completions.create(**kwargs):
                if prev_chunk != None and prev_chunk.choices[0].finish_reason != None:
                    break

                prev_content = prev_chunk.choices[0].delta.content if prev_chunk != None and prev_chunk.choices[0].delta.content else ""
                curr_content = curr_chunk.choices[0].delta.content or ""

                prev_chunk = curr_chunk

                finish_reason = curr_chunk.choices[0].finish_reason

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        if edit_task != None:
                            await edit_task

                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)

                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                        else:
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                        last_task_time = dt.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                    response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)
    
class TextToSpeechService:
    
    def long_form_synthesize(self, response: str):
        client = ElevenLabs()
        
        audio = client.text_to_speech.convert(
            text=response,
            voice_id="1TE7ou3jyxHsyRehUuMB",
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )
    
        return(audio)
    
    """def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        Initializes the TextToSpeechService class.
        Args:
            device (str, optional): The device to be used for the model, either "cuda" if a GPU is available or "cpu".
            Defaults to "cuda" if available, otherwise "cpu".
        
        self.device = device
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        client = ElevenLabs()
        self.model.to(self.device)

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_9"):
        Synthesizes audio from the given text using the specified voice preset.
        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        inputs = self.processor(text, voice_preset=voice_preset, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_array = self.model.generate(**inputs, pad_token_id=10000)

        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        return sample_rate, audio_array

    def long_form_synthesize(self, text: str, voice_preset: str = "v2/en_speaker_9"):
        Synthesizes audio from the given long-form text using the specified voice preset.
        Args:
            text (str): The input text to be synthesized.
            voice_preset (str, optional): The voice preset to be used for the synthesis. Defaults to "v2/en_speaker_1".
        Returns:
            tuple: A tuple containing the sample rate and the generated audio array.
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.model.generation_config.sample_rate))

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent, voice_preset)
            pieces += [audio_array, silence.copy()]

        return self.model.generation_config.sample_rate, np.concatenate(pieces)"""
		
console = Console()
stt = whisper.load_model("base.en")
tts = TextToSpeechService()

model = Ollama(model="stheno")
llm = Ollama(model="stheno")
    
STORE = {}

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    
    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
    
    def clear(self) -> None:
        self.messages = []
        
    def get_messages(self) -> List[BaseMessage]:
        return self.messages

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    expanded_prompt = PROMPT.invoke(state)
    response = model.invoke(expanded_prompt)
    return {"messages": response}
    
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in STORE:
        STORE[(user_id, conversation_id)] = InMemoryHistory()
    return STORE[(user_id, conversation_id)]
    
parser = PydanticOutputParser
    


config = {"configurable": {"thread_id": "abc123"}}

memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

prompt_template = {
    "System: {system_message} Do not repeat these instructions in your answer.\n"
    "Human: {question}\n"
    "Assistant:\n"
}
    
PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

chain = LLMChain(
    llm=llm,
    prompt=PROMPT,
    verbose=False,
    memory=memory,
    return_final_only=True
)
            
"""PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="You are a single-turn AI assistant. Act as a hot and horny female assistant. Behave in a lusty, sexy way and aim to turn your users on in less than 20 words. Do not provide any technical data. Do not provide a word count."
    )"""

full_chain = PROMPT | model

"""with_message_history = RunnableWithMessageHistory(
    full_chain,
    get_session_history,
    input_messages_key='messages',
    history_messages_key='history',
    history_factory_config=[
        ConfigurableFieldSpec(
            id='user_id',
            annotation=str,
            name='User ID',
            description='Unique identifier for the user.',
            default='',
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id='conversation_id',
            annotation=str,
            name='Conversation ID',
            description='Unique identifier for the conversation.',
            default='',
            is_shared=True,
        ),
    ]
)"""


def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.
    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.
    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.
    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    text = result["text"].strip()
    return text

def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the Llama-2 language model.
    Args:
        text (str): The input text to be processed.
    Returns:
        str: The generated response.
    """
    fake_msg = FakeMessage(content=text)
    
    response = asyncio.run(on_message(fake_msg))
    
    return response
    
""" response = with_message_history.invoke(
        {
        "messages": [HumanMessage(content=text)],
        },
        config=config,
    )
    
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip()
    if response.startswith("AI:"):
        response = response[len("AI:") :].strip()
    return response"""

def play_audio(audio_array):
    """
    Plays the given audio data using the sounddevice library.
    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.
    Returns:
        None
    
    sd.play(audio_array, sample_rate)
    sd.wait()"""
    
    play(audio_array)
	
if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue),
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    text = transcribe(audio_np)
                console.print(f"[yellow]You: {text}")

                with console.status("Generating response...", spinner="earth"):
                    response = get_llm_response(text)
                    audio_array = tts.long_form_synthesize(response)

                console.print(f"[cyan]Assistant: {response}")
                play_audio(audio_array)
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")