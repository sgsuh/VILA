import argparse
import base64
import copy
import json
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from threading import Lock, Thread
from typing import Any, Dict, List, Literal, Optional, Union, get_args

import requests
import torch
import uvicorn
import tempfile
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image as PILImage
from PIL.Image import Image
from pydantic import BaseModel
from transformers.generation.streamers import TextIteratorStreamer
from llava.utils.logging import logger
from llava.media import Video

from llava import conversation
from llava.constants import MEDIA_TOKENS
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.utils.stop_strings import StopStringFilter, truncate_at_stop
from llava.utils.tool_calls import (
    TOOL_RESPONSE_TAG,
    build_tools_system_prompt,
    format_tool_call,
    format_tool_response,
    parse_tool_calls,
)


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class MediaURL(BaseModel):
    url: str


class ImageContent(BaseModel):
    type: Literal["image_url"]
    image_url: MediaURL

class VideoContent(BaseModel):
    type: Literal["video_url"]
    video_url: MediaURL
    frames: Optional[int] = 8
    fps: Optional[int] = 2


IMAGE_CONTENT_BASE64_REGEX = re.compile(r"^data:image/(png|jpe?g);base64,(.*)$")
VIDEO_CONTENT_BASE64_REGEX = re.compile(r"^data:video/(mp4);base64,(.*)$")


def load_video(video_url: str) -> str:
    # download or parse video from base64
    if video_url.startswith("http") or video_url.startswith("https"):
        response = requests.get(video_url)
        video = BytesIO(response.content)
    else:
        match_results = VIDEO_CONTENT_BASE64_REGEX.match(video_url)
        if match_results is None:
            raise ValueError(f"Invalid video url: {video_url[:64]}")
        image_base64 = match_results.groups()[1]
        video = BytesIO(base64.b64decode(image_base64))

    temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    temp_fpath = os.path.join(temp_dir, f"{uuid.uuid5(uuid.NAMESPACE_DNS, video_url)}.mp4")
    with open(temp_fpath, "wb") as f:
        f.write(video.getbuffer())

    return temp_fpath

class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[Union[TextContent, ImageContent, VideoContent]]]] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamedFunction(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: NamedFunction


class ChatCompletionRequest(BaseModel):
    model: Literal[
        "NVILA-15B",
        "NVILA-Lite-2B",
        "VILA1.5-3B",
        "VILA1.5-3B-AWQ",
        "VILA1.5-3B-S2",
        "VILA1.5-3B-S2-AWQ",
        "Llama-3-VILA1.5-8B",
        "Llama-3-VILA1.5-8B-AWQ",
        "VILA1.5-13B",
        "VILA1.5-13B-AWQ",
        "VILA1.5-40B",
        "VILA1.5-40B-AWQ",
    ]
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    use_cache: Optional[bool] = True
    num_beams: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], NamedToolChoice]] = None


model = None
model_name = None
tokenizer = None
image_processor = None
context_len = None

# The model is shared across requests; concurrent generate calls corrupt its state (NaN logits).
generation_lock = Lock()


def generate_to_streamer(streamer: TextIteratorStreamer, **kwargs) -> None:
    try:
        with generation_lock:
            model.generate_content(streamer=streamer, **kwargs)
    except Exception:
        logger.exception("Streaming generation failed")
        # Unblock the consumer so the response terminates instead of hanging.
        streamer.end()


def load_image(image_url: str) -> Image:
    if image_url.startswith("http") or image_url.startswith("https"):
        print(f"[Server] Loading image from URL: {image_url}")
        response = requests.get(image_url)
        image = PILImage.open(BytesIO(response.content)).convert("RGB")
        print("[Server] Image loaded from URL successfully.")
    else:
        match_results = IMAGE_CONTENT_BASE64_REGEX.match(image_url)
        if match_results is None:
            raise ValueError(f"Invalid image url format: {image_url}")
        image_base64 = match_results.groups()[1]
        try:
            image = PILImage.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
            print("[Server] Base64 image loaded successfully.")
        except Exception as e:
            print(f"[Server] Failed to decode base64 image: {e}")
            raise e
    return image



def get_literal_values(cls, field_name: str):
    field_type = cls.__annotations__.get(field_name)
    if field_type is None:
        raise ValueError(f"{field_name} is not a valid field name")
    if hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
        return get_args(field_type)
    raise ValueError(f"{field_name} is not a Literal type")


VILA_MODELS = get_literal_values(ChatCompletionRequest, "model")


def content_parts(content) -> List[Any]:
    """Convert OpenAI message content into generate_content prompt parts (text, images, videos)."""
    if content is None:
        return []
    if isinstance(content, str):
        return [content]
    parts = []
    for item in content:
        if item.type == "text":
            parts.append(item.text)
        elif item.type == "image_url":
            parts.append(load_image(item.image_url.url))
        elif item.type == "video_url":
            video = load_video(item.video_url.url)
            logger.info(f"loading {item.frames} frames from {video}")
            model.config.num_video_frames = item.frames
            model.config.fps = item.fps
            parts.append(Video(video))
        else:
            raise NotImplementedError(f"Unsupported content type: {item.type}")
    return parts


def content_text(content) -> str:
    return "".join(part for part in content_parts(content) if isinstance(part, str))


def build_conversation(messages: List[ChatMessage], tools_prompt: Optional[str]) -> List[Dict[str, Any]]:
    """Convert OpenAI messages into a generate_content conversation.

    Tool calls and results are rendered as <tool_call>/<tool_response> text; tool results go in
    user turns, and consecutive turns from the same sender are merged.
    """
    system_texts = [content_text(m.content) for m in messages if m.role == "system"]
    if tools_prompt is not None:
        system_texts = [*(system_texts or ["You are a helpful assistant."]), tools_prompt]

    conversation = [{"from": "system", "value": "\n\n".join(system_texts)}] if system_texts else []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            sender, parts = "human", [format_tool_response(content_text(message.content))]
        elif message.role == "assistant":
            calls = [format_tool_call(c.function.name, c.function.arguments) for c in message.tool_calls or []]
            sender, parts = "gpt", ["\n".join([content_text(message.content), *calls]).strip()]
        else:
            sender, parts = "human", content_parts(message.content)

        if conversation and conversation[-1]["from"] == sender:
            conversation[-1]["value"] += ["\n", *parts]
        else:
            conversation.append({"from": sender, "value": parts})
    return conversation


def completion_message(text: str, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    message = {"role": "assistant", "content": [{"type": "text", "text": text}] if text else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_name, tokenizer, image_processor, context_len
    disable_torch_init()
    model_path = app.args.model_path
    model_name = get_model_name_from_path(model_path)
    conversation.default_conversation = conv_templates[app.args.conv_mode].copy()
    if app.args.backend == "tinychat":
        from llava.model.tinychat_backend import TinyChatNVILA

        model = TinyChatNVILA(model_path, app.args.quant_dir)
        tokenizer = model.tokenizer
        print(f"Model {model_name} loaded successfully with the TinyChat backend.")
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, None)
        print(f"Model {model_name} loaded successfully. Context length: {context_len}")
    yield


app = FastAPI(lifespan=lifespan)


# Load model upon startup
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        global model, tokenizer, image_processor, context_len

        if request.model != model_name:
            raise ValueError(
                f"The endpoint is configured to use the model {model_name}, "
                f"but the request model is {request.model}"
            )

        generation_config = copy.deepcopy(model.default_generation_config)

        generation_config.max_new_tokens = request.max_tokens
        generation_config.temperature = request.temperature
        generation_config.top_p = request.top_p
        generation_config.do_sample = request.temperature > 0
        generation_config.num_beams = request.num_beams
        generation_config.use_cache = request.use_cache

        conv = conv_templates[app.args.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        stop = [request.stop] if isinstance(request.stop, str) else list(request.stop or [])
        tool_names = [tool.function.name for tool in request.tools or []]
        use_tools = bool(tool_names) and request.tool_choice != "none"
        tools_prompt = None
        if use_tools:
            if isinstance(request.tool_choice, NamedToolChoice):
                required = request.tool_choice.function.name
            else:
                required = request.tool_choice == "required"
            tools_prompt = build_tools_system_prompt([tool.dict(exclude_none=True) for tool in request.tools], required)
            # Keep the model from hallucinating the tool result after its call.
            stop.append(TOOL_RESPONSE_TAG)

        prompt = build_conversation(request.messages, tools_prompt)
        generate_kwargs = dict(prompt=prompt, generation_config=generation_config, stop=stop)

        def make_chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": time.time(),
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk)}\n\n"

        completion_id = uuid.uuid4().hex

        with torch.inference_mode():
            if request.stream and not use_tools:
                streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, skip_special_tokens=True)
                Thread(target=generate_to_streamer, args=(streamer,), kwargs=generate_kwargs).start()

                def chunk_generator():
                    prepend_space = False
                    stop_filter = StopStringFilter(stop)
                    for new_text in streamer:
                        if new_text == " ":
                            prepend_space = True
                            continue
                        if new_text.endswith(stop_str):
                            new_text = new_text[: -len(stop_str)].strip()
                            prepend_space = False
                        elif prepend_space:
                            new_text = " " + new_text
                            prepend_space = False
                        new_text = stop_filter.feed(new_text)
                        if len(new_text):
                            yield make_chunk({"content": new_text})
                    tail = stop_filter.flush()
                    if tail:
                        yield make_chunk({"content": tail})
                    yield make_chunk({}, finish_reason="stop")
                    yield "data: [DONE]\n\n"

                return StreamingResponse(chunk_generator())

            with generation_lock:
                outputs = model.generate_content(**generate_kwargs)
            outputs = truncate_at_stop(outputs, stop)[0]
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            print("\nAssistant: ", outputs)

            text, tool_calls = parse_tool_calls(outputs, tool_names) if use_tools else (outputs, [])
            if not text and not tool_calls:
                raise ValueError("The model response is empty or malformed.")
            finish_reason = "tool_calls" if tool_calls else "stop"

            if request.stream:
                # Tool calls can only be parsed from the full output, so it is sent as a single chunk.
                def buffered_chunk_generator():
                    delta = {"role": "assistant"}
                    if text:
                        delta["content"] = text
                    if tool_calls:
                        delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(tool_calls)]
                    yield make_chunk(delta)
                    yield make_chunk({}, finish_reason=finish_reason)
                    yield "data: [DONE]\n\n"

                return StreamingResponse(buffered_chunk_generator())

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": time.time(),
                "model": request.model,
                "choices": [
                    {"index": 0, "message": completion_message(text, tool_calls), "finish_reason": finish_reason}
                ],
            }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


if __name__ == "__main__":

    host = os.getenv("VILA_HOST", "0.0.0.0")
    port = os.getenv("VILA_PORT", 8000)
    model_path = os.getenv("VILA_MODEL_PATH", "Efficient-Large-Model/VILA1.5-3B")
    conv_mode = os.getenv("VILA_CONV_MODE", "vicuna_v1")
    workers = os.getenv("VILA_WORKERS", 1)
    backend = os.getenv("VILA_BACKEND", "hf")
    quant_dir = os.getenv("VILA_QUANT_DIR") or None

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=host)
    parser.add_argument("--port", type=int, default=port)
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--conv-mode", type=str, default=conv_mode)
    parser.add_argument("--workers", type=int, default=workers)
    parser.add_argument("--backend", choices=["hf", "tinychat"], default=backend)
    parser.add_argument(
        "--quant-dir", type=str, default=quant_dir, help="AWQ checkpoint dir for --backend tinychat (default: runs/awq/<model>)"
    )
    app.args = parser.parse_args()

    uvicorn.run(app, host=app.args.host, port=app.args.port, workers=app.args.workers)


