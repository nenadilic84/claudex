import uvicorn
import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse # Added JSONResponse explicitly
import logging
import os
import json
import uuid
import time # <--- IMPORT ADDED
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Literal
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv() # Load variables from .env file

# Target OpenAI-compatible endpoint configuration
TARGET_API_BASE = os.environ.get("TARGET_API_BASE")
TARGET_API_KEY = os.environ.get("TARGET_API_KEY")
BIG_MODEL_TARGET = os.environ.get("BIG_MODEL_TARGET")
SMALL_MODEL_TARGET = os.environ.get("SMALL_MODEL_TARGET")

# Proxy configuration
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 8082 # Port this proxy listens on (can be changed)
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Basic validation
if not TARGET_API_BASE or not TARGET_API_KEY or not BIG_MODEL_TARGET or not SMALL_MODEL_TARGET:
    raise ValueError("Missing required environment variables: TARGET_API_BASE, TARGET_API_KEY, BIG_MODEL_TARGET, SMALL_MODEL_TARGET")

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AnthropicProxy")
# Silence overly verbose libraries if needed (optional)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# --- Pydantic Models (Simplified Anthropic Format) ---
class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    source: Optional[Dict[str, Any]] = None # For image
    id: Optional[str] = None # For tool_use
    name: Optional[str] = None # For tool_use
    input: Optional[Dict[str, Any]] = None # For tool_use
    tool_use_id: Optional[str] = None # For tool_result
    content: Optional[Union[str, List[Dict], Dict, List[Any]]] = None # For tool_result

class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]

class AnthropicTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class AnthropicMessagesRequest(BaseModel):
    model: str # Original model name from client
    max_tokens: int
    messages: List[AnthropicMessage]
    # --- MODIFIED system type ---
    system: Optional[Union[str, List[ContentBlock]]] = None # Allow string or list of blocks
    # --- END MODIFICATION ---
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None # Keep for potential passthrough

# --- Helper Functions ---

def get_mapped_model(original_model: str) -> str:
    """Maps incoming Claude model names to target names."""
    lower_model = original_model.lower()
    if "/" in lower_model:
        lower_model = lower_model.split("/")[-1]

    if "sonnet" in lower_model:
        logger.info(f"Mapping '{original_model}' -> '{BIG_MODEL_TARGET}'")
        return BIG_MODEL_TARGET
    elif "haiku" in lower_model:
        logger.info(f"Mapping '{original_model}' -> '{SMALL_MODEL_TARGET}'")
        return SMALL_MODEL_TARGET
    else:
        logger.warning(f"No mapping rule for '{original_model}'. Using target: '{BIG_MODEL_TARGET}' as default.")
        return BIG_MODEL_TARGET

def convert_anthropic_to_openai_request(
    anthropic_req: AnthropicMessagesRequest, mapped_model: str
) -> Dict[str, Any]:
    """Converts Anthropic request format to OpenAI format."""
    openai_messages = []

    # --- MODIFIED system prompt handling ---
    system_content_str = None
    if isinstance(anthropic_req.system, str):
        system_content_str = anthropic_req.system
    elif isinstance(anthropic_req.system, list):
        # Concatenate text from all text blocks in the system list
        system_content_str = "\n".join(
            block.text for block in anthropic_req.system if block.type == "text" and block.text
        )
    if system_content_str and system_content_str.strip():
        openai_messages.append({"role": "system", "content": system_content_str.strip()})
    # --- END MODIFICATION ---

    # Process conversation messages (same logic as before for tool results/calls)
    for msg in anthropic_req.messages:
        openai_msg = {"role": msg.role}
        if isinstance(msg.content, str):
            openai_msg["content"] = msg.content if msg.content else "..." # Ensure not empty
            openai_messages.append(openai_msg)
        elif isinstance(msg.content, list):
            combined_text = ""
            tool_results_for_openai = []

            for block in msg.content:
                if block.type == "text" and block.text:
                    combined_text += block.text + "\n"
                elif block.type == "tool_result" and block.tool_use_id:
                    tool_content = ""
                    if isinstance(block.content, str): tool_content = block.content
                    elif isinstance(block.content, list):
                        for item in block.content:
                           if isinstance(item, dict) and item.get("type") == "text": tool_content += item.get("text", "") + "\n"
                           else:
                               try: tool_content += json.dumps(item) + "\n"
                               except Exception: tool_content += str(item) + "\n"
                    else:
                        try: tool_content += json.dumps(block.content) + "\n"
                        except Exception: tool_content += str(block.content) + "\n"
                    tool_results_for_openai.append({"role": "tool", "tool_call_id": block.tool_use_id, "content": tool_content.strip()})
                elif block.type == "image": logger.warning("Ignoring image block for OpenAI conversion.")
                elif block.type == "tool_use" and msg.role == "user": logger.warning("Ignoring tool_use block found in user message during conversion.")

            # Add text content for the user/assistant message itself
            if combined_text.strip():
                openai_msg["content"] = combined_text.strip()
                openai_messages.append(openai_msg)
            # If only tool results, don't add an empty user/assistant message block
            elif not combined_text.strip() and tool_results_for_openai and msg.role == "user":
                 pass # Avoid adding user message if it *only* contained tool results

            # Append tool results following the message they belong to
            openai_messages.extend(tool_results_for_openai)

        # Handle tool calls from assistant messages
        if msg.role == "assistant":
            tool_calls_for_openai = []
            assistant_text_content = "" # Capture potential text alongside tool calls
            if isinstance(msg.content, list):
                for block in msg.content:
                    if block.type == "text" and block.text:
                         assistant_text_content += block.text + "\n" # Add text if assistant provided it
                    elif block.type == "tool_use" and block.id and block.name and block.input is not None:
                        tool_calls_for_openai.append({
                            "id": block.id,
                            "type": "function",
                            "function": {"name": block.name, "arguments": json.dumps(block.input)} # Arguments must be JSON string
                        })

            # Ensure the assistant message exists before adding tool_calls or content
            assistant_msg_exists = any(m is openai_msg for m in openai_messages)
            if not assistant_msg_exists and (assistant_text_content.strip() or tool_calls_for_openai):
                openai_messages.append(openai_msg) # Add the base assistant message first

            if assistant_text_content.strip():
                 openai_msg["content"] = assistant_text_content.strip() # Add text content
            if tool_calls_for_openai:
                 openai_msg["tool_calls"] = tool_calls_for_openai
                 # If there was no text content, OpenAI requires content to be explicitly None
                 if not assistant_text_content.strip():
                    openai_msg["content"] = None

    # --- Rest of the conversion logic (tools, tool_choice, optional params) ---
    # (Keep the existing logic for these parts)
    openai_request = {
        "model": mapped_model,
        "messages": openai_messages,
        "max_tokens": min(anthropic_req.max_tokens, 16384),
        "stream": anthropic_req.stream,
    }
    if anthropic_req.temperature is not None: openai_request["temperature"] = anthropic_req.temperature
    if anthropic_req.top_p is not None: openai_request["top_p"] = anthropic_req.top_p
    if anthropic_req.stop_sequences: openai_request["stop"] = anthropic_req.stop_sequences
    if anthropic_req.metadata and "user" in anthropic_req.metadata: openai_request["user"] = str(anthropic_req.metadata["user"])

    if anthropic_req.tools:
        openai_request["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
            for t in anthropic_req.tools
        ]
    if anthropic_req.tool_choice:
        choice_type = anthropic_req.tool_choice.get("type")
        if choice_type == "auto" or choice_type == "any": openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in anthropic_req.tool_choice:
            openai_request["tool_choice"] = {"type": "function", "function": {"name": anthropic_req.tool_choice["name"]}}
        else: openai_request["tool_choice"] = "auto"

    logger.debug(f"Converted OpenAI Request: {json.dumps(openai_request, indent=2)}")
    return openai_request

# --- Keep convert_openai_to_anthropic_response ---
# --- Keep handle_openai_to_anthropic_streaming ---
# (No changes needed in these response conversion functions based on the error)
def convert_openai_to_anthropic_response(
    openai_dict: Dict[str, Any], original_model: str
) -> Dict[str, Any]:
    """Converts OpenAI response format to Anthropic format."""
    anthropic_content = []
    stop_reason = "end_turn"
    usage = {"input_tokens": 0, "output_tokens": 0}

    try:
        choice = openai_dict.get("choices", [{}])[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "stop")

        # Text content
        if message.get("content"):
            anthropic_content.append({"type": "text", "text": message["content"]})

        # Tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                if tc.get("type") == "function" and tc.get("function"):
                    try:
                        # Arguments from OpenAI are already JSON strings
                        tool_input = json.loads(tc["function"].get("arguments", "{}"))
                    except json.JSONDecodeError:
                        tool_input = {"raw_string_args": tc["function"].get("arguments", "")} # Handle non-JSON args
                    anthropic_content.append({
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:10]}"),
                        "name": tc["function"].get("name"),
                        "input": tool_input
                    })

        # Map stop reason
        if finish_reason == "length": stop_reason = "max_tokens"
        elif finish_reason == "stop": stop_reason = "end_turn"
        elif finish_reason == "tool_calls": stop_reason = "tool_use"

        # Usage
        if openai_dict.get("usage"):
            usage["input_tokens"] = openai_dict["usage"].get("prompt_tokens", 0)
            usage["output_tokens"] = openai_dict["usage"].get("completion_tokens", 0)

        if not anthropic_content: anthropic_content.append({"type": "text", "text": ""})

        anthropic_response = {
            "id": openai_dict.get("id", f"msg_{uuid.uuid4().hex[:10]}"),
            "type": "message", "role": "assistant", "model": original_model,
            "content": anthropic_content, "stop_reason": stop_reason,
            "stop_sequence": None, "usage": usage,
        }
        logger.debug(f"Converted Anthropic Response: {json.dumps(anthropic_response, indent=2)}")
        return anthropic_response

    except Exception as e:
        logger.error(f"Error converting OpenAI response: {e}", exc_info=True)
        return {
            "id": f"msg_error_{uuid.uuid4().hex[:10]}", "type": "message",
            "role": "assistant", "model": original_model,
            "content": [{"type": "text", "text": f"Error processing backend response: {e}"}],
            "stop_reason": "error", "usage": {"input_tokens": 0, "output_tokens": 0},
        }

async def handle_openai_to_anthropic_streaming(openai_stream, original_model: str):
    """Converts OpenAI streaming chunks to Anthropic Server-Sent Events."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0
    final_stop_reason = "end_turn"

    try:
        # 1. Send message_start
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': original_model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens}}})}\n\n"
        # 2. Send ping
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        content_block_index = -1
        current_tool_id = None
        current_tool_name = None
        accumulated_tool_args = ""
        text_block_started = False
        tool_blocks = {} # Track tool blocks by index {anthropic_index: {id:.., name:.., args:...}}

        async for chunk_bytes in openai_stream:
            chunk_str = chunk_bytes.decode('utf-8').strip()
            lines = chunk_str.splitlines() # Handle multiple SSE events in one chunk

            for line in lines:
                if not line or line == "data: [DONE]":
                    continue
                if not line.startswith("data:"): continue

                chunk_str_data = line[len("data: "):]

                try:
                    chunk_data = json.loads(chunk_str_data)
                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})

                    # --- Handle Text Content ---
                    if delta.get("content"):
                        if not text_block_started:
                            content_block_index = 0 # Text is always index 0
                            text_block_started = True
                            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_index, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_index, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
                        output_tokens += 1

                    # --- Handle Tool Calls ---
                    if delta.get("tool_calls"):
                         # Stop previous text block if it was started
                        if text_block_started and content_block_index == 0:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            content_block_index = -1 # Reset index for tools

                        for tc in delta["tool_calls"]:
                            tc_index = tc.get("index", 0) # OpenAI provides index
                            anthropic_index = tc_index + 1 # Anthropic index starts after text block (if any)

                            if tc.get("id"): # Start of a new tool call
                                current_tool_id = tc["id"]
                                current_tool_name = tc.get("function", {}).get("name", "")
                                accumulated_tool_args = tc.get("function", {}).get("arguments", "")
                                tool_blocks[anthropic_index] = {"id": current_tool_id, "name": current_tool_name, "args": accumulated_tool_args}

                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_index, 'content_block': {'type': 'tool_use', 'id': current_tool_id, 'name': current_tool_name, 'input': {}}})}\n\n"
                                if accumulated_tool_args:
                                     yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_index, 'delta': {'type': 'input_json_delta', 'partial_json': accumulated_tool_args}})}\n\n"

                            elif tc.get("function", {}).get("arguments") and anthropic_index in tool_blocks: # Continuation
                                 args_delta = tc["function"]["arguments"]
                                 tool_blocks[anthropic_index]["args"] += args_delta
                                 yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_delta}})}\n\n"

                    # --- Handle Finish Reason ---
                    finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                    if finish_reason:
                        if finish_reason == "length": final_stop_reason = "max_tokens"
                        elif finish_reason == "stop": final_stop_reason = "end_turn"
                        elif finish_reason == "tool_calls": final_stop_reason = "tool_use"

                    # --- Handle Usage ---
                    if chunk_data.get("usage"):
                         # In streaming, usage might appear in chunks or only at the end
                         if chunk_data["usage"].get("prompt_tokens"):
                              input_tokens = chunk_data["usage"]["prompt_tokens"]
                         if chunk_data["usage"].get("completion_tokens"):
                              output_tokens = chunk_data["usage"]["completion_tokens"] # Update if total provided


                except json.JSONDecodeError: logger.warning(f"Could not decode stream chunk data: {chunk_str_data}")
                except Exception as e: logger.error(f"Error processing stream chunk data: {e}", exc_info=True)

        # 3. Send content_block_stop for all blocks started
        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        for index in tool_blocks:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"


        # 4. Send message_delta with stop reason and final usage
        final_usage = {"output_tokens": output_tokens}
        # Try to add input tokens if we got them
        if input_tokens > 0:
             final_usage["input_tokens"] = input_tokens

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': final_usage})}\n\n"

        # 5. Send message_stop
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    except Exception as e:
        logger.error(f"Error during streaming conversion: {e}", exc_info=True)
        try:
            error_payload = json.dumps({'type': 'error', 'error': {'type': 'internal_server_error', 'message': str(e)}})
            yield f"event: error\ndata: {error_payload}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        except Exception: logger.critical("Failed to send error event during streaming.")
    finally:
        logger.debug("Finished Anthropic stream conversion.")

# --- FastAPI Application ---
app = FastAPI(title="Anthropic to OpenAI Proxy")
http_client = httpx.AsyncClient()

@app.post("/v1/messages")
async def proxy_anthropic_request(anthropic_request: AnthropicMessagesRequest, raw_request: Request):
    """Receives Anthropic request, converts, proxies, converts back."""
    start_time = time.time() # Now time is defined
    original_model = anthropic_request.model
    mapped_model = get_mapped_model(original_model)

    logger.info(f"--> Request for '{original_model}' mapped to '{mapped_model}' (Stream: {anthropic_request.stream})")

    try:
        # 1. Convert Request
        openai_request = convert_anthropic_to_openai_request(anthropic_request, mapped_model)

        # 2. Prepare headers for target
        target_headers = {
            "Authorization": f"Bearer {TARGET_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not anthropic_request.stream else "text/event-stream",
        }
        if "user-agent" in raw_request.headers:
             target_headers["User-Agent"] = raw_request.headers["user-agent"]


        # 3. Forward Request
        target_url = f"{TARGET_API_BASE.rstrip('/')}/chat/completions"

        logger.debug(f"Forwarding to URL: {target_url}")
        # logger.debug(f"Forwarding Headers: {target_headers}") # Can be noisy
        # logger.debug(f"Forwarding Body: {json.dumps(openai_request)}") # Very noisy, be careful

        response = await http_client.post(
            target_url,
            json=openai_request,
            headers=target_headers,
            timeout=300.0
        )

        # 4. Handle Response
        elapsed_time = time.time() - start_time
        logger.info(f"<-- Response status from '{mapped_model}': {response.status_code} ({elapsed_time:.2f}s)")

        if response.status_code >= 400:
             error_content = await response.aread()
             logger.error(f"Target API Error ({response.status_code}): {error_content.decode()}")
             try: error_detail = response.json()
             except Exception: error_detail = error_content.decode()
             # Use status_code from target, detail from target error
             raise HTTPException(status_code=response.status_code, detail=error_detail)


        # Handle Streaming Response
        if anthropic_request.stream:
            if 'text/event-stream' not in response.headers.get('content-type', '').lower():
                 error_body = await response.aread()
                 logger.error(f"Backend did not stream as expected. Status: {response.status_code}. Body: {error_body.decode()}")
                 raise HTTPException(status_code=500, detail="Backend did not return a stream.")

            return StreamingResponse(
                handle_openai_to_anthropic_streaming(response.aiter_bytes(), original_model),
                media_type="text/event-stream" # Set correct content type for Anthropic client
            )
        # Handle Non-Streaming Response
        else:
            openai_response_dict = response.json()
            anthropic_response_dict = convert_openai_to_anthropic_response(openai_response_dict, original_model)
            return JSONResponse(content=anthropic_response_dict) # Use JSONResponse

    except httpx.RequestError as e:
        logger.error(f"HTTPX Request Error: {e}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"Error connecting to target API: {e}")
    except ValueError as e:
        logger.error(f"Configuration or Value Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except HTTPException as e:
        raise e # Re-raise FastAPI/proxy errors correctly
    except Exception as e:
        logger.error(f"Unhandled Exception: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.get("/")
async def health_check():
    return {"status": "ok", "message": "Anthropic-OpenAI Proxy is running"}

# --- Main Execution ---
if __name__ == "__main__":
    logger.info(f"Starting Anthropic-OpenAI Proxy on {LISTEN_HOST}:{LISTEN_PORT}")
    logger.info(f"Target API Base: {TARGET_API_BASE}")
    logger.info(f"Mapping Sonnet -> {BIG_MODEL_TARGET}")
    logger.info(f"Mapping Haiku -> {SMALL_MODEL_TARGET}")
    uvicorn.run(app, host=LISTEN_HOST, port=LISTEN_PORT, log_config=None)