### **Project Goal**
Build an automated email response system that:
1. Receives emails with attachments via an HTTP endpoint.
2. Uploads attachments to **DigitalOcean Spaces** for secure storage.
3. Uses **GPT-4** for attachment labelling, parsing, and analysis to extract structured insights.
4. Forwards the email content, analysis results, and attachment summaries to **o1** for generating a tailored response.
5. Sends the AI-generated response back to the original email sender using **Resend**.

---

### **Requirements**

#### **1. Email Reception**
- **HTTP Endpoint:**
  - Use a Python web framework (e.g., Flask or FastAPI) to create an endpoint to handle incoming emails.
- **Email Data:**
  - Parse the incoming email to extract:
    - Sender address
    - Subject
    - Body
    - Attachments (names, types, and raw content)
- **Validation:**
  - Validate attachment size and type for compatibility.
  - Reject unsupported formats with an appropriate error response.

#### **2. Attachment Storage**
- **DigitalOcean Spaces:**
  - Upload validated attachments to DigitalOcean Spaces.
  - Generate signed URLs for temporary access to these files.
- **Metadata:**
  - Record metadata for each attachment, including filename, size, type, and storage URL.

#### **3. Preprocessing with GPT-4**
- **Labelling and Parsing:**
  - Use GPT-4 to analyze attachments and generate:
    - Labels (e.g., "Invoice", "Contract", "Image").
    - Summaries of the content.
    - Key-value pairs or JSON structures for extracted data (e.g., totals, names, dates).
- **Preprocessing Tasks:**
  - If attachments are non-text (e.g., images, scanned PDFs), preprocess them using OCR (e.g., Tesseract or a cloud-based OCR API) to extract text before analysis.

#### **4. Main AI Response with o1**
- **Payload:**
  - Pass the email content and GPT-4’s structured analysis results to **o1** for generating a response.
  - Include all relevant data (e.g., email body, extracted attachment details).
- **Custom Responses:**
  - Use o1’s capabilities to generate tailored responses based on email context and attachment analysis.

#### **5. Response Delivery**
- **Resend Integration:**
  - Use Resend to send the AI-generated response back to the sender.
  - Include summaries or key details from attachments in the email response.

#### **6. Workflow Automation**
- **QStash for Queueing:**
  - Queue email and attachment data for processing using QStash.
  - Ensure asynchronous handling to decouple email receipt, attachment analysis, and AI response generation.

---

### **Implementation Details**

#### **Step 1: Email Reception**
- **Tools:** Flask or FastAPI.
- **Workflow:**
  1. Receive email payload via HTTP POST.
  2. Parse email content into structured fields.
  3. Validate and preprocess attachments.

**Example Code:**
python
from flask import Flask, request
import boto3
from botocore.client import Config

app = Flask(__name__)

# DigitalOcean Spaces client setup
session = boto3.session.Session()
s3_client = session.client(
    "s3",
    region_name="nyc3",
    endpoint_url="https://nyc3.digitaloceanspaces.com",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY"
)

# Upload file to DigitalOcean Spaces
def upload_to_spaces(file_name, file_content):
    bucket_name = "your-space-name"
    s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=file_content)
    return f"https://{bucket_name}.nyc3.digitaloceanspaces.com/{file_name}"

@app.route("/email-receive", methods=["POST"])
def email_receive():
    data = request.json
    sender = data.get("from")
    subject = data.get("subject")
    body = data.get("text")
    attachments = data.get("attachments", [])

    attachment_urls = []
    for attachment in attachments:
        file_name = attachment["filename"]
        file_content = attachment["content"]
        file_url = upload_to_spaces(file_name, file_content)
        attachment_urls.append(file_url)

    # Queue the email for further processing
    queue_email_for_processing(sender, subject, body, attachment_urls)
    return "Email received and queued", 200


---

#### **Step 2: Queue Email Data Using QStash**
- **Tools:** QStash Python SDK.
- **Workflow:**
  1. Publish email data and attachment URLs to a processing endpoint.
  2. Handle retries and ensure reliable delivery.

**Example Code:**
python
from qstash import QStash

qstash_token = "YOUR_QSTASH_TOKEN"
qstash_client = QStash(qstash_token)

def queue_email_for_processing(sender, subject, body, attachment_urls):
    qstash_client.message.publish_json(
        url="https://your-domain.com/process-email",
        body={
            "sender": sender,
            "subject": subject,
            "body": body,
            "attachments": attachment_urls
        },
        headers={"Content-Type": "application/json"},
    )


---

#### **Step 3: Preprocess Attachments with GPT-4**
- **Workflow:**
  1. Send attachment URLs to GPT-4 for analysis.
  2. Extract and return structured labels, summaries, and key insights.

**Example Code:**
python
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def analyze_attachment(attachment_url):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI that labels, summarizes, and extracts insights from files."},
            {"role": "user", "content": f"Analyze this attachment: {attachment_url}."}
        ],
    )
    return response["choices"][0]["message"]["content"]


---

#### **Step 4: Generate Response with o1**
- **Workflow:**
  1. Pass email content and GPT-4 analysis results to o1.
  2. Generate a tailored response.

---

#### **Step 5: Send Response via Resend**
- **Tools:** Resend Python SDK.
- **Workflow:**
  1. Use Resend to email the response back to the sender.
  2. Include summaries or extracted details from attachments.

**Example Code:**
python
from resend import Resend

resend_client = Resend(api_key="YOUR_RESEND_API_KEY")

def send_response(to_email, subject, body):
    resend_client.emails.send({
        "from": "Your Name <your_email@example.com>",
        "to": [to_email],
        "subject": subject,
        "html": f"<p>{body}</p>",
    })

Here is the sdk for https://github.com/upstash/qstash-py

<source type="github_repository" url="https://github.com/upstash/qstash-py">
<file name=".env.example">
QSTASH_TOKEN="YOUR_TOKEN"
QSTASH_CURRENT_SIGNING_KEY="&lt;YOUR_CURRENT_SIGNING_KEY&gt;"
QSTASH_NEXT_SIGNING_KEY="&lt;YOUR_NEXT_SIGNING_KEY&gt;"
OPENAI_API_KEY = "&lt;YOUR_OPENAI_API_KEY&gt;"

</file>
<file name="README.md">
# Upstash Python QStash SDK

&gt; [!NOTE]  
&gt; **This project is in GA Stage.**
&gt;
&gt; The Upstash Professional Support fully covers this project. It receives regular updates, and bug fixes.
&gt; The Upstash team is committed to maintaining and improving its functionality.

**QStash** is an HTTP based messaging and scheduling solution for serverless and edge runtimes.

[QStash Documentation](https://upstash.com/docs/qstash)

### Install
shell
pip install qstash
### Usage

You can get your QStash token from the [Upstash Console](https://console.upstash.com/qstash).

#### Publish a JSON message
python
from qstash import QStash

client = QStash("&lt;QSTASH_TOKEN&gt;")

res = client.message.publish_json(
    url="https://example.com",
    body={"hello": "world"},
    headers={
        "test-header": "test-value",
    },
)

print(res.message_id)
#### [Create a scheduled message](https://upstash.com/docs/qstash/features/schedules)
python
from qstash import QStash

client = QStash("&lt;QSTASH_TOKEN&gt;")

schedule_id = client.schedule.create(
    destination="https://example.com",
    cron="*/5 * * * *",
)

print(schedule_id)
#### [Receiving messages](https://upstash.com/docs/qstash/howto/receiving)
python
from qstash import Receiver

# Keys available from the QStash console
receiver = Receiver(
    current_signing_key="CURRENT_SIGNING_KEY",
    next_signing_key="NEXT_SIGNING_KEY",
)

# ... in your request handler

signature, body = req.headers["Upstash-Signature"], req.body

receiver.verify(
    body=body,
    signature=signature,
    url="https://example.com",  # Optional
)
#### Create Chat Completions
python
from qstash import QStash
from qstash.chat import upstash

client = QStash("&lt;QSTASH_TOKEN&gt;")

res = client.chat.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    provider=upstash(),
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Turkey?",
        }
    ],
)

print(res.choices[0].message.content)
#### Create Chat Completions Using Custom Providers
python
from qstash import QStash
from qstash.chat import openai

client = QStash("&lt;QSTASH_TOKEN&gt;")

res = client.chat.create(
    model="gpt-3.5-turbo",
    provider=openai("&lt;OPENAI_API_KEY&gt;"),
    messages=[
        {
            "role": "user",
            "content": "What is the capital of Turkey?",
        }
    ],
)

print(res.choices[0].message.content)
#### Publish a JSON message to LLM
python
from qstash import QStash
from qstash.chat import upstash

client = QStash("&lt;QSTASH_TOKEN&gt;")

res = client.message.publish_json(
    api={"name": "llm", "provider": upstash()},
    body={
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Turkey?",
            }
        ],
    },
    callback="https://example-cb.com",
)

print(res.message_id)
#### Publish a JSON message to LLM Using Custom Providers
python
from qstash import QStash
from qstash.chat import openai

client = QStash("&lt;QSTASH_TOKEN&gt;")

res = client.message.publish_json(
    api={"name": "llm", "provider": openai("&lt;OPENAI_API_KEY&gt;")},
    body={
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Turkey?",
            }
        ],
    },
    callback="https://example-cb.com",
)

print(res.message_id)
#### Additional configuration
python
from qstash import QStash

# Create a client with a custom retry configuration. This is
# for sending messages to QStash, not for sending messages to
# your endpoints.
# The default configuration is:
# {
#   "retries": 5,
#   "backoff": lambda retry_count: math.exp(retry_count) * 50,
# }
client = QStash(
    token="&lt;QSTASH_TOKEN&gt;",
    retry={
        "retries": 1,
        "backoff": lambda retry_count: (2 ** retry_count) * 20,
    },
)

# Publish to URL
client.message.publish_json(
    url="https://example.com",
    body={"key": "value"},
    # Retry sending message to API 3 times
    # https://upstash.com/docs/qstash/features/retry
    retries=3,
    # Schedule message to be sent 4 seconds from now
    delay="4s",
    # When message is sent, send a request to this URL
    # https://upstash.com/docs/qstash/features/callbacks
    callback="https://example.com/callback",
    # When message fails to send, send a request to this URL
    failure_callback="https://example.com/failure_callback",
    # Headers to forward to the endpoint
    headers={
        "test-header": "test-value",
    },
    # Enable content-based deduplication
    # https://upstash.com/docs/qstash/features/deduplication#content-based-deduplication
    content_based_deduplication=True,
)
Additional methods are available for managing url groups, schedules, and messages. See the examples folder for more.

### Development

1. Clone the repository
2. Install [Poetry](https://python-poetry.org/docs/#installation)
3. Install dependencies with `poetry install`
4. Create a .env file with `cp .env.example .env` and fill in the `QSTASH_TOKEN`
5. Run tests with `poetry run pytest`
6. Format with `poetry run ruff format .`

</file>
<file name="examples/async_publish.py">
"""
Uses asyncio to asynchronously publish a JSON message with a 3s delay to a URL using QStash.
"""

import asyncio

from qstash import AsyncQStash


async def main():
    client = AsyncQStash(
        token="&lt;QSTASH-TOKEN&gt;",
    )

    res = await client.message.publish_json(
        url="https://example.com",
        body={"hello": "world"},
        headers={
            "test-header": "test-value",
        },
        delay="3s",
    )

    print(res.message_id)


if __name__ == "__main__":
    asyncio.run(main())

</file>
<file name="examples/basic_publish.py">
"""
Publishes a JSON message with a 3s delay to a URL using QStash.
"""

from qstash import QStash


def main():
    client = QStash(
        token="&lt;QSTASH-TOKEN&gt;",
    )

    res = client.message.publish_json(
        url="https://example.com",
        body={"hello": "world"},
        headers={
            "test-header": "test-value",
        },
        delay="3s",
    )

    print(res.message_id)


if __name__ == "__main__":
    main()

</file>
<file name="examples/basic_schedule.py">
"""
Create a schedule that publishes a message every minute.
"""

from qstash import QStash


def main():
    client = QStash(
        token="&lt;QSTASH-TOKEN&gt;",
    )

    schedule_id = client.schedule.create_json(
        cron="* * * * *",
        destination="https://example..com",
        body={"hello": "world"},
    )

    # Print out the schedule ID
    print(schedule_id)

    # You can also get a schedule by ID
    schedule = client.schedule.get(schedule_id)
    print(schedule.cron)


if __name__ == "__main__":
    main()

</file>
<file name="examples/callback.py">
"""
Publish a message to a URL and send the response to a callback URL.

This is useful if you have a time consuming API call
and you want to send the response to your API URL without having
to wait for the response in a serverless function.
"""

from qstash import QStash


def main():
    client = QStash(
        token="&lt;QSTASH-TOKEN&gt;",
    )

    client.message.publish_json(
        url="https://expensive.com",
        callback="https://example-cb.com",
        # We want to send a GET request to https://expensive.com and have the response
        # sent to https://example-cb.com
        method="GET",
    )


if __name__ == "__main__":
    main()

</file>
<file name="examples/chat.py">
"""
Create a chat completion request and receive the response, either
at once, or streaming chunk by chunk.
"""

from qstash import QStash


def main():
    client = QStash(
        token="&lt;QSTASH-TOKEN&gt;",
    )

    res = client.chat.create(
        messages=[{"role": "user", "content": "How are you?"}],
        model="meta-llama/Meta-Llama-3-8B-Instruct",
    )

    # Get the response at once
    print(res.choices[0].message.content)

    stream_res = client.chat.create(
        messages=[{"role": "user", "content": "How are you again?"}],
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        stream=True,
    )

    # Get the response in chunks over time
    for chunk in stream_res:
        content = chunk.choices[0].delta.content
        if content is None:
            # Content is none for the first chunk
            continue

        print(content, end="")


if __name__ == "__main__":
    main()

</file>
<file name="examples/llm.py">
"""
Create a chat completion request and send the response to a callback URL.

This is useful to send the response to your API without having
to wait for the response in a serverless function.
"""

from qstash import QStash
from qstash.chat import upstash


def main():
    client = QStash(
        token="&lt;QSTASH-TOKEN&gt;",
    )

    client.message.publish_json(
        api={"name": "llm", "provider": upstash()},
        body={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of Turkey?",
                }
            ],
        },
        callback="https://example-cb.com",
        # We want to send the response to https://example-cb.com
    )


if __name__ == "__main__":
    main()

</file>
<file name="qstash/__init__.py">
from qstash.asyncio.client import AsyncQStash
from qstash.client import QStash
from qstash.receiver import Receiver

__version__ = "2.0.3"
__all__ = ["QStash", "AsyncQStash", "Receiver"]

</file>
<file name="qstash/asyncio/__init__.py">

</file>
<file name="qstash/asyncio/chat.py">
import json
from types import TracebackType
from typing import AsyncIterator, Dict, List, Optional, Union, Type

import httpx

from qstash.asyncio.http import AsyncHttpClient
from qstash.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatModel,
    ChatResponseFormat,
    convert_to_chat_messages,
    parse_chat_completion_chunk_response,
    parse_chat_completion_response,
    prepare_chat_request_body,
    LlmProvider,
    UPSTASH_LLM_PROVIDER,
)


class AsyncChatCompletionChunkStream:
    """
    An async iterable that yields completion chunks.

    To not leak any resources, either
    - the chunks most be read to completion
    - close() must be called
    - context manager must be used
    """

    def __init__(self, response: httpx.Response) -&gt; None:
        self._response = response
        self._iterator = self._chunk_iterator()

    async def close(self) -&gt; None:
        """
        Closes the underlying resources.

        No need to call it if the iterator is read to completion.
        """
        await self._response.aclose()

    async def __anext__(self) -&gt; ChatCompletionChunk:
        return await self._iterator.__anext__()

    def __aiter__(self) -&gt; AsyncIterator[ChatCompletionChunk]:
        return self

    async def __aenter__(self) -&gt; "AsyncChatCompletionChunkStream":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -&gt; None:
        await self.close()

    async def _chunk_iterator(self) -&gt; AsyncIterator[ChatCompletionChunk]:
        it = self._data_iterator()
        async for data in it:
            if data == b"[DONE]":
                break

            yield parse_chat_completion_chunk_response(json.loads(data))

        async for _ in it:
            pass

    async def _data_iterator(self) -&gt; AsyncIterator[bytes]:
        pending = None

        async for data in self._response.aiter_bytes():
            if pending is not None:
                data = pending + data

            parts = data.split(b"\n\n")

            if parts and parts[-1] and data and parts[-1][-1] == data[-1]:
                pending = parts.pop()
            else:
                pending = None

            for part in parts:
                if part.startswith(b"data: "):
                    part = part[6:]
                    yield part

        if pending is not None:
            if pending.startswith(b"data: "):
                pending = pending[6:]
                yield pending


class AsyncChatApi:
    def __init__(self, http: AsyncHttpClient) -&gt; None:
        self._http = http

    async def create(
        self,
        *,
        messages: List[ChatCompletionMessage],
        model: ChatModel,
        provider: Optional[LlmProvider] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -&gt; Union[ChatCompletion, AsyncChatCompletionChunkStream]:
        """
        Creates a model response for the given chat conversation.

        When `stream` is set to `True`, it returns an iterable
        that can be used to receive chat completion delta chunks
        one by one.

        Otherwise, response is returned in one go as a chat
        completion object.

        :param messages: One or more chat messages.
        :param model: Name of the model.
        :param provider: LLM provider for the chat completion request. By default,
            Upstash will be used.
        :param frequency_penalty: Number between `-2.0` and `2.0`.
            Positive values penalize new tokens based on their existing
            frequency in the text so far, decreasing the model's likelihood
            to repeat the same line verbatim.
        :param logit_bias: Modify the likelihood of specified tokens appearing
            in the completion. Accepts a dictionary that maps tokens (specified
            by their token ID in the tokenizer) to an associated bias value
            from `-100` to `100`. Mathematically, the bias is added to the
            logits generated by the model prior to sampling. The exact effect
            will vary per model, but values between `-1` and `1` should
            decrease or increase likelihood of selection; values like `-100` or
            `100` should result in a ban or exclusive selection of the
            relevant token.
        :param logprobs: Whether to return log probabilities of the output
            tokens or not. If true, returns the log probabilities of each
            output token returned in the content of message.
        :param top_logprobs: An integer between `0` and `20` specifying the
            number of most likely tokens to return at each token position,
            each with an associated log probability. logprobs must be set
            to true if this parameter is used.
        :param max_tokens: The maximum number of tokens that can be generated
            in the chat completion.
        :param n: How many chat completion choices to generate for each input
            message. Note that you will be charged based on the number of
            generated tokens across all of the choices. Keep `n` as `1` to
            minimize costs.
        :param presence_penalty: Number between `-2.0` and `2.0`. Positive
            values penalize new tokens based on whether they appear in the
            text so far, increasing the model's likelihood to talk about
            new topics.
        :param response_format: An object specifying the format that the
            model must output.
            Setting to `{ "type": "json_object" }` enables JSON mode,
            which guarantees the message the model generates is valid JSON.

            **Important**: when using JSON mode, you must also instruct the
            model to produce JSON yourself via a system or user message.
            Without this, the model may generate an unending stream of
            whitespace until the generation reaches the token limit, resulting
            in a long-running and seemingly "stuck" request. Also note that
            the message content may be partially cut off if
            `finish_reason="length"`, which indicates the generation exceeded
            `max_tokens` or the conversation exceeded the max context length.
        :param seed: If specified, our system will make a best effort to sample
            deterministically, such that repeated requests with the same seed
            and parameters should return the same result. Determinism is not
            guaranteed, and you should refer to the `system_fingerprint`
            response parameter to monitor changes in the backend.
        :param stop: Up to 4 sequences where the API will stop generating
            further tokens.
        :param stream: If set, partial message deltas will be sent. Tokens
            will be sent as data-only server-sent events as they become
            available.
        :param temperature: What sampling temperature to use, between `0`
            and `2`. Higher values like `0.8` will make the output more random,
            while lower values like `0.2` will make it more focused and
            deterministic.
            We generally recommend altering this or `top_p` but not both.
        :param top_p: An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the tokens
            with `top_p` probability mass. So `0.1` means only the tokens
            comprising the top `10%`` probability mass are considered.
            We generally recommend altering this or `temperature` but not both.
        """
        body = prepare_chat_request_body(
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
        )

        base_url = None
        token = None
        path = "/llm/v1/chat/completions"

        if provider is not None and provider.name != UPSTASH_LLM_PROVIDER.name:
            base_url = provider.base_url
            token = f"Bearer {provider.token}"
            path = "/v1/chat/completions"

        if stream:
            stream_response = await self._http.stream(
                path=path,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Connection": "keep-alive",
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                body=body,
                base_url=base_url,
                token=token,
            )

            return AsyncChatCompletionChunkStream(stream_response)

        response = await self._http.request(
            path=path,
            method="POST",
            headers={"Content-Type": "application/json"},
            body=body,
            base_url=base_url,
            token=token,
        )

        return parse_chat_completion_response(response)

    async def prompt(
        self,
        *,
        user: str,
        system: Optional[str] = None,
        model: ChatModel,
        provider: Optional[LlmProvider] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -&gt; Union[ChatCompletion, AsyncChatCompletionChunkStream]:
        """
        Creates a model response for the given user and optional
        system prompt. It is a utility method that converts
        the given user and system prompts to message history
        expected in the `create` method. It is only useful for
        single turn chat completions.

        When `stream` is set to `True`, it returns an iterable
        that can be used to receive chat completion delta chunks
        one by one.

        Otherwise, response is returned in one go as a chat
        completion object.

        :param user: User prompt.
        :param system: System prompt.
        :param model: Name of the model.
        :param provider: LLM provider for the chat completion request. By default,
            Upstash will be used.
        :param frequency_penalty: Number between `-2.0` and `2.0`.
            Positive values penalize new tokens based on their existing
            frequency in the text so far, decreasing the model's likelihood
            to repeat the same line verbatim.
        :param logit_bias: Modify the likelihood of specified tokens appearing
            in the completion. Accepts a dictionary that maps tokens (specified
            by their token ID in the tokenizer) to an associated bias value
            from `-100` to `100`. Mathematically, the bias is added to the
            logits generated by the model prior to sampling. The exact effect
            will vary per model, but values between `-1` and `1` should
            decrease or increase likelihood of selection; values like `-100` or
            `100` should result in a ban or exclusive selection of the
            relevant token.
        :param logprobs: Whether to return log probabilities of the output
            tokens or not. If true, returns the log probabilities of each
            output token returned in the content of message.
        :param top_logprobs: An integer between `0` and `20` specifying the
            number of most likely tokens to return at each token position,
            each with an associated log probability. logprobs must be set
            to true if this parameter is used.
        :param max_tokens: The maximum number of tokens that can be generated
            in the chat completion.
        :param n: How many chat completion choices to generate for each input
            message. Note that you will be charged based on the number of
            generated tokens across all of the choices. Keep `n` as `1` to
            minimize costs.
        :param presence_penalty: Number between `-2.0` and `2.0`. Positive
            values penalize new tokens based on whether they appear in the
            text so far, increasing the model's likelihood to talk about
            new topics.
        :param response_format: An object specifying the format that the
            model must output.
            Setting to `{ "type": "json_object" }` enables JSON mode,
            which guarantees the message the model generates is valid JSON.

            **Important**: when using JSON mode, you must also instruct the
            model to produce JSON yourself via a system or user message.
            Without this, the model may generate an unending stream of
            whitespace until the generation reaches the token limit, resulting
            in a long-running and seemingly "stuck" request. Also note that
            the message content may be partially cut off if
            `finish_reason="length"`, which indicates the generation exceeded
            `max_tokens` or the conversation exceeded the max context length.
        :param seed: If specified, our system will make a best effort to sample
            deterministically, such that repeated requests with the same seed
            and parameters should return the same result. Determinism is not
            guaranteed, and you should refer to the `system_fingerprint`
            response parameter to monitor changes in the backend.
        :param stop: Up to 4 sequences where the API will stop generating
            further tokens.
        :param stream: If set, partial message deltas will be sent. Tokens
            will be sent as data-only server-sent events as they become
            available.
        :param temperature: What sampling temperature to use, between `0`
            and `2`. Higher values like `0.8` will make the output more random,
            while lower values like `0.2` will make it more focused and
            deterministic.
            We generally recommend altering this or `top_p` but not both.
        :param top_p: An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the tokens
            with `top_p` probability mass. So `0.1` means only the tokens
            comprising the top `10%`` probability mass are considered.
            We generally recommend altering this or `temperature` but not both.
        """
        return await self.create(
            messages=convert_to_chat_messages(user, system),
            model=model,
            provider=provider,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
        )

</file>
<file name="qstash/asyncio/client.py">
from typing import Literal, Optional, Union

from qstash.asyncio.chat import AsyncChatApi
from qstash.asyncio.dlq import AsyncDlqApi
from qstash.asyncio.event import AsyncEventApi
from qstash.asyncio.http import AsyncHttpClient
from qstash.asyncio.message import AsyncMessageApi
from qstash.asyncio.queue import AsyncQueueApi
from qstash.asyncio.schedule import AsyncScheduleApi
from qstash.asyncio.signing_key import AsyncSigningKeyApi
from qstash.asyncio.url_group import AsyncUrlGroupApi
from qstash.http import RetryConfig


class AsyncQStash:
    def __init__(
        self,
        token: str,
        *,
        retry: Optional[Union[Literal[False], RetryConfig]] = None,
        base_url: Optional[str] = None,
    ) -&gt; None:
        """
        :param token: The authorization token from the Upstash console.
        :param retry: Configures how the client should retry requests.
        """
        self.http = AsyncHttpClient(
            token,
            retry,
            base_url,
        )
        self.message = AsyncMessageApi(self.http)
        """Message api."""

        self.url_group = AsyncUrlGroupApi(self.http)
        """Url group api."""

        self.queue = AsyncQueueApi(self.http)
        """Queue api."""

        self.schedule = AsyncScheduleApi(self.http)
        """Schedule api."""

        self.signing_key = AsyncSigningKeyApi(self.http)
        """Signing key api."""

        self.event = AsyncEventApi(self.http)
        """Event api."""

        self.dlq = AsyncDlqApi(self.http)
        """Dlq (Dead Letter Queue) api."""

        self.chat = AsyncChatApi(self.http)
        """Chat api."""

</file>
<file name="qstash/asyncio/dlq.py">
import json
from typing import List, Optional

from qstash.asyncio.http import AsyncHttpClient
from qstash.dlq import (
    DlqMessage,
    ListDlqMessagesResponse,
    parse_dlq_message_response,
    DlqFilter,
    prepare_list_dlq_messages_params,
)


class AsyncDlqApi:
    def __init__(self, http: AsyncHttpClient) -&gt; None:
        self._http = http

    async def get(self, dlq_id: str) -&gt; DlqMessage:
        """
        Gets a message from DLQ.

        :param dlq_id: The unique id within the DLQ to get.
        """
        response = await self._http.request(
            path=f"/v2/dlq/{dlq_id}",
            method="GET",
        )

        return parse_dlq_message_response(response, dlq_id)

    async def list(
        self,
        *,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        filter: Optional[DlqFilter] = None,
    ) -&gt; ListDlqMessagesResponse:
        """
        Lists all messages currently inside the DLQ.

        :param cursor: Optional cursor to start listing DLQ messages from.
        :param count: The maximum number of DLQ messages to return.
            Default and max is `100`.
        :param filter: Filter to use.
        """
        params = prepare_list_dlq_messages_params(
            cursor=cursor,
            count=count,
            filter=filter,
        )

        response = await self._http.request(
            path="/v2/dlq",
            method="GET",
            params=params,
        )

        messages = [parse_dlq_message_response(r) for r in response["messages"]]

        return ListDlqMessagesResponse(
            cursor=response.get("cursor"),
            messages=messages,
        )

    async def delete(self, dlq_id: str) -&gt; None:
        """
        Deletes a message from the DLQ.

        :param dlq_id: The unique id within the DLQ to delete.
        """
        await self._http.request(
            path=f"/v2/dlq/{dlq_id}",
            method="DELETE",
            parse_response=False,
        )

    async def delete_many(self, dlq_ids: List[str]) -&gt; int:
        """
        Deletes multiple messages from the DLQ and
        returns how many of them are deleted.

        :param dlq_ids: The unique ids within the DLQ to delete.
        """
        body = json.dumps({"dlqIds": dlq_ids})

        response = await self._http.request(
            path="/v2/dlq",
            method="DELETE",
            headers={"Content-Type": "application/json"},
            body=body,
        )

        return response["deleted"]

</file>
<file name="qstash/asyncio/event.py">
from typing import Optional

from qstash.asyncio.http import AsyncHttpClient
from qstash.event import (
    EventFilter,
    ListEventsResponse,
    parse_events_response,
    prepare_list_events_request_params,
)


class AsyncEventApi:
    def __init__(self, http: AsyncHttpClient) -&gt; None:
        self._http = http

    async def list(
        self,
        *,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        filter: Optional[EventFilter] = None,
    ) -&gt; ListEventsResponse:
        """
        Lists all events that happened, such as message creation or delivery.

        :param cursor: Optional cursor to start listing events from.
        :param count: The maximum number of events to return.
            Default and max is `1000`.
        :param filter: Filter to use.
        """
        params = prepare_list_events_request_params(
            cursor=cursor,
            count=count,
            filter=filter,
        )

        response = await self._http.request(
            path="/v2/events",
            method="GET",
            params=params,
        )

        events = parse_events_response(response["events"])

        return ListEventsResponse(
            cursor=response.get("cursor"),
            events=events,
        )

</file>
<file name="qstash/asyncio/http.py">
import asyncio
from typing import Any, Dict, Literal, Optional, Union

import httpx

from qstash.http import (
    BASE_URL,
    DEFAULT_RETRY,
    NO_RETRY,
    HttpMethod,
    RetryConfig,
    raise_for_non_ok_status,
    DEFAULT_TIMEOUT,
)


class AsyncHttpClient:
    def __init__(
        self,
        token: str,
        retry: Optional[Union[Literal[False], RetryConfig]],
        base_url: Optional[str] = None,
    ) -&gt; None:
        self._token = f"Bearer {token}"

        if retry is None:
            self._retry = DEFAULT_RETRY
        elif retry is False:
            self._retry = NO_RETRY
        else:
            self._retry = retry

        self._client = httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
        )

        self._base_url = base_url.rstrip("/") if base_url else BASE_URL

    async def request(
        self,
        *,
        path: str,
        method: HttpMethod,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[str, bytes]] = None,
        params: Optional[Dict[str, str]] = None,
        parse_response: bool = True,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -&gt; Any:
        base_url = base_url or self._base_url
        token = token or self._token

        url = base_url + path
        headers = {"Authorization": token, **(headers or {})}

        max_attempts = 1 + max(0, self._retry["retries"])
        last_error = None
        response = None
        for attempt in range(max_attempts):
            try:
                response = await self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    content=body,
                )
                break  # Break the loop as soon as we receive a proper response
            except Exception as e:
                last_error = e
                backoff = self._retry["backoff"](attempt) / 1000
                await asyncio.sleep(backoff)

        if not response:
            # Can't be None at this point
            raise last_error  # type:ignore[misc]

        raise_for_non_ok_status(response)

        if parse_response:
            return response.json()

        return response.text

    async def stream(
        self,
        *,
        path: str,
        method: HttpMethod,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[str, bytes]] = None,
        params: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -&gt; httpx.Response:
        base_url = base_url or self._base_url
        token = token or self._token

        url = base_url + path
        headers = {"Authorization": token, **(headers or {})}

        max_attempts = 1 + max(0, self._retry["retries"])
        last_error = None
        response = None
        for attempt in range(max_attempts):
            try:
                request = self._client.build_request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    content=body,
                )
                response = await self._client.send(
                    request,
                    stream=True,
                )
                break  # Break the loop as soon as we receive a proper response
            except Exception as e:
                last_error = e
                backoff = self._retry["backoff"](attempt) / 1000
                await asyncio.sleep(backoff)

        if not response:
            # Can't be None at this point
            raise last_error  # type:ignore[misc]

        try:
            raise_for_non_ok_status(response)
        except Exception as e:
            await response.aclose()
            raise e

        return response

</file>
<file name="qstash/asyncio/message.py">
import json
from typing import Any, Dict, List, Optional, Union

from qstash.asyncio.http import AsyncHttpClient
from qstash.http import HttpMethod
from qstash.message import (
    ApiT,
    BatchJsonRequest,
    BatchRequest,
    BatchResponse,
    BatchUrlGroupResponse,
    EnqueueResponse,
    EnqueueUrlGroupResponse,
    Message,
    PublishResponse,
    PublishUrlGroupResponse,
    convert_to_batch_messages,
    get_destination,
    parse_batch_response,
    parse_enqueue_response,
    parse_message_response,
    parse_publish_response,
    prepare_batch_message_body,
    prepare_headers,
)


class AsyncMessageApi:
    def __init__(self, http: AsyncHttpClient):
        self._http = http

    async def publish(
        self,
        *,
        url: Optional[str] = None,
        url_group: Optional[str] = None,
        api: Optional[ApiT] = None,
        body: Optional[Union[str, bytes]] = None,
        content_type: Optional[str] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        delay: Optional[Union[str, int]] = None,
        not_before: Optional[int] = None,
        deduplication_id: Optional[str] = None,
        content_based_deduplication: Optional[bool] = None,
        timeout: Optional[Union[str, int]] = None,
    ) -&gt; Union[PublishResponse, List[PublishUrlGroupResponse]]:
        """
        Publishes a message to QStash.

        If the destination is a `url` or an `api`, `PublishResponse`
        is returned.

        If the destination is a `url_group`, then a list of
        `PublishUrlGroupResponse`s are returned, one for each url
        in the url group.

        :param url: Url to send the message to.
        :param url_group: Url group to send the message to.
        :param api: Api to send the message to.
        :param body: The raw request message body passed to the destination as is.
        :param content_type: MIME type of the message.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param delay: Delay the message delivery. The format for the delay string is a
            number followed by duration abbreviation, like `10s`. Available durations
            are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
            it is also possible to specify the delay as an integer, which will be
            interpreted as delay in seconds.
        :param not_before: Delay the message until a certain time in the future.
            The format is a unix timestamp in seconds, based on the UTC timezone.
        :param deduplication_id: Id to use while deduplicating messages.
        :param content_based_deduplication: Automatically deduplicate messages based on
            their content.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        """
        headers = headers or {}
        destination = get_destination(
            url=url,
            url_group=url_group,
            api=api,
            headers=headers,
        )

        req_headers = prepare_headers(
            content_type=content_type,
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=delay,
            not_before=not_before,
            deduplication_id=deduplication_id,
            content_based_deduplication=content_based_deduplication,
            timeout=timeout,
        )

        response = await self._http.request(
            path=f"/v2/publish/{destination}",
            method="POST",
            headers=req_headers,
            body=body,
        )

        return parse_publish_response(response)

    async def publish_json(
        self,
        *,
        url: Optional[str] = None,
        url_group: Optional[str] = None,
        api: Optional[ApiT] = None,
        body: Optional[Any] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        delay: Optional[Union[str, int]] = None,
        not_before: Optional[int] = None,
        deduplication_id: Optional[str] = None,
        content_based_deduplication: Optional[bool] = None,
        timeout: Optional[Union[str, int]] = None,
    ) -&gt; Union[PublishResponse, List[PublishUrlGroupResponse]]:
        """
        Publish a message to QStash, automatically serializing the
        body as JSON string, and setting content type to `application/json`.

        If the destination is a `url` or an `api`, `PublishResponse`
        is returned.

        If the destination is a `url_group`, then a list of
        `PublishUrlGroupResponse`s are returned, one for each url
        in the url group.

        :param url: Url to send the message to.
        :param url_group: Url group to send the message to.
        :param api: Api to send the message to.
        :param body: The request message body passed to the destination after being
            serialized as JSON string.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param delay: Delay the message delivery. The format for the delay string is a
            number followed by duration abbreviation, like `10s`. Available durations
            are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
            it is also possible to specify the delay as an integer, which will be
            interpreted as delay in seconds.
        :param not_before: Delay the message until a certain time in the future.
            The format is a unix timestamp in seconds, based on the UTC timezone.
        :param deduplication_id: Id to use while deduplicating messages.
        :param content_based_deduplication: Automatically deduplicate messages based on
            their content.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        """
        return await self.publish(
            url=url,
            url_group=url_group,
            api=api,
            body=json.dumps(body),
            content_type="application/json",
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=delay,
            not_before=not_before,
            deduplication_id=deduplication_id,
            content_based_deduplication=content_based_deduplication,
            timeout=timeout,
        )

    async def enqueue(
        self,
        *,
        queue: str,
        url: Optional[str] = None,
        url_group: Optional[str] = None,
        api: Optional[ApiT] = None,
        body: Optional[Union[str, bytes]] = None,
        content_type: Optional[str] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        deduplication_id: Optional[str] = None,
        content_based_deduplication: Optional[bool] = None,
        timeout: Optional[Union[str, int]] = None,
    ) -&gt; Union[EnqueueResponse, List[EnqueueUrlGroupResponse]]:
        """
        Enqueues a message, after creating the queue if it does
        not exist.

        If the destination is a `url` or an `api`, `EnqueueResponse`
        is returned.

        If the destination is a `url_group`, then a list of
        `EnqueueUrlGroupResponse`s are returned, one for each url
        in the url group.

        :param queue: The name of the queue.
        :param url: Url to send the message to.
        :param url_group: Url group to send the message to.
        :param api: Api to send the message to.
        :param body: The raw request message body passed to the destination as is.
        :param content_type: MIME type of the message.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param deduplication_id: Id to use while deduplicating messages.
        :param content_based_deduplication: Automatically deduplicate messages based on
            their content.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        """
        headers = headers or {}
        destination = get_destination(
            url=url,
            url_group=url_group,
            api=api,
            headers=headers,
        )

        req_headers = prepare_headers(
            content_type=content_type,
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=None,
            not_before=None,
            deduplication_id=deduplication_id,
            content_based_deduplication=content_based_deduplication,
            timeout=timeout,
        )

        response = await self._http.request(
            path=f"/v2/enqueue/{queue}/{destination}",
            method="POST",
            headers=req_headers,
            body=body,
        )

        return parse_enqueue_response(response)

    async def enqueue_json(
        self,
        *,
        queue: str,
        url: Optional[str] = None,
        url_group: Optional[str] = None,
        api: Optional[ApiT] = None,
        body: Optional[Any] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        deduplication_id: Optional[str] = None,
        content_based_deduplication: Optional[bool] = None,
        timeout: Optional[Union[str, int]] = None,
    ) -&gt; Union[EnqueueResponse, List[EnqueueUrlGroupResponse]]:
        """
        Enqueues a message, after creating the queue if it does
        not exist. It automatically serializes the body as JSON string,
        and setting content type to `application/json`.

        If the destination is a `url` or an `api`, `EnqueueResponse`
        is returned.

        If the destination is a `url_group`, then a list of
        `EnqueueUrlGroupResponse`s are returned, one for each url
        in the url group.

        :param queue: The name of the queue.
        :param url: Url to send the message to.
        :param url_group: Url group to send the message to.
        :param api: Api to send the message to.
        :param body: The request message body passed to the destination after being
            serialized as JSON string.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param deduplication_id: Id to use while deduplicating messages.
        :param content_based_deduplication: Automatically deduplicate messages based on
            their content.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        """
        return await self.enqueue(
            queue=queue,
            url=url,
            url_group=url_group,
            api=api,
            body=json.dumps(body),
            content_type="application/json",
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            deduplication_id=deduplication_id,
            content_based_deduplication=content_based_deduplication,
            timeout=timeout,
        )

    async def batch(
        self, messages: List[BatchRequest]
    ) -&gt; List[Union[BatchResponse, List[BatchUrlGroupResponse]]]:
        """
        Publishes or enqueues multiple messages in a single request.

        Returns a list of publish or enqueue responses, one for each
        message in the batch.

        If the message in the batch is sent to a url or an API,
        the corresponding item in the response is `BatchResponse`.

        If the message in the batch is sent to a url group,
        the corresponding item in the response is list of
        `BatchUrlGroupResponse`s, one for each url in the url group.
        """
        body = prepare_batch_message_body(messages)

        response = await self._http.request(
            path="/v2/batch",
            body=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        return parse_batch_response(response)

    async def batch_json(
        self, messages: List[BatchJsonRequest]
    ) -&gt; List[Union[BatchResponse, List[BatchUrlGroupResponse]]]:
        """
        Publishes or enqueues multiple messages in a single request,
        automatically serializing the message bodies as JSON strings,
        and setting content type to `application/json`.

        Returns a list of publish or enqueue responses, one for each
        message in the batch.

        If the message in the batch is sent to a url or an API,
        the corresponding item in the response is `BatchResponse`.

        If the message in the batch is sent to a url group,
        the corresponding item in the response is list of
        `BatchUrlGroupResponse`s, one for each url in the url group.
        """
        batch_messages = convert_to_batch_messages(messages)
        return await self.batch(batch_messages)

    async def get(self, message_id: str) -&gt; Message:
        """
        Gets the message by its id.
        """
        response = await self._http.request(
            path=f"/v2/messages/{message_id}",
            method="GET",
        )

        return parse_message_response(response)

    async def cancel(self, message_id: str) -&gt; None:
        """
        Cancels delivery of an existing message.

        Cancelling a message will remove it from QStash and stop it from being
        delivered in the future. If a message is in flight to your API,
        it might be too late to cancel.
        """
        await self._http.request(
            path=f"/v2/messages/{message_id}",
            method="DELETE",
            parse_response=False,
        )

    async def cancel_many(self, message_ids: List[str]) -&gt; int:
        """
        Cancels delivery of existing messages.

        Cancelling a message will remove it from QStash and stop it from being
        delivered in the future. If a message is in flight to your API,
        it might be too late to cancel.

        Returns how many of the messages are cancelled.
        """
        body = json.dumps({"messageIds": message_ids})

        response = await self._http.request(
            path="/v2/messages",
            method="DELETE",
            headers={"Content-Type": "application/json"},
            body=body,
        )

        return response["cancelled"]

    async def cancel_all(self):
        """
        Cancels delivery of all the existing messages.

        Cancelling a message will remove it from QStash and stop it from being
        delivered in the future. If a message is in flight to your API,
        it might be too late to cancel.

        Returns how many messages are cancelled.
        """
        response = await self._http.request(
            path="/v2/messages",
            method="DELETE",
        )

        return response["cancelled"]

</file>
<file name="qstash/asyncio/queue.py">
from typing import List

from qstash.asyncio.http import AsyncHttpClient
from qstash.queue import Queue, parse_queue_response, prepare_upsert_body


class AsyncQueueApi:
    def __init__(self, http: AsyncHttpClient) -&gt; None:
        self._http = http

    async def upsert(
        self,
        queue: str,
        *,
        parallelism: int = 1,
        paused: bool = False,
    ) -&gt; None:
        """
        Updates or creates a queue.

        :param queue: The name of the queue.
        :param parallelism: The number of parallel consumers consuming from the queue.
        :param paused: Whether to pause the queue or not. A paused queue will not
            deliver new messages until it is resumed.
        """
        body = prepare_upsert_body(queue, parallelism, paused)

        await self._http.request(
            path="/v2/queues",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=body,
            parse_response=False,
        )

    async def get(self, queue: str) -&gt; Queue:
        """
        Gets the queue by its name.
        """
        response = await self._http.request(
            path=f"/v2/queues/{queue}",
            method="GET",
        )

        return parse_queue_response(response)

    async def list(self) -&gt; List[Queue]:
        """
        Lists all the queues.
        """
        response = await self._http.request(
            path="/v2/queues",
            method="GET",
        )

        return [parse_queue_response(r) for r in response]

    async def delete(self, queue: str) -&gt; None:
        """
        Deletes the queue.
        """
        await self._http.request(
            path=f"/v2/queues/{queue}",
            method="DELETE",
            parse_response=False,
        )

    async def pause(self, queue: str) -&gt; None:
        """
        Pauses the queue.

        A paused queue will not deliver messages until
        it is resumed.
        """
        await self._http.request(
            path=f"/v2/queues/{queue}/pause",
            method="POST",
            parse_response=False,
        )

    async def resume(self, queue: str) -&gt; None:
        """
        Resumes the queue.
        """
        await self._http.request(
            path=f"/v2/queues/{queue}/resume",
            method="POST",
            parse_response=False,
        )

</file>
<file name="qstash/asyncio/schedule.py">
import json
from typing import Any, Dict, List, Optional, Union

from qstash.asyncio.http import AsyncHttpClient
from qstash.http import HttpMethod
from qstash.schedule import (
    Schedule,
    parse_schedule_response,
    prepare_schedule_headers,
)


class AsyncScheduleApi:
    def __init__(self, http: AsyncHttpClient) -&gt; None:
        self._http = http

    async def create(
        self,
        *,
        destination: str,
        cron: str,
        body: Optional[Union[str, bytes]] = None,
        content_type: Optional[str] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        delay: Optional[Union[str, int]] = None,
        timeout: Optional[Union[str, int]] = None,
        schedule_id: Optional[str] = None,
    ) -&gt; str:
        """
        Creates a schedule to send messages periodically.

        Returns the created schedule id.

        :param destination: The destination url or url group.
        :param cron: The cron expression to use to schedule the messages.
        :param body: The raw request message body passed to the destination as is.
        :param content_type: MIME type of the message.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param delay: Delay the message delivery. The format for the delay string is a
            number followed by duration abbreviation, like `10s`. Available durations
            are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
            it is also possible to specify the delay as an integer, which will be
            interpreted as delay in seconds.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        :param schedule_id: Schedule id to use. Can be used to update the settings of an existing schedule.
        """
        req_headers = prepare_schedule_headers(
            cron=cron,
            content_type=content_type,
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=delay,
            timeout=timeout,
            schedule_id=schedule_id,
        )

        response = await self._http.request(
            path=f"/v2/schedules/{destination}",
            method="POST",
            headers=req_headers,
            body=body,
        )

        return response["scheduleId"]

    async def create_json(
        self,
        *,
        destination: str,
        cron: str,
        body: Optional[Any] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        delay: Optional[Union[str, int]] = None,
        timeout: Optional[Union[str, int]] = None,
        schedule_id: Optional[str] = None,
    ) -&gt; str:
        """
        Creates a schedule to send messages periodically, automatically serializing the
        body as JSON string, and setting content type to `application/json`.

        Returns the created schedule id.

        :param destination: The destination url or url group.
        :param cron: The cron expression to use to schedule the messages.
        :param body: The request message body passed to the destination after being
            serialized as JSON string.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param delay: Delay the message delivery. The format for the delay string is a
            number followed by duration abbreviation, like `10s`. Available durations
            are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
            it is also possible to specify the delay as an integer, which will be
            interpreted as delay in seconds.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        :param schedule_id: Schedule id to use. Can be used to update the settings of an existing schedule.
        """
        return await self.create(
            destination=destination,
            cron=cron,
            body=json.dumps(body),
            content_type="application/json",
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=delay,
            timeout=timeout,
            schedule_id=schedule_id,
        )

    async def get(self, schedule_id: str) -&gt; Schedule:
        """
        Gets the schedule by its id.
        """
        response = await self._http.request(
            path=f"/v2/schedules/{schedule_id}",
            method="GET",
        )

        return parse_schedule_response(response)

    async def list(self) -&gt; List[Schedule]:
        """
        Lists all the schedules.
        """
        response = await self._http.request(
            path="/v2/schedules",
            method="GET",
        )

        return [parse_schedule_response(r) for r in response]

    async def delete(self, schedule_id: str) -&gt; None:
        """
        Deletes the schedule.
        """
        await self._http.request(
            path=f"/v2/schedules/{schedule_id}",
            method="DELETE",
            parse_response=False,
        )

    async def pause(self, schedule_id: str) -&gt; None:
        """
        Pauses the schedule.

        A paused schedule will not produce new messages until
        it is resumed.
        """
        await self._http.request(
            path=f"/v2/schedules/{schedule_id}/pause",
            method="PATCH",
            parse_response=False,
        )

    async def resume(self, schedule_id: str) -&gt; None:
        """
        Resumes the schedule.
        """
        await self._http.request(
            path=f"/v2/schedules/{schedule_id}/resume",
            method="PATCH",
            parse_response=False,
        )

</file>
<file name="qstash/asyncio/signing_key.py">
from qstash.asyncio.http import AsyncHttpClient
from qstash.signing_key import SigningKey, parse_signing_key_response


class AsyncSigningKeyApi:
    def __init__(self, http: AsyncHttpClient) -&gt; None:
        self._http = http

    async def get(self) -&gt; SigningKey:
        """
        Gets the current and next signing keys.
        """
        response = await self._http.request(
            path="/v2/keys",
            method="GET",
        )

        return parse_signing_key_response(response)

    async def rotate(self) -&gt; SigningKey:
        """
        Rotates the current signing key and gets the new signing key.

        The next signing key becomes the current signing
        key, and a new signing key is assigned to the
        next signing key.
        """
        response = await self._http.request(
            path="/v2/keys/rotate",
            method="POST",
        )

        return parse_signing_key_response(response)

</file>
<file name="qstash/asyncio/url_group.py">
from typing import List

from qstash.asyncio.http import AsyncHttpClient
from qstash.url_group import (
    RemoveEndpointRequest,
    UpsertEndpointRequest,
    UrlGroup,
    parse_url_group_response,
    prepare_add_endpoints_body,
    prepare_remove_endpoints_body,
)


class AsyncUrlGroupApi:
    def __init__(self, http: AsyncHttpClient):
        self._http = http

    async def upsert_endpoints(
        self,
        url_group: str,
        endpoints: List[UpsertEndpointRequest],
    ) -&gt; None:
        """
        Add or updates an endpoint to a url group.

        If the url group or the endpoint does not exist, it will be created.
        If the endpoint exists, it will be updated.
        """
        body = prepare_add_endpoints_body(endpoints)

        await self._http.request(
            path=f"/v2/topics/{url_group}/endpoints",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=body,
            parse_response=False,
        )

    async def remove_endpoints(
        self,
        url_group: str,
        endpoints: List[RemoveEndpointRequest],
    ) -&gt; None:
        """
        Remove one or more endpoints from a url group.

        If all endpoints have been removed, the url group will be deleted.
        """
        body = prepare_remove_endpoints_body(endpoints)

        await self._http.request(
            path=f"/v2/topics/{url_group}/endpoints",
            method="DELETE",
            headers={"Content-Type": "application/json"},
            body=body,
            parse_response=False,
        )

    async def get(self, url_group: str) -&gt; UrlGroup:
        """
        Gets the url group by its name.
        """
        response = await self._http.request(
            path=f"/v2/topics/{url_group}",
            method="GET",
        )

        return parse_url_group_response(response)

    async def list(self) -&gt; List[UrlGroup]:
        """
        Lists all the url groups.
        """
        response = await self._http.request(
            path="/v2/topics",
            method="GET",
        )

        return [parse_url_group_response(r) for r in response]

    async def delete(self, url_group: str) -&gt; None:
        """
        Deletes the url group and all its endpoints.
        """
        await self._http.request(
            path=f"/v2/topics/{url_group}",
            method="DELETE",
            parse_response=False,
        )

</file>
<file name="qstash/chat.py">
import dataclasses
import json
import re
from types import TracebackType
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
    Type,
)

import httpx

from qstash.errors import QStashError
from qstash.http import HttpClient


@dataclasses.dataclass
class LlmProvider:
    name: str
    """Name of the LLM provider."""

    base_url: str
    """Base URL of the provider."""

    token: str
    """
    The token for the provider.
    
    The provided key will be passed to the
    endpoint as a bearer token.
    """


def openai(token: str) -&gt; LlmProvider:
    return LlmProvider(
        name="OpenAI",
        base_url="https://api.openai.com",
        token=token,
    )


UPSTASH_LLM_PROVIDER = LlmProvider(
    name="Upstash",
    base_url="",
    token="",
)


def upstash() -&gt; LlmProvider:
    return UPSTASH_LLM_PROVIDER


def custom(base_url: str, token: str) -&gt; LlmProvider:
    base_url = re.sub("/(v1/)?chat/completions$", "", base_url)
    return LlmProvider(
        name="custom",
        base_url=base_url,
        token=token,
    )


class ChatCompletionMessage(TypedDict):
    role: Literal["system", "assistant", "user"]
    """The role of the message author."""

    content: str
    """The content of the message."""


ChatModel = Union[
    Literal[
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ],
    str,
]


class ChatResponseFormat(TypedDict, total=False):
    type: Literal["text", "json_object"]
    """Must be one of `text` or `json_object`."""


@dataclasses.dataclass
class ChatTopLogprob:
    token: str
    """The token."""

    bytes: Optional[List[int]]
    """A list of integers representing the UTF-8 bytes representation of the token.

    Useful in instances where characters are represented by multiple tokens and
    their byte representations must be combined to generate the correct text
    representation. Can be `null` if there is no bytes representation for the token.
    """

    logprob: float
    """The log probability of this token, if it is within the top 20 most likely
    tokens.

    Otherwise, the value `-9999.0` is used to signify that the token is very
    unlikely.
    """


@dataclasses.dataclass
class ChatCompletionTokenLogprob:
    token: str
    """The token."""

    bytes: Optional[List[int]]
    """A list of integers representing the UTF-8 bytes representation of the token.

    Useful in instances where characters are represented by multiple tokens and
    their byte representations must be combined to generate the correct text
    representation. Can be `null` if there is no bytes representation for the token.
    """

    logprob: float
    """The log probability of this token, if it is within the top 20 most likely
    tokens.

    Otherwise, the value `-9999.0` is used to signify that the token is very
    unlikely.
    """

    top_logprobs: List[ChatTopLogprob]
    """List of the most likely tokens and their log probability, at this token
    position.

    In rare cases, there may be fewer than the number of requested `top_logprobs`
    returned.
    """


@dataclasses.dataclass
class ChatChoiceLogprobs:
    content: Optional[List[ChatCompletionTokenLogprob]]
    """A list of message content tokens with log probability information."""


@dataclasses.dataclass
class ChatChoiceMessage:
    role: Literal["system", "assistant", "user"]
    """The role of the message author."""

    content: str
    """The content of the message."""


@dataclasses.dataclass
class ChatChunkChoiceMessage:
    role: Optional[Literal["system", "assistant", "user"]]
    """The role of the message author."""

    content: Optional[str]
    """The content of the message."""


@dataclasses.dataclass
class ChatChoice:
    message: ChatChoiceMessage
    """A chat completion message generated by the model."""

    index: int
    """The index of the choice in the list of choices."""

    finish_reason: Literal["stop", "length"]
    """The reason the model stopped generating tokens."""

    logprobs: Optional[ChatChoiceLogprobs]
    """Log probability information for the choice."""


@dataclasses.dataclass
class ChatCompletionUsage:
    completion_tokens: int
    """Number of tokens in the generated completion."""

    prompt_tokens: int
    """Number of tokens in the prompt."""

    total_tokens: int
    """Total number of tokens used in the request (prompt + completion)."""


@dataclasses.dataclass
class ChatCompletion:
    id: str
    """A unique identifier for the chat completion."""

    choices: List[ChatChoice]
    """A list of chat completion choices.

    Can be more than one if `n` is greater than 1.
    """

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created."""

    model: str
    """The model used for the chat completion."""

    object: Literal["chat.completion"]
    """The object type, which is always `chat.completion`."""

    system_fingerprint: Optional[str]
    """This fingerprint represents the backend configuration that the model runs with.

    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    usage: ChatCompletionUsage
    """Usage statistics for the completion request."""


@dataclasses.dataclass
class ChatChunkChoice:
    delta: ChatChunkChoiceMessage
    """A chat completion delta generated by streamed model responses."""

    finish_reason: Literal["stop", "length"]
    """The reason the model stopped generating tokens."""

    index: int
    """The index of the choice in the list of choices."""

    logprobs: Optional[ChatChoiceLogprobs]
    """Log probability information for the choice."""


@dataclasses.dataclass
class ChatCompletionChunk:
    id: str
    """A unique identifier for the chat completion. Each chunk has the same ID."""

    choices: List[ChatChunkChoice]
    """A list of chat completion choices.

    Can contain more than one elements if `n` is greater than 1. Can also be empty
    for the last chunk.
    """

    created: int
    """The Unix timestamp (in seconds) of when the chat completion was created.

    Each chunk has the same timestamp.
    """

    model: str
    """The model to generate the completion."""

    object: Literal["chat.completion.chunk"]
    """The object type, which is always `chat.completion.chunk`."""

    system_fingerprint: Optional[str]
    """
    This fingerprint represents the backend configuration that the model runs with.
    Can be used in conjunction with the `seed` request parameter to understand when
    backend changes have been made that might impact determinism.
    """

    usage: Optional[ChatCompletionUsage]
    """
    Contains a null value except for the last chunk which contains
    the token usage statistics for the entire request.
    """


class ChatCompletionChunkStream:
    """
    An iterable that yields completion chunks.

    To not leak any resources, either
    - the chunks most be read to completion
    - close() must be called
    - context manager must be used
    """

    def __init__(self, response: httpx.Response) -&gt; None:
        self._response = response
        self._iterator = self._chunk_iterator()

    def close(self) -&gt; None:
        """
        Closes the underlying resources.

        No need to call it if the iterator is read to completion.
        """
        self._response.close()

    def __next__(self) -&gt; ChatCompletionChunk:
        return self._iterator.__next__()

    def __iter__(self) -&gt; Iterator[ChatCompletionChunk]:
        return self

    def __enter__(self) -&gt; "ChatCompletionChunkStream":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -&gt; None:
        self.close()

    def _chunk_iterator(self) -&gt; Iterator[ChatCompletionChunk]:
        it = self._data_iterator()
        for data in it:
            if data == b"[DONE]":
                break

            yield parse_chat_completion_chunk_response(json.loads(data))

        for _ in it:
            pass

    def _data_iterator(self) -&gt; Iterator[bytes]:
        pending = None

        for data in self._response.iter_bytes():
            if pending is not None:
                data = pending + data

            parts = data.split(b"\n\n")

            if parts and parts[-1] and data and parts[-1][-1] == data[-1]:
                pending = parts.pop()
            else:
                pending = None

            for part in parts:
                if part.startswith(b"data: "):
                    part = part[6:]
                    yield part

        if pending is not None:
            if pending.startswith(b"data: "):
                pending = pending[6:]
                yield pending


def prepare_chat_request_body(
    *,
    messages: List[ChatCompletionMessage],
    model: ChatModel,
    frequency_penalty: Optional[float],
    logit_bias: Optional[Dict[str, int]],
    logprobs: Optional[bool],
    top_logprobs: Optional[int],
    max_tokens: Optional[int],
    n: Optional[int],
    presence_penalty: Optional[float],
    response_format: Optional[ChatResponseFormat],
    seed: Optional[int],
    stop: Optional[Union[str, List[str]]],
    stream: Optional[bool],
    temperature: Optional[float],
    top_p: Optional[float],
) -&gt; str:
    for msg in messages:
        if "role" not in msg or "content" not in msg:
            raise QStashError("`role` and `content` must be provided in messages.")

    body: Dict[str, Any] = {
        "messages": messages,
        "model": model,
    }

    if frequency_penalty is not None:
        body["frequency_penalty"] = frequency_penalty

    if logit_bias is not None:
        body["logit_bias"] = logit_bias

    if logprobs is not None:
        body["logprobs"] = logprobs

    if top_logprobs is not None:
        body["top_logprobs"] = top_logprobs

    if max_tokens is not None:
        body["max_tokens"] = max_tokens

    if n is not None:
        body["n"] = n

    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty

    if response_format is not None:
        body["response_format"] = response_format

    if seed is not None:
        body["seed"] = seed

    if stop is not None:
        body["stop"] = stop

    if stream is not None:
        body["stream"] = stream

    if temperature is not None:
        body["temperature"] = temperature

    if top_p is not None:
        body["top_p"] = top_p

    return json.dumps(body)


def parse_chat_completion_top_logprobs(
    response: List[Dict[str, Any]],
) -&gt; List[ChatTopLogprob]:
    result = []

    for top_logprob in response:
        result.append(
            ChatTopLogprob(
                token=top_logprob["token"],
                bytes=top_logprob.get("bytes"),
                logprob=top_logprob["logprob"],
            )
        )

    return result


def parse_chat_completion_logprobs(
    response: Optional[Dict[str, Any]],
) -&gt; Optional[ChatChoiceLogprobs]:
    if response is None:
        return None

    if "content" not in response:
        return ChatChoiceLogprobs(content=None)

    content = []
    for token_logprob in response["content"]:
        content.append(
            ChatCompletionTokenLogprob(
                token=token_logprob["token"],
                bytes=token_logprob.get("bytes"),
                logprob=token_logprob["logprob"],
                top_logprobs=parse_chat_completion_top_logprobs(
                    token_logprob["top_logprobs"]
                ),
            )
        )

    return ChatChoiceLogprobs(content=content)


def parse_chat_completion_choices(
    response: List[Dict[str, Any]],
) -&gt; List[ChatChoice]:
    result = []

    for choice in response:
        result.append(
            ChatChoice(
                message=ChatChoiceMessage(
                    role=choice["message"]["role"],
                    content=choice["message"]["content"],
                ),
                finish_reason=choice["finish_reason"],
                index=choice["index"],
                logprobs=parse_chat_completion_logprobs(choice.get("logprobs")),
            )
        )

    return result


def parse_chat_completion_chunk_choices(
    response: List[Dict[str, Any]],
) -&gt; List[ChatChunkChoice]:
    result = []

    for choice in response:
        result.append(
            ChatChunkChoice(
                delta=ChatChunkChoiceMessage(
                    role=choice["delta"].get("role"),
                    content=choice["delta"].get("content"),
                ),
                finish_reason=choice["finish_reason"],
                index=choice["index"],
                logprobs=parse_chat_completion_logprobs(choice.get("logprobs")),
            )
        )

    return result


def parse_chat_completion_usage(
    response: Dict[str, Any],
) -&gt; ChatCompletionUsage:
    return ChatCompletionUsage(
        completion_tokens=response["completion_tokens"],
        prompt_tokens=response["prompt_tokens"],
        total_tokens=response["total_tokens"],
    )


def parse_chat_completion_response(response: Dict[str, Any]) -&gt; ChatCompletion:
    return ChatCompletion(
        id=response["id"],
        choices=parse_chat_completion_choices(response["choices"]),
        created=response["created"],
        model=response["model"],
        object=response["object"],
        system_fingerprint=response.get("system_fingerprint"),
        usage=parse_chat_completion_usage(response["usage"]),
    )


def parse_chat_completion_chunk_response(
    response: Dict[str, Any],
) -&gt; ChatCompletionChunk:
    if "usage" in response:
        usage = parse_chat_completion_usage(response["usage"])
    else:
        usage = None

    return ChatCompletionChunk(
        id=response["id"],
        choices=parse_chat_completion_chunk_choices(response["choices"]),
        created=response["created"],
        model=response["model"],
        object=response["object"],
        system_fingerprint=response.get("system_fingerprint"),
        usage=usage,
    )


def convert_to_chat_messages(
    user: str,
    system: Optional[str],
) -&gt; List[ChatCompletionMessage]:
    if system is None:
        return [
            {
                "role": "user",
                "content": user,
            },
        ]

    return [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": user,
        },
    ]


class ChatApi:
    def __init__(self, http: HttpClient) -&gt; None:
        self._http = http

    def create(
        self,
        *,
        messages: List[ChatCompletionMessage],
        model: ChatModel,
        provider: Optional[LlmProvider] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -&gt; Union[ChatCompletion, ChatCompletionChunkStream]:
        """
        Creates a model response for the given chat conversation.

        When `stream` is set to `True`, it returns an iterable
        that can be used to receive chat completion delta chunks
        one by one.

        Otherwise, response is returned in one go as a chat
        completion object.

        :param messages: One or more chat messages.
        :param model: Name of the model.
        :param frequency_penalty: Number between `-2.0` and `2.0`.
            Positive values penalize new tokens based on their existing
            frequency in the text so far, decreasing the model's likelihood
            to repeat the same line verbatim.
        :param provider: LLM provider for the chat completion request. By default,
            Upstash will be used.
        :param logit_bias: Modify the likelihood of specified tokens appearing
            in the completion. Accepts a dictionary that maps tokens (specified
            by their token ID in the tokenizer) to an associated bias value
            from `-100` to `100`. Mathematically, the bias is added to the
            logits generated by the model prior to sampling. The exact effect
            will vary per model, but values between `-1` and `1` should
            decrease or increase likelihood of selection; values like `-100` or
            `100` should result in a ban or exclusive selection of the
            relevant token.
        :param logprobs: Whether to return log probabilities of the output
            tokens or not. If true, returns the log probabilities of each
            output token returned in the content of message.
        :param top_logprobs: An integer between `0` and `20` specifying the
            number of most likely tokens to return at each token position,
            each with an associated log probability. logprobs must be set
            to true if this parameter is used.
        :param max_tokens: The maximum number of tokens that can be generated
            in the chat completion.
        :param n: How many chat completion choices to generate for each input
            message. Note that you will be charged based on the number of
            generated tokens across all of the choices. Keep `n` as `1` to
            minimize costs.
        :param presence_penalty: Number between `-2.0` and `2.0`. Positive
            values penalize new tokens based on whether they appear in the
            text so far, increasing the model's likelihood to talk about
            new topics.
        :param response_format: An object specifying the format that the
            model must output.
            Setting to `{ "type": "json_object" }` enables JSON mode,
            which guarantees the message the model generates is valid JSON.

            **Important**: when using JSON mode, you must also instruct the
            model to produce JSON yourself via a system or user message.
            Without this, the model may generate an unending stream of
            whitespace until the generation reaches the token limit, resulting
            in a long-running and seemingly "stuck" request. Also note that
            the message content may be partially cut off if
            `finish_reason="length"`, which indicates the generation exceeded
            `max_tokens` or the conversation exceeded the max context length.
        :param seed: If specified, our system will make a best effort to sample
            deterministically, such that repeated requests with the same seed
            and parameters should return the same result. Determinism is not
            guaranteed, and you should refer to the `system_fingerprint`
            response parameter to monitor changes in the backend.
        :param stop: Up to 4 sequences where the API will stop generating
            further tokens.
        :param stream: If set, partial message deltas will be sent. Tokens
            will be sent as data-only server-sent events as they become
            available.
        :param temperature: What sampling temperature to use, between `0`
            and `2`. Higher values like `0.8` will make the output more random,
            while lower values like `0.2` will make it more focused and
            deterministic.
            We generally recommend altering this or `top_p` but not both.
        :param top_p: An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the tokens
            with `top_p` probability mass. So `0.1` means only the tokens
            comprising the top `10%`` probability mass are considered.
            We generally recommend altering this or `temperature` but not both.
        """
        body = prepare_chat_request_body(
            messages=messages,
            model=model,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
        )

        base_url = None
        token = None
        path = "/llm/v1/chat/completions"

        if provider is not None and provider.name != UPSTASH_LLM_PROVIDER.name:
            base_url = provider.base_url
            token = f"Bearer {provider.token}"
            path = "/v1/chat/completions"

        if stream:
            stream_response = self._http.stream(
                path=path,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Connection": "keep-alive",
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                body=body,
                base_url=base_url,
                token=token,
            )

            return ChatCompletionChunkStream(stream_response)

        response = self._http.request(
            path=path,
            method="POST",
            headers={"Content-Type": "application/json"},
            body=body,
            base_url=base_url,
            token=token,
        )

        return parse_chat_completion_response(response)

    def prompt(
        self,
        *,
        user: str,
        system: Optional[str] = None,
        model: ChatModel,
        provider: Optional[LlmProvider] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[ChatResponseFormat] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -&gt; Union[ChatCompletion, ChatCompletionChunkStream]:
        """
        Creates a model response for the given user and optional
        system prompt. It is a utility method that converts
        the given user and system prompts to message history
        expected in the `create` method. It is only useful for
        single turn chat completions.

        When `stream` is set to `True`, it returns an iterable
        that can be used to receive chat completion delta chunks
        one by one.

        Otherwise, response is returned in one go as a chat
        completion object.

        :param user: User prompt.
        :param system: System prompt.
        :param model: Name of the model.
        :param provider: LLM provider for the chat completion request. By default,
            Upstash will be used.
        :param frequency_penalty: Number between `-2.0` and `2.0`.
            Positive values penalize new tokens based on their existing
            frequency in the text so far, decreasing the model's likelihood
            to repeat the same line verbatim.
        :param logit_bias: Modify the likelihood of specified tokens appearing
            in the completion. Accepts a dictionary that maps tokens (specified
            by their token ID in the tokenizer) to an associated bias value
            from `-100` to `100`. Mathematically, the bias is added to the
            logits generated by the model prior to sampling. The exact effect
            will vary per model, but values between `-1` and `1` should
            decrease or increase likelihood of selection; values like `-100` or
            `100` should result in a ban or exclusive selection of the
            relevant token.
        :param logprobs: Whether to return log probabilities of the output
            tokens or not. If true, returns the log probabilities of each
            output token returned in the content of message.
        :param top_logprobs: An integer between `0` and `20` specifying the
            number of most likely tokens to return at each token position,
            each with an associated log probability. logprobs must be set
            to true if this parameter is used.
        :param max_tokens: The maximum number of tokens that can be generated
            in the chat completion.
        :param n: How many chat completion choices to generate for each input
            message. Note that you will be charged based on the number of
            generated tokens across all of the choices. Keep `n` as `1` to
            minimize costs.
        :param presence_penalty: Number between `-2.0` and `2.0`. Positive
            values penalize new tokens based on whether they appear in the
            text so far, increasing the model's likelihood to talk about
            new topics.
        :param response_format: An object specifying the format that the
            model must output.
            Setting to `{ "type": "json_object" }` enables JSON mode,
            which guarantees the message the model generates is valid JSON.

            **Important**: when using JSON mode, you must also instruct the
            model to produce JSON yourself via a system or user message.
            Without this, the model may generate an unending stream of
            whitespace until the generation reaches the token limit, resulting
            in a long-running and seemingly "stuck" request. Also note that
            the message content may be partially cut off if
            `finish_reason="length"`, which indicates the generation exceeded
            `max_tokens` or the conversation exceeded the max context length.
        :param seed: If specified, our system will make a best effort to sample
            deterministically, such that repeated requests with the same seed
            and parameters should return the same result. Determinism is not
            guaranteed, and you should refer to the `system_fingerprint`
            response parameter to monitor changes in the backend.
        :param stop: Up to 4 sequences where the API will stop generating
            further tokens.
        :param stream: If set, partial message deltas will be sent. Tokens
            will be sent as data-only server-sent events as they become
            available.
        :param temperature: What sampling temperature to use, between `0`
            and `2`. Higher values like `0.8` will make the output more random,
            while lower values like `0.2` will make it more focused and
            deterministic.
            We generally recommend altering this or `top_p` but not both.
        :param top_p: An alternative to sampling with temperature, called
            nucleus sampling, where the model considers the results of the tokens
            with `top_p` probability mass. So `0.1` means only the tokens
            comprising the top `10%`` probability mass are considered.
            We generally recommend altering this or `temperature` but not both.
        """
        return self.create(
            messages=convert_to_chat_messages(user, system),
            model=model,
            provider=provider,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
        )

</file>
<file name="qstash/client.py">
from typing import Optional, Union, Literal

from qstash.chat import ChatApi
from qstash.dlq import DlqApi
from qstash.event import EventApi
from qstash.http import RetryConfig, HttpClient
from qstash.message import MessageApi
from qstash.queue import QueueApi
from qstash.schedule import ScheduleApi
from qstash.signing_key import SigningKeyApi
from qstash.url_group import UrlGroupApi


class QStash:
    """Synchronous SDK for the Upstash QStash."""

    def __init__(
        self,
        token: str,
        *,
        retry: Optional[Union[Literal[False], RetryConfig]] = None,
        base_url: Optional[str] = None,
    ) -&gt; None:
        """
        :param token: The authorization token from the Upstash console.
        :param retry: Configures how the client should retry requests.
        """
        self.http = HttpClient(
            token,
            retry,
            base_url,
        )
        self.message = MessageApi(self.http)
        """Message api."""

        self.url_group = UrlGroupApi(self.http)
        """Url group api."""

        self.queue = QueueApi(self.http)
        """Queue api."""

        self.schedule = ScheduleApi(self.http)
        """Schedule api."""

        self.signing_key = SigningKeyApi(self.http)
        """Signing key api."""

        self.event = EventApi(self.http)
        """Event api."""

        self.dlq = DlqApi(self.http)
        """Dlq (Dead Letter Queue) api."""

        self.chat = ChatApi(self.http)
        """Chat api."""

</file>
<file name="qstash/dlq.py">
import dataclasses
import json
from typing import Any, Dict, List, Optional, TypedDict

from qstash.http import HttpClient
from qstash.message import Message


@dataclasses.dataclass
class DlqMessage(Message):
    dlq_id: str
    """The unique id within the DLQ."""

    response_status: int
    """The HTTP status code of the last failed delivery attempt."""

    response_headers: Optional[Dict[str, List[str]]]
    """The response headers of the last failed delivery attempt."""

    response_body: Optional[str]
    """
    The response body of the last failed delivery attempt if it is
    composed of UTF-8 characters only, `None` otherwise.
    """

    response_body_base64: Optional[str]
    """
    The base64 encoded response body of the last failed delivery attempt
    if the response body contains non-UTF-8 characters, `None` otherwise.
    """


class DlqFilter(TypedDict, total=False):
    message_id: str
    """Filter DLQ entries by message id."""

    url: str
    """Filter DLQ entries by url."""

    url_group: str
    """Filter DLQ entries by url group name."""

    api: str
    """Filter DLQ entries by api name."""

    queue: str
    """Filter DLQ entries by queue name."""

    schedule_id: str
    """Filter DLQ entries by schedule id."""

    from_time: int
    """Filter DLQ entries by starting time, in milliseconds"""

    to_time: int
    """Filter DLQ entries by ending time, in milliseconds"""

    response_status: int
    """Filter DLQ entries by HTTP status of the response"""

    caller_ip: str
    """Filter DLQ entries by IP address of the publisher of the message"""


@dataclasses.dataclass
class ListDlqMessagesResponse:
    cursor: Optional[str]
    """
    A cursor which can be used in subsequent requests to paginate through
    all messages. If `None`, end of the DLQ messages are reached.
    """

    messages: List[DlqMessage]
    """List of DLQ messages."""


def parse_dlq_message_response(
    response: Dict[str, Any],
    dlq_id: str = "",
) -&gt; DlqMessage:
    return DlqMessage(
        message_id=response["messageId"],
        url=response["url"],
        url_group=response.get("topicName"),
        endpoint=response.get("endpointName"),
        api=response.get("api"),
        queue=response.get("queueName"),
        body=response.get("body"),
        body_base64=response.get("bodyBase64"),
        method=response["method"],
        headers=response.get("header"),
        max_retries=response["maxRetries"],
        not_before=response["notBefore"],
        created_at=response["createdAt"],
        callback=response.get("callback"),
        failure_callback=response.get("failureCallback"),
        schedule_id=response.get("scheduleId"),
        caller_ip=response.get("callerIP"),
        dlq_id=response.get("dlqId", dlq_id),
        response_status=response["responseStatus"],
        response_headers=response.get("responseHeader"),
        response_body=response.get("responseBody"),
        response_body_base64=response.get("responseBodyBase64"),
    )


def prepare_list_dlq_messages_params(
    *,
    cursor: Optional[str],
    count: Optional[int],
    filter: Optional[DlqFilter],
) -&gt; Dict[str, str]:
    params = {}

    if cursor is not None:
        params["cursor"] = cursor

    if count is not None:
        params["count"] = str(count)

    if filter is not None:
        if "message_id" in filter:
            params["messageId"] = filter["message_id"]

        if "url" in filter:
            params["url"] = filter["url"]

        if "url_group" in filter:
            params["topicName"] = filter["url_group"]

        if "api" in filter:
            params["api"] = filter["api"]

        if "queue" in filter:
            params["queueName"] = filter["queue"]

        if "schedule_id" in filter:
            params["scheduleId"] = filter["schedule_id"]

        if "from_time" in filter:
            params["fromDate"] = str(filter["from_time"])

        if "to_time" in filter:
            params["toDate"] = str(filter["to_time"])

        if "response_status" in filter:
            params["responseStatus"] = str(filter["response_status"])

        if "caller_ip" in filter:
            params["callerIp"] = filter["caller_ip"]

    return params


class DlqApi:
    def __init__(self, http: HttpClient) -&gt; None:
        self._http = http

    def get(self, dlq_id: str) -&gt; DlqMessage:
        """
        Gets a message from DLQ.

        :param dlq_id: The unique id within the DLQ to get.
        """
        response = self._http.request(
            path=f"/v2/dlq/{dlq_id}",
            method="GET",
        )

        return parse_dlq_message_response(response, dlq_id)

    def list(
        self,
        *,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        filter: Optional[DlqFilter] = None,
    ) -&gt; ListDlqMessagesResponse:
        """
        Lists all messages currently inside the DLQ.

        :param cursor: Optional cursor to start listing DLQ messages from.
        :param count: The maximum number of DLQ messages to return.
            Default and max is `100`.
        :param filter: Filter to use.
        """
        params = prepare_list_dlq_messages_params(
            cursor=cursor,
            count=count,
            filter=filter,
        )

        response = self._http.request(
            path="/v2/dlq",
            method="GET",
            params=params,
        )

        messages = [parse_dlq_message_response(r) for r in response["messages"]]

        return ListDlqMessagesResponse(
            cursor=response.get("cursor"),
            messages=messages,
        )

    def delete(self, dlq_id: str) -&gt; None:
        """
        Deletes a message from the DLQ.

        :param dlq_id: The unique id within the DLQ to delete.
        """
        self._http.request(
            path=f"/v2/dlq/{dlq_id}",
            method="DELETE",
            parse_response=False,
        )

    def delete_many(self, dlq_ids: List[str]) -&gt; int:
        """
        Deletes multiple messages from the DLQ and
        returns how many of them are deleted.

        :param dlq_ids: The unique ids within the DLQ to delete.
        """
        body = json.dumps({"dlqIds": dlq_ids})

        response = self._http.request(
            path="/v2/dlq",
            method="DELETE",
            headers={"Content-Type": "application/json"},
            body=body,
        )

        return response["deleted"]

</file>
<file name="qstash/errors.py">
from typing import Optional


class QStashError(Exception): ...


class SignatureError(QStashError):
    def __init__(self, *args: object) -&gt; None:
        super().__init__(*args)


class RateLimitExceededError(QStashError):
    def __init__(
        self, limit: Optional[str], remaining: Optional[str], reset: Optional[str]
    ):
        super().__init__(
            f"Exceeded burst rate limit: Limit: {limit}, remaining: {remaining}, reset: {reset}"
        )
        self.limit = limit
        self.remaining = remaining
        self.reset = reset


class DailyMessageLimitExceededError(QStashError):
    def __init__(
        self, limit: Optional[str], remaining: Optional[str], reset: Optional[str]
    ):
        super().__init__(
            f"Exceeded daily message limit: Limit: {limit}, remaining: {remaining}, reset: {reset}"
        )
        self.limit = limit
        self.remaining = remaining
        self.reset = reset


class ChatRateLimitExceededError(QStashError):
    def __init__(
        self,
        limit_requests: Optional[str],
        limit_tokens: Optional[str],
        remaining_requests: Optional[str],
        remaining_tokens: Optional[str],
        reset_requests: Optional[str],
        reset_tokens: Optional[str],
    ):
        super().__init__(
            f"Exceeded chat rate limit: "
            f"Request limit: {limit_requests}, remaining: {remaining_requests}, reset: {reset_requests}; "
            f"token limit: {limit_tokens}, remaining: {remaining_tokens}, reset: {reset_tokens}"
        )
        self.limit_requests = limit_requests
        self.limit_tokens = limit_tokens
        self.remaining_requests = remaining_requests
        self.remaining_tokens = remaining_tokens
        self.reset_requests = reset_requests
        self.reset_tokens = reset_tokens

</file>
<file name="qstash/event.py">
import dataclasses
import enum
from typing import Any, Dict, List, Optional, TypedDict

from qstash.http import HttpClient


class EventState(enum.Enum):
    """The state of the message."""

    CREATED = "CREATED"
    """The message has been accepted and stored in QStash"""

    ACTIVE = "ACTIVE"
    """The task is currently being processed by a worker."""

    RETRY = "RETRY"
    """The task has been scheduled to retry."""

    ERROR = "ERROR"
    """
    The execution threw an error and the task is waiting to be retried
    or failed.
    """

    DELIVERED = "DELIVERED"
    """The message was successfully delivered."""

    FAILED = "FAILED"
    """
    The task has failed too many times or encountered an error that it
    cannot recover from.
    """

    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    """The cancel request from the user is recorded."""

    CANCELED = "CANCELED"
    """The cancel request from the user is honored."""


@dataclasses.dataclass
class Event:
    time: int
    """Timestamp of this log entry, in milliseconds"""

    message_id: str
    """The associated message id."""

    state: EventState
    """The current state of the message at this point in time."""

    error: Optional[str]
    """An explanation what went wrong."""

    next_delivery_time: Optional[int]
    """The next scheduled timestamp of the message, milliseconds."""

    url: str
    """The destination url."""

    url_group: Optional[str]
    """The name of the url group if this message was sent through a url group."""

    endpoint: Optional[str]
    """The name of the endpoint if this message was sent through a url group."""

    api: Optional[str]
    """The name of the api if this message was sent to an api."""

    queue: Optional[str]
    """The name of the queue if this message is enqueued on a queue."""

    schedule_id: Optional[str]
    """The schedule id of the message if the message is triggered by a schedule."""

    headers: Optional[Dict[str, List[str]]]
    """Headers of the message"""

    body_base64: Optional[str]
    """The base64 encoded body of the message."""


class EventFilter(TypedDict, total=False):
    message_id: str
    """Filter events by message id."""

    state: EventState
    """Filter events by state."""

    url: str
    """Filter events by url."""

    url_group: str
    """Filter events by url group name."""

    api: str
    """Filter events by api name."""

    queue: str
    """Filter events by queue name."""

    schedule_id: str
    """Filter events by schedule id."""

    from_time: int
    """Filter events by starting time, in milliseconds"""

    to_time: int
    """Filter events by ending time, in milliseconds"""


@dataclasses.dataclass
class ListEventsResponse:
    cursor: Optional[str]
    """
    A cursor which can be used in subsequent requests to paginate through 
    all events. If `None`, end of the events are reached.
    """

    events: List[Event]
    """List of events."""


def prepare_list_events_request_params(
    *,
    cursor: Optional[str],
    count: Optional[int],
    filter: Optional[EventFilter],
) -&gt; Dict[str, str]:
    params = {}

    if cursor is not None:
        params["cursor"] = cursor

    if count is not None:
        params["count"] = str(count)

    if filter is not None:
        if "message_id" in filter:
            params["messageId"] = filter["message_id"]

        if "state" in filter:
            params["state"] = filter["state"].value

        if "url" in filter:
            params["url"] = filter["url"]

        if "url_group" in filter:
            params["topicName"] = filter["url_group"]

        if "api" in filter:
            params["api"] = filter["api"]

        if "queue" in filter:
            params["queueName"] = filter["queue"]

        if "schedule_id" in filter:
            params["scheduleId"] = filter["schedule_id"]

        if "from_time" in filter:
            params["fromDate"] = str(filter["from_time"])

        if "to_time" in filter:
            params["toDate"] = str(filter["to_time"])

    return params


def parse_events_response(response: List[Dict[str, Any]]) -&gt; List[Event]:
    events = []

    for event in response:
        events.append(
            Event(
                time=event["time"],
                message_id=event["messageId"],
                state=EventState(event["state"]),
                error=event.get("error"),
                next_delivery_time=event.get("nextDeliveryTime"),
                url=event["url"],
                url_group=event.get("topicName"),
                endpoint=event.get("endpointName"),
                api=event.get("api"),
                queue=event.get("queueName"),
                schedule_id=event.get("scheduleId"),
                headers=event.get("header"),
                body_base64=event.get("body"),
            )
        )

    return events


class EventApi:
    def __init__(self, http: HttpClient) -&gt; None:
        self._http = http

    def list(
        self,
        *,
        cursor: Optional[str] = None,
        count: Optional[int] = None,
        filter: Optional[EventFilter] = None,
    ) -&gt; ListEventsResponse:
        """
        Lists all events that happened, such as message creation or delivery.

        :param cursor: Optional cursor to start listing events from.
        :param count: The maximum number of events to return.
            Default and max is `1000`.
        :param filter: Filter to use.
        """
        params = prepare_list_events_request_params(
            cursor=cursor,
            count=count,
            filter=filter,
        )

        response = self._http.request(
            path="/v2/events",
            method="GET",
            params=params,
        )

        events = parse_events_response(response["events"])

        return ListEventsResponse(
            cursor=response.get("cursor"),
            events=events,
        )

</file>
<file name="qstash/http.py">
import math
import time
from typing import TypedDict, Callable, Optional, Union, Literal, Any, Dict

import httpx

from qstash.errors import (
    RateLimitExceededError,
    QStashError,
    ChatRateLimitExceededError,
    DailyMessageLimitExceededError,
)


class RetryConfig(TypedDict, total=False):
    retries: int
    """Maximum number of retries will be performed after the initial request fails."""

    backoff: Callable[[int], float]
    """A function that returns how many milliseconds to backoff before the given retry attempt."""


DEFAULT_TIMEOUT = httpx.Timeout(
    timeout=600.0,
    connect=5.0,
)

DEFAULT_RETRY = RetryConfig(
    retries=5,
    backoff=lambda retry_count: math.exp(1 + retry_count) * 50,
)

NO_RETRY = RetryConfig(
    retries=0,
    backoff=lambda _: 0,
)

BASE_URL = "https://qstash.upstash.io"

HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


def chat_rate_limit_error(headers: httpx.Headers) -&gt; ChatRateLimitExceededError:
    limit_requests = headers.get("x-ratelimit-limit-requests")
    limit_tokens = headers.get("x-ratelimit-limit-tokens")
    remaining_requests = headers.get("x-ratelimit-remaining-requests")
    remaining_tokens = headers.get("x-ratelimit-remaining-tokens")
    reset_requests = headers.get("x-ratelimit-reset-requests")
    reset_tokens = headers.get("x-ratelimit-reset-tokens")
    return ChatRateLimitExceededError(
        limit_requests=limit_requests,
        limit_tokens=limit_tokens,
        remaining_requests=remaining_requests,
        remaining_tokens=remaining_tokens,
        reset_requests=reset_requests,
        reset_tokens=reset_tokens,
    )


def daily_message_limit_error(headers: httpx.Headers) -&gt; DailyMessageLimitExceededError:
    limit = headers.get("RateLimit-Limit")
    remaining = headers.get("RateLimit-Remaining")
    reset = headers.get("RateLimit-Reset")
    return DailyMessageLimitExceededError(
        limit=limit,
        remaining=remaining,
        reset=reset,
    )


def burst_rate_limit_error(headers: httpx.Headers) -&gt; RateLimitExceededError:
    limit = headers.get("Burst-RateLimit-Limit")
    remaining = headers.get("Burst-RateLimit-Remaining")
    reset = headers.get("Burst-RateLimit-Reset")
    return RateLimitExceededError(
        limit=limit,
        remaining=remaining,
        reset=reset,
    )


def raise_for_non_ok_status(response: httpx.Response) -&gt; None:
    if response.is_success:
        return

    if response.status_code == 429:
        headers = response.headers
        if "x-ratelimit-limit-requests" in headers:
            raise chat_rate_limit_error(headers)
        elif "RateLimit-Limit" in headers:
            raise daily_message_limit_error(headers)
        else:
            raise burst_rate_limit_error(headers)

    raise QStashError(
        f"Request failed with status: {response.status_code}, body: {response.text}"
    )


class HttpClient:
    def __init__(
        self,
        token: str,
        retry: Optional[Union[Literal[False], RetryConfig]],
        base_url: Optional[str] = None,
    ) -&gt; None:
        self._token = f"Bearer {token}"

        if retry is None:
            self._retry = DEFAULT_RETRY
        elif retry is False:
            self._retry = NO_RETRY
        else:
            self._retry = retry

        self._client = httpx.Client(
            timeout=DEFAULT_TIMEOUT,
        )

        self._base_url = base_url.rstrip("/") if base_url else BASE_URL

    def request(
        self,
        *,
        path: str,
        method: HttpMethod,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[str, bytes]] = None,
        params: Optional[Dict[str, str]] = None,
        parse_response: bool = True,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -&gt; Any:
        base_url = base_url or self._base_url
        token = token or self._token

        url = base_url + path
        headers = {"Authorization": token, **(headers or {})}

        max_attempts = 1 + max(0, self._retry["retries"])
        last_error = None
        response = None
        for attempt in range(max_attempts):
            try:
                response = self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    content=body,
                )
                break  # Break the loop as soon as we receive a proper response
            except Exception as e:
                last_error = e
                backoff = self._retry["backoff"](attempt) / 1000
                time.sleep(backoff)

        if not response:
            # Can't be None at this point
            raise last_error  # type:ignore[misc]

        raise_for_non_ok_status(response)

        if parse_response:
            return response.json()

        return response.text

    def stream(
        self,
        *,
        path: str,
        method: HttpMethod,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Union[str, bytes]] = None,
        params: Optional[Dict[str, str]] = None,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ) -&gt; httpx.Response:
        base_url = base_url or self._base_url
        token = token or self._token

        url = base_url + path
        headers = {"Authorization": token, **(headers or {})}

        max_attempts = 1 + max(0, self._retry["retries"])
        last_error = None
        response = None
        for attempt in range(max_attempts):
            try:
                request = self._client.build_request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    content=body,
                )
                response = self._client.send(
                    request,
                    stream=True,
                )
                break  # Break the loop as soon as we receive a proper response
            except Exception as e:
                last_error = e
                backoff = self._retry["backoff"](attempt) / 1000
                time.sleep(backoff)

        if not response:
            # Can't be None at this point
            raise last_error  # type:ignore[misc]

        try:
            raise_for_non_ok_status(response)
        except Exception as e:
            response.close()
            raise e

        return response

</file>
<file name="qstash/message.py">
import dataclasses
import json
from typing import (
    Union,
    Optional,
    Literal,
    Dict,
    Any,
    List,
    TypedDict,
)

from qstash.chat import LlmProvider, UPSTASH_LLM_PROVIDER
from qstash.errors import QStashError
from qstash.http import HttpClient, HttpMethod


class LlmApi(TypedDict):
    name: Literal["llm"]
    """The name of the API type."""

    provider: LlmProvider
    """
    The LLM provider for the API.
    """


ApiT = LlmApi  # In the future, this can be union of different API types


@dataclasses.dataclass
class PublishResponse:
    message_id: str
    """The unique id of the message."""

    deduplicated: bool
    """Whether the message is a duplicate and was not sent to the destination."""


@dataclasses.dataclass
class PublishUrlGroupResponse:
    message_id: str
    """The unique id of the message."""

    url: str
    """The url where the message was sent to."""

    deduplicated: bool
    """Whether the message is a duplicate and was not sent to the destination."""


@dataclasses.dataclass
class EnqueueResponse:
    message_id: str
    """The unique id of the message."""

    deduplicated: bool
    """Whether the message is a duplicate and was not sent to the destination."""


@dataclasses.dataclass
class EnqueueUrlGroupResponse:
    message_id: str
    """The unique id of the message."""

    url: str
    """The url where the message was sent to."""

    deduplicated: bool
    """Whether the message is a duplicate and was not sent to the destination."""


@dataclasses.dataclass
class BatchResponse:
    message_id: str
    """The unique id of the message."""

    deduplicated: bool
    """Whether the message is a duplicate and was not sent to the destination."""


@dataclasses.dataclass
class BatchUrlGroupResponse:
    message_id: str
    """The unique id of the message."""

    url: str
    """The url where the message was sent to."""

    deduplicated: bool
    """Whether the message is a duplicate and was not sent to the destination."""


class BatchRequest(TypedDict, total=False):
    queue: str
    """Name of the queue that message will be enqueued on."""

    url: str
    """Url to send the message to."""

    url_group: str
    """Url group to send the message to."""

    api: ApiT
    """Api to send the message to."""

    body: Union[str, bytes]
    """The raw request message body passed to the endpoints as is."""

    content_type: str
    """MIME type of the message."""

    method: HttpMethod
    """The HTTP method to use when sending a webhook to your API."""

    headers: Dict[str, str]
    """Headers to forward along with the message."""

    retries: int
    """
    How often should this message be retried in case the destination 
    API is not available.
    """

    callback: str
    """A callback url that will be called after each attempt."""

    failure_callback: str
    """
    A failure callback url that will be called when a delivery is failed, 
    that is when all the defined retries are exhausted.
    """

    delay: Union[str, int]
    """
    Delay the message delivery. 
    
    The format for the delay string is a
    number followed by duration abbreviation, like `10s`. Available durations
    are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
    it is also possible to specify the delay as an integer, which will be
    interpreted as delay in seconds.
    """

    not_before: int
    """
    Delay the message until a certain time in the future. 
    
    The format is a unix timestamp in seconds, based on the UTC timezone.
    """

    deduplication_id: str
    """Id to use while deduplicating messages."""

    content_based_deduplication: bool
    """Automatically deduplicate messages based on their content."""

    timeout: Union[str, int]
    """
    The HTTP timeout value to use while calling the destination URL.
    When a timeout is specified, it will be used instead of the maximum timeout
    value permitted by the QStash plan. It is useful in scenarios, where a message
    should be delivered with a shorter timeout.
    
    The format for the timeout string is a number followed by duration abbreviation, 
    like `10s`. Available durations are `s` (seconds), `m` (minutes), `h` (hours), 
    and `d` (days). As convenience, it is also possible to specify the timeout as 
    an integer, which will be interpreted as timeout in seconds.
    """


class BatchJsonRequest(TypedDict, total=False):
    queue: str
    """Name of the queue that message will be enqueued on."""

    url: str
    """Url to send the message to."""

    url_group: str
    """Url group to send the message to."""

    api: ApiT
    """Api to send the message to."""

    body: Any
    """
    The request body passed to the endpoints after being serialized to 
    JSON string.
    """

    method: HttpMethod
    """The HTTP method to use when sending a webhook to your API."""

    headers: Dict[str, str]
    """Headers to forward along with the message."""

    retries: int
    """
    How often should this message be retried in case the destination 
    API is not available.
    """

    callback: str
    """A callback url that will be called after each attempt."""

    failure_callback: str
    """
    A failure callback url that will be called when a delivery is failed, 
    that is when all the defined retries are exhausted.
    """

    delay: Union[str, int]
    """
    Delay the message delivery. 
    
    The format for the delay string is a
    number followed by duration abbreviation, like `10s`. Available durations
    are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
    it is also possible to specify the delay as an integer, which will be
    interpreted as delay in seconds.
    """

    not_before: int
    """
    Delay the message until a certain time in the future. 
    
    The format is a unix timestamp in seconds, based on the UTC timezone.
    """

    deduplication_id: str
    """Id to use while deduplicating messages."""

    content_based_deduplication: bool
    """Automatically deduplicate messages based on their content."""

    timeout: Union[str, int]
    """
    The HTTP timeout value to use while calling the destination URL.
    When a timeout is specified, it will be used instead of the maximum timeout
    value permitted by the QStash plan. It is useful in scenarios, where a message
    should be delivered with a shorter timeout.

    The format for the timeout string is a number followed by duration abbreviation, 
    like `10s`. Available durations are `s` (seconds), `m` (minutes), `h` (hours), 
    and `d` (days). As convenience, it is also possible to specify the timeout as 
    an integer, which will be interpreted as timeout in seconds.
    """

    provider: LlmProvider
    """
    LLM provider to use. 
    
    When specified, destination and headers will be
    set according to the LLM provider.
    """


@dataclasses.dataclass
class Message:
    message_id: str
    """The unique identifier of the message."""

    url: str
    """The url to which the message should be delivered."""

    url_group: Optional[str]
    """The url group name if this message was sent to a url group."""

    endpoint: Optional[str]
    """
    The endpoint name of the message if the endpoint is given a 
    name within the url group.
    """

    api: Optional[str]
    """The api name if this message was sent to an api."""

    queue: Optional[str]
    """The queue name if this message was enqueued to a queue."""

    body: Optional[str]
    """
    The body of the message if it is composed of UTF-8 characters only, 
    `None` otherwise.
    """

    body_base64: Optional[str]
    """
    The base64 encoded body if the body contains non-UTF-8 characters, 
    `None` otherwise.
    """

    method: HttpMethod
    """The HTTP method to use for the message."""

    headers: Optional[Dict[str, List[str]]]
    """The HTTP headers sent the endpoint."""

    max_retries: int
    """The number of retries that should be attempted in case of delivery failure."""

    not_before: int
    """The unix timestamp in milliseconds before which the message should not be delivered."""

    created_at: int
    """The unix timestamp in milliseconds when the message was created."""

    callback: Optional[str]
    """The url which is called each time the message is attempted to be delivered."""

    failure_callback: Optional[str]
    """The url which is called after the message is failed."""

    schedule_id: Optional[str]
    """The scheduleId of the message if the message is triggered by a schedule."""

    caller_ip: Optional[str]
    """IP address of the publisher of this message."""


def get_destination(
    *,
    url: Optional[str],
    url_group: Optional[str],
    api: Optional[ApiT],
    headers: Dict[str, str],
) -&gt; str:
    destination = None
    count = 0
    if url is not None:
        destination = url
        count += 1

    if url_group is not None:
        destination = url_group
        count += 1

    if api is not None:
        provider = api["provider"]
        if provider.name == UPSTASH_LLM_PROVIDER.name:
            destination = "api/llm"
        else:
            destination = provider.base_url + "/v1/chat/completions"
            headers["Authorization"] = f"Bearer {provider.token}"

        count += 1

    if count != 1:
        raise QStashError(
            "Only and only one of 'url', 'url_group', or 'api' must be provided."
        )

    # Can't be None at this point
    return destination  # type:ignore[return-value]


def prepare_headers(
    *,
    content_type: Optional[str],
    method: Optional[HttpMethod],
    headers: Optional[Dict[str, str]],
    retries: Optional[int],
    callback: Optional[str],
    failure_callback: Optional[str],
    delay: Optional[Union[str, int]],
    not_before: Optional[int],
    deduplication_id: Optional[str],
    content_based_deduplication: Optional[bool],
    timeout: Optional[Union[str, int]],
) -&gt; Dict[str, str]:
    h = {}

    if content_type is not None:
        h["Content-Type"] = content_type

    if method is not None:
        h["Upstash-Method"] = method

    if headers:
        for k, v in headers.items():
            if not k.lower().startswith("upstash-"):
                k = f"Upstash-Forward-{k}"

            h[k] = v

    if retries is not None:
        h["Upstash-Retries"] = str(retries)

    if callback is not None:
        h["Upstash-Callback"] = callback

    if failure_callback is not None:
        h["Upstash-Failure-Callback"] = failure_callback

    if delay is not None:
        if isinstance(delay, int):
            h["Upstash-Delay"] = f"{delay}s"
        else:
            h["Upstash-Delay"] = delay

    if not_before is not None:
        h["Upstash-Not-Before"] = str(not_before)

    if deduplication_id is not None:
        h["Upstash-Deduplication-Id"] = deduplication_id

    if content_based_deduplication is not None:
        h["Upstash-Content-Based-Deduplication"] = str(content_based_deduplication)

    if timeout is not None:
        if isinstance(timeout, int):
            h["Upstash-Timeout"] = f"{timeout}s"
        else:
            h["Upstash-Timeout"] = timeout

    return h


def parse_publish_response(
    response: Union[List[Dict[str, Any]], Dict[str, Any]],
) -&gt; Union[PublishResponse, List[PublishUrlGroupResponse]]:
    if isinstance(response, list):
        result = []
        for ug_resp in response:
            result.append(
                PublishUrlGroupResponse(
                    message_id=ug_resp["messageId"],
                    url=ug_resp["url"],
                    deduplicated=ug_resp.get("deduplicated", False),
                )
            )

        return result

    return PublishResponse(
        message_id=response["messageId"],
        deduplicated=response.get("deduplicated", False),
    )


def parse_enqueue_response(
    response: Union[List[Dict[str, Any]], Dict[str, Any]],
) -&gt; Union[EnqueueResponse, List[EnqueueUrlGroupResponse]]:
    if isinstance(response, list):
        result = []
        for ug_resp in response:
            result.append(
                EnqueueUrlGroupResponse(
                    message_id=ug_resp["messageId"],
                    url=ug_resp["url"],
                    deduplicated=ug_resp.get("deduplicated", False),
                )
            )

        return result

    return EnqueueResponse(
        message_id=response["messageId"],
        deduplicated=response.get("deduplicated", False),
    )


def prepare_batch_message_body(messages: List[BatchRequest]) -&gt; str:
    batch_messages = []

    for msg in messages:
        user_headers = msg.get("headers") or {}
        destination = get_destination(
            url=msg.get("url"),
            url_group=msg.get("url_group"),
            api=msg.get("api"),
            headers=user_headers,
        )

        headers = prepare_headers(
            content_type=msg.get("content_type"),
            method=msg.get("method"),
            headers=user_headers,
            retries=msg.get("retries"),
            callback=msg.get("callback"),
            failure_callback=msg.get("failure_callback"),
            delay=msg.get("delay"),
            not_before=msg.get("not_before"),
            deduplication_id=msg.get("deduplication_id"),
            content_based_deduplication=msg.get("content_based_deduplication"),
            timeout=msg.get("timeout"),
        )

        batch_messages.append(
            {
                "destination": destination,
                "headers": headers,
                "body": msg.get("body"),
                "queue": msg.get("queue"),
            }
        )

    return json.dumps(batch_messages)


def parse_batch_response(
    response: List[Union[List[Dict[str, Any]], Dict[str, Any]]],
) -&gt; List[Union[BatchResponse, List[BatchUrlGroupResponse]]]:
    result: List[Union[BatchResponse, List[BatchUrlGroupResponse]]] = []

    for resp in response:
        if isinstance(resp, list):
            ug_result = []
            for ug_resp in resp:
                ug_result.append(
                    BatchUrlGroupResponse(
                        message_id=ug_resp["messageId"],
                        url=ug_resp["url"],
                        deduplicated=ug_resp.get("deduplicated", False),
                    )
                )

            result.append(ug_result)
        else:
            result.append(
                BatchResponse(
                    message_id=resp["messageId"],
                    deduplicated=resp.get("deduplicated", False),
                )
            )

    return result


def convert_to_batch_messages(
    messages: List[BatchJsonRequest],
) -&gt; List[BatchRequest]:
    batch_messages = []

    for msg in messages:
        batch_msg: BatchRequest = {}
        if "queue" in msg:
            batch_msg["queue"] = msg["queue"]

        if "url" in msg:
            batch_msg["url"] = msg["url"]

        if "url_group" in msg:
            batch_msg["url_group"] = msg["url_group"]

        if "api" in msg:
            batch_msg["api"] = msg["api"]

        batch_msg["body"] = json.dumps(msg.get("body"))
        batch_msg["content_type"] = "application/json"

        if "method" in msg:
            batch_msg["method"] = msg["method"]

        if "headers" in msg:
            batch_msg["headers"] = msg["headers"]

        if "retries" in msg:
            batch_msg["retries"] = msg["retries"]

        if "callback" in msg:
            batch_msg["callback"] = msg["callback"]

        if "failure_callback" in msg:
            batch_msg["failure_callback"] = msg["failure_callback"]

        if "delay" in msg:
            batch_msg["delay"] = msg["delay"]

        if "not_before" in msg:
            batch_msg["not_before"] = msg["not_before"]

        if "deduplication_id" in msg:
            batch_msg["deduplication_id"] = msg["deduplication_id"]

        if "content_based_deduplication" in msg:
            batch_msg["content_based_deduplication"] = msg[
                "content_based_deduplication"
            ]

        if "timeout" in msg:
            batch_msg["timeout"] = msg["timeout"]

        batch_messages.append(batch_msg)

    return batch_messages


def parse_message_response(response: Dict[str, Any]) -&gt; Message:
    return Message(
        message_id=response["messageId"],
        url=response["url"],
        url_group=response.get("topicName"),
        endpoint=response.get("endpointName"),
        api=response.get("api"),
        queue=response.get("queueName"),
        body=response.get("body"),
        body_base64=response.get("bodyBase64"),
        method=response["method"],
        headers=response.get("header"),
        max_retries=response["maxRetries"],
        not_before=response["notBefore"],
        created_at=response["createdAt"],
        callback=response.get("callback"),
        failure_callback=response.get("failureCallback"),
        schedule_id=response.get("scheduleId"),
        caller_ip=response.get("callerIP"),
    )


class MessageApi:
    def __init__(self, http: HttpClient):
        self._http = http

    def publish(
        self,
        *,
        url: Optional[str] = None,
        url_group: Optional[str] = None,
        api: Optional[ApiT] = None,
        body: Optional[Union[str, bytes]] = None,
        content_type: Optional[str] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        delay: Optional[Union[str, int]] = None,
        not_before: Optional[int] = None,
        deduplication_id: Optional[str] = None,
        content_based_deduplication: Optional[bool] = None,
        timeout: Optional[Union[str, int]] = None,
    ) -&gt; Union[PublishResponse, List[PublishUrlGroupResponse]]:
        """
        Publishes a message to QStash.

        If the destination is a `url` or an `api`, `PublishResponse`
        is returned.

        If the destination is a `url_group`, then a list of
        `PublishUrlGroupResponse`s are returned, one for each url
        in the url group.

        :param url: Url to send the message to.
        :param url_group: Url group to send the message to.
        :param api: Api to send the message to.
        :param body: The raw request message body passed to the destination as is.
        :param content_type: MIME type of the message.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param delay: Delay the message delivery. The format for the delay string is a
            number followed by duration abbreviation, like `10s`. Available durations
            are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
            it is also possible to specify the delay as an integer, which will be
            interpreted as delay in seconds.
        :param not_before: Delay the message until a certain time in the future.
            The format is a unix timestamp in seconds, based on the UTC timezone.
        :param deduplication_id: Id to use while deduplicating messages.
        :param content_based_deduplication: Automatically deduplicate messages based on
            their content.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        """
        headers = headers or {}
        destination = get_destination(
            url=url,
            url_group=url_group,
            api=api,
            headers=headers,
        )

        req_headers = prepare_headers(
            content_type=content_type,
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=delay,
            not_before=not_before,
            deduplication_id=deduplication_id,
            content_based_deduplication=content_based_deduplication,
            timeout=timeout,
        )

        response = self._http.request(
            path=f"/v2/publish/{destination}",
            method="POST",
            headers=req_headers,
            body=body,
        )

        return parse_publish_response(response)

    def publish_json(
        self,
        *,
        url: Optional[str] = None,
        url_group: Optional[str] = None,
        api: Optional[ApiT] = None,
        body: Optional[Any] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        delay: Optional[Union[str, int]] = None,
        not_before: Optional[int] = None,
        deduplication_id: Optional[str] = None,
        content_based_deduplication: Optional[bool] = None,
        timeout: Optional[Union[str, int]] = None,
    ) -&gt; Union[PublishResponse, List[PublishUrlGroupResponse]]:
        """
        Publish a message to QStash, automatically serializing the
        body as JSON string, and setting content type to `application/json`.

        If the destination is a `url` or an `api`, `PublishResponse`
        is returned.

        If the destination is a `url_group`, then a list of
        `PublishUrlGroupResponse`s are returned, one for each url
        in the url group.

        :param url: Url to send the message to.
        :param url_group: Url group to send the message to.
        :param api: Api to send the message to.
        :param body: The request message body passed to the destination after being
            serialized as JSON string.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param delay: Delay the message delivery. The format for the delay string is a
            number followed by duration abbreviation, like `10s`. Available durations
            are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
            it is also possible to specify the delay as an integer, which will be
            interpreted as delay in seconds.
        :param not_before: Delay the message until a certain time in the future.
            The format is a unix timestamp in seconds, based on the UTC timezone.
        :param deduplication_id: Id to use while deduplicating messages.
        :param content_based_deduplication: Automatically deduplicate messages based on
            their content.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        """
        return self.publish(
            url=url,
            url_group=url_group,
            api=api,
            body=json.dumps(body),
            content_type="application/json",
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=delay,
            not_before=not_before,
            deduplication_id=deduplication_id,
            content_based_deduplication=content_based_deduplication,
            timeout=timeout,
        )

    def enqueue(
        self,
        *,
        queue: str,
        url: Optional[str] = None,
        url_group: Optional[str] = None,
        api: Optional[ApiT] = None,
        body: Optional[Union[str, bytes]] = None,
        content_type: Optional[str] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        deduplication_id: Optional[str] = None,
        content_based_deduplication: Optional[bool] = None,
        timeout: Optional[Union[str, int]] = None,
    ) -&gt; Union[EnqueueResponse, List[EnqueueUrlGroupResponse]]:
        """
        Enqueues a message, after creating the queue if it does
        not exist.

        If the destination is a `url` or an `api`, `EnqueueResponse`
        is returned.

        If the destination is a `url_group`, then a list of
        `EnqueueUrlGroupResponse`s are returned, one for each url
        in the url group.

        :param queue: The name of the queue.
        :param url: Url to send the message to.
        :param url_group: Url group to send the message to.
        :param api: Api to send the message to.
        :param body: The raw request message body passed to the destination as is.
        :param content_type: MIME type of the message.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param deduplication_id: Id to use while deduplicating messages.
        :param content_based_deduplication: Automatically deduplicate messages based on
            their content.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        """
        headers = headers or {}
        destination = get_destination(
            url=url,
            url_group=url_group,
            api=api,
            headers=headers,
        )

        req_headers = prepare_headers(
            content_type=content_type,
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=None,
            not_before=None,
            deduplication_id=deduplication_id,
            content_based_deduplication=content_based_deduplication,
            timeout=timeout,
        )

        response = self._http.request(
            path=f"/v2/enqueue/{queue}/{destination}",
            method="POST",
            headers=req_headers,
            body=body,
        )

        return parse_enqueue_response(response)

    def enqueue_json(
        self,
        *,
        queue: str,
        url: Optional[str] = None,
        url_group: Optional[str] = None,
        api: Optional[ApiT] = None,
        body: Optional[Any] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        deduplication_id: Optional[str] = None,
        content_based_deduplication: Optional[bool] = None,
        timeout: Optional[Union[str, int]] = None,
    ) -&gt; Union[EnqueueResponse, List[EnqueueUrlGroupResponse]]:
        """
        Enqueues a message, after creating the queue if it does
        not exist. It automatically serializes the body as JSON string,
        and setting content type to `application/json`.

        If the destination is a `url` or an `api`, `EnqueueResponse`
        is returned.

        If the destination is a `url_group`, then a list of
        `EnqueueUrlGroupResponse`s are returned, one for each url
        in the url group.

        :param queue: The name of the queue.
        :param url: Url to send the message to.
        :param url_group: Url group to send the message to.
        :param api: Api to send the message to.
        :param body: The request message body passed to the destination after being
            serialized as JSON string.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param deduplication_id: Id to use while deduplicating messages.
        :param content_based_deduplication: Automatically deduplicate messages based on
            their content.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        """
        return self.enqueue(
            queue=queue,
            url=url,
            url_group=url_group,
            api=api,
            body=json.dumps(body),
            content_type="application/json",
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            deduplication_id=deduplication_id,
            content_based_deduplication=content_based_deduplication,
            timeout=timeout,
        )

    def batch(
        self, messages: List[BatchRequest]
    ) -&gt; List[Union[BatchResponse, List[BatchUrlGroupResponse]]]:
        """
        Publishes or enqueues multiple messages in a single request.

        Returns a list of publish or enqueue responses, one for each
        message in the batch.

        If the message in the batch is sent to a url or an API,
        the corresponding item in the response is `BatchResponse`.

        If the message in the batch is sent to a url group,
        the corresponding item in the response is list of
        `BatchUrlGroupResponse`s, one for each url in the url group.
        """
        body = prepare_batch_message_body(messages)

        response = self._http.request(
            path="/v2/batch",
            body=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        return parse_batch_response(response)

    def batch_json(
        self, messages: List[BatchJsonRequest]
    ) -&gt; List[Union[BatchResponse, List[BatchUrlGroupResponse]]]:
        """
        Publishes or enqueues multiple messages in a single request,
        automatically serializing the message bodies as JSON strings,
        and setting content type to `application/json`.

        Returns a list of publish or enqueue responses, one for each
        message in the batch.

        If the message in the batch is sent to a url or an API,
        the corresponding item in the response is `BatchResponse`.

        If the message in the batch is sent to a url group,
        the corresponding item in the response is list of
        `BatchUrlGroupResponse`s, one for each url in the url group.
        """
        batch_messages = convert_to_batch_messages(messages)
        return self.batch(batch_messages)

    def get(self, message_id: str) -&gt; Message:
        """
        Gets the message by its id.
        """
        response = self._http.request(
            path=f"/v2/messages/{message_id}",
            method="GET",
        )

        return parse_message_response(response)

    def cancel(self, message_id: str) -&gt; None:
        """
        Cancels delivery of an existing message.

        Cancelling a message will remove it from QStash and stop it from being
        delivered in the future. If a message is in flight to your API,
        it might be too late to cancel.
        """
        self._http.request(
            path=f"/v2/messages/{message_id}",
            method="DELETE",
            parse_response=False,
        )

    def cancel_many(self, message_ids: List[str]) -&gt; int:
        """
        Cancels delivery of existing messages.

        Cancelling a message will remove it from QStash and stop it from being
        delivered in the future. If a message is in flight to your API,
        it might be too late to cancel.

        Returns how many of the messages are cancelled.
        """
        body = json.dumps({"messageIds": message_ids})

        response = self._http.request(
            path="/v2/messages",
            method="DELETE",
            headers={"Content-Type": "application/json"},
            body=body,
        )

        return response["cancelled"]

    def cancel_all(self):
        """
        Cancels delivery of all the existing messages.

        Cancelling a message will remove it from QStash and stop it from being
        delivered in the future. If a message is in flight to your API,
        it might be too late to cancel.

        Returns how many messages are cancelled.
        """
        response = self._http.request(
            path="/v2/messages",
            method="DELETE",
        )

        return response["cancelled"]

</file>
<file name="qstash/queue.py">
import dataclasses
import json
from typing import Any, Dict, List

from qstash.http import HttpClient


@dataclasses.dataclass
class Queue:
    name: str
    """The name of the queue."""

    parallelism: int
    """The number of parallel consumers consuming from the queue."""

    created_at: int
    """The creation time of the queue, in unix milliseconds."""

    updated_at: int
    """The last update time of the queue, in unix milliseconds."""

    lag: int
    """The number of unprocessed messages that exist in the queue."""

    paused: bool
    """Whether the queue is paused or not."""


def prepare_upsert_body(queue: str, parallelism: int, paused: bool) -&gt; str:
    return json.dumps(
        {
            "queueName": queue,
            "parallelism": parallelism,
            "paused": paused,
        }
    )


def parse_queue_response(response: Dict[str, Any]) -&gt; Queue:
    return Queue(
        name=response["name"],
        parallelism=response["parallelism"],
        created_at=response["createdAt"],
        updated_at=response["updatedAt"],
        lag=response["lag"],
        paused=response["paused"],
    )


class QueueApi:
    def __init__(self, http: HttpClient) -&gt; None:
        self._http = http

    def upsert(
        self,
        queue: str,
        *,
        parallelism: int = 1,
        paused: bool = False,
    ) -&gt; None:
        """
        Updates or creates a queue.

        :param queue: The name of the queue.
        :param parallelism: The number of parallel consumers consuming from the queue.
        :param paused: Whether to pause the queue or not. A paused queue will not
            deliver new messages until it is resumed.
        """
        body = prepare_upsert_body(queue, parallelism, paused)

        self._http.request(
            path="/v2/queues",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=body,
            parse_response=False,
        )

    def get(self, queue: str) -&gt; Queue:
        """
        Gets the queue by its name.
        """
        response = self._http.request(
            path=f"/v2/queues/{queue}",
            method="GET",
        )

        return parse_queue_response(response)

    def list(self) -&gt; List[Queue]:
        """
        Lists all the queues.
        """
        response = self._http.request(
            path="/v2/queues",
            method="GET",
        )

        return [parse_queue_response(r) for r in response]

    def delete(self, queue: str) -&gt; None:
        """
        Deletes the queue.
        """
        self._http.request(
            path=f"/v2/queues/{queue}",
            method="DELETE",
            parse_response=False,
        )

    def pause(self, queue: str) -&gt; None:
        """
        Pauses the queue.

        A paused queue will not deliver messages until
        it is resumed.
        """
        self._http.request(
            path=f"/v2/queues/{queue}/pause",
            method="POST",
            parse_response=False,
        )

    def resume(self, queue: str) -&gt; None:
        """
        Resumes the queue.
        """
        self._http.request(
            path=f"/v2/queues/{queue}/resume",
            method="POST",
            parse_response=False,
        )

</file>
<file name="qstash/receiver.py">
import base64
import hashlib
from typing import Optional

import jwt

from qstash.errors import SignatureError


def verify_with_key(
    key: str,
    *,
    signature: str,
    body: str,
    url: Optional[str] = None,
    clock_tolerance: int = 0,
) -&gt; None:
    try:
        decoded = jwt.decode(
            signature,
            key,
            algorithms=["HS256"],
            issuer="Upstash",
            options={
                "require": ["iss", "sub", "exp", "nbf"],
                "leeway": clock_tolerance,
            },
        )
    except jwt.ExpiredSignatureError:
        raise SignatureError("Signature has expired")
    except Exception as e:
        raise SignatureError(f"Error while decoding signature: {e}")

    if url is not None and decoded["sub"] != url:
        raise SignatureError(f"Invalid subject: {decoded['sub']}, want: {url}")

    body_hash = hashlib.sha256(body.encode()).digest()
    body_hash_b64 = base64.urlsafe_b64encode(body_hash).decode().rstrip("=")

    if decoded["body"].rstrip("=") != body_hash_b64:
        raise SignatureError(
            f"Invalid body hash: {decoded['body']}, want: {body_hash_b64}"
        )


class Receiver:
    """Receiver offers a simple way to verify the signature of a request."""

    def __init__(self, current_signing_key: str, next_signing_key: str) -&gt; None:
        """
        :param current_signing_key: The current signing key.
            Get it from `https://console.upstash.com/qstash
        :param next_signing_key: The next signing key.
            Get it from `https://console.upstash.com/qstash
        """
        self._current_signing_key = current_signing_key
        self._next_signing_key = next_signing_key

    def verify(
        self,
        *,
        signature: str,
        body: str,
        url: Optional[str] = None,
        clock_tolerance: int = 0,
    ) -&gt; None:
        """
        Verifies the signature of a request.

        Tries to verify the signature with the current signing key.
        If that fails, maybe because you have rotated the keys recently, it will
        try to verify the signature with the next signing key.

        If that fails, the signature is invalid and a `SignatureError` is thrown.

        :param signature: The signature from the `Upstash-Signature` header.
        :param body: The raw request body.
        :param url: Url of the endpoint where the request was sent to.
            When set to `None`, url is not check.
        :param clock_tolerance: Number of seconds to tolerate when checking
            `nbf` and `exp` claims, to deal with small clock differences
            among different servers.
        """
        try:
            verify_with_key(
                self._current_signing_key,
                signature=signature,
                body=body,
                url=url,
                clock_tolerance=clock_tolerance,
            )
        except SignatureError:
            verify_with_key(
                self._next_signing_key,
                signature=signature,
                body=body,
                url=url,
                clock_tolerance=clock_tolerance,
            )

</file>
<file name="qstash/schedule.py">
import dataclasses
import json
from typing import Any, Dict, List, Optional, Union

from qstash.http import HttpClient, HttpMethod


@dataclasses.dataclass
class Schedule:
    schedule_id: str
    """The id of the schedule."""

    destination: str
    """The destination url or url group."""

    cron: str
    """The cron expression used to schedule the messages."""

    created_at: int
    """The creation time of the schedule, in unix milliseconds."""

    body: Optional[str]
    """The body of the scheduled message if it is composed of UTF-8 characters only, 
    `None` otherwise.."""

    body_base64: Optional[str]
    """
    The base64 encoded body if the scheduled message body contains non-UTF-8 characters, 
    `None` otherwise.
    """

    method: HttpMethod
    """The HTTP method to use for the message."""

    headers: Optional[Dict[str, List[str]]]
    """The headers of the message."""

    retries: int
    """The number of retries that should be attempted in case of delivery failure."""

    callback: Optional[str]
    """The url which is called each time the message is attempted to be delivered."""

    failure_callback: Optional[str]
    """The url which is called after the message is failed."""

    delay: Optional[int]
    """The delay in seconds before the message is delivered."""

    caller_ip: Optional[str]
    """IP address of the creator of this schedule."""

    paused: bool
    """Whether the schedule is paused or not."""


def prepare_schedule_headers(
    *,
    cron: str,
    content_type: Optional[str],
    method: Optional[HttpMethod],
    headers: Optional[Dict[str, str]],
    retries: Optional[int],
    callback: Optional[str],
    failure_callback: Optional[str],
    delay: Optional[Union[str, int]],
    timeout: Optional[Union[str, int]],
    schedule_id: Optional[str],
) -&gt; Dict[str, str]:
    h = {
        "Upstash-Cron": cron,
    }

    if content_type is not None:
        h["Content-Type"] = content_type

    if method is not None:
        h["Upstash-Method"] = method

    if headers:
        for k, v in headers.items():
            if not k.lower().startswith("upstash-"):
                k = f"Upstash-Forward-{k}"

            h[k] = v

    if retries is not None:
        h["Upstash-Retries"] = str(retries)

    if callback is not None:
        h["Upstash-Callback"] = callback

    if failure_callback is not None:
        h["Upstash-Failure-Callback"] = failure_callback

    if delay is not None:
        if isinstance(delay, int):
            h["Upstash-Delay"] = f"{delay}s"
        else:
            h["Upstash-Delay"] = delay

    if timeout is not None:
        if isinstance(timeout, int):
            h["Upstash-Timeout"] = f"{timeout}s"
        else:
            h["Upstash-Timeout"] = timeout

    if schedule_id is not None:
        h["Upstash-Schedule-Id"] = schedule_id

    return h


def parse_schedule_response(response: Dict[str, Any]) -&gt; Schedule:
    return Schedule(
        schedule_id=response["scheduleId"],
        destination=response["destination"],
        cron=response["cron"],
        created_at=response["createdAt"],
        body=response.get("body"),
        body_base64=response.get("bodyBase64"),
        method=response["method"],
        headers=response.get("header"),
        retries=response["retries"],
        callback=response.get("callback"),
        failure_callback=response.get("failureCallback"),
        delay=response.get("delay"),
        caller_ip=response.get("callerIP"),
        paused=response.get("isPaused", False),
    )


class ScheduleApi:
    def __init__(self, http: HttpClient) -&gt; None:
        self._http = http

    def create(
        self,
        *,
        destination: str,
        cron: str,
        body: Optional[Union[str, bytes]] = None,
        content_type: Optional[str] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        delay: Optional[Union[str, int]] = None,
        timeout: Optional[Union[str, int]] = None,
        schedule_id: Optional[str] = None,
    ) -&gt; str:
        """
        Creates a schedule to send messages periodically.

        Returns the created schedule id.

        :param destination: The destination url or url group.
        :param cron: The cron expression to use to schedule the messages.
        :param body: The raw request message body passed to the destination as is.
        :param content_type: MIME type of the message.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param delay: Delay the message delivery. The format for the delay string is a
            number followed by duration abbreviation, like `10s`. Available durations
            are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
            it is also possible to specify the delay as an integer, which will be
            interpreted as delay in seconds.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        :param schedule_id: Schedule id to use. Can be used to update the settings of an existing schedule.
        """
        req_headers = prepare_schedule_headers(
            cron=cron,
            content_type=content_type,
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=delay,
            timeout=timeout,
            schedule_id=schedule_id,
        )

        response = self._http.request(
            path=f"/v2/schedules/{destination}",
            method="POST",
            headers=req_headers,
            body=body,
        )

        return response["scheduleId"]

    def create_json(
        self,
        *,
        destination: str,
        cron: str,
        body: Optional[Any] = None,
        method: Optional[HttpMethod] = None,
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None,
        callback: Optional[str] = None,
        failure_callback: Optional[str] = None,
        delay: Optional[Union[str, int]] = None,
        timeout: Optional[Union[str, int]] = None,
        schedule_id: Optional[str] = None,
    ) -&gt; str:
        """
        Creates a schedule to send messages periodically, automatically serializing the
        body as JSON string, and setting content type to `application/json`.

        Returns the created schedule id.

        :param destination: The destination url or url group.
        :param cron: The cron expression to use to schedule the messages.
        :param body: The request message body passed to the destination after being
            serialized as JSON string.
        :param method: The HTTP method to use when sending a webhook to your API.
        :param headers: Headers to forward along with the message.
        :param retries: How often should this message be retried in case the destination
            API is not available.
        :param callback: A callback url that will be called after each attempt.
        :param failure_callback: A failure callback url that will be called when a delivery
            is failed, that is when all the defined retries are exhausted.
        :param delay: Delay the message delivery. The format for the delay string is a
            number followed by duration abbreviation, like `10s`. Available durations
            are `s` (seconds), `m` (minutes), `h` (hours), and `d` (days). As convenience,
            it is also possible to specify the delay as an integer, which will be
            interpreted as delay in seconds.
        :param timeout: The HTTP timeout value to use while calling the destination URL.
            When a timeout is specified, it will be used instead of the maximum timeout
            value permitted by the QStash plan. It is useful in scenarios, where a message
            should be delivered with a shorter timeout.
        :param schedule_id: Schedule id to use. Can be used to update the settings of an existing schedule.
        """
        return self.create(
            destination=destination,
            cron=cron,
            body=json.dumps(body),
            content_type="application/json",
            method=method,
            headers=headers,
            retries=retries,
            callback=callback,
            failure_callback=failure_callback,
            delay=delay,
            timeout=timeout,
            schedule_id=schedule_id,
        )

    def get(self, schedule_id: str) -&gt; Schedule:
        """
        Gets the schedule by its id.
        """
        response = self._http.request(
            path=f"/v2/schedules/{schedule_id}",
            method="GET",
        )

        return parse_schedule_response(response)

    def list(self) -&gt; List[Schedule]:
        """
        Lists all the schedules.
        """
        response = self._http.request(
            path="/v2/schedules",
            method="GET",
        )

        return [parse_schedule_response(r) for r in response]

    def delete(self, schedule_id: str) -&gt; None:
        """
        Deletes the schedule.
        """
        self._http.request(
            path=f"/v2/schedules/{schedule_id}",
            method="DELETE",
            parse_response=False,
        )

    def pause(self, schedule_id: str) -&gt; None:
        """
        Pauses the schedule.

        A paused schedule will not produce new messages until
        it is resumed.
        """
        self._http.request(
            path=f"/v2/schedules/{schedule_id}/pause",
            method="PATCH",
            parse_response=False,
        )

    def resume(self, schedule_id: str) -&gt; None:
        """
        Resumes the schedule.
        """
        self._http.request(
            path=f"/v2/schedules/{schedule_id}/resume",
            method="PATCH",
            parse_response=False,
        )

</file>
<file name="qstash/signing_key.py">
import dataclasses
from typing import Any, Dict

from qstash.http import HttpClient


@dataclasses.dataclass
class SigningKey:
    current: str
    """The current signing key."""

    next: str
    """The next signing key."""


def parse_signing_key_response(response: Dict[str, Any]) -&gt; SigningKey:
    return SigningKey(
        current=response["current"],
        next=response["next"],
    )


class SigningKeyApi:
    def __init__(self, http: HttpClient) -&gt; None:
        self._http = http

    def get(self) -&gt; SigningKey:
        """
        Gets the current and next signing keys.
        """
        response = self._http.request(
            path="/v2/keys",
            method="GET",
        )

        return parse_signing_key_response(response)

    def rotate(self) -&gt; SigningKey:
        """
        Rotates the current signing key and gets the new signing key.

        The next signing key becomes the current signing
        key, and a new signing key is assigned to the
        next signing key.
        """
        response = self._http.request(
            path="/v2/keys/rotate",
            method="POST",
        )

        return parse_signing_key_response(response)

</file>
<file name="qstash/url_group.py">
import dataclasses
import json
from typing import Any, Dict, List, Optional, TypedDict

from qstash.errors import QStashError
from qstash.http import HttpClient


class UpsertEndpointRequest(TypedDict, total=False):
    url: str
    """The url of the endpoint"""

    name: str
    """The optional name of the endpoint"""


class RemoveEndpointRequest(TypedDict, total=False):
    url: str
    """The url of the endpoint"""

    name: str
    """The name of the endpoint"""


@dataclasses.dataclass
class Endpoint:
    url: str
    """The url of the endpoint"""

    name: Optional[str]
    """The name of the endpoint"""


@dataclasses.dataclass
class UrlGroup:
    name: str
    """The name of the url group."""

    created_at: int
    """The creation time of the url group, in unix milliseconds."""

    updated_at: int
    """The last update time of the url group, in unix milliseconds."""

    endpoints: List[Endpoint]
    """The list of endpoints."""


def prepare_add_endpoints_body(
    endpoints: List[UpsertEndpointRequest],
) -&gt; str:
    for e in endpoints:
        if "url" not in e:
            raise QStashError("`url` of the endpoint must be provided.")

    return json.dumps(
        {
            "endpoints": endpoints,
        }
    )


def prepare_remove_endpoints_body(
    endpoints: List[RemoveEndpointRequest],
) -&gt; str:
    for e in endpoints:
        if "url" not in e and "name" not in e:
            raise QStashError(
                "One of `url` or `name` of the endpoint must be provided."
            )

    return json.dumps(
        {
            "endpoints": endpoints,
        }
    )


def parse_url_group_response(response: Dict[str, Any]) -&gt; UrlGroup:
    endpoints = []
    for e in response["endpoints"]:
        endpoints.append(
            Endpoint(
                url=e["url"],
                name=e.get("name"),
            )
        )

    return UrlGroup(
        name=response["name"],
        created_at=response["createdAt"],
        updated_at=response["updatedAt"],
        endpoints=endpoints,
    )


class UrlGroupApi:
    def __init__(self, http: HttpClient) -&gt; None:
        self._http = http

    def upsert_endpoints(
        self,
        url_group: str,
        endpoints: List[UpsertEndpointRequest],
    ) -&gt; None:
        """
        Add or updates an endpoint to a url group.

        If the url group or the endpoint does not exist, it will be created.
        If the endpoint exists, it will be updated.
        """
        body = prepare_add_endpoints_body(endpoints)

        self._http.request(
            path=f"/v2/topics/{url_group}/endpoints",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=body,
            parse_response=False,
        )

    def remove_endpoints(
        self,
        url_group: str,
        endpoints: List[RemoveEndpointRequest],
    ) -&gt; None:
        """
        Remove one or more endpoints from a url group.

        If all endpoints have been removed, the url group will be deleted.
        """
        body = prepare_remove_endpoints_body(endpoints)

        self._http.request(
            path=f"/v2/topics/{url_group}/endpoints",
            method="DELETE",
            headers={"Content-Type": "application/json"},
            body=body,
            parse_response=False,
        )

    def get(self, url_group: str) -&gt; UrlGroup:
        """
        Gets the url group by its name.
        """
        response = self._http.request(
            path=f"/v2/topics/{url_group}",
            method="GET",
        )

        return parse_url_group_response(response)

    def list(self) -&gt; List[UrlGroup]:
        """
        Lists all the url groups.
        """
        response = self._http.request(
            path="/v2/topics",
            method="GET",
        )

        return [parse_url_group_response(r) for r in response]

    def delete(self, url_group: str) -&gt; None:
        """
        Deletes the url group and all its endpoints.
        """
        self._http.request(
            path=f"/v2/topics/{url_group}",
            method="DELETE",
            parse_response=False,
        )

</file>
<file name="tests/__init__.py">
import asyncio
import os
import time
from typing import Callable, Coroutine

import dotenv

QSTASH_TOKEN = os.environ.get(
    "QSTASH_TOKEN",
    dotenv.dotenv_values().get("QSTASH_TOKEN"),
)

QSTASH_CURRENT_SIGNING_KEY = os.environ.get(
    "QSTASH_CURRENT_SIGNING_KEY",
    dotenv.dotenv_values().get("QSTASH_CURRENT_SIGNING_KEY"),
)

QSTASH_NEXT_SIGNING_KEY = os.environ.get(
    "QSTASH_NEXT_SIGNING_KEY",
    dotenv.dotenv_values().get("QSTASH_NEXT_SIGNING_KEY"),
)

OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    dotenv.dotenv_values().get("OPENAI_API_KEY"),
)


def assert_eventually(
    assertion: Callable[[], None],
    initial_delay: float = 0,
    retry_delay: float = 0.5,
    timeout: float = 10.0,
) -&gt; None:
    if initial_delay &gt; 0:
        time.sleep(initial_delay)

    deadline = time.time() + timeout
    last_err = None

    while time.time() &lt; deadline:
        try:
            assertion()
            return
        except AssertionError as e:
            last_err = e
            time.sleep(retry_delay)

    if last_err is None:
        raise AssertionError("Couldn't run the assertion")

    raise last_err


async def assert_eventually_async(
    assertion: Callable[[], Coroutine[None, None, None]],
    initial_delay: float = 0,
    retry_delay: float = 0.5,
    timeout: float = 10.0,
) -&gt; None:
    if initial_delay &gt; 0:
        await asyncio.sleep(initial_delay)

    deadline = time.time() + timeout
    last_err = None

    while time.time() &lt; deadline:
        try:
            await assertion()
            return
        except AssertionError as e:
            last_err = e
            await asyncio.sleep(retry_delay)

    if last_err is None:
        raise AssertionError("Couldn't run the assertion")

    raise last_err

</file>
<file name="tests/asyncio/test_chat.py">
import pytest

from qstash import AsyncQStash
from qstash.asyncio.chat import AsyncChatCompletionChunkStream
from qstash.chat import ChatCompletion, upstash, openai
from tests import OPENAI_API_KEY


@pytest.mark.asyncio
async def test_chat_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.chat.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "just say hello"}],
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


@pytest.mark.asyncio
async def test_chat_streaming_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.chat.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "just say hello"}],
        stream=True,
    )

    assert isinstance(res, AsyncChatCompletionChunkStream)

    i = 0
    async for r in res:
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert r.choices[0].delta.content is not None

        i += 1


@pytest.mark.asyncio
async def test_prompt_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.chat.prompt(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        user="just say hello",
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


@pytest.mark.asyncio
async def test_prompt_streaming_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.chat.prompt(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        user="just say hello",
        stream=True,
    )

    assert isinstance(res, AsyncChatCompletionChunkStream)

    i = 0
    async for r in res:
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert r.choices[0].delta.content is not None

        i += 1


@pytest.mark.asyncio
async def test_chat_explicit_upstash_provider_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.chat.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "just say hello"}],
        provider=upstash(),
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


@pytest.mark.asyncio
async def test_chat_explicit_upstash_provider_streaming_async(
    async_client: AsyncQStash,
) -&gt; None:
    res = await async_client.chat.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "just say hello"}],
        provider=upstash(),
        stream=True,
    )

    assert isinstance(res, AsyncChatCompletionChunkStream)

    i = 0
    async for r in res:
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert r.choices[0].delta.content is not None

        i += 1


@pytest.mark.asyncio
async def test_prompt_explicit_upstash_provider_async(
    async_client: AsyncQStash,
) -&gt; None:
    res = await async_client.chat.prompt(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        user="just say hello",
        provider=upstash(),
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


@pytest.mark.asyncio
async def test_prompt_explicit_upstash_provider_streaming_async(
    async_client: AsyncQStash,
) -&gt; None:
    res = await async_client.chat.prompt(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        user="just say hello",
        provider=upstash(),
        stream=True,
    )

    assert isinstance(res, AsyncChatCompletionChunkStream)

    i = 0
    async for r in res:
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert r.choices[0].delta.content is not None

        i += 1


@pytest.mark.asyncio
async def test_chat_custom_provider_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.chat.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "just say hello"}],
        provider=openai(token=OPENAI_API_KEY),  # type:ignore[arg-type]
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


@pytest.mark.asyncio
async def test_chat_custom_provider_streaming_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.chat.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "just say hello"}],
        provider=openai(token=OPENAI_API_KEY),  # type:ignore[arg-type]
        stream=True,
    )

    assert isinstance(res, AsyncChatCompletionChunkStream)

    i = 0
    async for r in res:
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert (
                r.choices[0].delta.content is not None
                or r.choices[0].finish_reason is not None
            )

        i += 1


@pytest.mark.asyncio
async def test_prompt_custom_provider_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.chat.prompt(
        model="gpt-3.5-turbo",
        user="just say hello",
        provider=openai(token=OPENAI_API_KEY),  # type:ignore[arg-type]
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


@pytest.mark.asyncio
async def test_prompt_custom_provider_streaming_async(
    async_client: AsyncQStash,
) -&gt; None:
    res = await async_client.chat.prompt(
        model="gpt-3.5-turbo",
        user="just say hello",
        provider=openai(token=OPENAI_API_KEY),  # type:ignore[arg-type]
        stream=True,
    )

    assert isinstance(res, AsyncChatCompletionChunkStream)

    i = 0
    async for r in res:
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert (
                r.choices[0].delta.content is not None
                or r.choices[0].finish_reason is not None
            )

        i += 1

</file>
<file name="tests/asyncio/test_dlq.py">
import pytest

from qstash import AsyncQStash
from qstash.message import PublishResponse
from tests import assert_eventually_async


async def assert_failed_eventually_async(
    async_client: AsyncQStash, *msg_ids: str
) -&gt; None:
    async def assertion() -&gt; None:
        messages = (await async_client.dlq.list()).messages

        matched_messages = [msg for msg in messages if msg.message_id in msg_ids]
        assert len(matched_messages) == len(msg_ids)

        for msg in matched_messages:
            dlq_msg = await async_client.dlq.get(msg.dlq_id)
            assert dlq_msg.response_body == "404 Not Found"
            assert msg.response_body == "404 Not Found"

        if len(msg_ids) == 1:
            await async_client.dlq.delete(matched_messages[0].dlq_id)
        else:
            deleted = await async_client.dlq.delete_many(
                [m.dlq_id for m in matched_messages]
            )
            assert deleted == len(msg_ids)

        messages = (await async_client.dlq.list()).messages
        matched = any(True for msg in messages if msg.message_id in msg_ids)
        assert not matched

    await assert_eventually_async(
        assertion,
        initial_delay=2.0,
        retry_delay=1.0,
        timeout=10.0,
    )


@pytest.mark.asyncio
async def test_dlq_get_and_delete_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.message.publish_json(
        url="http://httpstat.us/404",
        retries=0,
    )

    assert isinstance(res, PublishResponse)

    await assert_failed_eventually_async(async_client, res.message_id)


@pytest.mark.asyncio
async def test_dlq_get_and_delete_many_async(async_client: AsyncQStash) -&gt; None:
    msg_ids = []
    for _ in range(5):
        res = await async_client.message.publish_json(
            url="http://httpstat.us/404",
            retries=0,
        )

        assert isinstance(res, PublishResponse)
        msg_ids.append(res.message_id)

    await assert_failed_eventually_async(async_client, *msg_ids)


@pytest.mark.asyncio
async def test_dlq_filter_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.message.publish_json(
        url="http://httpstat.us/404",
        retries=0,
    )

    assert isinstance(res, PublishResponse)

    async def assertion():
        messages = (
            await async_client.dlq.list(
                filter={"message_id": res.message_id},
                count=1,
            )
        ).messages

        assert len(messages) == 1
        assert messages[0].message_id == res.message_id

        await async_client.dlq.delete(messages[0].dlq_id)

    await assert_eventually_async(
        assertion,
        initial_delay=2.0,
        retry_delay=1.0,
        timeout=10.0,
    )

</file>
<file name="tests/asyncio/test_message.py">
from typing import Callable

import pytest

from qstash import AsyncQStash
from qstash.chat import upstash, openai
from qstash.errors import QStashError
from qstash.event import EventState
from qstash.message import (
    BatchJsonRequest,
    BatchRequest,
    BatchResponse,
    EnqueueResponse,
    PublishResponse,
)
from tests import assert_eventually_async, OPENAI_API_KEY


async def assert_delivered_eventually_async(
    async_client: AsyncQStash, msg_id: str
) -&gt; None:
    async def assertion() -&gt; None:
        events = (
            await async_client.event.list(
                filter={
                    "message_id": msg_id,
                    "state": EventState.DELIVERED,
                }
            )
        ).events

        assert len(events) == 1

    await assert_eventually_async(
        assertion,
        initial_delay=1.0,
        retry_delay=1.0,
        timeout=60.0,
    )


@pytest.mark.asyncio
async def test_publish_to_url_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.message.publish(
        body="test-body",
        url="https://httpstat.us/200",
        headers={
            "test-header": "test-value",
        },
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    await assert_delivered_eventually_async(async_client, res.message_id)


@pytest.mark.asyncio
async def test_publish_to_url_json_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.message.publish_json(
        body={"ex_key": "ex_value"},
        url="https://httpstat.us/200",
        headers={
            "test-header": "test-value",
        },
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    await assert_delivered_eventually_async(async_client, res.message_id)


@pytest.mark.asyncio
async def test_disallow_multiple_destinations_async(async_client: AsyncQStash) -&gt; None:
    with pytest.raises(QStashError):
        await async_client.message.publish_json(
            url="https://httpstat.us/200",
            url_group="test-url-group",
        )

    with pytest.raises(QStashError):
        await async_client.message.publish_json(
            url="https://httpstat.us/200",
            api={"name": "llm", "provider": upstash()},
        )

    with pytest.raises(QStashError):
        await async_client.message.publish_json(
            url_group="test-url-group",
            api={"name": "llm", "provider": upstash()},
        )


@pytest.mark.asyncio
async def test_batch_async(async_client: AsyncQStash) -&gt; None:
    N = 3
    messages = []
    for i in range(N):
        messages.append(
            BatchRequest(
                body=f"hi {i}",
                url="https://httpstat.us/200",
                retries=0,
                headers={
                    f"test-header-{i}": f"test-value-{i}",
                    "content-type": "text/plain",
                },
            )
        )

    res = await async_client.message.batch(messages)

    assert len(res) == N

    for r in res:
        assert isinstance(r, BatchResponse)
        assert len(r.message_id) &gt; 0


@pytest.mark.asyncio
async def test_batch_json_async(async_client: AsyncQStash) -&gt; None:
    N = 3
    messages = []
    for i in range(N):
        messages.append(
            BatchJsonRequest(
                body={"hi": i},
                url="https://httpstat.us/200",
                retries=0,
                headers={
                    f"test-header-{i}": f"test-value-{i}",
                },
            )
        )

    res = await async_client.message.batch_json(messages)

    assert len(res) == N

    for r in res:
        assert isinstance(r, BatchResponse)
        assert len(r.message_id) &gt; 0


@pytest.mark.asyncio
async def test_publish_to_api_llm_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.message.publish_json(
        api={"name": "llm", "provider": upstash()},
        body={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "just say hello",
                }
            ],
        },
        callback="https://httpstat.us/200",
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    await assert_delivered_eventually_async(async_client, res.message_id)


@pytest.mark.asyncio
async def test_batch_api_llm_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.message.batch_json(
        [
            {
                "api": {"name": "llm", "provider": upstash()},
                "body": {
                    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": "just say hello",
                        }
                    ],
                },
                "callback": "https://httpstat.us/200",
            },
            {
                "api": {
                    "name": "llm",
                    "provider": openai(OPENAI_API_KEY),  # type:ignore[arg-type]
                },
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "user",
                            "content": "just say hello",
                        }
                    ],
                },
                "callback": "https://httpstat.us/200",
            },
        ]
    )

    assert len(res) == 2

    assert isinstance(res[0], BatchResponse)
    assert len(res[0].message_id) &gt; 0

    assert isinstance(res[1], BatchResponse)
    assert len(res[1].message_id) &gt; 0

    await assert_delivered_eventually_async(async_client, res[0].message_id)
    await assert_delivered_eventually_async(async_client, res[1].message_id)


@pytest.mark.asyncio
async def test_enqueue_async(
    async_client: AsyncQStash,
    cleanup_queue_async: Callable[[AsyncQStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue_async(async_client, name)

    res = await async_client.message.enqueue(
        queue=name,
        body="test-body",
        url="https://httpstat.us/200",
        headers={
            "test-header": "test-value",
        },
    )

    assert isinstance(res, EnqueueResponse)

    assert len(res.message_id) &gt; 0


@pytest.mark.asyncio
async def test_enqueue_json_async(
    async_client: AsyncQStash,
    cleanup_queue_async: Callable[[AsyncQStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue_async(async_client, name)

    res = await async_client.message.enqueue_json(
        queue=name,
        body={"test": "body"},
        url="https://httpstat.us/200",
        headers={
            "test-header": "test-value",
        },
    )

    assert isinstance(res, EnqueueResponse)

    assert len(res.message_id) &gt; 0


@pytest.mark.asyncio
async def test_enqueue_api_llm_async(
    async_client: AsyncQStash,
    cleanup_queue_async: Callable[[AsyncQStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue_async(async_client, name)

    res = await async_client.message.enqueue_json(
        queue=name,
        body={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "just say hello",
                }
            ],
        },
        api={"name": "llm", "provider": upstash()},
        callback="https://httpstat.us/200",
    )

    assert isinstance(res, EnqueueResponse)

    assert len(res.message_id) &gt; 0


@pytest.mark.asyncio
async def test_publish_to_url_group_async(async_client: AsyncQStash) -&gt; None:
    name = "python_url_group"
    await async_client.url_group.delete(name)

    await async_client.url_group.upsert_endpoints(
        url_group=name,
        endpoints=[
            {"url": "https://httpstat.us/200"},
            {"url": "https://httpstat.us/201"},
        ],
    )

    res = await async_client.message.publish(
        body="test-body",
        url_group=name,
    )

    assert isinstance(res, list)
    assert len(res) == 2

    await assert_delivered_eventually_async(async_client, res[0].message_id)
    await assert_delivered_eventually_async(async_client, res[1].message_id)


@pytest.mark.asyncio
async def test_timeout_async(async_client: AsyncQStash) -&gt; None:
    res = await async_client.message.publish_json(
        body={"ex_key": "ex_value"},
        url="https://httpstat.us/200",
        timeout=90,
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    await assert_delivered_eventually_async(async_client, res.message_id)


@pytest.mark.asyncio
async def test_cancel_many_async(async_client: AsyncQStash) -&gt; None:
    res0 = await async_client.message.publish(
        url="http://httpstat.us/404",
        retries=3,
    )

    assert isinstance(res0, PublishResponse)

    res1 = await async_client.message.publish(
        url="http://httpstat.us/404",
        retries=3,
    )

    assert isinstance(res1, PublishResponse)

    cancelled = await async_client.message.cancel_many(
        [res0.message_id, res1.message_id]
    )

    assert cancelled == 2


@pytest.mark.asyncio
async def test_cancel_all_async(async_client: AsyncQStash) -&gt; None:
    res0 = await async_client.message.publish(
        url="http://httpstat.us/404",
        retries=3,
    )

    assert isinstance(res0, PublishResponse)

    res1 = await async_client.message.publish(
        url="http://httpstat.us/404",
        retries=3,
    )

    assert isinstance(res1, PublishResponse)

    cancelled = await async_client.message.cancel_all()

    assert cancelled &gt;= 2


@pytest.mark.asyncio
async def test_publish_to_api_llm_custom_provider_async(
    async_client: AsyncQStash,
) -&gt; None:
    res = await async_client.message.publish_json(
        api={
            "name": "llm",
            "provider": openai(OPENAI_API_KEY),  # type:ignore[arg-type]
        },
        body={
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "just say hello",
                }
            ],
        },
        callback="https://httpstat.us/200",
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    await assert_delivered_eventually_async(async_client, res.message_id)


@pytest.mark.asyncio
async def test_enqueue_api_llm_custom_provider_async(
    async_client: AsyncQStash,
    cleanup_queue: Callable[[AsyncQStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue(async_client, name)

    res = await async_client.message.enqueue_json(
        queue=name,
        body={
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "just say hello",
                }
            ],
        },
        api={
            "name": "llm",
            "provider": openai(OPENAI_API_KEY),  # type:ignore[arg-type]
        },
        callback="https://httpstat.us/200",
    )

    assert isinstance(res, EnqueueResponse)

    assert len(res.message_id) &gt; 0

</file>
<file name="tests/asyncio/test_queue.py">
from typing import Callable

import pytest

from qstash import AsyncQStash


@pytest.mark.asyncio
async def test_queue_async(
    async_client: AsyncQStash,
    cleanup_queue_async: Callable[[AsyncQStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue_async(async_client, name)

    await async_client.queue.upsert(queue=name, parallelism=1)

    queue = await async_client.queue.get(name)
    assert queue.name == name
    assert queue.parallelism == 1

    await async_client.queue.upsert(queue=name, parallelism=2)

    queue = await async_client.queue.get(name)
    assert queue.name == name
    assert queue.parallelism == 2

    all_queues = await async_client.queue.list()
    assert any(True for q in all_queues if q.name == name)

    await async_client.queue.delete(name)

    all_queues = await async_client.queue.list()
    assert not any(True for q in all_queues if q.name == name)


@pytest.mark.asyncio
async def test_queue_pause_resume_async(
    async_client: AsyncQStash,
    cleanup_queue_async: Callable[[AsyncQStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue_async(async_client, name)

    await async_client.queue.upsert(queue=name)

    queue = await async_client.queue.get(name)
    assert queue.paused is False

    await async_client.queue.pause(name)

    queue = await async_client.queue.get(name)
    assert queue.paused is True

    await async_client.queue.resume(name)

    queue = await async_client.queue.get(name)
    assert queue.paused is False

    await async_client.queue.upsert(name, paused=True)

    queue = await async_client.queue.get(name)
    assert queue.paused is True

    await async_client.queue.upsert(name, paused=False)

    queue = await async_client.queue.get(name)
    assert queue.paused is False

</file>
<file name="tests/asyncio/test_schedules.py">
from typing import Callable

import pytest

from qstash import AsyncQStash


@pytest.mark.asyncio
async def test_schedule_lifecycle_async(
    async_client: AsyncQStash,
    cleanup_schedule_async: Callable[[AsyncQStash, str], None],
) -&gt; None:
    schedule_id = await async_client.schedule.create_json(
        cron="1 1 1 1 1",
        destination="https://httpstat.us/200",
        body={"ex_key": "ex_value"},
    )

    cleanup_schedule_async(async_client, schedule_id)

    assert len(schedule_id) &gt; 0

    res = await async_client.schedule.get(schedule_id)
    assert res.schedule_id == schedule_id
    assert res.cron == "1 1 1 1 1"

    list_res = await async_client.schedule.list()
    assert any(s.schedule_id == schedule_id for s in list_res)

    await async_client.schedule.delete(schedule_id)

    list_res = await async_client.schedule.list()
    assert not any(s.schedule_id == schedule_id for s in list_res)


@pytest.mark.asyncio
async def test_schedule_pause_resume_async(
    async_client: AsyncQStash,
    cleanup_schedule_async: Callable[[AsyncQStash, str], None],
) -&gt; None:
    schedule_id = await async_client.schedule.create_json(
        cron="1 1 1 1 1",
        destination="https://httpstat.us/200",
        body={"ex_key": "ex_value"},
    )

    cleanup_schedule_async(async_client, schedule_id)

    assert len(schedule_id) &gt; 0

    res = await async_client.schedule.get(schedule_id)
    assert res.schedule_id == schedule_id
    assert res.cron == "1 1 1 1 1"
    assert res.paused is False

    await async_client.schedule.pause(schedule_id)

    res = await async_client.schedule.get(schedule_id)
    assert res.paused is True

    await async_client.schedule.resume(schedule_id)

    res = await async_client.schedule.get(schedule_id)
    assert res.paused is False

</file>
<file name="tests/asyncio/test_signing_key.py">
import pytest

from qstash import AsyncQStash
from tests import QSTASH_CURRENT_SIGNING_KEY, QSTASH_NEXT_SIGNING_KEY


@pytest.mark.asyncio
async def test_get_async(async_client: AsyncQStash) -&gt; None:
    key = await async_client.signing_key.get()
    assert key.current == QSTASH_CURRENT_SIGNING_KEY
    assert key.next == QSTASH_NEXT_SIGNING_KEY

</file>
<file name="tests/asyncio/test_url_group.py">
import pytest

from qstash import AsyncQStash


@pytest.mark.asyncio
async def test_url_group_async(async_client: AsyncQStash) -&gt; None:
    name = "python_url_group"
    await async_client.url_group.delete(name)

    await async_client.url_group.upsert_endpoints(
        url_group=name,
        endpoints=[
            {"url": "https://httpstat.us/200"},
            {"url": "https://httpstat.us/201"},
        ],
    )

    url_group = await async_client.url_group.get(name)
    assert url_group.name == name
    assert any(True for e in url_group.endpoints if e.url == "https://httpstat.us/200")
    assert any(True for e in url_group.endpoints if e.url == "https://httpstat.us/201")

    url_groups = await async_client.url_group.list()
    assert any(True for ug in url_groups if ug.name == name)

    await async_client.url_group.remove_endpoints(
        url_group=name,
        endpoints=[
            {
                "url": "https://httpstat.us/201",
            }
        ],
    )

    url_group = await async_client.url_group.get(name)
    assert url_group.name == name
    assert any(True for e in url_group.endpoints if e.url == "https://httpstat.us/200")
    assert not any(
        True for e in url_group.endpoints if e.url == "https://httpstat.us/201"
    )

</file>
<file name="tests/conftest.py">
import asyncio
from typing import Callable

import pytest
import pytest_asyncio

from qstash import QStash, AsyncQStash
from tests import QSTASH_TOKEN


@pytest.fixture
def client():
    return QStash(token=QSTASH_TOKEN)


@pytest_asyncio.fixture
async def async_client():
    return AsyncQStash(token=QSTASH_TOKEN)


@pytest.fixture
def cleanup_queue(request: pytest.FixtureRequest) -&gt; Callable[[QStash, str], None]:
    queue_names = []

    def register(client: QStash, queue_name: str) -&gt; None:
        queue_names.append((client, queue_name))

    def delete():
        for client, queue_name in queue_names:
            try:
                client.queue.delete(queue_name)
            except Exception:
                pass

    request.addfinalizer(delete)

    return register


@pytest_asyncio.fixture
def cleanup_queue_async(
    request: pytest.FixtureRequest,
) -&gt; Callable[[AsyncQStash, str], None]:
    queue_names = []

    def register(async_client: AsyncQStash, queue_name: str) -&gt; None:
        queue_names.append((async_client, queue_name))

    def finalizer():
        async def delete():
            for async_client, queue_name in queue_names:
                try:
                    await async_client.queue.delete(queue_name)
                except Exception:
                    pass

        asyncio.run(delete())

    request.addfinalizer(finalizer)

    return register


@pytest.fixture
def cleanup_schedule(request: pytest.FixtureRequest) -&gt; Callable[[QStash, str], None]:
    schedule_ids = []

    def register(client: QStash, schedule_id: str) -&gt; None:
        schedule_ids.append((client, schedule_id))

    def delete():
        for client, schedule_id in schedule_ids:
            try:
                client.schedule.delete(schedule_id)
            except Exception:
                pass

    request.addfinalizer(delete)

    return register


@pytest_asyncio.fixture
def cleanup_schedule_async(
    request: pytest.FixtureRequest,
) -&gt; Callable[[AsyncQStash, str], None]:
    schedule_ids = []

    def register(async_client: AsyncQStash, schedule_id: str) -&gt; None:
        schedule_ids.append((async_client, schedule_id))

    def finalizer():
        async def delete():
            for async_client, schedule_id in schedule_ids:
                try:
                    await async_client.schedule.delete(schedule_id)
                except Exception:
                    pass

        asyncio.run(delete())

    request.addfinalizer(finalizer)

    return register

</file>
<file name="tests/test_chat.py">
from qstash import QStash
from qstash.chat import (
    ChatCompletion,
    ChatCompletionChunkStream,
    upstash,
    openai,
)
from tests import OPENAI_API_KEY


def test_chat(client: QStash) -&gt; None:
    res = client.chat.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "just say hello"}],
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


def test_chat_streaming(client: QStash) -&gt; None:
    res = client.chat.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "just say hello"}],
        stream=True,
    )

    assert isinstance(res, ChatCompletionChunkStream)

    for i, r in enumerate(res):
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert r.choices[0].delta.content is not None


def test_prompt(client: QStash) -&gt; None:
    res = client.chat.prompt(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        user="just say hello",
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


def test_prompt_streaming(client: QStash) -&gt; None:
    res = client.chat.prompt(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        user="just say hello",
        stream=True,
    )

    assert isinstance(res, ChatCompletionChunkStream)

    for i, r in enumerate(res):
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert r.choices[0].delta.content is not None


def test_chat_explicit_upstash_provider(client: QStash) -&gt; None:
    res = client.chat.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "just say hello"}],
        provider=upstash(),
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


def test_chat_explicit_upstash_provider_streaming(client: QStash) -&gt; None:
    res = client.chat.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "just say hello"}],
        provider=upstash(),
        stream=True,
    )

    assert isinstance(res, ChatCompletionChunkStream)

    for i, r in enumerate(res):
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert r.choices[0].delta.content is not None


def test_prompt_explicit_upstash_provider(client: QStash) -&gt; None:
    res = client.chat.prompt(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        user="just say hello",
        provider=upstash(),
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


def test_prompt_explicit_upstash_provider_streaming(client: QStash) -&gt; None:
    res = client.chat.prompt(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        user="just say hello",
        provider=upstash(),
        stream=True,
    )

    assert isinstance(res, ChatCompletionChunkStream)

    for i, r in enumerate(res):
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert r.choices[0].delta.content is not None


def test_chat_custom_provider(client: QStash) -&gt; None:
    res = client.chat.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "just say hello"}],
        provider=openai(token=OPENAI_API_KEY),  # type:ignore[arg-type]
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


def test_chat_custom_provider_streaming(client: QStash) -&gt; None:
    res = client.chat.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "just say hello"}],
        provider=openai(token=OPENAI_API_KEY),  # type:ignore[arg-type]
        stream=True,
    )

    assert isinstance(res, ChatCompletionChunkStream)

    for i, r in enumerate(res):
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert (
                r.choices[0].delta.content is not None
                or r.choices[0].finish_reason is not None
            )


def test_prompt_custom_provider(client: QStash) -&gt; None:
    res = client.chat.prompt(
        model="gpt-3.5-turbo",
        user="just say hello",
        provider=openai(token=OPENAI_API_KEY),  # type:ignore[arg-type]
    )

    assert isinstance(res, ChatCompletion)

    assert len(res.choices[0].message.content) &gt; 0
    assert res.choices[0].message.role == "assistant"


def test_prompt_custom_provider_streaming(client: QStash) -&gt; None:
    res = client.chat.prompt(
        model="gpt-3.5-turbo",
        user="just say hello",
        provider=openai(token=OPENAI_API_KEY),  # type:ignore[arg-type]
        stream=True,
    )

    assert isinstance(res, ChatCompletionChunkStream)

    for i, r in enumerate(res):
        if i == 0:
            assert r.choices[0].delta.role is not None
        else:
            assert (
                r.choices[0].delta.content is not None
                or r.choices[0].finish_reason is not None
            )

</file>
<file name="tests/test_dlq.py">
from qstash import QStash
from qstash.message import PublishResponse
from tests import assert_eventually


def assert_failed_eventually(client: QStash, *msg_ids: str) -&gt; None:
    def assertion() -&gt; None:
        messages = client.dlq.list().messages

        matched_messages = [msg for msg in messages if msg.message_id in msg_ids]
        assert len(matched_messages) == len(msg_ids)

        for msg in matched_messages:
            dlq_msg = client.dlq.get(msg.dlq_id)
            assert dlq_msg.response_body == "404 Not Found"
            assert msg.response_body == "404 Not Found"

        if len(msg_ids) == 1:
            client.dlq.delete(matched_messages[0].dlq_id)
        else:
            deleted = client.dlq.delete_many([m.dlq_id for m in matched_messages])
            assert deleted == len(msg_ids)

        messages = client.dlq.list().messages
        matched = any(True for msg in messages if msg.message_id in msg_ids)
        assert not matched

    assert_eventually(
        assertion,
        initial_delay=2.0,
        retry_delay=1.0,
        timeout=10.0,
    )


def test_dlq_get_and_delete(client: QStash) -&gt; None:
    res = client.message.publish_json(
        url="http://httpstat.us/404",
        retries=0,
    )

    assert isinstance(res, PublishResponse)

    assert_failed_eventually(client, res.message_id)


def test_dlq_get_and_delete_many(client: QStash) -&gt; None:
    msg_ids = []
    for _ in range(5):
        res = client.message.publish_json(
            url="http://httpstat.us/404",
            retries=0,
        )

        assert isinstance(res, PublishResponse)
        msg_ids.append(res.message_id)

    assert_failed_eventually(client, *msg_ids)


def test_dlq_filter(client: QStash) -&gt; None:
    res = client.message.publish_json(
        url="http://httpstat.us/404",
        retries=0,
    )

    assert isinstance(res, PublishResponse)

    def assertion():
        messages = client.dlq.list(
            filter={"message_id": res.message_id},
            count=1,
        ).messages

        assert len(messages) == 1
        assert messages[0].message_id == res.message_id

        client.dlq.delete(messages[0].dlq_id)

    assert_eventually(
        assertion,
        initial_delay=2.0,
        retry_delay=1.0,
        timeout=10.0,
    )

</file>
<file name="tests/test_message.py">
from typing import Callable

import pytest

from qstash import QStash
from qstash.chat import upstash, openai
from qstash.errors import QStashError
from qstash.event import EventState
from qstash.message import (
    BatchJsonRequest,
    BatchRequest,
    BatchResponse,
    EnqueueResponse,
    PublishResponse,
)
from tests import assert_eventually, OPENAI_API_KEY


def assert_delivered_eventually(client: QStash, msg_id: str) -&gt; None:
    def assertion() -&gt; None:
        events = client.event.list(
            filter={
                "message_id": msg_id,
                "state": EventState.DELIVERED,
            }
        ).events

        assert len(events) == 1

    assert_eventually(
        assertion,
        initial_delay=1.0,
        retry_delay=1.0,
        timeout=60.0,
    )


def test_publish_to_url(client: QStash) -&gt; None:
    res = client.message.publish(
        body="test-body",
        url="https://httpstat.us/200",
        headers={
            "test-header": "test-value",
        },
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    assert_delivered_eventually(client, res.message_id)


def test_publish_to_url_json(client: QStash) -&gt; None:
    res = client.message.publish_json(
        body={"ex_key": "ex_value"},
        url="https://httpstat.us/200",
        headers={
            "test-header": "test-value",
        },
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    assert_delivered_eventually(client, res.message_id)


def test_disallow_multiple_destinations(client: QStash) -&gt; None:
    with pytest.raises(QStashError):
        client.message.publish_json(
            url="https://httpstat.us/200",
            url_group="test-url-group",
        )

    with pytest.raises(QStashError):
        client.message.publish_json(
            url="https://httpstat.us/200",
            api={"name": "llm", "provider": upstash()},
        )

    with pytest.raises(QStashError):
        client.message.publish_json(
            url_group="test-url-group",
            api={"name": "llm", "provider": upstash()},
        )


def test_batch(client: QStash) -&gt; None:
    N = 3
    messages = []
    for i in range(N):
        messages.append(
            BatchRequest(
                body=f"hi {i}",
                url="https://httpstat.us/200",
                retries=0,
                headers={
                    f"test-header-{i}": f"test-value-{i}",
                    "content-type": "text/plain",
                },
            )
        )

    res = client.message.batch(messages)

    assert len(res) == N

    for r in res:
        assert isinstance(r, BatchResponse)
        assert len(r.message_id) &gt; 0


def test_batch_json(client: QStash) -&gt; None:
    N = 3
    messages = []
    for i in range(N):
        messages.append(
            BatchJsonRequest(
                body={"hi": i},
                url="https://httpstat.us/200",
                retries=0,
                headers={
                    f"test-header-{i}": f"test-value-{i}",
                },
            )
        )

    res = client.message.batch_json(messages)

    assert len(res) == N

    for r in res:
        assert isinstance(r, BatchResponse)
        assert len(r.message_id) &gt; 0


def test_publish_to_api_llm(client: QStash) -&gt; None:
    res = client.message.publish_json(
        api={"name": "llm", "provider": upstash()},
        body={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "just say hello",
                }
            ],
        },
        callback="https://httpstat.us/200",
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    assert_delivered_eventually(client, res.message_id)


def test_batch_api_llm(client: QStash) -&gt; None:
    res = client.message.batch_json(
        [
            {
                "api": {"name": "llm", "provider": upstash()},
                "body": {
                    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": "just say hello",
                        }
                    ],
                },
                "callback": "https://httpstat.us/200",
            },
            {
                "api": {
                    "name": "llm",
                    "provider": openai(OPENAI_API_KEY),  # type:ignore[arg-type]
                },  # type:ignore[arg-type]
                "body": {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {
                            "role": "user",
                            "content": "just say hello",
                        }
                    ],
                },
                "callback": "https://httpstat.us/200",
            },
        ]
    )

    assert len(res) == 2

    assert isinstance(res[0], BatchResponse)
    assert len(res[0].message_id) &gt; 0

    assert isinstance(res[1], BatchResponse)
    assert len(res[1].message_id) &gt; 0

    assert_delivered_eventually(client, res[0].message_id)
    assert_delivered_eventually(client, res[1].message_id)


def test_enqueue(
    client: QStash,
    cleanup_queue: Callable[[QStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue(client, name)

    res = client.message.enqueue(
        queue=name,
        body="test-body",
        url="https://httpstat.us/200",
        headers={
            "test-header": "test-value",
        },
    )

    assert isinstance(res, EnqueueResponse)

    assert len(res.message_id) &gt; 0


def test_enqueue_json(
    client: QStash,
    cleanup_queue: Callable[[QStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue(client, name)

    res = client.message.enqueue_json(
        queue=name,
        body={"test": "body"},
        url="https://httpstat.us/200",
        headers={
            "test-header": "test-value",
        },
    )

    assert isinstance(res, EnqueueResponse)

    assert len(res.message_id) &gt; 0


def test_enqueue_api_llm(
    client: QStash,
    cleanup_queue: Callable[[QStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue(client, name)

    res = client.message.enqueue_json(
        queue=name,
        body={
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "just say hello",
                }
            ],
        },
        api={"name": "llm", "provider": upstash()},
        callback="https://httpstat.us/200",
    )

    assert isinstance(res, EnqueueResponse)

    assert len(res.message_id) &gt; 0


def test_publish_to_url_group(client: QStash) -&gt; None:
    name = "python_url_group"
    client.url_group.delete(name)

    client.url_group.upsert_endpoints(
        url_group=name,
        endpoints=[
            {"url": "https://httpstat.us/200"},
            {"url": "https://httpstat.us/201"},
        ],
    )

    res = client.message.publish(
        body="test-body",
        url_group=name,
    )

    assert isinstance(res, list)
    assert len(res) == 2

    assert_delivered_eventually(client, res[0].message_id)
    assert_delivered_eventually(client, res[1].message_id)


def test_timeout(client: QStash) -&gt; None:
    res = client.message.publish_json(
        body={"ex_key": "ex_value"},
        url="https://httpstat.us/200",
        timeout=90,
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    assert_delivered_eventually(client, res.message_id)


def test_cancel_many(client: QStash) -&gt; None:
    res0 = client.message.publish(
        url="http://httpstat.us/404",
        retries=3,
    )

    assert isinstance(res0, PublishResponse)

    res1 = client.message.publish(
        url="http://httpstat.us/404",
        retries=3,
    )

    assert isinstance(res1, PublishResponse)

    cancelled = client.message.cancel_many([res0.message_id, res1.message_id])

    assert cancelled == 2


def test_cancel_all(client: QStash) -&gt; None:
    res0 = client.message.publish(
        url="http://httpstat.us/404",
        retries=3,
    )

    assert isinstance(res0, PublishResponse)

    res1 = client.message.publish(
        url="http://httpstat.us/404",
        retries=3,
    )

    assert isinstance(res1, PublishResponse)

    cancelled = client.message.cancel_all()

    assert cancelled &gt;= 2


def test_publish_to_api_llm_custom_provider(client: QStash) -&gt; None:
    res = client.message.publish_json(
        api={
            "name": "llm",
            "provider": openai(OPENAI_API_KEY),  # type:ignore[arg-type]
        },
        body={
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "just say hello",
                }
            ],
        },
        callback="https://httpstat.us/200",
    )

    assert isinstance(res, PublishResponse)
    assert len(res.message_id) &gt; 0

    assert_delivered_eventually(client, res.message_id)


def test_enqueue_api_llm_custom_provider(
    client: QStash,
    cleanup_queue: Callable[[QStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue(client, name)

    res = client.message.enqueue_json(
        queue=name,
        body={
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": "just say hello",
                }
            ],
        },
        api={
            "name": "llm",
            "provider": openai(OPENAI_API_KEY),  # type:ignore[arg-type]
        },
        callback="https://httpstat.us/200",
    )

    assert isinstance(res, EnqueueResponse)

    assert len(res.message_id) &gt; 0

</file>
<file name="tests/test_queue.py">
from typing import Callable

from qstash import QStash


def test_queue(
    client: QStash,
    cleanup_queue: Callable[[QStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue(client, name)

    client.queue.upsert(queue=name, parallelism=1)

    queue = client.queue.get(name)
    assert queue.name == name
    assert queue.parallelism == 1

    client.queue.upsert(queue=name, parallelism=2)

    queue = client.queue.get(name)
    assert queue.name == name
    assert queue.parallelism == 2

    all_queues = client.queue.list()
    assert any(True for q in all_queues if q.name == name)

    client.queue.delete(name)

    all_queues = client.queue.list()
    assert not any(True for q in all_queues if q.name == name)


def test_queue_pause_resume(
    client: QStash,
    cleanup_queue: Callable[[QStash, str], None],
) -&gt; None:
    name = "test_queue"
    cleanup_queue(client, name)

    client.queue.upsert(queue=name)

    queue = client.queue.get(name)
    assert queue.paused is False

    client.queue.pause(name)

    queue = client.queue.get(name)
    assert queue.paused is True

    client.queue.resume(name)

    queue = client.queue.get(name)
    assert queue.paused is False

    client.queue.upsert(name, paused=True)

    queue = client.queue.get(name)
    assert queue.paused is True

    client.queue.upsert(name, paused=False)

    queue = client.queue.get(name)
    assert queue.paused is False

</file>
<file name="tests/test_receiver.py">
import base64
import hashlib
import json
import time
from typing import Optional

import jwt
import pytest

from qstash import Receiver
from qstash.errors import SignatureError
from tests import QSTASH_CURRENT_SIGNING_KEY, QSTASH_NEXT_SIGNING_KEY


@pytest.fixture
def receiver():
    return Receiver(
        current_signing_key=QSTASH_CURRENT_SIGNING_KEY,
        next_signing_key=QSTASH_NEXT_SIGNING_KEY,
    )


def get_signature(body: str, key: Optional[str]) -&gt; str:
    body_hash = hashlib.sha256(body.encode()).digest()
    body_hash_b64 = base64.urlsafe_b64encode(body_hash).decode().rstrip("=")
    payload = {
        "aud": "",
        "body": body_hash_b64,
        "exp": int(time.time()) + 300,
        "iat": int(time.time()),
        "iss": "Upstash",
        "jti": time.time(),
        "nbf": int(time.time()),
        "sub": "https://httpstat.us/200",
    }
    signature = jwt.encode(
        payload, key, algorithm="HS256", headers={"alg": "HS256", "typ": "JWT"}
    )
    return signature


def test_receiver(receiver: Receiver) -&gt; None:
    body = json.dumps({"hello": "world"})
    sig = get_signature(body, QSTASH_CURRENT_SIGNING_KEY)

    receiver.verify(
        signature=sig,
        body=body,
        url="https://httpstat.us/200",
    )


def test_failed_verification(receiver: Receiver) -&gt; None:
    body = json.dumps({"hello": "world"})
    sig = get_signature(body, QSTASH_CURRENT_SIGNING_KEY)

    with pytest.raises(SignatureError):
        receiver.verify(
            signature=sig,
            body=body,
            url="https://httpstat.us/201",
        )

</file>
<file name="tests/test_schedules.py">
from typing import Callable

import pytest

from qstash import QStash


@pytest.fixture
def cleanup_schedule(request: pytest.FixtureRequest) -&gt; Callable[[QStash, str], None]:
    schedule_ids = []

    def register(client: QStash, schedule_id: str) -&gt; None:
        schedule_ids.append((client, schedule_id))

    def delete():
        for client, schedule_id in schedule_ids:
            try:
                client.schedule.delete(schedule_id)
            except Exception:
                pass

    request.addfinalizer(delete)

    return register


def test_schedule_lifecycle(
    client: QStash,
    cleanup_schedule: Callable[[QStash, str], None],
) -&gt; None:
    schedule_id = client.schedule.create_json(
        cron="1 1 1 1 1",
        destination="https://httpstat.us/200",
        body={"ex_key": "ex_value"},
    )

    cleanup_schedule(client, schedule_id)

    assert len(schedule_id) &gt; 0

    res = client.schedule.get(schedule_id)
    assert res.schedule_id == schedule_id
    assert res.cron == "1 1 1 1 1"

    list_res = client.schedule.list()
    assert any(s.schedule_id == schedule_id for s in list_res)

    client.schedule.delete(schedule_id)

    list_res = client.schedule.list()
    assert not any(s.schedule_id == schedule_id for s in list_res)


def test_schedule_pause_resume(
    client: QStash,
    cleanup_schedule: Callable[[QStash, str], None],
) -&gt; None:
    schedule_id = client.schedule.create_json(
        cron="1 1 1 1 1",
        destination="https://httpstat.us/200",
        body={"ex_key": "ex_value"},
    )

    cleanup_schedule(client, schedule_id)

    assert len(schedule_id) &gt; 0

    res = client.schedule.get(schedule_id)
    assert res.schedule_id == schedule_id
    assert res.cron == "1 1 1 1 1"
    assert res.paused is False

    client.schedule.pause(schedule_id)

    res = client.schedule.get(schedule_id)
    assert res.paused is True

    client.schedule.resume(schedule_id)

    res = client.schedule.get(schedule_id)
    assert res.paused is False

</file>
<file name="tests/test_signing_key.py">
from qstash import QStash
from tests import QSTASH_CURRENT_SIGNING_KEY, QSTASH_NEXT_SIGNING_KEY


def test_get(client: QStash) -&gt; None:
    key = client.signing_key.get()
    assert key.current == QSTASH_CURRENT_SIGNING_KEY
    assert key.next == QSTASH_NEXT_SIGNING_KEY

</file>
<file name="tests/test_url_group.py">
from qstash import QStash


def test_url_group(client: QStash) -&gt; None:
    name = "python_url_group"
    client.url_group.delete(name)

    client.url_group.upsert_endpoints(
        url_group=name,
        endpoints=[
            {"url": "https://httpstat.us/200"},
            {"url": "https://httpstat.us/201"},
        ],
    )

    url_group = client.url_group.get(name)
    assert url_group.name == name
    assert any(True for e in url_group.endpoints if e.url == "https://httpstat.us/200")
    assert any(True for e in url_group.endpoints if e.url == "https://httpstat.us/201")

    url_groups = client.url_group.list()
    assert any(True for ug in url_groups if ug.name == name)

    client.url_group.remove_endpoints(
        url_group=name,
        endpoints=[
            {
                "url": "https://httpstat.us/201",
            }
        ],
    )

    url_group = client.url_group.get(name)
    assert url_group.name == name
    assert any(True for e in url_group.endpoints if e.url == "https://httpstat.us/200")
    assert not any(
        True for e in url_group.endpoints if e.url == "https://httpstat.us/201"
    )

</file>
</source>