# 1. High-Level Architecture Overview

The system orchestrates the following steps:  

1. **Inbound Email** arrives at your webhook endpoint (`/email-receive`).  
2. **Flask** decodes attachments and uploads them to **DigitalOcean Spaces**.  
3. **QStash** asynchronously queues the request to your worker endpoint (`/process-email`).  
4. **Worker** logic (in `workers.py`) invokes:  
   - **GPT-4** analysis on attachments,  
   - **o1** external API to finalize a response,  
   - **Resend** to send the final email.  

## 1.1 ASCII Architecture Diagram

```
                 +------------------------+
  (A) Email ---> |   Email Service /     |
  (B) Webhook -> |   Incoming Webhook    |
                 +---------+-------------+
                           |
                           v
                      (1) POST /email-receive
                           |
                           v
          +----------------+----------------+
          |       Flask (app.py)           |
          |                                |
          |  - Decodes & Uploads           |
          |    attachments to DO Spaces    |
          |  - Publishes message via       |
          |    QStash                      |
          +----------------+----------------+
                           |
                     (2) QStash Publish
                           v
                 +------------------------+
                 |       QStash Queue    |
                 +-----------+-----------+
                             |
                    (3) POST /process-email
                             |
                             v
              +--------------+--------------+
              |  workers.py (process_bp)   |
              |    - GPT analysis          |
              |    - Call o1 to build resp |
              |    - Send final email      |
              +--------------+-------------+
                             |
                (4) GPT / o1 / Resend Calls
                             v
            +-------------------------------+
            | External Services:           |
            |  - GPT-4 (OpenAI)            |
            |  - o1 API                    |
            |  - Resend                    |
            |  - DO Spaces                 |
            +-------------------------------+
```

Key Flow:  
1. An HTTP POST with the email arrives at `/email-receive`.  
2. Flask uploads attachments to **DO Spaces**, then publishes a message to QStash.  
3. QStash calls `/process-email` (in the workers).  
4. Worker does GPT analysis, calls o1, and sends the final email via Resend.  

---

# 2. Data Flow Diagram

Below is a simplified data flow focusing on **where data moves** between components.

```
[Email Inbound JSON] 
   |
   v
(1) /email-receive (Flask)  
   |          \
   |           \ base64 decode attachments
   |            \
   |             v
   |        DO Spaces (store attachments)
   |
   v
   QStash publish (Email metadata + attachments' URLs)
   |
   v
[ QStash ]  
   |
   v
(2) /process-email (workers.py)  
   |          \
   |           \ GPT-4 (analysis of each attachment URL)
   |            \
   |             o1 (construct final response)
   |
   v
 Resend (send final email)
   |
   v
[User's Inbox receives final message]
```

**Main Artifacts:**  
- **Inbound JSON** from your email provider.  
- **Attachments** stored in DO Spaces.  
- **Analysis** results from GPT-4.  
- **Tailored** response from o1.  
- **Final Email** from Resend.  

---

# 3. Sequence Diagram

This shows time-ordered steps from inbound email to final outbound email.

```
   Email Service          Flask (app.py)          QStash            workers.py           GPT-4        o1       Resend
       |                       |                     |                  |                  |           |          |
       | (A) Email arrives     |                     |                  |                  |           |          |
(1)    |---------------------->| /email-receive      |                  |                  |           |          |
       | POST JSON incl atts  |                     |                  |                  |           |          |
(2)    |                       |-- decode base64 --> (Local memory)     |                  |           |          |
(3)    |                       |-- upload_to_spaces -> DO Spaces        |                  |           |          |
       |                       |(gets URLs for each) |                  |                  |           |          |
(4)    |                       |-- qstash_client.publish_json --------->|  (QStash)        |           |          |
       |                       |               (returns 200)            |                  |           |          |
       |                       |<-------------------- 200 --------------|                  |           |          |
       |                       |                     |                  |                  |           |          |
(5)    |                       |                     |     callback --->| /process-email   |           |          |
       |                       |                     |  (body w/ attach |                  |           |          |
       |                       |                     |   URLs, etc.)    |                  |           |          |
(6)    |                       |                     |                  |-- analyze_attach->| GPT-4     |          |
       |                       |                     |                  |   ... returns text|<----------|          |
       |                       |                     |                  | accumulate results|           |          |
(7)    |                       |                     |                  |-----> o1 -------->|           |  (o1)    |
       |                       |                     |                  |   (get response)  |<----------|          |
       |                       |                     |                  |   subject/body_html|          |          |
(8)    |                       |                     |                  |---> Resend ------------------>|          |
       |                       |                     |                  |        (send final email)     |<---------|
(9)    |                       |                     |                  |<---------------- 200 ---------|
       |                       |                     |                  |
```

---

# 4. Call Graph (Simplified)

Below is a top-down representation of function calls across key files:

```
app.py:
 └── email_receive() [Flask route]
      ├── base64.b64decode(attachment) for each
      ├── upload_to_spaces(...) in do_spaces.py
      ├── qstash_client.message.publish_json(...) in qstash.py
      └── return HTTP 200

workers.py:
 └── process_email_data() [Flask route for QStash]
      ├── for each attachment_url:
      │    └── analyze_attachment(url) in gpt_processing.py 
      │         └── openai.ChatCompletion.create(...)
      ├── requests.post(...) to o1 service
      ├── send_response(...) in resend_client.py
      │    └── resend_client.emails.send(...)
      └── return HTTP 200
```

### Explanation

- `email_receive()` is invoked by the inbound email.  
  - Decodes attachments → calls `upload_to_spaces(file_content)` → calls `qstash_client.message.publish_json(...)`.  
- `process_email_data()` is invoked by QStash.  
  - For each attachment URL → calls `analyze_attachment()` → GPT-4.  
  - Then calls `o1`.  
  - Finally calls `send_response()` → Resend.  

---

# 5. Key Architecture Details

1. **Flask**  
   - **`/email-receive`** handles **inbound** emails.  
   - Synchronous decoding + DO Spaces upload.  
   - Publishes to QStash for asynchronous processing.  

2. **DigitalOcean Spaces** (in `do_spaces.py`)  
   - `upload_to_spaces()` stores attachments.  
   - Returns a URL for each uploaded file.  

3. **QStash**  
   - The system uses QStash to queue heavy tasks and decouple the immediate /email-receive from the longer GPT analysis.  
   - `publish_json(...)` -> QStash -> calls `/process-email`.  

4. **Workers** (in `workers.py`)  
   - **`/process-email`** route processes the queued job.  
   - Calls GPT-4 with each attachment URL (`gpt_processing.py`), calls `o1`, sends final email.  

5. **GPT-4**  
   - Provided by OpenAI (`openai.ChatCompletion.create`).  
   - Potentially large or expensive calls, so done asynchronously after QStash triggers.  

6. **o1**  
   - External service (example: `api.o1.example.com/generate-response`).  
   - Final transformation or logic before sending.  

7. **Resend**  
   - Sends the final email back to the original sender.  
   - `send_response()` in `resend_client.py`.  

8. **Error Handling & Retries**  
   - QStash will **retry** if the worker endpoint fails.  
   - DO Spaces or GPT calls can raise exceptions if not handled.  

---

## 6. Operational Concerns

- **Attachment Size**: Large attachments can consume memory when base64-decoding. Production usage might want streaming uploads.  
- **Concurrency**: For large traffic, consider a WSGI server (Gunicorn) and background workers or container orchestration.  
- **Logging**: Use structured logs to trace GPT calls, DO upload times, etc.  
- **Security**: Validate inbound attachments, scan for malware if needed.  
- **Production**: Turn off `debug=True`, use environment variables for secrets (OpenAI, DO keys, Resend, QStash).  

---

# 7. Summary

The project’s core design pattern is:  
1. **Receive** email → **store** attachments in DO → **queue** request via QStash.  
2. **Process** asynchronously → **GPT** for analysis → **o1** for final response → **Resend** the result.  

This setup allows **non-blocking** email ingestion and a robust pipeline for advanced text analysis with GPT-4.
