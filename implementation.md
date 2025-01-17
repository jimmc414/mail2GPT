Below is a **step-by-step** plan that assumes **all the code files** (`app.py`, `workers.py`, `gpt_processing.py`, `do_spaces.py`, `resend_client.py`, etc.) already exist in the `my_email_automation/` folder as shown in your folder structure. We also assume **Python is installed** on the developer’s machine. This guide will walk you through everything from installing dependencies, setting environment variables, running the app, and testing.

---

# 1. Navigate to Project Directory

1. **Open your terminal** (on Windows, you can use Command Prompt or PowerShell; on macOS/Linux, use a shell).
2. **Change directory** to your project folder.

```bash
cd path/to/my_email_automation
```

(Replace `path/to/my_email_automation` with the actual path where your project code resides.)

---

# 2. Create a Virtual Environment (Optional but Recommended)

If you’d like to keep your dependencies isolated:

```bash
python -m venv venv
```

- **Windows**:  
  ```bash
  venv\Scripts\activate
  ```
- **macOS/Linux**:  
  ```bash
  source venv/bin/activate
  ```

*(If you’re comfortable installing packages globally, you can skip creating a virtual environment.)*

---

# 3. Install Dependencies

Use `pip` to install all required libraries from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Wait until the process finishes. This will install packages like:

- **Flask** (or possibly FastAPI)
- **boto3** (for DigitalOcean Spaces)
- **qstash**
- **openai**
- **resend**
- etc.

---

# 4. Set Environment Variables

You must set all the environment variables used by the code. Below is a sample list (your actual keys and variable names may differ):

- **`DO_ACCESS_KEY`**: Your DigitalOcean Access Key  
- **`DO_SECRET_KEY`**: Your DigitalOcean Secret Key  
- **`DO_BUCKET_NAME`**: The name of your DigitalOcean Space (e.g., `my-space`)  
- **`OPENAI_API_KEY`**: Your GPT-4/OpenAI API Key  
- **`RESEND_API_KEY`**: Your Resend API Key  
- **`QSTASH_TOKEN`**: Your QStash publishing token  
- **`QSTASH_CURRENT_SIGNING_KEY`** and **`QSTASH_NEXT_SIGNING_KEY`** (if verifying signatures)  

### 4.1 Option A: Using a `.env` File

1. **Create a new file** in your project called `.env`.
2. **Open** it in a text editor and add the variables, for example:

   ```plaintext
   DO_ACCESS_KEY=your_digitalocean_access_key
   DO_SECRET_KEY=your_digitalocean_secret_key
   DO_BUCKET_NAME=your-bucket-name
   OPENAI_API_KEY=your_openai_api_key
   RESEND_API_KEY=your_resend_api_key
   QSTASH_TOKEN=your_qstash_token
   QSTASH_CURRENT_SIGNING_KEY=your_qstash_current_signing_key
   QSTASH_NEXT_SIGNING_KEY=your_qstash_next_signing_key
   ```

3. **Save** the `.env` file.

*(Your code might already import and use `python-dotenv` to load these automatically.)*

### 4.2 Option B: Export Variables in Terminal

Alternatively, set them **manually** in your terminal. For example, on macOS/Linux:

```bash
export DO_ACCESS_KEY="your_digitalocean_access_key"
export DO_SECRET_KEY="your_digitalocean_secret_key"
export DO_BUCKET_NAME="your-bucket-name"
export OPENAI_API_KEY="your_openai_api_key"
export RESEND_API_KEY="your_resend_api_key"
export QSTASH_TOKEN="your_qstash_token"
export QSTASH_CURRENT_SIGNING_KEY="your_qstash_current_signing_key"
export QSTASH_NEXT_SIGNING_KEY="your_qstash_next_signing_key"
```

On Windows (Command Prompt):

```bat
set DO_ACCESS_KEY=your_digitalocean_access_key
set DO_SECRET_KEY=your_digitalocean_secret_key
set DO_BUCKET_NAME=your-bucket-name
set OPENAI_API_KEY=your_openai_api_key
set RESEND_API_KEY=your_resend_api_key
set QSTASH_TOKEN=your_qstash_token
set QSTASH_CURRENT_SIGNING_KEY=your_qstash_current_signing_key
set QSTASH_NEXT_SIGNING_KEY=your_qstash_next_signing_key
```

*(If you close the terminal, you’ll need to re-export the variables unless you’re using a `.env` file.)*

---

# 5. Verify the Folder Structure & Code

Make sure your folder structure looks like this:

```
my_email_automation/
  ├─ app.py           
  ├─ config.py        
  ├─ workers.py       
  ├─ gpt_processing.py
  ├─ do_spaces.py     
  ├─ resend_client.py 
  ├─ requirements.txt
  └─ ...
```

Confirm that each file contains the code as described in the reference implementation. For instance:

- **`do_spaces.py`** has the `upload_to_spaces()` function that uses `boto3`.
- **`gpt_processing.py`** has the `analyze_attachment()` function using `openai.ChatCompletion`.
- **`resend_client.py`** has `send_response()` that uses the `resend` package.
- **`app.py`** has your Flask routes (`/email-receive`, etc.).
- **`workers.py`** (or a blueprint in the same file) has the `/process-email` route.

---

# 6. Run the Flask App

1. **Open your terminal** in the `my_email_automation` directory (make sure your virtual environment is activated if you set one up).
2. Start the Flask app. There are two main ways:

   **Option A:** Using `flask run`  
   ```bash
   export FLASK_APP=app.py     # or set FLASK_APP=app.py on Windows
   flask run --port=5000
   ```

   **Option B:** Using `python app.py` directly  
   ```bash
   python app.py
   ```

3. You should see output like:

   ```plaintext
   * Serving Flask app 'app'
   * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
   ```
   
   This means your Flask server is running on port `5000`.

*(If you encounter errors, verify you installed all dependencies and set environment variables.)*

---

# 7. Make the Server Publicly Accessible

Your email webhook service (and QStash) needs a **public endpoint** to call:

1. **Deploy** your app to a cloud platform (e.g., AWS, Heroku, DigitalOcean App Platform, etc.) **OR**
2. Use a tool like [**ngrok**](https://ngrok.com/) to expose your local server.

For instance, using ngrok:

```bash
ngrok http 5000
```

This will output a forwarding URL like:  
```
Forwarding                    https://<some-random-subdomain>.ngrok.io -> http://localhost:5000
```

Use that `https://<some-random-subdomain>.ngrok.io` as your public endpoint.  

---

# 8. Configure QStash & Email Webhook

1. **In your QStash dashboard** (or via Upstash console/API), set the **publish URL** to your publicly accessible endpoint for `/process-email`. You might do something like:

   ```
   https://<your-public-domain>/process-email
   ```
   
   Or if you’re using the QStash library in the code, make sure your code references the correct public domain in:
   ```python
   qstash_client.message.publish_json(
       url="https://<your-public-domain>/process-email",
       ...
   )
   ```

2. **Configure your Email Service** (e.g., Mailgun, SendGrid, or whichever you’re using) to send incoming emails via webhook to:

   ```
   https://<your-public-domain>/email-receive
   ```

   This ensures that any inbound email triggers a POST request to your Flask `/email-receive` route with the JSON payload (including attachments).

---

# 9. Test the Workflow

## 9.1 Send a Test Email

1. Send a **test email** from any email client (e.g., Gmail) to the address configured in your email service’s inbound routing.
2. Attach a small PDF, text file, or image to test the attachment uploading flow.

## 9.2 Observe the Logs

- **Check your Flask app console** (the terminal where you ran `flask run` or `python app.py`).  
- You should see a `POST /email-receive` log message once your email webhook calls it.  
- Shortly thereafter, QStash will POST to `POST /process-email` to run the GPT-4 analysis, call `o1`, and then send a response via Resend.

## 9.3 Confirm the Response Email

- You (the sender) should receive a **reply** from your app (via Resend) with a subject/body that references the data from the GPT-4 analysis and the `o1` service.

---

# 10. Verify QStash Signatures (Production Step)

1. In your code (`workers.py`, etc.), ensure you have the logic for:

   ```python
   from qstash import Receiver
   
   receiver.verify(
       signature=signature,
       body=raw_body,
       url="https://<your-public-domain>/process-email"
   )
   ```

2. Confirm you have the environment variables **`QSTASH_CURRENT_SIGNING_KEY`** and **`QSTASH_NEXT_SIGNING_KEY`** set properly.

*(For local testing, you can skip signature verification, but in production, it’s highly recommended.)*

---

# 11. Production Considerations

- **Error Handling**: If GPT-4 or `o1` fails, ensure your code logs the issue or retries.
- **Security**: 
  - Verify QStash signatures.  
  - Ensure your Resend and DigitalOcean credentials are safely stored.
- **Logging**: Add robust logging (e.g., use Python’s built-in `logging` module).
- **Deployment**: For a final production deployment, host it on a cloud server or a container-based platform with SSL, etc.
- **Scaling**: If you expect high volume, consider concurrency or queue-based worker setups.

---

# 12. Summary of Keystrokes & Commands

Here’s a compact reference:

```bash
# 1) Navigate to the project
cd path/to/my_email_automation

# 2) (Optional) Create & activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optionally) create a .env file with your secrets
#     DO_ACCESS_KEY=...
#     DO_SECRET_KEY=...
#     ...
#   or export them manually in terminal

# 5) Run the Flask app
export FLASK_APP=app.py    # or set FLASK_APP=app.py on Windows
flask run --port=5000

# 6) Expose or deploy your app (e.g., with ngrok)
ngrok http 5000
# => forward traffic to http://localhost:5000

# 7) Configure QStash & Email webhooks to call:
#    https://<your-public-domain>/email-receive
#    https://<your-public-domain>/process-email
```

---

## Done!

You now have a working, **end-to-end** automated email response system that:

1. **Receives Emails** (with attachments).
2. **Uploads Attachments** to DigitalOcean Spaces.
3. **Analyzes Attachments** with GPT-4.
4. **Generates a Custom Reply** via `o1`.
5. **Sends the Response** back to the original sender via Resend.
