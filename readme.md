# mail2GPT: AI Email Analysis and Response Pipeline with GPT-4 and o1 Pro

A scalable email automation system that combines GPT-4's analysis capabilities with o1 Pro's advanced response generation to deliver intelligent, context-aware email responses. The system processes incoming emails and attachments, analyzes them using GPT-4, and leverages o1 Pro's sophisticated models to generate tailored, professional responses.

## Core Components

1. **Initial Analysis**: GPT-4 processes attachments and email content
2. **Advanced Processing**: o1 Pro analyzes the GPT-4 output and email context
3. **Response Generation**: o1 Pro generates tailored, contextually appropriate responses
4. **Delivery**: Automated response delivery via Resend

## System Architecture

```
                 +------------------------+
  Email/Webhook->|   Flask Endpoint      |
                 |   /email-receive      |
                 +---------+-------------+
                           |
                           v
          +----------------+----------------+
          |       Flask App                |
          |  - Decodes attachments        |
          |  - Uploads to DO Spaces       |
          |  - Publishes to QStash        |
          +----------------+---------------+
                           |
                     QStash Queue
                           |
                           v
              +-----------+-----------+
              |     Worker Process    |
              |  1. GPT-4 analysis   |
              |  2. o1 Pro process   |
              |  3. Email response   |
              +---------------------+
```

## Prerequisites

- Python 3.7+
- Flask web framework
- API access:
  - o1 Pro API credentials
  - OpenAI GPT-4 API key
  - DigitalOcean Spaces credentials
  - QStash account
  - Resend API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jimmc414/mail2GPT.git
cd mail2GPT
```

2. Set up Python environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables in `.env`:
```
O1_API_KEY=your_o1_pro_api_key
DO_ACCESS_KEY=your_digitalocean_access_key
DO_SECRET_KEY=your_digitalocean_secret_key
DO_BUCKET_NAME=your-bucket-name
OPENAI_API_KEY=your_openai_api_key
RESEND_API_KEY=your_resend_api_key
QSTASH_TOKEN=your_qstash_token
QSTASH_CURRENT_SIGNING_KEY=your_current_signing_key
QSTASH_NEXT_SIGNING_KEY=your_next_signing_key
```

## Processing Flow

1. **Email Reception**
   - System receives email via webhook
   - Attachments are extracted and decoded
   - Files are securely stored in DO Spaces

2. **Initial Analysis**
   - GPT-4 analyzes attachments and email content
   - Generates structured analysis and insights
   - Prepares data for o1 Pro processing

3. **o1 Pro Processing**
   - Receives GPT-4 analysis and email context
   - Applies advanced contextual understanding
   - Generates professional, tailored responses
   - Handles complex business logic and rules

4. **Response Delivery**
   - Final response formatted and prepared
   - Sent via Resend email service
   - Delivery tracking and confirmation

## Project Structure

```
mail2GPT/
├── app.py              # Main Flask application
├── workers.py          # QStash worker process
├── do_spaces.py        # DO Spaces integration
├── gpt_processing.py   # GPT-4 analysis
├── readme.md           # readme
├── architecture.md     # architecture document
├── implementation.md   # step by step implementation guide
├── overview.md         # project description
├── prompt.md           # initial buidl prompt
├── todos.md            # todos
├── code_review.md      # Code review
├── o1_client.py        # o1 Pro API integration
├── resend_client.py    # Email sending
└── requirements.txt    # Dependencies
```

## Key Components

### o1 Pro Integration (`o1_client.py`)
- Processes GPT-4 analysis results
- Applies business logic and rules
- Generates contextual responses
- Handles complex decision making

### Email Reception (`app.py`)
- Webhook endpoint for incoming emails
- Attachment processing
- QStash queue integration

### Analysis Pipeline (`workers.py`)
- Coordinates GPT-4 and o1 Pro processing
- Manages asynchronous workflows
- Handles response generation

## Configuration Options

Key configuration areas:

- o1 Pro API settings and preferences
- GPT-4 analysis parameters
- QStash queue settings
- Email handling preferences
- Storage configurations

## Security Features

- Secure credential management
- Encrypted file storage
- QStash signature verification
- API authentication
- Rate limiting and monitoring

## Deployment Guidelines

1. **Development Environment**
   ```bash
   python app.py
   ```

2. **Production Deployment**
   - Use production WSGI server
   - Configure SSL/TLS
   - Set up monitoring
   - Enable error tracking
   - Configure backup systems

## Performance Optimization

- Asynchronous processing via QStash
- Efficient file handling
- Response caching where appropriate
- Concurrent processing support
- Error recovery mechanisms

## Testing

```bash
# Run test suite
pytest

# Test o1 Pro integration
pytest tests/test_o1_integration.py

# Test email processing
pytest tests/test_email_processing.py
```

## Monitoring and Logging

The system includes comprehensive logging for:
- o1 Pro API interactions
- GPT-4 analysis results
- Email processing status
- Error tracking and alerts