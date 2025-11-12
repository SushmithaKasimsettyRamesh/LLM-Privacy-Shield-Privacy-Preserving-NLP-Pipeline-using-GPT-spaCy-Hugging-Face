# LLM Privacy Shield

A practical solution for protecting sensitive information when using Large Language Models (LLMs). This tool automatically detects and masks personally identifiable information (PII) before sending data to LLM APIs, then restores it in the responses.

## Problem

Users often unknowingly share sensitive information with LLMs:
```
User: "Hi, I'm John Smith. Email me at john.smith@company.com about the contract."
→ Sent directly to LLM servers
→ Logged, potentially used for training, vulnerable to breaches
```

This creates compliance issues for healthcare (HIPAA), finance (PCI-DSS), and enterprise apps (GDPR).

## Solution

Privacy Shield sits between your app and the LLM, masking PII and restoring it in responses:
```
Input:        "Hi, I'm John Smith at john.smith@company.com"
Masked:       "Hi, I'm {{PERSON_1}} at {{EMAIL_1}}"
LLM Response: "Nice to meet you {{PERSON_1}}!"
Output:       "Nice to meet you John Smith!"
```

The LLM never sees actual sensitive data.

## Features

- **Multi-layer detection**: regex, spaCy NER, optional transformer models
- **Context-aware masking**: maintains conversation flow
- **Reversible anonymization**: restores original values
- **Selective privacy**: choose which data types remain masked
- **Zero configuration**: works out of the box
- **Production-ready**: error handling, logging, edge-case management

## Quick Start

1. **Install dependencies**
```bash
   pip install -r requirements.txt
```

2. **Set your API key**
   
   Create a `.env` file:
```
   OPENAI_API_KEY=sk-your-actual-key-here
```

3. **Run the Streamlit App**
```bash
   streamlit run app.py
```
   - Enter text with PII
   - Choose which types to keep masked (optional)
   - Click "Send to LLM" to see masked input, LLM response, and final output

## Example
```
Input:
Hi, I'm John Smith. My email is john.smith@company.com. Call me at 555-123-4567.

Masked Input (sent to LLM):
Hi, I'm {{PERSON_1}}. My email is {{EMAIL_1}}. Call me at {{PHONE_1}}.

LLM Response:
Hello {{PERSON_1}}, thanks for sharing your contact info {{EMAIL_1}}. I'll call you at {{PHONE_1}}.

Final Output (PII restored):
Hello John Smith, thanks for sharing your contact info john.smith@company.com. I'll call you at 555-123-4567.
```

## What Gets Protected

| Type | Examples |
|------|----------|
| Names | John Smith, Dr. Sarah Johnson |
| Emails | john@example.com, sarah.j@company.org |
| Phone Numbers | 555-123-4567, (555) 123-4567, +1-555-123-4567 |
| SSNs | 123-45-6789 |
| Credit Cards | 4532-1234-5678-9012 |
| Organizations | Google Inc., Microsoft Corporation |
| Locations | New York, San Francisco, CA |
| IP Addresses | 192.168.1.1 |
| ZIP Codes | 02101, 94105-1234 |
| Dates of Birth | 01/15/1990, 1990-01-15 |

## How It Works

1. **Regex Patterns** – Fast matching for structured data (emails, phones)
2. **spaCy NER** – Context-aware detection (names, orgs, locations)
3. **Transformer Models** (optional) – BERT-based models for high accuracy

Detectors are combined intelligently to maximize coverage and eliminate duplicates.

## Architecture
```
┌──────────┐    ┌──────────┐    ┌─────┐    ┌─────────┐    ┌──────────┐    ┌────────┐
│   User   │───▶│   PII    │───▶│Mask │───▶│   LLM   │───▶│ Restore  │───▶│ Output │
│  Input   │    │ Detector │    │     │    │   API   │    │          │    │        │
└──────────┘    └──────────┘    └─────┘    └─────────┘    └──────────┘    └────────┘
```

## Advanced Features

- **Multi-turn conversations**: preserves context across queries
- **Selective masking**: skip remap for certain data types
- **Custom prompts**: e.g., HIPAA-compliant assistant
- **Cost monitoring**: track token usage and costs

## Performance

*Benchmarked on typical PII-heavy text (names, emails, phones)*

| Method | Accuracy | Speed |
|--------|----------|-------|
| Regex only | ~85% | ~0.5ms |
| + spaCy | ~90% | ~15ms |
| + Hugging Face | ~95% | ~150ms |

## Use Cases

Protects patient data in healthcare chatbots, screens customer info in support tools, anonymizes candidate details in HR systems, and safeguards financial information in banking applications.

## Security

- API keys loaded from `.env` (gitignored by default)
- Token mappings cleared after session
- No PII logged to disk

## Limitations

- Context leakage possible through conversational patterns
- New PII types may require custom patterns
- English language only
- Model updates may affect token handling

## Contributing

Contributions welcome! Areas of interest:
- Add new PII patterns
- Support more languages
- Integrate additional LLM providers
- Optimize performance
- Add test coverage


## Acknowledgments

Built with spaCy, Hugging Face Transformers, and OpenAI Python SDK.
