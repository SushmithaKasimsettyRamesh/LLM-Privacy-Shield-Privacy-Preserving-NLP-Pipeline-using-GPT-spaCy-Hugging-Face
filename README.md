# LLM-Privacy-Shield-Privacy-Preserving-NLP-Pipeline-using-GPT-spaCy-Hugging-Face
Developed an anonymization engine that detects and masks sensitive PII (names, orgs, emails) using spaCy and transformer-based NER (Hugging Face), enabling secure GPT interactions.

Integrated OpenAI API to process masked input and remapped outputs with consistent token tracking and reversible logic.

Implemented memory-aware token mapping to ensure repeat entities (e.g., same name) are assigned persistent tokens across user input.

Designed selective remapping logic, allowing developers to control which PII types remain visible post-processing.

Modularized the entire system into a callable function (anonymize_and_run(text)) for future deployment as a Streamlit or FastAPI application.

Tech Stack: Python · spaCy · Hugging Face Transformers · OpenAI API · Pydantic (planned) · FastAPI (next) · Streamlit · Regex · JSON

