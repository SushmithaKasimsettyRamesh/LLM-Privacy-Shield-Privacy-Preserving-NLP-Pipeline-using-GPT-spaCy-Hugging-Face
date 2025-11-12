# app.py
import streamlit as st
import os
from llm_privacy_shield import PIIDetector, PrivacyShieldPipeline, openai_llm

# ======================================
# Streamlit UI
# ======================================
st.set_page_config(page_title="LLM Privacy Shield", page_icon="🛡️")
st.title("🛡️ LLM Privacy Shield Demo")

# Initialize detector and pipeline
detector = PIIDetector()
pipeline = PrivacyShieldPipeline(detector)

# User input
user_input = st.text_area("Enter text with PII:", height=150)

# Optional: select PII to keep masked
skip_masked_labels = st.multiselect(
    "Keep these PII types masked in the response (optional):",
    ["PERSON", "EMAIL", "PHONE"]
)

# Button to send to LLM
if st.button("Send to LLM") and user_input.strip():
    with st.spinner("Processing…"):
        result = pipeline.process(
            user_input=user_input,
            llm_function=openai_llm,
            skip_remap=skip_masked_labels,
            verbose=False
        )

    # Display results
    st.subheader("Masked Input")
    st.code(result['masked_input'])

    st.subheader("LLM Response (masked)")
    st.code(result['llm_response'])

    st.subheader("Final Output (PII remapped)")
    st.code(result['final_output'])

    if result['skipped_tokens']:
        st.info(f"Kept masked tokens: {', '.join(result['skipped_tokens'])}")
