"""
LLM Privacy Shield - Consolidated Implementation
A robust system for detecting, masking, and remapping PII when using LLMs

CORE CONCEPT:
1. User sends: "Hi, I'm John Smith at john@email.com"
2. We mask it: "Hi, I'm {{PERSON_1}} at {{EMAIL_1}}"
3. Send masked version to LLM (no privacy leak!)
4. LLM responds with tokens: "Nice to meet you {{PERSON_1}}!"
5. We remap: "Nice to meet you John Smith!"
"""

import spacy
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json

# Optional: Hugging Face transformers (install with: pip install transformers)
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Hugging Face transformers not available. Install with: pip install transformers")


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class PIIEntity:
    """Represents a detected PII entity with metadata"""
    text: str           # Original text (e.g., "john@email.com")
    label: str          # Entity type (e.g., "EMAIL")
    start: int          # Character start position
    end: int            # Character end position
    confidence: float   # Detection confidence (0-1)
    source: str         # Detection method ('spacy', 'huggingface', 'regex')


# =============================================================================
# MAIN PII DETECTOR CLASS
# =============================================================================

class PIIDetector:
    """
    Unified PII detection system using multiple methods:
    - Regex patterns for structured data (email, phone, SSN, etc.)
    - spaCy NER for names, organizations, locations
    - Hugging Face transformers for advanced NER (optional)
    """
    
    def __init__(self, use_huggingface: bool = False, use_gpu: bool = False):
        """
        Initialize the detector with specified models
        
        Args:
            use_huggingface: Whether to use HF transformers (slower but more accurate)
            use_gpu: Whether to use GPU for HF model (requires CUDA)
        """
        print("🔧 Loading PII Detection Models...")
        
        # === REGEX PATTERNS ===
        # These patterns catch structured PII that follows predictable formats
        self.regex_patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b(?:\+?1[-.●]?)?(?:\([0-9]{3}\)|[0-9]{3})[-.●]?[0-9]{3}[-.●]?[0-9]{4}\b',
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'CREDIT_CARD': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'IP_ADDRESS': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'DATE_OF_BIRTH': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            'ZIP_CODE': r'\b\d{5}(?:-\d{4})?\b'
        }
        
        # === SPACY MODEL ===
        # spaCy is great for detecting names, organizations, and locations
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded")
            self.spacy_available = True
        except OSError:
            print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            self.spacy_available = False
        
        # Map spaCy's labels to our standard labels
        self.spacy_label_map = {
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'GPE': 'LOCATION',        # Geopolitical entity (cities, countries)
            'LOC': 'LOCATION',
            'FAC': 'LOCATION',        # Facilities
            'DATE': 'DATE'
        }
        
        # === HUGGING FACE MODEL (Optional) ===
        # More accurate than spaCy but slower
        self.hf_available = False
        if use_huggingface and HF_AVAILABLE:
            try:
                model_name = "dslim/bert-base-NER"
                self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.hf_model = AutoModelForTokenClassification.from_pretrained(model_name)
                
                device = 0 if use_gpu else -1
                self.hf_pipeline = pipeline(
                    "ner",
                    model=self.hf_model,
                    tokenizer=self.hf_tokenizer,
                    aggregation_strategy="simple",
                    device=device
                )
                print("✅ Hugging Face BERT-NER model loaded")
                self.hf_available = True
            except Exception as e:
                print(f"⚠️  Could not load Hugging Face model: {e}")
                self.hf_available = False
        
        # Map HF's labels to our standard labels
        self.hf_label_map = {
            'PER': 'PERSON',
            'ORG': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'MISC': 'MISC'
        }
        
        print("✅ PIIDetector initialized\n")
    
    # =========================================================================
    # DETECTION METHODS
    # =========================================================================
    
    def _detect_regex(self, text: str) -> List[PIIEntity]:
        """
        Detect PII using regex patterns
        
        This catches things like emails, phone numbers, SSNs that have
        predictable formats
        """
        entities = []
        
        for label, pattern in self.regex_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(PIIEntity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,  # High confidence for regex matches
                    source='regex'
                ))
        
        return entities
    
    def _detect_spacy(self, text: str) -> List[PIIEntity]:
        """
        Detect PII using spaCy NER
        
        spaCy is good at finding names, organizations, and locations
        in natural text
        """
        if not self.spacy_available:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.spacy_label_map:
                entities.append(PIIEntity(
                    text=ent.text,
                    label=self.spacy_label_map[ent.label_],
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,  # spaCy doesn't provide confidence scores
                    source='spacy'
                ))
        
        return entities
    
    def _detect_huggingface(self, text: str) -> List[PIIEntity]:
        """
        Detect PII using Hugging Face transformer model
        
        More accurate than spaCy but slower. Good for production use.
        """
        if not self.hf_available:
            return []
        
        try:
            predictions = self.hf_pipeline(text)
            entities = []
            
            for pred in predictions:
                # Extract and map label
                hf_label = pred.get('entity_group', pred.get('entity', ''))
                clean_label = hf_label.replace('B-', '').replace('I-', '')
                standard_label = self.hf_label_map.get(clean_label, clean_label)
                
                entities.append(PIIEntity(
                    text=pred['word'],
                    label=standard_label,
                    start=pred['start'],
                    end=pred['end'],
                    confidence=pred['score'],
                    source='huggingface'
                ))
            
            return entities
        
        except Exception as e:
            print(f"⚠️  HF detection error: {e}")
            return []
    
    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Remove overlapping entities, keeping the highest confidence one
        
        Example: If both spaCy and HF detect "John Smith", keep the one
        with higher confidence.
        
        Strategy:
        1. Sort by confidence (highest first)
        2. Keep entity only if it doesn't overlap with any already kept
        3. Re-sort by position for output
        """
        if not entities:
            return []
        
        # Sort by confidence (highest first)
        entities.sort(key=lambda x: -x.confidence)
        
        deduplicated = []
        
        for entity in entities:
            # Check if this entity overlaps with any already accepted
            overlaps = False
            for accepted in deduplicated:
                if self._entities_overlap(entity, accepted):
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        # Sort by start position for final output
        deduplicated.sort(key=lambda x: x.start)
        
        return deduplicated
    
    @staticmethod
    def _entities_overlap(e1: PIIEntity, e2: PIIEntity) -> bool:
        """
        Check if two entities overlap in the text
        
        Example:
        - "John" (0-4) and "Smith" (5-10) → No overlap
        - "John" (0-4) and "John Smith" (0-10) → Overlap!
        """
        return not (e1.end <= e2.start or e2.end <= e1.start)
    
    def detect_pii(self, text: str, 
                   use_spacy: bool = True,
                   use_hf: bool = True,
                   use_regex: bool = True) -> List[PIIEntity]:
        """
        Detect all PII in text using enabled methods
        
        This is the main entry point for PII detection.
        
        Args:
            text: Input text to analyze
            use_spacy: Enable spaCy detection
            use_hf: Enable Hugging Face detection
            use_regex: Enable regex detection
        
        Returns:
            List of deduplicated PIIEntity objects, sorted by position
        """
        all_entities = []
        
        # Collect entities from all enabled sources
        if use_regex:
            all_entities.extend(self._detect_regex(text))
        
        if use_spacy and self.spacy_available:
            all_entities.extend(self._detect_spacy(text))
        
        if use_hf and self.hf_available:
            all_entities.extend(self._detect_huggingface(text))
        
        # Remove duplicates/overlaps (keep highest confidence)
        final_entities = self._deduplicate_entities(all_entities)
        
        return final_entities
    
    # =========================================================================
    # ANONYMIZATION & DEANONYMIZATION
    # =========================================================================
    
    def anonymize(self, text: str, **detection_kwargs) -> Tuple[str, Dict[str, str], List[PIIEntity]]:
        """
        Replace PII with tokens like {{PERSON_1}}, {{EMAIL_1}}, etc.
        
        IMPORTANT: We replace from END to START to preserve character positions!
        
        Example:
        Input:  "Hi, I'm John Smith at john@email.com"
        Output: "Hi, I'm {{PERSON_1}} at {{EMAIL_1}}"
        
        Args:
            text: Input text to anonymize
            **detection_kwargs: Arguments to pass to detect_pii()
        
        Returns:
            tuple: (anonymized_text, token_to_original_mapping, detected_entities)
        """
        # Step 1: Detect all PII
        entities = self.detect_pii(text, **detection_kwargs)
        
        if not entities:
            return text, {}, []
        
        # Step 2: Create token mapping
        token_map = {}
        label_counters = defaultdict(int)  # Track count per label type
        
        # Step 3: Replace entities from END to START
        # This is critical! If we go start to end, positions shift as we replace
        anonymized = text
        for entity in sorted(entities, key=lambda x: x.start, reverse=True):
            # Generate unique token for this entity
            label_counters[entity.label] += 1
            token = f"{{{{{entity.label}_{label_counters[entity.label]}}}}}"
            
            # Store the mapping: token -> original value
            token_map[token] = entity.text
            
            # Replace in text using exact positions
            anonymized = anonymized[:entity.start] + token + anonymized[entity.end:]
        
        return anonymized, token_map, entities
    
    def deanonymize(self, text: str, token_map: Dict[str, str], 
                    skip_tokens: Optional[List[str]] = None) -> str:
        """
        Restore original values from anonymized text
        
        This takes the LLM's response (which contains tokens) and replaces
        them back with the original PII values.
        
        Example:
        Input:  "Nice to meet you {{PERSON_1}}!"
        Output: "Nice to meet you John Smith!"
        
        Args:
            text: Anonymized text with tokens
            token_map: Mapping from tokens to original values
            skip_tokens: List of specific tokens to NOT replace (keep masked)
        
        Returns:
            Text with original values restored
        """
        if skip_tokens is None:
            skip_tokens = []
        
        result = text
        for token, original in token_map.items():
            # Only replace if not in skip list
            if token not in skip_tokens:
                result = result.replace(token, original)
        
        return result
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of text with detection statistics
        
        Useful for debugging and understanding what was detected.
        
        Returns detailed breakdown of detected PII
        """
        entities = self.detect_pii(text)
        
        # Group by label (PERSON, EMAIL, etc.)
        by_label = defaultdict(list)
        for entity in entities:
            by_label[entity.label].append(entity)
        
        # Group by source (spacy, regex, etc.)
        by_source = defaultdict(list)
        for entity in entities:
            by_source[entity.source].append(entity)
        
        return {
            'text': text,
            'total_entities': len(entities),
            'entities': entities,
            'by_label': dict(by_label),
            'by_source': dict(by_source),
            'label_counts': {label: len(ents) for label, ents in by_label.items()},
            'source_counts': {source: len(ents) for source, ents in by_source.items()}
        }


# =============================================================================
# LLM PIPELINE
# =============================================================================

class PrivacyShieldPipeline:
    """
    Complete pipeline: User Input → Mask PII → LLM → Remap PII → Final Output
    
    This orchestrates the entire flow:
    1. User sends text with PII
    2. We detect and mask PII
    3. Send masked text to LLM
    4. LLM responds (possibly using tokens)
    5. We remap tokens back to original values
    6. Return final output to user
    """
    
    def __init__(self, detector: PIIDetector):
        """
        Initialize pipeline with a PII detector
        
        Args:
            detector: PIIDetector instance
        """
        self.detector = detector
        self.conversation_memory = {}  # Track PII across entire conversation
    
    def process(self, user_input: str, 
                llm_function: callable,
                skip_remap: Optional[List[str]] = None,
                verbose: bool = True) -> Dict[str, Any]:
        """
        Complete privacy-preserving LLM pipeline
        
        This is the main function you'll call to process user input through
        the LLM while protecting their privacy.
        
        Args:
            user_input: User's original input (contains PII)
            llm_function: Function that takes masked text and returns LLM response
                         Should have signature: def my_llm(text: str) -> str
            skip_remap: PII label types to keep masked even after LLM responds
                       Example: ['EMAIL', 'PHONE'] to keep those masked
            verbose: Print detailed logs of each stage
        
        Returns:
            Dictionary with all pipeline stages and results
        """
        if verbose:
            print("🛡️  LLM Privacy Shield - Processing Request")
            print("=" * 60)
        
        # === STAGE 1: DETECT & MASK PII ===
        if verbose:
            print("\n🔍 Stage 1: Detecting and masking PII...")
        
        # Anonymize the user's input
        masked_text, token_map, entities = self.detector.anonymize(user_input)
        
        # Update conversation memory (track PII across multiple turns)
        for token, original in token_map.items():
            self.conversation_memory[token] = original
        
        if verbose:
            print(f"Original:  {user_input}")
            print(f"Masked:    {masked_text}")
            print(f"Detected:  {len(entities)} PII entities")
            for entity in entities:
                print(f"  • {entity.text} → {entity.label} ({entity.source}, {entity.confidence:.2f})")
        
        # === STAGE 2: CALL LLM ===
        if verbose:
            print("\n🤖 Stage 2: Calling LLM with masked text...")
        
        try:
            # Call the LLM with the MASKED text (no PII exposed!)
            llm_response = llm_function(masked_text)
        except Exception as e:
            if verbose:
                print(f"❌ LLM Error: {e}")
            return {
                'error': str(e),
                'original_input': user_input,
                'masked_input': masked_text
            }
        
        if verbose:
            print(f"LLM Response: {llm_response}")
        
        # === STAGE 3: REMAP ORIGINAL VALUES ===
        if verbose:
            print("\n🔄 Stage 3: Remapping original values...")
        
        # Determine which tokens to skip (keep masked)
        skip_tokens = []
        if skip_remap:
            for token, original in token_map.items():
                # Extract label from token (e.g., "{{EMAIL_1}}" → "EMAIL")
                label = token.split('_')[0].replace('{', '').replace('}', '')
                if label in skip_remap:
                    skip_tokens.append(token)
        
        # Restore original PII values in the LLM's response
        final_output = self.detector.deanonymize(llm_response, token_map, skip_tokens)
        
        if verbose:
            print(f"Final Output: {final_output}")
            if skip_tokens:
                print(f"Kept Masked: {skip_tokens}")
            print("\n" + "=" * 60)
        
        # Return all the information about this processing
        return {
            'original_input': user_input,
            'masked_input': masked_text,
            'detected_entities': entities,
            'token_mapping': token_map,
            'llm_response': llm_response,
            'final_output': final_output,
            'skipped_tokens': skip_tokens,
            'conversation_memory': dict(self.conversation_memory)
        }


# =============================================================================
# DEMO & TESTING
# =============================================================================

def openai_llm(masked_text: str) -> str:
    """
    Mock LLM for testing - simulates how GPT might respond
    
    This is just for demonstration. In production, you'd replace this
    with actual OpenAI API calls, Anthropic API, etc.
    """
    # Extract tokens from the masked text
    tokens = re.findall(r'\{\{[^}]+\}\}', masked_text)
    
    # Generate a response that might include some of the tokens
    if "email" in masked_text.lower():
        name_token = next((t for t in tokens if 'PERSON' in t), '{{PERSON_1}}')
        email_token = next((t for t in tokens if 'EMAIL' in t), '{{EMAIL_1}}')
        return f"I can help you draft an email. {name_token} can be reached at {email_token}."
    
    elif "meeting" in masked_text.lower():
        return f"I'd be happy to help schedule a meeting with {tokens[0] if tokens else 'the person'}."
    
    elif "call" in masked_text.lower() or "phone" in masked_text.lower():
        phone_token = next((t for t in tokens if 'PHONE' in t), None)
        if phone_token:
            return f"You can reach them at {phone_token}."
        return "I can help you with that phone call."
    
    else:
        response = "I understand you're asking about "
        if tokens:
            response += f"{', '.join(tokens[:2])}. "
        response += "I'll help you with that."
        return response


'''def run_demo():
    """Run interactive demo with test cases"""
    print("🚀 LLM Privacy Shield - Demo\n")
    
    # Initialize detector (set use_huggingface=True if you have it installed)
    detector = PIIDetector(use_huggingface=False)
    pipeline = PrivacyShieldPipeline(detector)
    
    # Test cases demonstrating different types of PII
    test_cases = [
        "Hi, I'm John Smith. Email me at john.smith@gmail.com or call 555-123-4567.",
        "I work at Google Inc. in New York. My SSN is 123-45-6789.",
        "Contact Sarah Johnson at sarah@company.org about the project.",
        "My credit card 4532-1234-5678-9012 was charged twice yesterday.",
        "Call Dr. Martinez at (555) 987-6543 or visit 123 Main Street, Boston, MA 02101.",
    ]
    
    print("📋 Running Test Cases:\n")
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case #{i}")
        print(f"{'='*60}")
        
        result = pipeline.process(
            user_input=test_input,
            llm_function=openai_llm,
            verbose=True
        )
    
    # Interactive mode
    print("\n\n🎮 Interactive Mode (type 'quit' to exit)")
    print("-" * 60)
    print("\nTry entering text with PII like:")
    print("  • 'My email is john@example.com'")
    print("  • 'Call me at 555-1234'")
    print("  • 'I'm Sarah from Microsoft'\n")
   
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        result = pipeline.process(
            user_input=user_input,
            llm_function=openai_llm,
            verbose=True
        )


if __name__ == "__main__":
    run_demo()

'''