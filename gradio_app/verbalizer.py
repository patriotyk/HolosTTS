import os
import time
import ctranslate2
from transformers import M2M100Tokenizer
from huggingface_hub import snapshot_download


class Verbalizer():
    MODEL_PATH = os.getenv("MODEL_PATH", "skypro1111/m2m100-ukr-verbalization-ct2")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "skypro1111/m2m100-ukr-verbalization")
    def __init__(self):
        
        print("\nInitializing CTranslate2 model and tokenizer...")

    
        # Download the model from HuggingFace Hub
        local_model_path = snapshot_download(
            repo_id=self.MODEL_PATH,
            allow_patterns=["*.bin", "*.json", "tokenizer.json", "vocab.json"],
        )
        
        self.translator = ctranslate2.Translator(
            local_model_path,
            device='cpu',
            compute_type="int8",
        )
        
        # Load tokenizer
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.TOKENIZER_PATH)
        self.tokenizer.src_lang = "uk"
        
    def process_text(self, text: str):
        """Process a single text input using the CTranslate2 model."""
        start_time = time.time()
        
        # Tokenize input
        source = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        target_prefix = [self.tokenizer.lang_code_to_token["uk"]]
        
        # Run inference
        results = self.translator.translate_batch(
            [source],
            target_prefix=[target_prefix],
            beam_size=1,
            num_hypotheses=1,
            use_vmap=True,
        )
        
        # Get target tokens and decode
        target = results[0].hypotheses[0][1:]  # Remove language token
        output = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target))
        
        inference_time = time.time() - start_time
        return output, inference_time

