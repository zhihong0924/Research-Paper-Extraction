import os
import time
import re
import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError, APIStatusError
from pathlib import Path


# ======================
# CONFIG
# ======================
MEDIA_DIR = Path(__file__).parent / "media"
OUTPUT_DIR = Path(__file__).parent / "output"

CHUNK_MODEL_POOL = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile"
]

SYNTHESIS_MODEL_POOL = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile"
]

MAX_OUTPUT_TOKENS = 300
SYNTHESIS_OUTPUT_TOKENS = 700

SAFE_TPM = 10000
REQUEST_DELAY = 1.5
MODEL_COOLDOWN = 300  # seconds
CHUNK_CHAR_LIMIT = 8000  # ~2000 tokens

# ======================
# UTILS
# ======================
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"References.*", "", text, flags=re.S | re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def safe_filename(name):
    return re.sub(r"[^\w\-_\. ]", "_", name)

# ======================
# MAIN CLASS
# ======================
class ResearchPaperSummarizer:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.tokens_used_this_minute = 0
        self.minute_start = time.time()
        self.model_cooldowns = {}
        self.chunk_prompt_template = self._load_prompt_template(Path(__file__).parent / "prompt_read")
        self.synthesis_prompt_template = self._load_prompt_template(Path(__file__).parent / "prompt_synthesis")

    # ----------------------
    # Prompt Templates
    # ----------------------
    def _load_prompt_template(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    # ----------------------
    # Rate Limiting
    # ----------------------
    def _throttle(self, estimated_tokens):
        now = time.time()
        if now - self.minute_start >= 60:
            self.tokens_used_this_minute = 0
            self.minute_start = now

        if self.tokens_used_this_minute + estimated_tokens > SAFE_TPM:
            sleep_time = 60 - (now - self.minute_start)
            print(f"⏳ TPM limit — sleeping {sleep_time:.1f}s")
            time.sleep(max(1, sleep_time))
            self.tokens_used_this_minute = 0
            self.minute_start = time.time()

        self.tokens_used_this_minute += estimated_tokens

    # ----------------------
    # PDF Extraction
    # ----------------------
    def extract_pdf_text(self, path):
        print("📄 Extracting PDF text...")
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(
                    x_tolerance=2,
                    y_tolerance=2,
                    layout=True
                )
                if page_text:
                    text += page_text + "\n"

        return clean_text(text)

    # ----------------------
    # Chunking
    # ----------------------
    def chunk_text(self, text):
        return [
            text[i:i + CHUNK_CHAR_LIMIT]
            for i in range(0, len(text), CHUNK_CHAR_LIMIT)
        ]

    # ----------------------
    # LLM Call (Failover + Cooldown)
    # ----------------------
    def _call_llm(self, prompt, max_tokens, model_pool):
        est_tokens = estimate_tokens(prompt) + max_tokens
        self._throttle(est_tokens)

        last_error = None

        for model in model_pool:
            now = time.time()

            if model in self.model_cooldowns:
                if now < self.model_cooldowns[model]:
                    continue
                else:
                    del self.model_cooldowns[model]

            try:
                print(f"⚡ Using model: {model}")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )

                time.sleep(REQUEST_DELAY)
                return response.choices[0].message.content

            except (RateLimitError, APIStatusError) as e:
                print(f"🚫 Model {model} blocked")
                self.model_cooldowns[model] = time.time() + MODEL_COOLDOWN
                last_error = e

        raise RuntimeError(f"All models failed. Last error: {last_error}")

    # ----------------------
    # Chunk Summarization
    # ----------------------
    def summarize_chunk(self, chunk, idx, total):
        prompt = self.chunk_prompt_template.format(
            part_num=idx,
            total_parts=total,
            chunk_text=chunk
        )
        return self._call_llm(
            prompt,
            MAX_OUTPUT_TOKENS,
            CHUNK_MODEL_POOL
        )

    # ----------------------
    # Final Synthesis
    # ----------------------
    def synthesize_summary(self, partial_summaries):
        prompt = self.synthesis_prompt_template.format(
            summaries_text="\n\n".join(partial_summaries)
        )
        return self._call_llm(
            prompt,
            SYNTHESIS_OUTPUT_TOKENS,
            SYNTHESIS_MODEL_POOL
        )

    # ----------------------
    # Pipeline
    # ----------------------
    def run(self, pdf_path):
        text = self.extract_pdf_text(pdf_path)
        chunks = self.chunk_text(text)

        print(f"🔹 Total chunks: {len(chunks)}")
        partial_summaries = []

        for i, chunk in enumerate(chunks, 1):
            print(f"🧠 Summarizing chunk {i}/{len(chunks)}...")
            summary = self.summarize_chunk(chunk, i, len(chunks))
            partial_summaries.append(summary)

        print("📊 Synthesizing final summary...")
        return self.synthesize_summary(partial_summaries)

# ======================
# RUN
# ======================
if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / "env")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groq_api_key = os.getenv("GROQ_API_KEY")
    summarizer = ResearchPaperSummarizer(groq_api_key)

    for file in os.listdir(MEDIA_DIR):
        if file.lower().endswith(".pdf"):
            print(f"\n=== PROCESSING FILE: {file} ===\n")

            pdf_path = os.path.join(MEDIA_DIR, file)
            output_name = safe_filename(file.replace(".pdf", "_summary.txt"))
            output_path = os.path.join(OUTPUT_DIR, output_name)

            try:
                result = summarizer.run(pdf_path)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(f"SUMMARY FOR: {file}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(result)

                print(f"✅ Saved summary to: {output_path}")

            except Exception as e:
                print(f"❌ Failed to process {file}: {e}")
