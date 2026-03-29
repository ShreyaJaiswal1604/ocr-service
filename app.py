"""
InkLens OCR Service — LightOnOCR backend with image enhancement.

Usage:
    pip install -r requirements.txt
    python app.py
"""

import io
import json
import logging
import os
import re
import time
import threading

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from PIL import Image

from enhance import enhance_for_ocr

# ── Config ───────────────────────────────────────────────────────────────────────

PORT = int(os.getenv("PORT", "8090"))
HOST = os.getenv("HOST", "0.0.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inklens")

# ── Model (lazy loading) ────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

_model = {}

def get_lighton():
    if "proc" not in _model:
        logger.info("📦 Loading LightOnOCR-2-1B...")
        t = time.time()
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
        _model["proc"] = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")
        mdl = LightOnOcrForConditionalGeneration.from_pretrained("lightonai/LightOnOCR-2-1B")
        mdl.eval()
        if DEVICE != "cpu":
            mdl = mdl.to(DEVICE)
        _model["mdl"] = mdl
        logger.info(f"✅ LightOnOCR loaded on {DEVICE} in {time.time()-t:.1f}s")
    return _model["proc"], _model["mdl"]


# ── OCR ──────────────────────────────────────────────────────────────────────────

def run_ocr(image: Image.Image) -> str:
    proc, mdl = get_lighton()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<ocr>"}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(images=image, text=prompt, return_tensors="pt")
    if DEVICE != "cpu":
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        generated_ids = mdl.generate(**inputs, max_new_tokens=512)

    new_ids = generated_ids[:, prompt_len:]
    raw = proc.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    cleaned = _clean_latex(raw)
    return raw, cleaned


def stream_ocr_generator(image: Image.Image):
    from transformers import TextIteratorStreamer
    proc, mdl = get_lighton()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<ocr>"}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(images=image, text=prompt, return_tensors="pt")
    if DEVICE != "cpu":
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    streamer = TextIteratorStreamer(proc.tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = {**inputs, "max_new_tokens": 512, "streamer": streamer}

    start = time.time()
    thread = threading.Thread(target=lambda: _gen(mdl, kwargs))
    thread.start()

    raw = ""
    for chunk in streamer:
        if chunk:
            raw += chunk

    thread.join()
    elapsed_ms = (time.time() - start) * 1000
    cleaned = _clean_latex(raw)
    return raw, cleaned, elapsed_ms


def _gen(mdl, kwargs):
    with torch.no_grad():
        mdl.generate(**kwargs)


# ── LaTeX Cleanup ────────────────────────────────────────────────────────────────

def _clean_latex(text: str) -> str:
    """Strip LaTeX/Markdown from LightOnOCR output, return plain text."""
    text = re.sub(r'\\(?:overline|underline|hat|dot|vec|bar|tilde|widetilde|widehat)\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:text|mathrm|mathbf|mathit|mathcal|mathbb|mathsf|textbf|textit|operatorname)\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', text)
    text = re.sub(r'\$\$([^$]*)\$\$', r'\1', text)
    text = re.sub(r'\$([^$]*)\$', r'\1', text)
    text = re.sub(r'\\(?:quad|qquad|,|;|!|enspace|thinspace)', ' ', text)
    text = text.replace('\\\\', '\n')
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '').replace('$', '')
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*([^*]*)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]*)\*', r'\1', text)
    text = text.replace('|', ' ')
    # Strip HTML tags (tables, divs, etc.)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Invoice Number Extraction ────────────────────────────────────────────────────

def extract_invoice_number(text: str) -> str | None:
    """
    Run heuristic regex patterns on OCR text to find an invoice number.
    Handles OCR noise: extra spaces between letters, slight misspellings.
    """
    # Step 1: Normalize the text for better matching
    # Remove spaces between individual letters: "I N V O I C E" → "INVOICE"
    normalized = _collapse_spaced_letters(text)
    logger.info(f"   🔎 Searching for invoice # in: {repr(normalized[:200])}")

    # Step 2: Try patterns on both original and normalized text
    for search_text in [normalized, text]:
        result = _try_patterns(search_text)
        if result:
            return result

    # Step 3: Last resort — just find any number that follows invoice-like words
    fallback = re.search(
        r'(?:inv(?:oice)?|receipt|bill|slip)\s*(?:no\.?|number|num|#)?\s*[:\-]?\s*([A-Za-z0-9/\-_]{3,})',
        normalized, re.IGNORECASE
    )
    if fallback:
        extracted = fallback.group(1)
        if re.search(r'\d', extracted):
            logger.info(f"   🔢 Invoice number found via fallback: {extracted}")
            return extracted

    return None


def _collapse_spaced_letters(text: str) -> str:
    """
    Collapse single-letter sequences with spaces: "I N V O I C E" → "INVOICE"
    Also handles "I N V 1 2 3" → "INV 123"
    """
    result = text
    # Find sequences of single chars separated by spaces (3+ in a row)
    # e.g. "I N V O I C E" or "1 2 3 4 5"
    def collapse(m):
        chars = m.group(0).replace(' ', '')
        return chars

    # Match: word-boundary, then (single-char space)+ single-char, word-boundary
    # At least 3 single chars in a row
    result = re.sub(r'\b([A-Za-z0-9] ){2,}[A-Za-z0-9]\b', collapse, result)
    return result


def _try_patterns(text: str) -> str | None:
    """Try all invoice number patterns against text."""
    patterns = [
        (r"INV#\s*:?\s*([A-Za-z0-9/\-]+)", "INV#"),
        (r"INV:\s*([^\s]+)", "INV:"),
        (r"INV-NO\.\s*([^\s]+)", "INV-NO"),
        (r"INV\s*\.?\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "INV NO"),
        (r"INVOICE\s*\.?\s*NO\.?\s*[:\-]?\s*([A-Za-z0-9/\-_]+)", "INVOICE NO"),
        (r"INVOICE\s*\.?\s*NUMBER\s*[:\-]?\s*([A-Za-z0-9/\-_]+)", "INVOICE NUMBER"),
        (r"INVOICE\s*#\s*:?[\s]*([A-Za-z0-9/\-]+)", "INVOICE#"),
        (r"INVOICE\s*[:\-]\s*([A-Za-z0-9/\-_]+)", "INVOICE:"),
        (r"INVOICE\s+([A-Za-z0-9/\-_]+\d[A-Za-z0-9/\-_]*)", "INVOICE <num>"),
        (r"SLIP\s*(?:NO\.?|NUMBER)?\s*[^A-Za-z0-9]*\s*([A-Za-z0-9/\-]+)", "SLIP"),
        (r"RECEIPT\s*(?:NO\.?|NUMBER)?\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "RECEIPT"),
        (r"BILL\s*(?:NO\.?|NUMBER)?\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "BILL"),
        (r"CB#\s*:\s*([A-Za-z0-9/\-]+)", "CB#"),
        (r"C/N\s*NO\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "C/N NO"),
        (r"TRANSACTION\s*(?:NO\.?|NUMBER)?\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "TRANSACTION"),
        (r"TRN\s*[:\-]?\s*([A-Za-z0-9/\-]+)", "TRN"),
        (r"RCPT#\s*:\s*([A-Za-z0-9]+)", "RCPT"),
    ]

    for pattern, name in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Must be > 2 chars
            if len(extracted) <= 2:
                continue
            # Must contain at least one digit
            if not re.search(r"\d", extracted):
                continue
            # Skip dates (MM/DD/YYYY)
            if extracted.count("/") == 2 and re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", extracted):
                continue
            # Take first value if comma-separated
            if "," in extracted:
                extracted = extracted.split(",")[0]
            logger.info(f"   🔢 Invoice number found via '{name}': {extracted}")
            return extracted

    return None


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


async def _read_image(upload: UploadFile) -> Image.Image:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image, got: {upload.content_type}")
    try:
        contents = await upload.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")


# ── App ──────────────────────────────────────────────────────────────────────────

app = FastAPI(title="InkLens OCR Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE, "model_loaded": "proc" in _model}


@app.post("/enhance")
async def enhance_endpoint(image: UploadFile = File(...)):
    logger.info("🖼️  /enhance — received image")
    pil_image = await _read_image(image)
    t = time.time()
    enhanced = enhance_for_ocr(pil_image)
    logger.info(f"🖼️  /enhance — done in {(time.time()-t)*1000:.0f}ms")
    return Response(content=_pil_to_png_bytes(enhanced), media_type="image/png")


@app.post("/ocr")
async def ocr_endpoint(
    image: UploadFile = File(...),
    enhance: str = Form("false"),
):
    logger.info(f"📝 /ocr — received image (enhance={enhance})")
    pil_image = await _read_image(image)

    if enhance.lower() == "true":
        t = time.time()
        pil_image = enhance_for_ocr(pil_image)
        logger.info(f"   ⚡ Enhancement done in {(time.time()-t)*1000:.0f}ms")

    logger.info("   🔍 Running LightOnOCR...")
    start = time.time()
    raw, cleaned = run_ocr(pil_image)
    elapsed_ms = (time.time() - start) * 1000

    invoice_number = extract_invoice_number(cleaned)

    logger.info(f"   📄 Raw output:     {repr(raw[:150])}")
    logger.info(f"   ✅ Cleaned output:  {repr(cleaned[:150])}")
    if invoice_number:
        logger.info(f"   🔢 Invoice #:      {invoice_number}")
    logger.info(f"   ⏱️  Completed in {elapsed_ms:.0f}ms")

    return {"text": cleaned, "invoice_number": invoice_number, "processing_time_ms": round(elapsed_ms)}


@app.post("/ocr/stream")
async def ocr_stream_endpoint(
    image: UploadFile = File(...),
    model: str = Form("lighton"),
    enhance: str = Form("false"),
):
    logger.info(f"📝 /ocr/stream — received image (model={model}, enhance={enhance})")
    pil_image = await _read_image(image)
    enhanced_image = None

    if enhance.lower() == "true":
        t = time.time()
        enhanced_image = enhance_for_ocr(pil_image)
        logger.info(f"   ⚡ Enhancement done in {(time.time()-t)*1000:.0f}ms")

    ocr_input = enhanced_image or pil_image

    async def generate():
        import base64

        # Send enhanced image preview
        if enhanced_image is not None:
            png_bytes = _pil_to_png_bytes(enhanced_image)
            b64 = base64.b64encode(png_bytes).decode("ascii")
            logger.info(f"   🖼️  Sending enhanced image preview ({len(png_bytes)} bytes)")
            yield f"data: {json.dumps({'type': 'enhanced_image', 'data_url': f'data:image/png;base64,{b64}'})}\n\n"

        # Run OCR
        logger.info("   🔍 Running LightOnOCR...")
        raw, cleaned, elapsed_ms = stream_ocr_generator(ocr_input)

        invoice_number = extract_invoice_number(cleaned)

        logger.info(f"   📄 Raw output:     {repr(raw[:150])}")
        logger.info(f"   ✅ Cleaned output:  {repr(cleaned[:150])}")
        if invoice_number:
            logger.info(f"   🔢 Invoice #:      {invoice_number}")
        logger.info(f"   ⏱️  Completed in {elapsed_ms:.0f}ms")

        if cleaned:
            yield f"data: {json.dumps({'type': 'token', 'text': cleaned})}\n\n"
        if invoice_number:
            yield f"data: {json.dumps({'type': 'invoice_number', 'invoice_number': invoice_number})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'processing_time_ms': round(elapsed_ms)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════╗
║         InkLens OCR Service              ║
║                                          ║
║  Device:  {DEVICE:<30s}║
║  Port:    {PORT:<30d}║
║  Host:    {HOST:<30s}║
║                                          ║
║  Endpoints:                              ║
║    GET  /health                          ║
║    POST /ocr         (non-streaming)     ║
║    POST /ocr/stream  (SSE streaming)     ║
║    POST /enhance     (image only)        ║
║                                          ║
║  Model loads on first request.           ║
╚══════════════════════════════════════════╝
""")
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
