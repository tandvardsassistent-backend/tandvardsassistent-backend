# main.py
import os
import tempfile
import re
import io       # Importera io för BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("FATAL: OPENAI_API_KEY environment variable not found.")
client = OpenAI(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(title="Tandvårdsassistent Backend", version="1.0.3") # Bump version

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: Allows all origins. Change in production.
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class TranscriptionResponse(BaseModel):
    corrected_text: str
    raw_whisper_text: str

# --- Helper Functions ---

def post_process_corrections(text: str) -> str:
    """
    Applies rule-based regex substitutions AFTER GPT processing
    to enforce specific formatting and correct common, predictable errors.
    THIS IS A SAFETY NET - Ideally GPT should handle most corrections.
    """
    if not text:
        return ""

    # --- Formatting Rules (More Specific) ---
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text) # 4 - 8 -> 4-8
    # Handle "4 8" -> "48" only after specific keywords to avoid changing normal numbers
    text = re.sub(r'(?i)(tand|tänder|regio)\s+(\d)\s+(\d)\b', r'\1 \2\3', text)
    # Handle number ranges like "46 till 48" -> "46-48"
    text = re.sub(r'(?i)(regio|område)\s+(\d{2})\s+(till|--|-)\s+(\d{2})\b', r'\1 \2-\4', text)
    # Sutur format "4 noll" or "4 - 0" -> "4-0"
    text = re.sub(r'\b(\d)\s*-\s*0\b', r'\1-0', text)
    text = re.sub(r'\b(\d)\s+(noll|Noll)\b', r'\1-0', text)

    # --- Terminology Corrections (More Aggressive - Use with caution) ---
    # Use \b for whole word matching where appropriate. Case-insensitive (?i).
    # Lambå corrections
    text = re.sub(r'(?i)\b(myokopadiost|mukoperiost|myoko)\s+(lambor|lambå)\b', 'mucoperiostlambå', text)
    text = re.sub(r'(?i)\blambor\b', 'lambå', text)
    # Ramus corrections
    text = re.sub(r'(?i)\b(ranus|ramus|rammus)\s*(framkant)?\b', 'ramus framkant', text)
    # Buckalt corrections
    text = re.sub(r'(?i)\b(buccal|bockalt|buktalt)\s*(ben)?\b', 'buckalt ben', text)
    # Rot corrections
    text = re.sub(r'(?i)\b(mesial|messial)\b', 'mesial', text)
    text = re.sub(r'(?i)\b(distal|distalråt|stalrot)\b', 'distal', text) # Keep it simple first
    text = re.sub(r'(?i)distal\s*råt\b', 'distal rot', text) # Specific fix

    # Sutur corrections - Look for context like "sutur", "suturerat"
    # This one is tricky with regex alone, GPT prompt is better.
    # Example: Replace "vi kryr" or similar ONLY if near "sutur"
    # Simple replacement (less safe):
    text = re.sub(r'(?i)\b(vi kryr|vicryr|vikryl)\b', 'Vicryl', text)
    # Combine Vicryl + Number-0 if separated
    text = re.sub(r'(?i)\b(Vicryl)\s+(\d-0)\b', r'\1 \2', text) # Ensure space: Vicryl 4-0

    # General cleanup like extra commas or weird phrasing
    text = re.sub(r'(?i)\bkomma\s+separerade\b', ', separerade', text) # "Komma separerade" -> ", separerade"
    text = re.sub(r'(?i)^Så tur är\b', 'Sutur', text) # Fix "Så tur är" -> "Sutur" if at start

    # --- Capitalization and Punctuation (Keep as before) ---
    if text and text[0].isalpha() and not text[0].isupper():
         text = text[0].upper() + text[1:]
    if text and text[-1].isalnum():
        text += "."
    # Remove potential double periods
    text = text.replace("..", ".")

    return text.strip()

async def get_gpt_correction(text_to_correct: str, previous_context: str | None = None) -> str:
    """
    Sends the transcribed text (from Whisper) to GPT-4 for correction,
    formatting, and applying medical/dental terminology context. (v2 with stronger prompt)
    """
    if not text_to_correct or text_to_correct.isspace():
        print("GPT Correction: Skipping empty or whitespace input.")
        return ""

    print(f"GPT Correction: Processing text: '{text_to_correct}'")
    if previous_context: print(f"GPT Correction: Using context: '{previous_context}'")

    # --- STÄRKT SYSTEM PROMPT ---
    system_instruction = (
        "Du är en **extremt noggrann** svensk medicinsk sekreterare specialiserad på **tandvårdsjournaler**. "
        "Din uppgift är att omvandla **potentiellt felaktig** transkriberad text från Whisper till **korrekt, professionell och kliniskt precis** journaltext.\n\n"
        "**MYCKET VIKTIGT:**\n"
        "1.  **FÖRVÄNTA DIG FEL:** Whisper kan misstolka facktermer grovt (t.ex. 'bockalt', 'vi kryr', 'lambor'). Din **främsta uppgift** är att identifiera och **aggressivt korrigera** dessa till korrekt svensk odontologisk terminologi (t.ex. 'buckalt', 'Vicryl', 'lambå'). Ändra även om orden verkar helt felstavade.\n"
        "2.  **KORREKT FORMATERING:** Säkerställ ALLTID korrekt format för:\n"
        "    *   **Tandnummer:** Enstaka (t.ex. 48), intervall (t.ex. 46-48). Ändra 'fyra åtta' till '48', 'fyra sex till fyra åtta' till '46-48'.\n"
        "    *   **Suturer:** Material + storlek (t.ex. 'Vicryl 4-0', 'Supramid 3-0'). Ändra 'fyra noll' till '4-0'. Korrigera felstavade materialnamn ('vikryl' -> 'Vicryl').\n"
        "3.  **SPRÅK OCH GRAMMATIK:** Korrigera alla grammatiska fel och stavfel. Gör texten flytande och professionell.\n"
        "4.  **BEHÅLL BETYDELSE:** Ändra inte den medicinska innebörden. Lägg inte till information.\n"
        "5.  **MENINGSBYGGNAD:** Skapa fullständiga meningar med stor bokstav och punkt. Ta bort talspråk som 'komma'.\n\n"
        "**Exempel på KORREKT hantering av FELAKTIG input:**\n"
        "*   Input: 'extraktion tand fyra åtta'\n    Output: 'Extraktion tand 48.'\n"
        "*   Input: 'fäller mockå päråst lamborg regio 46 till 48'\n    Output: 'Fäller mucoperiostlambå regio 46-48.'\n"
        "*   Input: 'avlägsna buccal ben komma separerade tanden i mesial och distalråt'\n    Output: 'Avlägsnar buckalt ben, separerar tanden i mesial och distal rot.'\n"
        "*   Input: 'så tur är vi kryr fyra noll'\n    Output: 'Sutur Vicryl 4-0.'\n"
        "*   Input: 'snitt mot ranusframkant'\n    Output: 'Avlastningssnitt mot ramus framkant.' (Om kontexten antyder avlastning)\n\n"
        "**AGERA:** Bearbeta följande text enligt **alla** dessa instruktioner. Var **inte** rädd för att göra stora ändringar för att uppnå klinisk korrekthet. Returnera **endast** den färdiga, korrigerade journaltexten."
    )

    messages = [{"role": "system", "content": system_instruction}]
    user_prompt_parts = []
    if previous_context and previous_context.strip():
        user_prompt_parts.append(f"Föregående mening (för kontext): \"{previous_context.strip()}\"")
    # Skicka den råa texten från Whisper
    user_prompt_parts.append(f"Transkriberad text att korrigera: \"{text_to_correct}\"")
    messages.append({"role": "user", "content": "\n\n".join(user_prompt_parts)})

    try:
        print("GPT Correction: Sending request to OpenAI...")
        response = await client.chat.completions.create(
            model="gpt-4-turbo", # Eller gpt-4o om du vill testa
            messages=messages,
            temperature=0.1, # Behåll låg temp för precision
            max_tokens=300, # Öka lite om meningarna kan bli längre efter korrigering
            n=1,
            stop=None
        )
        corrected_raw = response.choices[0].message.content.strip()
        # Ta bort eventuella citattecken som GPT ibland lägger runt svaret
        if corrected_raw.startswith('"') and corrected_raw.endswith('"'):
            corrected_raw = corrected_raw[1:-1]
        print(f"GPT Correction: Raw response: '{corrected_raw}'")

        # Applicera post-processing som en sista putsning/säkerhetsåtgärd
        final_corrected = post_process_corrections(corrected_raw)
        print(f"GPT Correction: Post-processed response: '{final_corrected}'")
        return final_corrected

    except Exception as e:
        print(f"ERROR during GPT-4 correction request:")
        traceback.print_exc()
        print("GPT Correction: Falling back to post-processed Whisper text due to GPT error.")
        # Applicera post-processing även på fallback för att få grundläggande fixar
        return post_process_corrections(text_to_correct)
# --- /Helper Functions ---


@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...),
    previous_context: str | None = Form(None)
    ):
    """
    API endpoint using io.BytesIO. Includes Whisper prompting.
    """
    print("==> DEBUG: Inne i transcribe_audio_chunk v7 (io.BytesIO + Whisper Prompt) <==") # Version tag

    content_type = audio_chunk.content_type or 'audio/webm'
    print(f"Received chunk with content_type: {content_type}")

    if previous_context and len(previous_context) > 500:
        previous_context = previous_context[-500:]

    audio_data_bytesio = None

    try:
        content = await audio_chunk.read()
        if not content:
            raise HTTPException(status_code=400, detail="Received empty audio file content.")
        audio_data_bytesio = io.BytesIO(content)
        print(f"Audio chunk read into BytesIO object, size: {len(content)} bytes")

        # --- Step 1: Transcribe audio using Whisper ---
        raw_transcript = ""
        try:
             base_mime_type = content_type.split(';')[0].strip()
             file_extension = base_mime_type.split('/')[-1] if '/' in base_mime_type else 'webm'
             valid_extensions = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
             if file_extension not in valid_extensions: file_extension = 'webm'
             filename = f"audio_chunk.{file_extension}"

             # ---> WHISPER PROMPT <---
             # Lägg till vanliga och svåra termer här
             whisper_prompt = (
                 "Tandvård journal diktering svenska. Vanliga termer: patienten, tand, tänder, extraktion, extrahera, "
                 "mucoperiostlambå, lambå, regio, sutur, suturerat, Vicryl, Supramid, Ethilon, avlastningssnitt, "
                 "ramus, buckalt, lingualt, palatinalt, mesial, distal, rot, krona, preparation, fyllning, "
                 "komposit, amalgam, karies, parodontit, gingivit, bedövning, Xylocain, Adrenalin, utan adrenalin, "
                 "anestesi, fullständigt, rensat, spolat, kofferdam. Tandnummer som 48, 36, 11, 25. Intervall som 46-48. "
                 "Suturformat som 4-0, 3-0."
             )
             print(f"Sending data to Whisper with prompt: '{whisper_prompt[:100]}...'")

             transcript_result = client.audio.transcriptions.create(
                 model="whisper-1",
                 file=(filename, audio_data_bytesio, content_type),
                 language="sv",
                 prompt=whisper_prompt # <--- Lägg till prompten här
             )
             raw_transcript = transcript_result.text.strip() if transcript_result and transcript_result.text else ""
             print(f"Whisper Raw Transcription: '{raw_transcript}'")

        except Exception as e:
             print(f"ERROR during Whisper transcription:")
             traceback.print_exc()
             error_detail = f"Whisper transcription failed: {str(e)}"
             if hasattr(e, 'status_code') and e.status_code == 400:
                  error_detail = f"Whisper transcription failed: Bad request (400) - Likely invalid audio format or parameters using BytesIO. Original error: {str(e)}"
             elif hasattr(e, 'status_code'):
                 error_detail = f"Whisper transcription failed: API error {e.status_code} - {str(e)}"
             raise HTTPException(status_code=500, detail=error_detail)

        # --- Step 2: Correct transcription using GPT-4 ---
        corrected_text = ""
        if raw_transcript:
            corrected_text = await get_gpt_correction(raw_transcript, previous_context)
            print(f"GPT Corrected Text (final): '{corrected_text}'")
        else:
             print("Whisper returned empty transcription. Skipping GPT correction.")
             corrected_text = ""

        # --- Step 3: Return the result ---
        return TranscriptionResponse(corrected_text=corrected_text, raw_whisper_text=raw_transcript)

    except HTTPException as http_exc:
        print(f"HTTP Exception caught and re-raised: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        print(f"ERROR: Unhandled exception in /api/transcribe-chunk endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    finally:
        # --- Resource Cleanup ---
        if audio_data_bytesio:
            try: audio_data_bytesio.close(); print("BytesIO object closed.")
            except Exception as bio_e: print(f"Warning: Error closing BytesIO object: {bio_e}")
        if audio_chunk:
             try: await audio_chunk.close()
             except Exception as e: print(f"Warning: Error closing audio chunk resource: {e}")

# --- Uvicorn Runner Configuration ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Uvicorn server locally on http://0.0.0.0:{port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False # Set to False for Render
    )
