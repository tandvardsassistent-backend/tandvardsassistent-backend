# main.py
import os
import tempfile # Fortfarande bra att ha för andra syften ev.
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
app = FastAPI(title="Tandvårdsassistent Backend", version="1.0.2") # Bump version

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
# (post_process_corrections och get_gpt_correction som tidigare - inga ändringar där)
def post_process_corrections(text: str) -> str:
    if not text: return ""
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    text = re.sub(r'(?i)(tand|regio)\s+(\d)\s+(\d)\b', r'\1 \2\3', text)
    text = re.sub(r'(\d)\s*-\s*0', r'\1-0', text)
    text = re.sub(r'(\d)\s+(noll|Noll)\b', r'\1-0', text)
    if text and text[0].isalpha() and not text[0].isupper(): text = text[0].upper() + text[1:]
    if text and text[-1].isalnum(): text += "."
    text = re.sub(r'(?i)\bVicryl\.\b', 'Vicryl', text)
    text = re.sub(r'(?i)\blamborg\b', 'lambå', text)
    text = re.sub(r'(?i)\bockalt\b', 'buckalt', text)
    text = re.sub(r'(?i)\bmessial\b', 'mesial', text)
    text = re.sub(r'(?i)\bstalrot\b', 'distalrot', text)
    return text.strip()

async def get_gpt_correction(text_to_correct: str, previous_context: str | None = None) -> str:
    if not text_to_correct or text_to_correct.isspace():
        print("GPT Correction: Skipping empty or whitespace input.")
        return ""
    print(f"GPT Correction: Processing text: '{text_to_correct}'")
    if previous_context: print(f"GPT Correction: Using context: '{previous_context}'")
    system_instruction = (
        "Du är en expert på svensk medicinsk och odontologisk terminologi och journalföring. "
        # ... (Fullständig system prompt som tidigare) ...
        "Din uppgift är att korrigera och formatera korta, dikterade textsegment från en tandläkare. "
        "Transkriptionen kommer från Whisper och kan innehålla fel.\n\n"
        "Fokusera på:\n"
        "1.  **Rätta stavfel och grammatik.** Se till att meningen är korrekt på svenska.\n"
        "2.  **Omvandla felaktigt igenkända ord till korrekt fackterminologi.** Exempel: 'bockalt'->'buckalt', 'vid kiryl'->'Vicryl', 'stalrot'->'distalrot'. Var observant på vanliga fonetiska misstag.\n"
        "3.  **Formatera korrekt:** Särskilt tandnummer (t.ex. 'fyra åtta'->'48', 'fyra sex till fyra åtta'->'46-48'), suturmaterial och storlekar (t.ex. 'fyra noll'->'4-0'). Använd bindestreck för intervall.\n"
        "4.  **Skapa en fullständig, naturlig mening:** Börja med stor bokstav och avsluta med punkt (eller annan lämplig interpunktion), om det inte redan är gjort.\n"
        "5.  **Behåll den ursprungliga betydelsen och detaljnivån.** Gör inga egna tolkningar eller tillägg utöver ren korrigering och formatering. Omvandla inte passiv till aktiv form eller vice versa om det inte är uppenbart fel.\n\n"
        "Var inte rädd att göra nödvändiga ändringar för att uppnå korrekt kliniskt språk, även om den ursprungliga transkriptionen var otydlig eller fragmenterad.\n\n"
        "**VIKTIGT:** Returnera ENDAST den korrigerade texten, utan någon extra förklaring eller inledning.\n\n"
        "Exempel på transformation:\n"
        "Input: 'extraktion tand fyra åtta'\nOutput: 'Extraktion tand 48.'\n"
        "Input: 'fäller mockå päråst lamborg regio 46 till 48'\nOutput: 'Fäller mucoperiostlambå regio 46-48.'\n"
        "Input: 'sutur vid kiryl fyra noll'\nOutput: 'Sutur Vicryl 4-0.'\n"
        "Input: 'avlägsnar bockalt ben'\nOutput: 'Avlägsnar buckalt ben.'\n"
    )
    messages = [{"role": "system", "content": system_instruction}]
    user_prompt_parts = []
    if previous_context and previous_context.strip():
        user_prompt_parts.append(f"Föregående mening (för kontext): \"{previous_context.strip()}\"")
    user_prompt_parts.append(f"Dikterad text att korrigera: \"{text_to_correct}\"")
    user_prompt_parts.append("Korrigera och formatera denna text enligt de givna instruktionerna. Returnera endast den färdiga texten.")
    messages.append({"role": "user", "content": "\n\n".join(user_prompt_parts)})
    try:
        print("GPT Correction: Sending request to OpenAI...")
        response = await client.chat.completions.create(
            model="gpt-4-turbo", messages=messages, temperature=0.1, max_tokens=250, n=1, stop=None
        )
        corrected_raw = response.choices[0].message.content.strip()
        print(f"GPT Correction: Raw response: '{corrected_raw}'")
        final_corrected = post_process_corrections(corrected_raw)
        print(f"GPT Correction: Post-processed response: '{final_corrected}'")
        return final_corrected
    except Exception as e:
        print(f"ERROR during GPT-4 correction request:")
        traceback.print_exc()
        print("GPT Correction: Falling back to post-processed Whisper text due to GPT error.")
        return post_process_corrections(text_to_correct)
# --- /Helper Functions ---


@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...),
    previous_context: str | None = Form(None)
    ):
    """
    API endpoint to receive an audio chunk, transcribe it using OpenAI Whisper,
    correct the transcription using OpenAI GPT-4, and return the result.
    Uses io.BytesIO to handle the audio data entirely in memory.
    """
    print("==> DEBUG: Inne i transcribe_audio_chunk v6 (io.BytesIO) <==") # Version tag

    content_type = audio_chunk.content_type or 'audio/webm'
    print(f"Received chunk with content_type: {content_type}")

    if previous_context and len(previous_context) > 500:
        previous_context = previous_context[-500:]

    audio_data_bytesio = None # Variable to hold the BytesIO object

    try:
        # --- Read audio data into BytesIO object ---
        content = await audio_chunk.read()
        if not content:
            print("ERROR: Received empty audio file content.")
            raise HTTPException(status_code=400, detail="Received empty audio file content.")

        # Create a BytesIO object from the read content
        audio_data_bytesio = io.BytesIO(content)
        # Det är viktigt att inte stänga BytesIO-objektet här,
        # OpenAI-biblioteket behöver kunna läsa från det.
        print(f"Audio chunk read into BytesIO object, size: {len(content)} bytes")

        # --- Step 1: Transcribe audio using Whisper ---
        raw_transcript = ""
        try:
             # Bestäm filnamn för API-hint (behövs för tuple-formatet)
             base_mime_type = content_type.split(';')[0].strip()
             file_extension = base_mime_type.split('/')[-1] if '/' in base_mime_type else 'webm'
             valid_extensions = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
             if file_extension not in valid_extensions:
                 file_extension = 'webm'
             filename = f"audio_chunk.{file_extension}"

             # ---> SKICKA SOM TUPLE MED BytesIO-OBJEKTET <---
             # OpenAI-biblioteket bör kunna hantera ett fil-liknande objekt
             # direkt i tuple-formatet.
             print(f"Sending data to Whisper as tuple: ('{filename}', <BytesIO object>, '{content_type}')")

             # Skicka tuple: (filnamn, fil-liknande objekt, content_type)
             transcript_result = client.audio.transcriptions.create(
                 model="whisper-1",
                 file=(filename, audio_data_bytesio, content_type), # Skicka tuple med BytesIO
                 language="sv"
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
        # Stäng BytesIO-objektet (även om GC borde hantera det)
        if audio_data_bytesio:
            try:
                audio_data_bytesio.close()
                print("BytesIO object closed.")
            except Exception as bio_e:
                print(f"Warning: Error closing BytesIO object: {bio_e}")
        # Stäng även den ursprungliga UploadFile-resursen
        if audio_chunk:
             try:
                await audio_chunk.close()
             except Exception as e:
                  print(f"Warning: Error closing audio chunk resource: {e}")

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
