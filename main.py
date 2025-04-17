# main.py
import os
import tempfile
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables or .env file.")

app = FastAPI()

# Tillåt alla ursprung för enkelhetens skull under utveckling
# I produktion bör du begränsa detta till din frontend-domän
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---
class TranscriptionResponse(BaseModel):
    corrected_text: str
    raw_whisper_text: str # Bra för felsökning

# --- Helper Functions ---

def post_process_corrections(text: str) -> str:
    """
    Apply rule-based corrections AFTER GPT processing for common formatting issues.
    """
    if not text:
        return ""

    # Ensure correct spacing around hyphens in ranges (e.g., 46 - 48 -> 46-48)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    # Handle cases like "tand 4 8" -> "tand 48" (simple cases)
    text = re.sub(r'(?i)(tand|regio)\s+(\d)\s+(\d)\b', r'\1 \2\3', text)
    # Ensure suture format "4 - 0" or "4 noll" -> "4-0"
    text = re.sub(r'(\d)\s*-\s*0', r'\1-0', text)
    text = re.sub(r'(\d)\s+(noll| Noll)\b', r'\1-0', text)
    # Capitalize start of sentence if not already
    if text and not text[0].isupper():
         text = text[0].upper() + text[1:]
    # Ensure sentences end with a period if they don't have other punctuation
    if text and text[-1].isalnum():
        text += "."

    # Fix common misinterpretations GPT might retain or introduce
    text = text.replace("Vicryl.", "Vicryl") # Avoid period directly after names like Vicryl
    text = text.replace("lamborg", "lambå") # Common Whisper/GPT confusion

    return text.strip()

async def get_gpt_correction(text_to_correct: str, previous_context: str | None = None) -> str:
    """
    Sends text to GPT-4 for correction and formatting based on dental context.
    """
    if not text_to_correct or text_to_correct.isspace():
        return "" # Return empty if input is empty

    system_instruction = (
        "Du är en expert på svensk medicinsk och odontologisk terminologi och journalföring. "
        "Din uppgift är att korrigera och formatera korta, dikterade textsegment från en tandläkare. "
        "Transkriptionen kommer från Whisper och kan innehålla fel. \n\n"
        "Fokusera på:\n"
        "1.  **Rätta stavfel och grammatik.**\n"
        "2.  **Omvandla felaktigt igenkända ord till korrekt fackterminologi** (t.ex. 'bockalt'->'buckalt', 'vid kiryl'->'Vicryl').\n"
        "3.  **Formatera korrekt:** Särskilt tandnummer (t.ex. 'fyra åtta'->'48', 'fyra sex till fyra åtta'->'46-48'), suturmaterial och storlekar (t.ex. 'fyra noll'->'4-0').\n"
        "4.  **Skapa en fullständig, naturlig mening:** Börja med stor bokstav och avsluta med punkt, om det inte redan är gjort.\n"
        "5.  **Behåll den ursprungliga betydelsen och detaljnivån.** Gör inga egna tolkningar eller tillägg utöver ren korrigering och formatering.\n\n"
        "Var inte rädd att göra nödvändiga ändringar för att uppnå korrekt kliniskt språk, även om den ursprungliga transkriptionen var otydlig.\n\n"
        "Exempel på transformation:\n"
        "Input: 'extraktion tand fyra åtta'\nOutput: 'Extraktion tand 48.'\n"
        "Input: 'fäller mockå päråst lamborg regio 46 till 48'\nOutput: 'Fäller mucoperiostlambå regio 46-48.'\n"
        "Input: 'sutur vid kiryl fyra noll'\nOutput: 'Sutur Vicryl 4-0.'\n"
    )

    messages = [
        {"role": "system", "content": system_instruction}
    ]

    user_prompt_parts = []
    if previous_context:
        user_prompt_parts.append(f"Föregående mening (kontext): \"{previous_context}\"")

    user_prompt_parts.append(f"Transkriberad text att korrigera: \"{text_to_correct}\"")
    user_prompt_parts.append("Korrigera och formatera denna text enligt instruktionerna ovan.")

    messages.append({"role": "user", "content": "\n\n".join(user_prompt_parts)})

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",  # Or "gpt-4o" when available and stable
            messages=messages,
            temperature=0.1,  # Low temperature for deterministic correction
            max_tokens=150, # Limit output length for sentence correction
            n=1,
            stop=None # Let the model decide when to stop
        )
        corrected = response.choices[0].message.content.strip()
        # Apply post-processing rules for consistency
        return post_process_corrections(corrected)
    except Exception as e:
        print(f"Error during GPT-4 correction: {e}")
        # Fallback: return the post-processed original text on GPT error
        return post_process_corrections(text_to_correct)


# --- API Endpoints ---

@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...),
    previous_context: str | None = Form(None) # Get previous sentence as form data
    ):
    """
    Receives an audio chunk, transcribes with Whisper, corrects with GPT-4.
    """
    if not audio_chunk.content_type.startswith("audio/"):
         # Basic check, more robust validation might be needed
         raise HTTPException(status_code=400, detail="Invalid file type. Expected audio.")

    # Ensure previous_context is not overly long
    if previous_context and len(previous_context) > 500:
        previous_context = previous_context[-500:] # Limit context length

    # Save temporary file
    tmp_path = None
    try:
        # Create a temporary file to store the audio chunk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp: # Adjust suffix if needed (e.g., .wav, .mp3)
            content = await audio_chunk.read()
            if not content:
                raise HTTPException(status_code=400, detail="Received empty audio file.")
            tmp.write(content)
            tmp_path = tmp.name
        print(f"Audio chunk saved to temporary file: {tmp_path}, size: {len(content)} bytes")

        # 1. Transcribe with Whisper
        raw_transcript = ""
        try:
             # Use 'await' for the async file read inside the context manager
             with open(tmp_path, "rb") as f:
                  transcript_result = await client.audio.transcriptions.create(
                      model="whisper-1",
                      file=f,
                      language="sv",
                      # Optional: Provide a prompt for Whisper? Might help with jargon.
                      # prompt="Patient, tand, extraktion, regio, lambå, sutur, Vicryl"
                  )
             raw_transcript = transcript_result.text.strip()
             print(f"Whisper Raw Transcription: {raw_transcript}")
        except Exception as e:
             print(f"Error during Whisper transcription: {e}")
             # Decide how to handle Whisper errors: return empty, return error message, etc.
             # Here, we'll try to proceed without transcription if Whisper fails, or raise error
             raise HTTPException(status_code=500, detail=f"Whisper transcription failed: {e}")

        # 2. Correct with GPT-4 (if transcription was successful)
        corrected_text = ""
        if raw_transcript:
            corrected_text = await get_gpt_correction(raw_transcript, previous_context)
            print(f"GPT Corrected Text: {corrected_text}")
        else:
            # If Whisper returned nothing, maybe log or return empty?
             print("Whisper returned empty transcription. Skipping GPT correction.")
             corrected_text = "" # Ensure it's an empty string

        return {"corrected_text": corrected_text, "raw_whisper_text": raw_transcript}

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        print(f"Unhandled error in /api/transcribe-chunk: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up the temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"Temporary file deleted: {tmp_path}")
            except Exception as e:
                print(f"Error deleting temporary file {tmp_path}: {e}")
        # Ensure the uploaded file resource is closed
        await audio_chunk.close()


# Endpoint för att starta servern (t.ex. med uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
