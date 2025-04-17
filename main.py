import os
import tempfile
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import traceback

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables or .env file.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    text = re.sub(r'(\d)\s+(noll| Noll)\b', r'\1-0', text)
    if text and not text[0].isupper(): text = text[0].upper() + text[1:]
    if text and text[-1].isalnum(): text += "."
    text = text.replace("Vicryl.", "Vicryl")
    text = text.replace("lamborg", "lambå")
    return text.strip()

async def get_gpt_correction(text_to_correct: str, previous_context: str | None = None) -> str:
    if not text_to_correct or text_to_correct.isspace(): return ""
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
    messages = [{"role": "system", "content": system_instruction}]
    user_prompt_parts = []
    if previous_context: user_prompt_parts.append(f"Föregående mening (kontext): \"{previous_context}\"")
    user_prompt_parts.append(f"Transkriberad text att korrigera: \"{text_to_correct}\"")
    user_prompt_parts.append("Korrigera och formatera denna text enligt instruktionerna ovan.")
    messages.append({"role": "user", "content": "\n\n".join(user_prompt_parts)})
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo", messages=messages, temperature=0.1, max_tokens=150, n=1, stop=None
        )
        corrected = response.choices[0].message.content.strip()
        return post_process_corrections(corrected)
    except Exception as e:
        print(f"Error during GPT-4 correction: {e}")
        return post_process_corrections(text_to_correct)
# --- /Helper Functions ---


@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...),
    previous_context: str | None = Form(None)
    ):
    # ---> DEBUG Utskrift <---
    print("==> DEBUG: Inne i transcribe_audio_chunk v3 (med tuple-fil) <==")

    content_type = audio_chunk.content_type or 'audio/webm' # Fånga content type
    print(f"Received chunk with content_type: {content_type}")

    if previous_context and len(previous_context) > 500:
        previous_context = previous_context[-500:]

    tmp_path = None # Behövs inte längre om vi läser direkt till minnet
    file_content = None # För att hålla ljuddatan

    try:
        # Läs innehållet direkt till minnet
        file_content = await audio_chunk.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Received empty audio file content.")
        print(f"Audio chunk read into memory, size: {len(file_content)} bytes")

        # 1. Transcribe with Whisper
        raw_transcript = ""
        try:
             # ---> ÄNDRING: Skicka fil som tuple (filnamn, innehåll, content_type) <---
             # Ge filen ett generiskt namn med korrekt extension om möjligt
             file_extension = content_type.split('/')[-1] if '/' in content_type else 'webm'
             filename = f"audio_chunk.{file_extension}"

             print(f"Sending data to Whisper as tuple: ('{filename}', <{len(file_content)} bytes>, '{content_type}')")

             # ----> KORRIGERING: Säkerställ att await ÄR BORTA här <----
             transcript_result = client.audio.transcriptions.create(
                 model="whisper-1",
                 file=(filename, file_content, content_type), # Skicka som tuple
                 language="sv"
             )
             raw_transcript = transcript_result.text.strip()
             print(f"Whisper Raw Transcription: {raw_transcript}")

        except Exception as e:
             print(f"ERROR during Whisper transcription:")
             traceback.print_exc()
             error_detail = f"Whisper transcription failed: {str(e)}"
             if hasattr(e, 'message') and isinstance(e.message, str) and 'Unsupported file format' in e.message:
                 error_detail = f"Whisper transcription failed: Unsupported file format '{content_type}' sent."
             elif hasattr(e, 'status_code'):
                 error_detail = f"Whisper transcription failed: Error code: {e.status_code} - {str(e)}"
             raise HTTPException(status_code=500, detail=error_detail)


        # 2. Correct with GPT-4 (om transkribering lyckades)
        corrected_text = ""
        if raw_transcript:
            corrected_text = await get_gpt_correction(raw_transcript, previous_context)
            print(f"GPT Corrected Text: {corrected_text}")
        else:
             print("Whisper returned empty transcription. Skipping GPT correction.")
             corrected_text = ""

        return {"corrected_text": corrected_text, "raw_whisper_text": raw_transcript}

    except HTTPException as http_exc:
        # Logga HTTP-exception innan den skickas vidare
        print(f"HTTP Exception caught: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        print(f"ERROR: Unhandled error in /api/transcribe-chunk:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        # Rensa filinnehåll från minnet (även om Python GC borde hantera det)
        file_content = None
        # Stäng den uppladdade filresursen
        if audio_chunk:
             try:
                await audio_chunk.close()
             except Exception as e:
                  print(f"Warning: Error closing audio chunk: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000)) # Standard för Render är ofta 10000
    print(f"Starting Uvicorn on host 0.0.0.0 port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False) # Ta bort reload=True i produktion/Render
