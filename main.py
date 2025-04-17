# main.py
import os
import tempfile
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import traceback # Importera för bättre fel-loggning

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

def post_process_corrections(text: str) -> str:
    # (Inga ändringar här, behåll som den var)
    if not text:
        return ""
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    text = re.sub(r'(?i)(tand|regio)\s+(\d)\s+(\d)\b', r'\1 \2\3', text)
    text = re.sub(r'(\d)\s*-\s*0', r'\1-0', text)
    text = re.sub(r'(\d)\s+(noll| Noll)\b', r'\1-0', text)
    if text and not text[0].isupper():
         text = text[0].upper() + text[1:]
    if text and text[-1].isalnum():
        text += "."
    text = text.replace("Vicryl.", "Vicryl")
    text = text.replace("lamborg", "lambå")
    return text.strip()

async def get_gpt_correction(text_to_correct: str, previous_context: str | None = None) -> str:
    # (Inga ändringar här, behåll som den var)
    if not text_to_correct or text_to_correct.isspace():
        return ""
    system_instruction = (
        "Du är en expert på svensk medicinsk och odontologisk terminologi och journalföring. "
        # ... (resten av prompten som tidigare) ...
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
        response = await client.chat.completions.create( # OBS: await här är KORREKT för chat.completions
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.1,
            max_tokens=150,
            n=1,
            stop=None
        )
        corrected = response.choices[0].message.content.strip()
        return post_process_corrections(corrected)
    except Exception as e:
        print(f"Error during GPT-4 correction: {e}")
        return post_process_corrections(text_to_correct)


@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...),
    previous_context: str | None = Form(None)
    ):
    if not audio_chunk.content_type or not audio_chunk.content_type.startswith("audio/"):
         print(f"Warning: Received potentially invalid content type: {audio_chunk.content_type}")
         # Raise HTTPException(status_code=400, detail="Invalid file type. Expected audio.") # Kanske för strikt?

    if previous_context and len(previous_context) > 500:
        previous_context = previous_context[-500:]

    tmp_path = None
    try:
        # Skapa temporär fil med korrekt suffix baserat på content_type om möjligt
        suffix = ".webm" # Default
        if audio_chunk.content_type and "/" in audio_chunk.content_type:
            potential_suffix = "." + audio_chunk.content_type.split("/")[-1]
            # Lägg till fler kända mappningar om nödvändigt
            if potential_suffix in [".webm", ".wav", ".mp3", ".ogg", ".mp4", ".m4a"]:
                 suffix = potential_suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio_chunk.read()
            if not content:
                raise HTTPException(status_code=400, detail="Received empty audio file.")
            tmp.write(content)
            tmp_path = tmp.name
        print(f"Audio chunk saved to temporary file: {tmp_path}, size: {len(content)} bytes, suffix: {suffix}")

        # 1. Transcribe with Whisper
        raw_transcript = ""
        try:
             # Öppna den sparade filen för läsning i binärt läge
             with open(tmp_path, "rb") as audio_file_handle:
                  # ----> KORRIGERING: Ta bort 'await' här <----
                  transcript_result = client.audio.transcriptions.create(
                      model="whisper-1",
                      file=audio_file_handle, # Skicka filhandtaget
                      language="sv"
                  )
             raw_transcript = transcript_result.text.strip()
             print(f"Whisper Raw Transcription: {raw_transcript}")
        except Exception as e:
             print(f"Error during Whisper transcription: {e}")
             # Skriv ut hela traceback för mer detaljer vid Whisper-fel
             traceback.print_exc()
             # Skicka tillbaka ett mer informativt felmeddelande till klienten
             error_detail = f"Whisper transcription failed: {str(e)}"
             # Försök fånga specifika API-fel från OpenAI om möjligt
             if hasattr(e, 'status_code'):
                 error_detail = f"Whisper transcription failed: Error code: {e.status_code} - {str(e)}"

             raise HTTPException(status_code=500, detail=error_detail) # Skicka 500 vid Whisper-fel


        # 2. Correct with GPT-4 (if transcription was successful)
        corrected_text = ""
        if raw_transcript:
            corrected_text = await get_gpt_correction(raw_transcript, previous_context)
            print(f"GPT Corrected Text: {corrected_text}")
        else:
             print("Whisper returned empty transcription. Skipping GPT correction.")
             corrected_text = ""

        return {"corrected_text": corrected_text, "raw_whisper_text": raw_transcript}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unhandled error in /api/transcribe-chunk:")
        # Skriv ut hela traceback för oväntade fel
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"Temporary file deleted: {tmp_path}")
            except Exception as e:
                print(f"Error deleting temporary file {tmp_path}: {e}")
        if audio_chunk: # Säkerställ att audio_chunk finns innan close anropas
             try:
                await audio_chunk.close()
             except Exception as e:
                  print(f"Error closing audio chunk: {e}")


if __name__ == "__main__":
    import uvicorn
    # Se till att porten matchar det du förväntar dig (Render använder oftast 10000)
    port = int(os.environ.get("PORT", 8000)) # Anpassa default om nödvändigt
    uvicorn.run(app, host="0.0.0.0", port=port)
