from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import tempfile
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionResult(BaseModel):
    transcription: str

class JournalRequest(BaseModel):
    transcription: str

class JournalResult(BaseModel):
    journal: str

@app.post("/api/transcribe", response_model=TranscriptionResult)
async def transcribe_audio(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="sv"
            )
        return {"transcription": transcript.text}
    except Exception as e:
        return {"transcription": f"Fel: {str(e)}"}
    finally:
        os.remove(tmp_path)

@app.post("/api/generate-journal", response_model=JournalResult)
async def generate_journal(request: JournalRequest):
    try:
        prompt = (
            "Text: \"{}\"\n"
            "Instruktion: Omvandla detta till en korrekt svensk tandvårdsjournal."
            " Använd fackspråk, klinisk struktur och var kortfattad."
        ).format(request.transcription.strip())

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Du är en erfaren tandläkare som skriver strukturerade journaler."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        return {"journal": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"journal": f"Fel: {str(e)}"}

@app.post("/api/correct-sentence", response_model=JournalResult)
async def correct_sentence(request: JournalRequest):
    try:
        prompt = (
            "Korrigera och tolka följande dikterade mening till korrekt svenska med kliniskt språk. "
            "Använd fackspråk, korta satser och skriv korrekt:\n\n{}"
        ).format(request.transcription.strip())

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Du är en kliniskt inriktad AI som tolkar talspråk till korrekt tandvårdsspråk."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return {"journal": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"journal": f"Fel: {str(e)}"}
