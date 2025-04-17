from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import tempfile
import os
import re

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
        system_instruction = (
            "Du är en skicklig svensk läkarsekreterare med inriktning på tandvård. "
            "Din uppgift är att bearbeta dikterad text från tandläkare och omvandla den till korrekt, välformulerad och professionell journaltext.\n\n"
            "Följande gäller:\n"
            "- Korrigera stavfel, grammatiska fel och teckenanvändning.\n"
            "- Använd korrekt svensk medicinsk och odontologisk terminologi.\n"
            "- Behåll talarens meningsuppbyggnad, innehåll och ton.\n"
            "- Gör inte om formuleringar i onödan.\n"
            "- Korta inte ner eller förändra innehållet utan tydlig anledning.\n"
            "- Om något är otydligt, anta det mest troliga utifrån kontexten och odontologisk praxis.\n"
            "- Använd alltid en vårdad, saklig och professionell ton.\n\n"
            "Syftet är att den bearbetade texten ska kunna klistras in direkt i en tandvårdsjournal."
        )

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": request.transcription.strip()}
            ],
            temperature=0.2
        )

        return {"journal": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"journal": f"Fel: {str(e)}"}

@app.post("/api/correct-sentence", response_model=JournalResult)
async def correct_sentence(request: JournalRequest):
    try:
        raw = request.transcription.strip()

        # Enkel pre-clean av vanliga feltolkningar
        corrections = {
            r"(?i)lamborg": "lambå",
            r"(?i)ramos": "ramus",
            r"(?i)bokalt": "buckalt",
            r"(?i)bokal": "buckal",
            r"(?i)komma": ",",
            r"(?i)avlägsnabbokalt": "avlägsnar buckalt",
            r"(?i)till stalrot": "distalrot",
            r"(?i)messial": "mesial",
            r"(?i)buktalt": "buckalt"
        }

        for pattern, replacement in corrections.items():
            raw = re.sub(pattern, replacement, raw)

        prompt = (
            "Text: \"{}\"\n"
            "Instruktion: Omvandla detta till en professionell tandvårdsjournalformulering på korrekt svenska. "
            "Förbättra grammatik, tolka talspråk, rätta felaktiga ordval och använd fackspråk. "
            "Uttryck meningen så som en tandläkare hade skrivit den i en journal."
        ).format(raw)

        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Du är en kliniskt inriktad AI som tolkar talspråk till korrekt tandvårdsspråk."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return {"journal": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"journal": f"Fel: {str(e)}"}
