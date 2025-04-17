# main.py
igen Render's infrastruktur framför din app) inte fick ett svar från din *import os
import re
import io
import asyncio # Importera asynciofaktiska* FastAPI-applikation.
    *   **T för to_thread och wait_for
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydrolig Orsak:** Att bearbeta en stor ljudfil (över 40antic import BaseModel
from openai import ( # Importera specifika fel
    OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError, BadRequestError
)
0KB) med både Whisper och sedan GPT tar tid och resurser (minnefrom dotenv import load_dotenv
import traceback
import functools # Behövs för att använda/CPU). Det är mycket möjligt att:
        *   **Timeout:** Request sync funktion med to_thread

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("FATAL: OPENAI_API_KEY environment variable not found.")
# Konfigurera global timeout
en tar längre tid än vad Render's gateway tillåter (ofta 30-60 sekunder), så gatewayen ger upp och skickar 502 tillclient = OpenAI(api_key=api_key, timeout=60 webbläsaren, *innan* din app är klar.
        *   **Res.0)

# Initialize FastAPI app
app = FastAPI(title="Tandvårdsassistentursbrist:** Bearbetningen kräver mer minne eller CPU än vad din Render-instans (speciellt gratisnivån) har tillgängligt, vilket gör att process Backend", version="1.0.6") # Bump version

# Configure CORS
app.add_middleware(
    CORSMiddleware,
en kraschar eller blir dödad av operativsystemet.

**    allow_origins=["*"], # WARNING: Change in production.
    allow_credentials=True,Kombinationen av felen:** 502-felet (
    allow_methods=["GET", "POST"],
    allow_headerstimeout/krasch) inträffar på servern *innan* ett=["*"],
)

# --- Pydantic Models ---
class TranscriptionResponse(BaseModel):
    corrected_text: str
    raw_whisper_text: str

 svar kan skickas tillbaka. Därför saknas CORS-headern,# --- Helper Functions ---
# (post_process_corrections - inga och webbläsaren rapporterar båda problemen.

**Lösningar ändringar)
def post_process_corrections(text: str) -> str:
    if not text: return ""
    text = re:**

1.  **Specifik CORS-Origin (God Praxis & F.sub(r'(\d)\s*-\s*(\d)',elsökning):** Även om det inte är grundorsaken till 502: r'\1-\2', text)
    text = re.sub(r'(?i)\btand\s+4-8\b',an, bör du i produktion *alltid* specificera exakta till 'tand 48', text)
    text = re.sub(åtna origins istället för `"*"`. Detta kan ibland också hjälpa tillr'(?i)(tand|tänder|regio)\s+(\d)\s+(\d)\b att felsöka om problemet ligger i Render's proxy-lager.
', r'\1 \2\3', text)
    text = re.sub(r'(2.  **Hantera Långa Bearbetningstider (V?i)(regio|område)\s+(\d+)\s*(-?\s*(till|\-|--)\s*\d+)\s+(\d+)\b', riktigast):** Detta är kärnan. Att bearbeta en stor fil direkt i en'\1 \2\5', text)
    text = re.sub(r'(?i)(regio|område)\s+(\d{ HTTP-request är inte robust. Den korrekta lösningen är **asynkron1,2})\s*(-|till|--)\s*(\d{1,2})\ bearbetning**:
    *   **Ändra API-flödet:**
b', r'\1 \2-\4', text)
    text =        a.  Frontend skickar ljudfilen till en endpoint (t.ex. `/api re.sub(r'\b(\d)\s*-\s*0/upload-audio`).
        b.  Backend tar emot filen *\b', r'\1-0', text)
    text = resnabbt*, sparar den (t.ex. på disk eller i.sub(r'\b(\d)\s+(noll|Noll molnlagring som S3) och startar en **bakgrundsprocess** (med t.ex. FastAPI:s `BackgroundTasks`, Cel)\b', r'\1-0', text)
    text = re.sub(r'(?i)\b(myokopadiost|mukoperiost|myoko)\s+(lambor|lambå)\bery, eller RQ) för att köra Whisper och GPT på filen.
        c.  Backend returnerar **omedelbart** ett `202 Accepted`-svar till', 'mucoperiostlambå', text)
    text = re frontend, kanske med ett jobb-ID.
        d.  Frontend får.sub(r'(?i)\blambor\b', 'lambå', text)
    text = re.sub(r'(?i 202-svaret och visar "Bearbetar..."
        e.  Frontend behöver)\bmotramus\b', 'mot ramus', text)
     sedan ett sätt att hämta resultatet:
            *   **Pollingtext = re.sub(r'(?i)\b(ranus|:** Frontend frågar en annan endpoint (t.ex. `/api/getrammus)\s*(framkant)?\b', 'ramus framkant', text)
    text-result/{job_id}`) med jämna mellanrum tills resultatet är klart.
            * = re.sub(r'(?i)\b(buccal|   **WebSockets:** En mer avancerad lösning där backend skickar resultbockalt|buktalt)\s*(ben)?\b', 'buckalt ben',atet till frontend via en WebSocket-anslutning när det är klart.
    *   ** text)
    text = re.sub(r'(?i)\b(mesial|messial)\b', 'mesial', text)
Fördelar:** Undviker HTTP-timeouts helt, ger bättre användarfeedback    text = re.sub(r'(?i)\b(distal|distalråt|stalrot)\b', 'distal', text)
    text, skalar bättre.
    *   **Nackdelar:** Mer = re.sub(r'(?i)distal\s*råt\b', ' komplex arkitektur att implementera.

3.  **Öka Resurser/distal rot', text)
    text = re.sub(r'(?i)\b(vi kryr|vicryr|vikryl)\b', 'VicrylTimeout (Potentiell Quick Fix - Kan kosta):**
    *   **', text)
    text = re.sub(r'(?i)\b(Vicryl)\s+(\d-0)\b', r'\1 \2',Render Timeout:** Kolla om du kan öka timeout för din tjänst på Render (kan vara begränsat på gratisnivån).
    *   **Render Instance Size text)
    text = re.sub(r'\b(\w+:** Uppgradera till en betald plan med mer RAM och CPU. Detta *),\s+(\w+)\b', r'\1 \2', text)kan* lösa problemet om det är ren resursbrist, men det är inte en
    text = re.sub(r'\b(\w+),\s+(\w+)\b', r'\1 \2', text)
     garanti och kostar pengar. Asynkron bearbetning är oftast en bättre lösning.

**text = re.sub(r'(?i)\bkomma\sOmedelbar Åtgärd (Fixa CORS-konfig + Grund+separerade\b', ', separerade', text)
    text = re.sub(r'(?läggande Timeout):**

Låt oss först fixa CORS-inställningen ochi)^Så tur är\b', 'Sutur', text)
 lägga in en global timeout för Uvicorn (även om Render'    if text and text[0].isalpha() and not text[0].s gateway-timeout troligen är lägre). Detta löser inte grundproblemet medisupper(): text = text[0].upper() + text[1:] lång bearbetningstid, men är korrekt konfiguration.

**Fullständig `main
    if text and text[-1].isalnum(): text += "."
    text = text.replace("..", ".")
    return text.strip()

# (get_gpt_correction - inga ändringar, den är redan async-kompatibel)
async def get.py` med Korrekt CORS + Uvicorn Timeout-parameter:**

```python
# main._gpt_correction(text_to_correct: str, previous_contextpy
import os
import re
import io
import asyncio
from fastapi import FastAPI, File,: str | None = None) -> str:
    if not text_to_correct or UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSM text_to_correct.isspace(): return ""
    print(f"GPT Correction: Processingiddleware
from pydantic import BaseModel
from openai import (
    OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError, BadRequestError
)
 text: '{text_to_correct}'")
    if previous_contextfrom dotenv import load_dotenv
import traceback

# Load environment variables
load: print(f"GPT Correction: Using context: '{previous_context}'")
    system_instruction =_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_ (
        # ... (Samma starka system prompt som tidigare) ...
        "Du är en **pedantisk** svensk medicinsk sekreterare för **tandvKEY")
if not api_key:
    raise ValueError("FATAL: OPENAI_API_KEYård**. Ditt enda mål är att omvandla rå, **ofta felaktig**, Whisper environment variable not found.")
# Global timeout for OpenAI client requests
client = OpenAI(api_key-transkriberad text till **perfekt** journaltext.\n\n"=api_key, timeout=60.0)

# Initialize FastAPI app
app = FastAPI(title="Tandvårdsassistent Backend
        "**ABSOLUT VIKTIGASTE REGLERNA", version="1.0.6") # Bump version

# --- Kon:**\n"
        "1.  **KORRIGERA ALLfigurera CORS Mer Specifikt ---
# Lista över tillåtna origins (din frontend)
# OBS! Ändra till HTTPS om din frontend använderA FACKTERMER:** Whisper **kommer** att stava fel. Rätta ** det
origins = [
    "https://www.i-media.se", # Din frontend-domän
    # Lägg eventuellt till localhost för lokalalltid** till korrekt svensk odontologisk term (mucoperiostlambå, buck utveckling om det behövs
    # "http://localhost",
    # "httpalt, ramus, mesial, distal, Vicryl, Supramid, etc.).://localhost:8080", # Eller vilken port du använder lokalt
]

app.add_ Ändra aggressivt även om det ser konstigt ut.\n"
        middleware(
    CORSMiddleware,
    allow_origins=origins"2.  **FIXA SIFFROR OCH RANGES:** Tänder anges, # Använd den specifika listan
    allow_credentials=True,
    allow_methods som siffror (t.ex. 48, 11). Intervall anges=["GET", "POST"], # Tillåt bara nödvändiga metoder
    allow med bindestreck (t.ex. 46-48). Suturer_headers=["*"], # Eller specificera headers som "Content-Type" anges som Material #Noll (t.ex. Vicryl 4-0
)

# --- Pydantic Models ---
class TranscriptionResponse(BaseModel):
    ). **Rätta ALLTID** felaktiga format som 'fyra åtta', '4corrected_text: str
    raw_whisper_text: str

 punkt 8', '4-8' (om det ska vara '48'), '46 till # --- Helper Functions ---
# (post_process_corrections - inga ändringar)
def post48', 'fyra noll'.\n"
        "3.  **TA_process_corrections(text: str) -> str:
    if BORT ONÖDIGA KOMMATECKEN:** Whisper lägger ofta in kommatecken mellan not text: return ""
    text = re.sub(r'(\ varje ord. **Ta bort dessa** och skapa naturliga, flytande meningar med korrekt interd)\s*-\s*(\d)', r'\1-\2',punktion.\n"
        "4.  **FIXA SAMMAN text)
    text = re.sub(r'(?i)\btSKRIVNA ORD:** Rätta ord som 'motramus' till 'motand\s+4-8\b', 'tand 48', text)
    text = re ramus'.\n"
        "5.  **GRAMMATIK & STAVNING.sub(r'(?i)(tand|tänder|regio)\:** Korrigera alla vanliga svenska fel.\n"
        "6.  s+(\d)\s+(\d)\b', r'\1 \2\3', text)
    text = re.sub(r'(?i)(regio|område)\s+(\d+)\s**PROFESSIONELLT SPRÅK:** Omvandla talspråk*(-?\s*(till|\-|--)\s*\d+)\s+(\d+)\b', r'\1 \2\5', text)
    text = re.sub till korrekt journalspråk. Skapa fullständiga meningar.\n"
        "(r'(?i)(regio|område)\s+(\d{7.  **BEHÅLL INNEHÅLL:** Ändra inte den medic1,2})\s*(-|till|--)\s*(\d{1,2})\inska innebörden.\n\n"
        "**Exempelb', r'\1 \2-\4', text)
    text = re.sub(r'\b(\d)\s*-\s*0 på hur du SKA transformera DÅLIG input:**\n"
        \b', r'\1-0', text)
    text = re"*   Input: 'extraktion, tand, 4-8'\n    Output: 'Extraktion tand 48.'\n"
        "*   Input: 'fäller.sub(r'\b(\d)\s+(noll|Noll, myokopadiost, lambor, regio, 4-6, till,)\b', r'\1-0', text)
    text = re 4-8'\n    Output: 'Fäller mucoperiostlambå regio.sub(r'(?i)\b(myokopadiost| 46-48.'\n"
        "*   Input:mukoperiost|myoko)\s+(lambor|lambå)\b 'avlägsnar, buccalben, komma, separerade, tanden, i', 'mucoperiostlambå', text)
    text = re, mesial, och, distalråt'\n    Output: 'Av.sub(r'(?i)\blambor\b', 'lambå', text)
lägsnar buckalt ben, separerar tanden i mesial och distal    text = re.sub(r'(?i)\bmotramus rot.'\n"
        "*   Input: 'avlastningssn\b', 'mot ramus', text)
    text = re.sub(r'(?i)\itt, motramus, framkant'\n    Output: 'Avlastningssnitt mot ramusb(ranus|rammus)\s*(framkant)?\b', framkant.'\n"
        "*   Input: 'så, tur, är, vi, 'ramus framkant', text)
    text = re.sub( kryr, 4, -, 0'\n    Output: 'Sutur Vicryl 4r'(?i)\b(buccal|bockalt|buktalt)\s-0.'\n\n"
        "**UTFÖR:** Korrigera följande*(ben)?\b', 'buckalt ben', text)
    text text enligt **exakt** dessa regler. Var **strikt**. Returnera **endast** den färdiga journaltexten."
    )
    messages = [{"role": "system", = re.sub(r'(?i)\b(mesial|messial)\b "content": system_instruction}]
    user_prompt_parts = []
    if previous_context and', 'mesial', text)
    text = re.sub(r previous_context.strip(): user_prompt_parts.append(f"Föregående mening (kontext): \"{previous_context.strip()}\"")
    user'(?i)\b(distal|distalråt|stalrot)\b', '_prompt_parts.append(f"Rå transkriberaddistal', text)
    text = re.sub(r'(? text att korrigera: \"{text_to_correct}\"")
    messages.append({"i)distal\s*råt\b', 'distal rot', text)
    role": "user", "content": "\n\n".join(usertext = re.sub(r'(?i)\b(vi kryr|vicryr|vikryl)\b', 'Vicryl', text)
    text = re.sub(r'(?i)\b(Vicryl)\s+(\d-0)\b', r'\1 \2', text)
    text = re.sub(r'\b(\w+),\s+(\w+)\b',_prompt_parts)})
    try:
        print("GPT Correction: Sending request to OpenAI API...")
        response = await asyncio.wait_for( # Behåll timeout för GPT
             client.chat.completions.create( model="gpt-4-turbo", messages=messages, temperature=0.1, max_tokens=350, n=1, stop=None ),
             timeout=45.0 )
        corrected_raw = response. r'\1 \2', text)
    text = re.sub(choices[0].message.content.strip()
        if corrected_rawr'\b(\w+),\s+(\w+)\b', r'\1 \2', text)
    text = re.sub(r'(.startswith('"') and corrected_raw.endswith('"'): corrected_raw =?i)\bkomma\s+separerade\b', ', separerade', text)
    text corrected_raw[1:-1]
        print(f"GPT Correction = re.sub(r'(?i)^Så tur är\b',: Raw response: '{corrected_raw}'")
        final_corrected = 'Sutur', text)
    if text and text[0]. post_process_corrections(corrected_raw)
        print(f"GPT Correction: Post-processed response: '{final_corrected}'")
        return final_corrected
isalpha() and not text[0].isupper(): text = text[0    except APITimeoutError: print("ERROR: OpenAI API request timed out."); raise HTTPException].upper() + text[1:]
    if text and text[-1(status_code=504, detail="Timeout vid kommunikation med AI-modellen.")
].isalnum(): text += "."
    text = text.replace("..", ".")
    return text.    except APIConnectionError as e: print(f"ERROR: OpenAI API connection error: {e}"); raise HTTPException(status_code=503, detail="Kunde inte ansluta till AI-modstrip()

# (get_gpt_correction - inga ändringar)
async def get_gpt_correction(text_to_correct: str, previous_context: str | None = None) -> str:
    if not text_to_correct or text_to_correct.isspace():
        print("GPT Correction: Skipping empty or whitespace input.")
        return ""
    print(f"GPT Correction: Processing text: '{text_to_correct}'")
    if previous_context: print(f"GPT Correction: Using context: '{previous_context}'")
    system_instruction = (
        "Du är en **pedantisk** svensk medicinsk sekreterare för **tandvård**. Ditt enda mål är att omvandla rå, **ofta felaktig**, Whisper-transkriberad text till **perfekt** journaltext.\n\n"
        "**ABSOLUT VIKTIGASTE REGLERNA:**\n"
        "1.  **KORRIGERA ALLA FACKTERMER:** Whisper **kommer** att stava fel. Rätta **alltid** till korrekt svensk odontologisk term (mucoperiostlambå, buckalt, ramus, mesial, distal, Vicryl, Supramid, etc.). Ändra aggressivt även om det ser konstigt ut.\n"
        "2.  **FIXA SIFFROR OCH RANGES:** Tänder anges som siffror (t.ex. 48, 11). Intervall anges med bindestreck (t.ex. 46-48). Suturer anges som Material #Noll (t.ex. Vicryl 4-0). **Rätta ALLTID** felaktiga format som 'fyra åtta', '4 punkt 8', '4-8' (om det ska vara '48'), '46 till 48', 'fyra noll'.\n"
        "3.  **TA BORT ONÖDIGA KOMMATECKEN:** Whisper lägger ofta in kommatecken mellan varje ord. **Ta bort dessa** och skapa naturliga, flytande meningar med korrekt interpunktion.\n"
        "4.  **FIXA SAMMANSKRIVNA ORD:** Rätta ord som 'motramus' till 'mot ramus'.\n"
        "5.  **GRAMMATIK & STAVNING:** Korrigera alla vanliga svenska fel.\n"
        "6.  **PROFESSIONELLT SPRÅK:** Omvandla talspråk till korrekt journalspråk. Skapa fullständiga meningar.\n"
        "7.  **BEHÅLL INNEHÅLL:** Ändra inte den medicinska innebörden.\n\n"
        "**Exempel på hur du SKA transformera DÅLIG input:**\n"
        "*   Input: 'extraktion, tand, 4-8'\n    Output: 'Extraktion tand 48.'\n"
        "*   Input: 'fäller, myokopadiost, lambor, regio, 4-6, till, 4-8'\n    Output: 'Fäller mucoperiostlambå regio 46-48.'\n"
        "*   Input: 'avlägsnar, buccalben, komma, separerade, tanden, i, mesial, och, distalråt'\n    Output: 'Avlägsnar buckalt ben, separerar tanden i mesial och distal rot.'\n"
        "*   Input: 'avlastningssnitt, motramus, framkant'\n    Output: 'Avlastningssnitt mot ramus framkant.'\n"
        "*   Input: 'så, tur, är, vi, kryr, 4, -, 0'\n    Output: 'Sutur Vicryl 4-0.'\n\n"
        "**UTFÖR:** Korrigera följande text enligt **exakt** dessa regler. Var **strikt**. Returnera **endast** den färdiga journaltexten."
    )
    messages = [{"role": "system", "content": system_instruction}]
    user_prompt_parts = []
    if previous_context and previous_context.strip():
        user_prompt_parts.append(f"Föregående mening (kontext): \"{previous_context.strip()}\"")
    user_prompt_parts.append(f"Rå transkriberad text att korrigera: \"{text_to_correct}\"")
    messages.append({"role": "user", "content": "\n\n".join(user_prompt_parts)})
    try:
        print("GPT Correction: Sending request to OpenAI API...")
        response = await asyncio.wait_for( # Behåll timeout för GPT
             client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.1,
                max_tokens=350,
                n=1,
                stop=None
            ),
            timeout=45.0 # Specifik timeout för GPT
        )
        corrected_raw = response.choices[0].message.content.strip()
        if corrected_raw.startswith('"') and corrected_raw.endswith('"'): corrected_raw = corrected_raw[1:-1]
        print(f"GPT Correction: Raw response: '{corrected_raw}'")
        final_corrected = post_process_corrections(corrected_raw)
        print(f"GPT Correction: Post-processed response: '{final_corrected}'")
        return final_corrected
    except APITimeoutError: print("ERROR: OpenAI API request timed out."); raise HTTPException(status_code=504, detail="Timeout vid kommunikation med AI-modellen.")
    except APIConnectionError as e: print(f"ERROR: OpenAI API connection error: {e}"); raise HTTPException(status_code=503, detail="Kunde inte ansluta till AI-modellen.")
    except RateLimitError: print("ERROR: OpenAI API rate limit exceeded."); raise HTTPException(status_code=429, detail="För många anrop till AI-modellen, försök igen senare.")
    except BadRequestError as e: print(f"ERROR: OpenAI API Bad Request (400): {e}"); raise HTTPException(status_code=400, detail=f"Ogiltig begäran till AI-modellen: {e}")
    except APIStatusError as e: print(f"ERROR: OpenAI API status error ({e.status_code}): {e.response}"); raise HTTPException(status_code=e.status_code, detail=f"Fel från AI-modellens API: {e.response}")
    except asyncio.TimeoutError: print("ERROR: GPT correction task timed out via asyncio."); raise HTTPException(status_code=504, detail="Timeout under AI-bearbetning.")
    except Exception as e:
        print(f"ERROR during GPT-4 correction (Unknown Exception):"); traceback.print_exc()
        print("GPT Correction: Falling back to post-processed Whisper text due to unknown GPT error.")
        return post_process_corrections(text_to_correct)
# --- /Helper Functions ---


@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
ellen.")
    except RateLimitError: print("ERROR: OpenAI API rate limit exceeded."); raise HTTPException(status_code=429, detail="För många anrop till AI-modellenasync def transcribe_audio_chunk(
    audio_chunk: UploadFile.")
    except BadRequestError as e: print(f"ERROR: OpenAI API Bad Request (400): {e}"); raise HTTPException(status_code=400, detail=f"Ogiltig begäran till AI-modellen: {e}")
    except APIStatusError as e: print(f"ERROR: OpenAI API status error ({e.status_code}): {e.response}"); raise HTTPException(status_code=e.status_code, detail=f"Fel från AI-modellens API: {e.response}")
    except asyncio.TimeoutError: print("ERROR: GPT correction task timed out via asyncio."); raise HTTPException(status_code=504, detail="Timeout under AI-bearbetning.")
    except Exception as e:
        print(f"ERROR during GPT-4 correction (Unknown Exception):"); traceback.print_exc()
        print("GPT Correction: Falling back to post-processed Whisper text due to unknown GPT error.")
        return post_process_corrections(text_to_correct)
# --- /Helper Functions ---


@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...),
    previous_context: str | None = Form(None)
    ):
    """
    API endpoint using io.BytesIO. Runs sync Whisper call in thread pool.
    """
    print("==> DEBUG: Inne i transcribe_audio_chunk v10 (to_thread for Whisper) <==") # Version tag

    content_type = audio_chunk.content_type or 'audio/webm'
    print(f"Received chunk with content_type: {content_type}")

    if previous_context and len(previous_context) > 500:
        previous_context = previous_context[-500:]

    audio_data_bytesio = None
    filename = None # Deklarera filename utanför try

    try:
        content = await audio_chunk.read()
        if not content: raise HTTPException(status_code=400, detail="Received empty audio file content.")
        audio_data_bytesio = io.BytesIO(content)
        print(f"Audio chunk read into BytesIO object, size: {len(content)} bytes")

        # --- Step 1: Transcribe audio using Whisper (in thread pool) ---
        raw_transcript = ""
        print("Whisper Transcription: Preparing sync call...")
        try:
             base_mime_type = content_type.split(';')[0].strip()
             file_extension = base_mime_type.split('/')[-1] if '/' in base_mime_type else 'webm'
             valid_extensions = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
             if file_extension not in valid_extensions: file_extension = 'webm'
             filename = f"audio_chunk.{file_extension}" # Sätt filnamn

             whisper_prompt = ( # Behåll prompten
                 "Svensk tandvårdsjournal diktering. Fokusera på termer som: patienten, tand 48, tand 11, tänder, "
                 # ... (samma prompt som tidigare) ...
                 "extraktion, extrahera, lambå, mucoperiostlambå, regio 46-48, regio 11-13, sutur, suturerat, "
                 "Vicryl 4-0, Supramid 3-0, Ethilon 5-0, avlastningssnitt, ramus, buckalt, lingualt, palatinalt, "
                 "mesial, distal, rot, rötter, krona, preparation, fyllning, komposit, amalgam, karies, parodontit, "
                 "gingivit, bedövning, Xylocain, Adrenalin, Carbocain, utan adrenalin, anestesi, fullständigt, "
                 "rensat, spolat, kofferdam. Undvik att sätta kommatecken mellan varje ord."
             )

             # ----> KÖR WHISPER I EN TRÅD <----
             # Definiera den synkrona funktionen som ska köras i tråden
             # Notera: client och whisper_prompt är tillgängliga via closure
             # Vi måste dock skicka filename och audio_data_bytesio som argument
             def sync_whisper_call(fname, audio_bytes_io, prompt_text):
                 print(f"Thread Pool: Running Whisper sync call for {fname}")
                 # Notera: BytesIO-objektet måste vara läsbart av tråden
                 result = client.audio.transcriptions.create(
                     model="whisper-1",
                     file=(fname, audio_bytes_io, content_type), # Använd argumenten
                     language="sv",
                     prompt=prompt_text
                 )
                 print(f"Thread Pool: Whisper sync call finished for {fname}")
                 return result

             # Kör den synkrona funktionen i trådpoolen och vänta på resultatet
             # Viktigt: Skicka med nödvändiga argument som inte är globala/i closure
             print("Whisper Transcription: Dispatching to thread pool...")
             transcript_result = await asyncio.to_thread(
                 sync_whisper_call, filename, audio_data_bytesio, whisper_prompt
             )
             # ----> SLUT TRÅDANROP <----

             raw_transcript = transcript_result.text.strip() if transcript_result and transcript_result.text else ""
             print(f"Whisper Raw Transcription: '{raw_transcript}'")

        except BadRequestError as e: # Fånga specifikt 400-fel
            print(f"ERROR during Whisper transcription (Bad Request 400): {e}")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Ogiltigt ljudformat/parameter för Whisper: {e}")
        except Exception as e: # Fånga andra Whisper-fel
             print(f"ERROR during Whisper transcription (Other):")
             traceback.print_exc()
             error_detail = f"Whisper transcription failed: {str(e)}"
             if hasattr(e, 'status_code'): error_detail = f"Whisper API error {e.status_code}: {str(e)}"
             raise HTTPException(status_code=500, detail=error_detail)

        # --- Step 2: Correct transcription using GPT-4 ---
        # (Ingen ändring här, den är redan async)
        corrected_text = ""
        if raw_transcript:
            corrected_text = await get_gpt_correction(raw_transcript, previous_context)
            print(f"GPT Corrected Text (final): '{corrected_text}'")
        else = File(...),
    previous_context: str | None = Form(None) # previous_context används inte i "hel fil"-läget
    ):
    """
    API endpoint using io.BytesIO. Includes Whisper prompting. Handles the entire audio recording.
    """
    print("==> DEBUG: Inne i transcribe_audio_chunk v9 (io.BytesIO, sync Whisper) <==") # Behåll version

    content_type = audio_chunk.content_type or 'audio/webm'
    print(f"Received chunk with content_type: {content_type}")

    # Ingen previous_context används här logiskt sett, men vi rensar den inte från signaturen
    # if previous_context and len(previous_context) > 500:
    #     previous_context = previous_context[-500:]

    audio_data_bytesio = None

    try:
        content = await audio_chunk.read()
        if not content:
            :
             print("Whisper returned empty transcription. Skipping GPT correction.")
             corrected_text = ""

        # --- Step 3: Return the result ---
        return TranscriptionResponse(
            corrected_text=corrected_text if corrected_text else raw_transcript,
            raw_whisper_text=raw_transcript
        )

    except HTTPException as http_exc:
        print(f"HTTP Exception caught and re-raised: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        print(f"ERROR: Unhandled exception in /api/transcribe-chunk endpoint:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    finally:
        # --- Resource Cleanup ---
        # BytesIO behöver stängas även om den skickades till en annan tråd
        if audio_data_bytesio:
            try: audio_data_bytesio.close(); print("BytesIO object closed.")
            except Exception as bio_e: print(f"Warningraise HTTPException(status_code=400, detail="Received empty audio file content.")
        audio_data_bytesio = io.BytesIO(content)
        print(f"Audio chunk read into BytesIO object, size: {len(content)} bytes")

        # --- Step 1: Transcribe audio using Whisper ---
        raw_transcript = ""
        print("Whisper Transcription: Error closing BytesIO object: {bio_e}")
        # Stäng även den ursprungliga UploadFile-resursen
        if audio_chunk:
             try: await audio_chunk.close()
             except Exception as e: print(f: Starting...")
        try:
             base_mime_type = content"Warning: Error closing audio chunk resource: {e}")

# --- U_type.split(';')[0].strip()
             file_extension = base_mime_type.split('/')[-1] if '/' in base_mime_type else 'webmvicorn Runner Configuration ---
if __name__ == "__main__":
    '
             valid_extensions = ['flac', 'm4a',import uvicorn
    port = int(os.environ.get("PORT", 1000 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg',0))
    print(f"Starting Uvicorn server locally on http://0.0.0 'wav', 'webm']
             if file_extension not in valid_.0:{port}")
    uvicorn.run(
        "mainextensions: file_extension = 'webm'
             filename = f"audio:app", host="0.0.0.0", port=port, reload_chunk.{file_extension}"

             whisper_prompt = (
                 "Svensk tandvårdsjournal diktering. Fokusera på=False # reload=False for Render
    )
