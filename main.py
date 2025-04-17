# main.py
import os
import re
import io
import asyncio # Importera asyncio för timeout
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError, BadRequestError
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("FATAL: OPENAI_API_KEY environment variable not found.")
# Konfigurera timeout (t.ex. 60 sekunder för hela requesten)
client = OpenAI(api_key=api_key, timeout=60.0)

# Initialize FastAPI app
app = FastAPI(title="Tandvårdsassistent Backend", version="1.0.4") # Bump version

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
    Applies rule-based regex substitutions AFTER GPT processing.
    Safety net for common errors. (v3 - more rules)
    """
    if not text: return ""

    # --- Formatting Rules ---
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text) # 4 - 8 -> 4-8
    # Fix specific case "tand 4-8" -> "tand 48"
    text = re.sub(r'(?i)\btand\s+4-8\b', 'tand 48', text)
    text = re.sub(r'(?i)(tand|tänder|regio)\s+(\d)\s+(\d)\b', r'\1 \2\3', text) # 4 8 -> 48 after keyword
    # Fix "regio 4-6 till 4-8" (and similar variations) -> "regio 46-48"
    # Handles digits, potential hyphen, space, till/-, space, digits
    text = re.sub(r'(?i)(regio|område)\s+(\d+)\s*(-?\s*(till|\-|--)\s*\d+)\s+(\d+)\b', r'\1 \2\5', text) # Förenklad: Fånga första och sista siffran i en konstig range
    text = re.sub(r'(?i)(regio|område)\s+(\d{1,2})\s*(-|till|--)\s*(\d{1,2})\b', r'\1 \2-\4', text) # Mer standard 46 till 48 -> 46-48
    text = re.sub(r'\b(\d)\s*-\s*0\b', r'\1-0', text) # 4 - 0 -> 4-0
    text = re.sub(r'\b(\d)\s+(noll|Noll)\b', r'\1-0', text) # 4 noll -> 4-0

    # --- Terminology Corrections ---
    text = re.sub(r'(?i)\b(myokopadiost|mukoperiost|myoko)\s+(lambor|lambå)\b', 'mucoperiostlambå', text)
    text = re.sub(r'(?i)\blambor\b', 'lambå', text)
    # Fix "motramus" -> "mot ramus" (handle potential case issues)
    text = re.sub(r'(?i)\bmotramus\b', 'mot ramus', text)
    text = re.sub(r'(?i)\b(ranus|rammus)\s*(framkant)?\b', 'ramus framkant', text) # Fix other ramus misspellings
    text = re.sub(r'(?i)\b(buccal|bockalt|buktalt)\s*(ben)?\b', 'buckalt ben', text)
    text = re.sub(r'(?i)\b(mesial|messial)\b', 'mesial', text)
    text = re.sub(r'(?i)\b(distal|distalråt|stalrot)\b', 'distal', text)
    text = re.sub(r'(?i)distal\s*råt\b', 'distal rot', text)
    text = re.sub(r'(?i)\b(vi kryr|vicryr|vikryl)\b', 'Vicryl', text)
    text = re.sub(r'(?i)\b(Vicryl)\s+(\d-0)\b', r'\1 \2', text)

    # --- Comma Cleanup (Cautious) ---
    # Remove comma AFTER a single word if followed by another single word (common Whisper artifact)
    # Example: "fäller, mucoperiostlambå" -> "fäller mucoperiostlambå"
    # This is tricky, use limited scope. \b = word boundary, \w+ = one or more word chars
    text = re.sub(r'\b(\w+),\s+(\w+)\b', r'\1 \2', text)
    # Maybe run it twice?
    text = re.sub(r'\b(\w+),\s+(\w+)\b', r'\1 \2', text)

    # General cleanup
    text = re.sub(r'(?i)\bkomma\s+separerade\b', ', separerade', text)
    text = re.sub(r'(?i)^Så tur är\b', 'Sutur', text)

    # --- Capitalization and Punctuation ---
    if text and text[0].isalpha() and not text[0].isupper():
         text = text[0].upper() + text[1:]
    if text and text[-1].isalnum():
        text += "."
    text = text.replace("..", ".") # Remove double periods

    return text.strip()

async def get_gpt_correction(text_to_correct: str, previous_context: str | None = None) -> str:
    """
    Sends the transcribed text to GPT-4 for correction. (v3 - even stronger prompt)
    Includes robust error handling for API calls.
    """
    if not text_to_correct or text_to_correct.isspace():
        print("GPT Correction: Skipping empty or whitespace input.")
        return ""

    print(f"GPT Correction: Processing text: '{text_to_correct}'")
    if previous_context: print(f"GPT Correction: Using context: '{previous_context}'")

    # --- ÄNNU STARKARE SYSTEM PROMPT ---
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
        # Öka timeout för just detta anrop om det behövs, annars används klientens default
        response = await asyncio.wait_for(
             client.chat.completions.create(
                model="gpt-4-turbo", # Eller gpt-4o
                messages=messages,
                temperature=0.1,
                max_tokens=350, # Ge lite mer utrymme
                n=1,
                stop=None
            ),
            timeout=45.0 # Specifik timeout för detta anrop (sekunder)
        )

        corrected_raw = response.choices[0].message.content.strip()
        if corrected_raw.startswith('"') and corrected_raw.endswith('"'):
            corrected_raw = corrected_raw[1:-1]
        print(f"GPT Correction: Raw response: '{corrected_raw}'")
        final_corrected = post_process_corrections(corrected_raw)
        print(f"GPT Correction: Post-processed response: '{final_corrected}'")
        return final_corrected

    # --- Robust Felhantering för OpenAI API ---
    except APITimeoutError:
        print("ERROR: OpenAI API request timed out.")
        raise HTTPException(status_code=504, detail="Timeout vid kommunikation med AI-modellen.")
    except APIConnectionError as e:
        print(f"ERROR: OpenAI API connection error: {e}")
        raise HTTPException(status_code=503, detail="Kunde inte ansluta till AI-modellen.")
    except RateLimitError:
        print("ERROR: OpenAI API rate limit exceeded.")
        raise HTTPException(status_code=429, detail="För många anrop till AI-modellen, försök igen senare.")
    except BadRequestError as e:
         print(f"ERROR: OpenAI API Bad Request (400): {e}")
         # Detta kan tyda på problem med prompten eller innehållet
         raise HTTPException(status_code=400, detail=f"Ogiltig begäran till AI-modellen: {e}")
    except APIStatusError as e:
        print(f"ERROR: OpenAI API status error ({e.status_code}): {e.response}")
        raise HTTPException(status_code=e.status_code, detail=f"Fel från AI-modellens API: {e.response}")
    except asyncio.TimeoutError:
        # Fånga timeout från asyncio.wait_for
        print("ERROR: GPT correction task timed out via asyncio.")
        raise HTTPException(status_code=504, detail="Timeout under AI-bearbetning.")
    except Exception as e:
        # Fånga alla andra oväntade fel
        print(f"ERROR during GPT-4 correction (Unknown Exception):")
        traceback.print_exc()
        print("GPT Correction: Falling back to post-processed Whisper text due to unknown GPT error.")
        # Returnera post-processad Whisper-text som fallback
        return post_process_corrections(text_to_correct)

# --- /Helper Functions ---


@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...),
    previous_context: str | None = Form(None)
    ):
    """
    API endpoint using io.BytesIO. Includes Whisper prompting and enhanced error handling.
    Handles the entire audio recording sent from the frontend upon stop.
    """
    print("==> DEBUG: Inne i transcribe_audio_chunk v8 (io.BytesIO + Prompts + Robust GPT Call) <==") # Version tag

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
        print("Whisper Transcription: Starting...")
        try:
             base_mime_type = content_type.split(';')[0].strip()
             file_extension = base_mime_type.split('/')[-1] if '/' in base_mime_type else 'webm'
             valid_extensions = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
             if file_extension not in valid_extensions: file_extension = 'webm'
             filename = f"audio_chunk.{file_extension}"

             # Förbättrad Whisper Prompt
             whisper_prompt = (
                 "Svensk tandvårdsjournal diktering. Fokusera på termer som: patienten, tand 48, tand 11, tänder, "
                 "extraktion, extrahera, lambå, mucoperiostlambå, regio 46-48, regio 11-13, sutur, suturerat, "
                 "Vicryl 4-0, Supramid 3-0, Ethilon 5-0, avlastningssnitt, ramus, buckalt, lingualt, palatinalt, "
                 "mesial, distal, rot, rötter, krona, preparation, fyllning, komposit, amalgam, karies, parodontit, "
                 "gingivit, bedövning, Xylocain, Adrenalin, Carbocain, utan adrenalin, anestesi, fullständigt, "
                 "rensat, spolat, kofferdam. Undvik att sätta kommatecken mellan varje ord."
             )
             print(f"Sending data to Whisper with prompt: '{whisper_prompt[:100]}...'")

             # Lägg till timeout även för Whisper? Kan vara långa filer.
             transcript_result = await asyncio.wait_for(
                 client.audio.transcriptions.create(
                     model="whisper-1",
                     file=(filename, audio_data_bytesio, content_type),
                     language="sv",
                     prompt=whisper_prompt
                 ),
                 timeout=60.0 # Max 60 sek för Whisper att bearbeta filen
             )
             raw_transcript = transcript_result.text.strip() if transcript_result and transcript_result.text else ""
             print(f"Whisper Raw Transcription: '{raw_transcript}'")

        except asyncio.TimeoutError:
             print("ERROR: Whisper transcription task timed out.")
             raise HTTPException(status_code=504, detail="Timeout vid transkribering av ljud.")
        except BadRequestError as e: # Fånga specifikt 400-fel från Whisper
            print(f"ERROR during Whisper transcription (Bad Request 400): {e}")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Ogiltigt ljudformat eller parameter för transkribering: {e}")
        except Exception as e: # Fånga andra Whisper-fel
             print(f"ERROR during Whisper transcription (Other):")
             traceback.print_exc()
             error_detail = f"Whisper transcription failed: {str(e)}"
             if hasattr(e, 'status_code'): # Om det är ett annat API-fel
                 error_detail = f"Whisper transcription failed: API error {e.status_code} - {str(e)}"
             raise HTTPException(status_code=500, detail=error_detail)

        # --- Step 2: Correct transcription using GPT-4 ---
        corrected_text = ""
        if raw_transcript:
             # Anropa den nu mer robusta GPT-funktionen
            corrected_text = await get_gpt_correction(raw_transcript, previous_context)
            print(f"GPT Corrected Text (final): '{corrected_text}'")
        else:
             print("Whisper returned empty transcription. Skipping GPT correction.")
             corrected_text = "" # Sätt till tom sträng om ingen rå text finns

        # --- Step 3: Return the result ---
        # Säkerställ att vi alltid returnerar en sträng, även om korrigering misslyckades helt
        return TranscriptionResponse(
            corrected_text=corrected_text if corrected_text else raw_transcript, # Fallback till rå text om korrigering är tom
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
