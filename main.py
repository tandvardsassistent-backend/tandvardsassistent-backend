# main.py
import os
import tempfile # Ensure tempfile is imported
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import traceback # Import for detailed error logging

# Load environment variables (works locally with .env and on Render with Env Vars)
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Critical error if the API key is missing
    raise ValueError("FATAL: OPENAI_API_KEY environment variable not found.")
client = OpenAI(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(title="Tandvårdsassistent Backend", version="1.0.1") # Version bump

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: Allows all origins. Change in production.
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Allow specific methods needed
    allow_headers=["*"], # Allow all headers, or specify necessary ones like "Content-Type"
)

# --- Pydantic Models (Data Transfer Objects) ---
class TranscriptionResponse(BaseModel):
    """Defines the structure of the response sent back to the client."""
    corrected_text: str
    raw_whisper_text: str # Include raw text for potential debugging on client side

# --- Helper Functions ---

def post_process_corrections(text: str) -> str:
    """
    Applies rule-based regex substitutions AFTER GPT processing
    to enforce specific formatting and correct common, predictable errors.
    """
    if not text:
        return "" # Return early if text is empty

    # --- Formatting Rules ---
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    text = re.sub(r'(?i)(tand|regio)\s+(\d)\s+(\d)\b', r'\1 \2\3', text)
    text = re.sub(r'(\d)\s*-\s*0', r'\1-0', text)
    text = re.sub(r'(\d)\s+(noll|Noll)\b', r'\1-0', text) # Case-insensitive "noll"

    # --- Capitalization and Punctuation ---
    if text and text[0].isalpha() and not text[0].isupper():
         text = text[0].upper() + text[1:]
    if text and text[-1].isalnum():
        text += "."

    # --- Terminology Corrections (Examples) ---
    text = re.sub(r'(?i)\bVicryl\.\b', 'Vicryl', text)
    text = re.sub(r'(?i)\blamborg\b', 'lambå', text)
    text = re.sub(r'(?i)\bockalt\b', 'buckalt', text)
    text = re.sub(r'(?i)\bmessial\b', 'mesial', text)
    text = re.sub(r'(?i)\bstalrot\b', 'distalrot', text)

    return text.strip() # Return the cleaned text

async def get_gpt_correction(text_to_correct: str, previous_context: str | None = None) -> str:
    """
    Sends the transcribed text (from Whisper) to GPT-4 for correction,
    formatting, and applying medical/dental terminology context.
    Includes optional previous sentence context for better coherence.
    """
    if not text_to_correct or text_to_correct.isspace():
        print("GPT Correction: Skipping empty or whitespace input.")
        return "" # Nothing to correct

    print(f"GPT Correction: Processing text: '{text_to_correct}'")
    if previous_context:
        print(f"GPT Correction: Using context: '{previous_context}'")

    # System prompt defining the AI's role and instructions
    system_instruction = (
        "Du är en expert på svensk medicinsk och odontologisk terminologi och journalföring. "
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

    # Construct messages for the ChatCompletion API
    messages = [
        {"role": "system", "content": system_instruction}
    ]

    user_prompt_parts = []
    if previous_context and previous_context.strip():
        user_prompt_parts.append(f"Föregående mening (för kontext): \"{previous_context.strip()}\"")
    user_prompt_parts.append(f"Dikterad text att korrigera: \"{text_to_correct}\"")
    user_prompt_parts.append("Korrigera och formatera denna text enligt de givna instruktionerna. Returnera endast den färdiga texten.")
    messages.append({"role": "user", "content": "\n\n".join(user_prompt_parts)})

    try:
        print("GPT Correction: Sending request to OpenAI...")
        response = await client.chat.completions.create(
            model="gpt-4-turbo", # Or "gpt-4o"
            messages=messages,
            temperature=0.1,
            max_tokens=250,
            n=1,
            stop=None
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
    audio_chunk: UploadFile = File(...), # The uploaded audio file chunk
    previous_context: str | None = Form(None) # Optional previous sentence for context
    ):
    """
    API endpoint to receive an audio chunk, transcribe it using OpenAI Whisper,
    correct the transcription using OpenAI GPT-4, and return the result.
    Uses a temporary file to store the audio chunk before sending to Whisper.
    """
    # Log entry into the function for tracking requests
    print("==> DEBUG: Inne i transcribe_audio_chunk v5 (med tempfile) <==") # Version tag

    # Determine the content type, default to webm if not provided
    content_type = audio_chunk.content_type or 'audio/webm'
    print(f"Received chunk with content_type: {content_type}")

    # Limit the length of the previous context
    if previous_context and len(previous_context) > 500:
        print(f"Context provided is long ({len(previous_context)} chars), trimming to last 500.")
        previous_context = previous_context[-500:]

    tmp_path = None # Variable to hold the path to the temporary file

    try:
        # --- Save audio chunk to a temporary file ---
        # Determine a suitable file extension based on content type
        base_mime_type = content_type.split(';')[0].strip()
        file_extension = base_mime_type.split('/')[-1] if '/' in base_mime_type else 'webm'
        # List of formats explicitly supported by Whisper
        valid_extensions = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
        # Fallback to .webm if the extracted extension is not recognized
        if file_extension not in valid_extensions:
            print(f"Warning: Derived extension '{file_extension}' from type '{base_mime_type}' not in Whisper's list, defaulting to 'webm'.")
            file_extension = 'webm'
        suffix = f".{file_extension}" # Suffix for the temp file (e.g., ".webm")

        # Create a named temporary file (ensures it has a path)
        # delete=False is important so we can reopen it by path later
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            # Read the content from the uploaded file chunk (async)
            content = await audio_chunk.read()
            # Check if content is empty
            if not content:
                print("ERROR: Received empty audio file content.")
                raise HTTPException(status_code=400, detail="Received empty audio file content.")
            # Write the content to the temporary file
            tmp.write(content)
            # Get the path of the temporary file
            tmp_path = tmp.name
        print(f"Audio chunk saved to temporary file: {tmp_path}, size: {len(content)} bytes")

        # --- Step 1: Transcribe audio using Whisper ---
        raw_transcript = ""
        try:
             # Reopen the temporary file in binary read mode ('rb')
             # The 'with' statement ensures the file handle is closed automatically
             with open(tmp_path, "rb") as audio_file_handle:
                 print(f"Sending temporary file handle ({tmp_path}) to Whisper...")
                 # Call the synchronous Whisper API using the file handle
                 # Ensure NO await here for the standard OpenAI client
                 transcript_result = client.audio.transcriptions.create(
                     model="whisper-1", # Specify the Whisper model
                     file=audio_file_handle, # Pass the file handle
                     language="sv" # Specify the language
                 )
             # Extract the transcribed text
             raw_transcript = transcript_result.text.strip() if transcript_result and transcript_result.text else ""
             print(f"Whisper Raw Transcription: '{raw_transcript}'")

        except Exception as e:
             # Handle errors during Whisper transcription
             print(f"ERROR during Whisper transcription:")
             traceback.print_exc() # Print full stack trace
             error_detail = f"Whisper transcription failed: {str(e)}"
             # Provide more specific detail if it's a known error type
             if hasattr(e, 'status_code') and e.status_code == 400:
                  error_detail = f"Whisper transcription failed: Bad request (400) - Likely invalid audio format or parameters. Check temp file '{tmp_path}'? Original error: {str(e)}"
             elif hasattr(e, 'status_code'):
                 error_detail = f"Whisper transcription failed: API error {e.status_code} - {str(e)}"
             # Raise an internal server error
             raise HTTPException(status_code=500, detail=error_detail)

        # --- Step 2: Correct transcription using GPT-4 ---
        corrected_text = ""
        # Proceed only if Whisper produced a non-empty transcription
        if raw_transcript:
            # Call the async helper function for GPT correction
            corrected_text = await get_gpt_correction(raw_transcript, previous_context)
            print(f"GPT Corrected Text (final): '{corrected_text}'")
        else:
             # Log if Whisper returned nothing (e.g., only silence)
             print("Whisper returned empty transcription. Skipping GPT correction.")
             corrected_text = "" # Ensure response consistency

        # --- Step 3: Return the result ---
        # Send the corrected text and the raw Whisper text back to the client
        return TranscriptionResponse(corrected_text=corrected_text, raw_whisper_text=raw_transcript)

    except HTTPException as http_exc:
        # If an HTTPException was raised deliberately, log and re-raise
        print(f"HTTP Exception caught and re-raised: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any unexpected errors
        print(f"ERROR: Unhandled exception in /api/transcribe-chunk endpoint:")
        traceback.print_exc() # Log full traceback
        # Return a generic 500 error to the client
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    finally:
        # --- Resource Cleanup ---
        # Delete the temporary file from disk if it was created
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"Temporary file deleted: {tmp_path}")
            except Exception as e:
                # Log error during deletion but don't crash
                print(f"Warning: Error deleting temporary file {tmp_path}: {e}")
        # Ensure the UploadFile resource is closed
        if audio_chunk:
             try:
                # Closing the UploadFile might be important for releasing resources
                await audio_chunk.close()
             except Exception as e:
                  # Log error during close but don't crash
                  print(f"Warning: Error closing audio chunk resource: {e}")

# --- Uvicorn Runner Configuration ---
if __name__ == "__main__":
    import uvicorn
    # Read the PORT environment variable (Render provides this) or default
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Uvicorn server locally on http://0.0.0.0:{port}")
    # Use reload=False for production/Render deployments
    uvicorn.run(
        "main:app",  # FastAPI app instance location
        host="0.0.0.0", # Listen on all network interfaces
        port=port,
        reload=False # Set to False for Render
    )
