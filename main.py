# main.py
import os
import tempfile # Keep import even if not used in main path, might be useful later
import re
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import traceback # Import for detailed error logging

# Load environment variables (works locally with .env and on Render with Env Vars)
# Ensure this runs before initializing the client that depends on the API key
load_dotenv()

# Initialize OpenAI client
# Retrieve the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Critical error if the API key is missing
    # Log this or raise an exception to prevent the app from starting incorrectly
    # On Render, check Environment Variables in the service settings.
    # Locally, ensure .env file exists and is loaded, or the variable is exported.
    raise ValueError("FATAL: OPENAI_API_KEY environment variable not found.")
client = OpenAI(api_key=api_key)

# Initialize FastAPI app
app = FastAPI(title="Tandvårdsassistent Backend", version="1.0.0")

# Configure CORS (Cross-Origin Resource Sharing)
# Allows frontend hosted on different origins (domains) to access the API.
# For production, restrict allow_origins to your actual frontend domain for security.
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
    # Consolidate spacing around hyphens in number ranges (e.g., "46 - 48" -> "46-48")
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    # Combine adjacent numbers after specific keywords (e.g., "tand 4 8" -> "tand 48")
    # Use word boundaries (\b) to avoid matching within larger numbers or words.
    text = re.sub(r'(?i)(tand|regio)\s+(\d)\s+(\d)\b', r'\1 \2\3', text)
    # Standardize suture format (e.g., "4 - 0" or "4 noll" -> "4-0")
    text = re.sub(r'(\d)\s*-\s*0', r'\1-0', text)
    text = re.sub(r'(\d)\s+(noll|Noll)\b', r'\1-0', text) # Case-insensitive "noll"

    # --- Capitalization and Punctuation ---
    # Capitalize the first letter if it's an alphabet character and not already capitalized
    if text and text[0].isalpha() and not text[0].isupper():
         text = text[0].upper() + text[1:]
    # Add a period at the end if the sentence ends with an alphanumeric character
    if text and text[-1].isalnum():
        text += "."

    # --- Terminology Corrections (Examples) ---
    # Correct common misinterpretations or GPT variations using case-insensitive matching
    # Ensure whole word matching using \b to avoid partial replacements (e.g., "Vicryl." -> "Vicryl")
    text = re.sub(r'(?i)\bVicryl\.\b', 'Vicryl', text)
    text = re.sub(r'(?i)\blamborg\b', 'lambå', text) # Common error for "lambå"
    text = re.sub(r'(?i)\bockalt\b', 'buckalt', text) # Common error for "buckalt"
    text = re.sub(r'(?i)\bmessial\b', 'mesial', text)   # Common error for "mesial"
    text = re.sub(r'(?i)\bstalrot\b', 'distalrot', text) # Common error for "distalrot"

    return text.strip() # Return the cleaned text, removing leading/trailing whitespace

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
    # Add previous sentence context if available and not empty
    if previous_context and previous_context.strip():
        user_prompt_parts.append(f"Föregående mening (för kontext): \"{previous_context.strip()}\"")

    # Add the main text to be corrected
    user_prompt_parts.append(f"Dikterad text att korrigera: \"{text_to_correct}\"")
    # Explicit instruction (though covered by system prompt, can reinforce)
    user_prompt_parts.append("Korrigera och formatera denna text enligt de givna instruktionerna. Returnera endast den färdiga texten.")

    messages.append({"role": "user", "content": "\n\n".join(user_prompt_parts)})

    try:
        # Use await as client.chat.completions.create is awaitable in async context (openai > 1.0)
        print("GPT Correction: Sending request to OpenAI...")
        response = await client.chat.completions.create(
            model="gpt-4-turbo", # Consider gpt-4o for potential speed/cost benefits
            messages=messages,
            temperature=0.1,  # Low temperature for more predictable, less "creative" corrections
            max_tokens=250, # Allow slightly more tokens for complex sentences or corrections
            n=1,            # Request a single completion choice
            stop=None       # Let the model decide when the corrected sentence ends
        )
        # Extract the corrected text content
        corrected_raw = response.choices[0].message.content.strip()
        print(f"GPT Correction: Raw response: '{corrected_raw}'")

        # Apply final rule-based post-processing for consistency
        final_corrected = post_process_corrections(corrected_raw)
        print(f"GPT Correction: Post-processed response: '{final_corrected}'")
        return final_corrected

    except Exception as e:
        # Log the error comprehensively if GPT call fails
        print(f"ERROR during GPT-4 correction request:")
        traceback.print_exc()
        # Fallback strategy: Return the Whisper text after applying post-processing rules
        # This preserves the transcription even if GPT fails, applying basic fixes.
        print("GPT Correction: Falling back to post-processed Whisper text due to GPT error.")
        return post_process_corrections(text_to_correct)

# --- API Endpoint ---

@app.post("/api/transcribe-chunk", response_model=TranscriptionResponse)
async def transcribe_audio_chunk(
    audio_chunk: UploadFile = File(...), # The uploaded audio file chunk
    previous_context: str | None = Form(None) # Optional previous sentence for context
    ):
    """
    API endpoint to receive an audio chunk, transcribe it using OpenAI Whisper,
    correct the transcription using OpenAI GPT-4, and return the result.
    """
    # Log entry into the function for tracking requests
    print("==> DEBUG: Inne i transcribe_audio_chunk v4 (rensat filnamn) <==")

    # Determine the content type, default to webm if not provided
    content_type = audio_chunk.content_type or 'audio/webm'
    print(f"Received chunk with content_type: {content_type}")

    # Limit the length of the previous context to prevent excessive prompt sizes
    if previous_context and len(previous_context) > 500:
        print(f"Context provided is long ({len(previous_context)} chars), trimming to last 500.")
        previous_context = previous_context[-500:]

    file_content = None # Variable to hold the audio bytes in memory

    try:
        # Read the entire content of the uploaded audio file into memory
        # `await` is needed as `read()` on UploadFile is async
        file_content = await audio_chunk.read()
        # Check if the file content is empty (e.g., zero-byte file upload)
        if not file_content:
            print("ERROR: Received empty audio file content.")
            raise HTTPException(status_code=400, detail="Received empty audio file content.")
        print(f"Audio chunk read into memory, size: {len(file_content)} bytes")

        # --- Step 1: Transcribe audio using Whisper ---
        raw_transcript = ""
        try:
             # Prepare filename and ensure valid extension for Whisper API
             # Extract base MIME type (e.g., 'audio/webm') by stripping parameters
             base_mime_type = content_type.split(';')[0].strip()
             # Extract the subtype as the extension (e.g., 'webm')
             file_extension = base_mime_type.split('/')[-1] if '/' in base_mime_type else 'webm'

             # List of formats explicitly supported by Whisper documentation/API
             valid_extensions = ['flac', 'm4a', 'mp3', 'mp4', 'mpeg', 'mpga', 'oga', 'ogg', 'wav', 'webm']
             # If the extracted extension isn't known, fallback to webm (common from browsers)
             if file_extension not in valid_extensions:
                 print(f"Warning: Derived extension '{file_extension}' from type '{base_mime_type}' is not explicitly listed by Whisper, defaulting to 'webm'.")
                 file_extension = 'webm'

             # Construct a clean filename using the validated/defaulted extension
             filename = f"audio_chunk.{file_extension}"

             # Log the parameters being sent to Whisper
             print(f"Sending data to Whisper as tuple: ('{filename}', <{len(file_content)} bytes> of audio data, '{content_type}')")

             # Call the synchronous Whisper API using the tuple format for the file
             # Ensure NO await here for the standard OpenAI client
             transcript_result = client.audio.transcriptions.create(
                 model="whisper-1", # Specify the Whisper model
                 file=(filename, file_content, content_type), # Pass file as tuple
                 language="sv" # Specify the language
             )
             # Extract the transcribed text, handle potential None or empty results
             raw_transcript = transcript_result.text.strip() if transcript_result and transcript_result.text else ""
             print(f"Whisper Raw Transcription: '{raw_transcript}'")

        except Exception as e:
             # Handle errors during the Whisper transcription process
             print(f"ERROR during Whisper transcription:")
             traceback.print_exc() # Print the full stack trace for debugging
             error_detail = f"Whisper transcription failed: {str(e)}"
             # Provide more specific detail if it's a known error type (like 400 Bad Request)
             if hasattr(e, 'status_code') and e.status_code == 400:
                  error_detail = f"Whisper transcription failed: Bad request (400) - Likely invalid audio format or parameters. Check audio data and content type. Original error: {str(e)}"
             elif hasattr(e, 'status_code'):
                 error_detail = f"Whisper transcription failed: API error {e.status_code} - {str(e)}"
             # Raise an internal server error, passing the detail back to the client
             raise HTTPException(status_code=500, detail=error_detail)

        # --- Step 2: Correct transcription using GPT-4 ---
        corrected_text = ""
        # Proceed only if Whisper produced a non-empty transcription
        if raw_transcript:
            # Call the async helper function to get the GPT correction
            corrected_text = await get_gpt_correction(raw_transcript, previous_context)
            print(f"GPT Corrected Text (final): '{corrected_text}'")
        else:
             # If Whisper returned nothing (e.g., silence), log it and return empty correction
             print("Whisper returned empty transcription. Skipping GPT correction.")
             corrected_text = "" # Ensure response consistency

        # --- Step 3: Return the result ---
        # Send the corrected text and the raw Whisper text back to the client
        return TranscriptionResponse(corrected_text=corrected_text, raw_whisper_text=raw_transcript)

    except HTTPException as http_exc:
        # If an HTTPException was raised deliberately (e.g., 400 for empty file),
        # log it and re-raise it to be sent to the client.
        print(f"HTTP Exception caught and re-raised: {http_exc.status_code} - {http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any unexpected errors not handled above
        print(f"ERROR: Unhandled exception in /api/transcribe-chunk endpoint:")
        traceback.print_exc() # Log the full traceback for server-side debugging
        # Return a generic 500 Internal Server Error to the client for security
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")
    finally:
        # --- Resource Cleanup ---
        # Explicitly set file_content to None to hint at garbage collection
        file_content = None
        # Ensure the UploadFile resource is closed to release file descriptors etc.
        if audio_chunk:
             try:
                await audio_chunk.close()
             except Exception as e:
                  # Log error during close but don't let it crash the response flow
                  print(f"Warning: Error closing audio chunk resource: {e}")

# --- Uvicorn Runner Configuration ---
# This block allows running the server directly using `python main.py`
# Useful for local development. Render typically uses a start command like
# `uvicorn main:app --host 0.0.0.0 --port $PORT`
if __name__ == "__main__":
    import uvicorn
    # Read the PORT environment variable provided by Render (or default to 10000)
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Uvicorn server locally on http://0.0.0.0:{port}")
    # Run the Uvicorn server
    # Set reload=True for local development auto-reloading on code changes.
    # Set reload=False when deploying to Render (Render handles restarts/deploys).
    # The host '0.0.0.0' makes the server accessible on the network (and required by Render).
    uvicorn.run(
        "main:app",  # Points to the FastAPI app instance 'app' in the 'main.py' file
        host="0.0.0.0",
        port=port,
        reload=False # CHANGE TO True FOR LOCAL DEVELOPMENT IF NEEDED
    )
