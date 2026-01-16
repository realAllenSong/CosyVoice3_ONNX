
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import uvicorn
import os
from typing import Optional
from cosyvoice_onnx import CosyVoiceTTS, CosyVoiceConfig

app = FastAPI(title="CosyVoice3 ONNX API")

# Global TTS instance
tts = None

class SynthesisRequest(BaseModel):
    text: str
    voice: Optional[str] = "zh_female_1"
    prompt_text: Optional[str] = None
    prompt_audio_path: Optional[str] = None
    speed: float = 1.0
    
@app.on_event("startup")
async def startup_event():
    global tts
    config = CosyVoiceConfig()
    # Configure for server usage
    config.model.precision = "fp16"
    config.model.preload = True 
    tts = CosyVoiceTTS(base_dir=os.path.abspath("."), config=config)
    print("TTS Engine initialized")

@app.post("/tts")
async def synthesize(request: SynthesisRequest):
    try:
        if request.prompt_audio_path:
            # Voice Cloning Mode
            output_path = tts.clone_voice(
                text=request.text,
                prompt_audio=request.prompt_audio_path,
                prompt_text=request.prompt_text or "",
                speed=request.speed
            )
        else:
            # Preset Mode
            output_path = tts.synthesize_preset(
                text=request.text,
                voice_name=request.voice,
                speed=request.speed
            )
        
        return {
            "status": "success",
            "audio_path": output_path,
            "message": "Audio synthesized successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
