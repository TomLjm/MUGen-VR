from ..common.utils import get_device

def create_app():
    try:
        from fastapi import FastAPI, File, Form, UploadFile
        app = FastAPI(title="MUGen-VR API", version="0.1.0")
        device = get_device()

        @app.get("/health")
        def health():
            return {"status": "ok", "device": str(device)}

        @app.post("/retrieve")
        async def retrieve(text: str = Form(None), image: UploadFile = File(None), top_k: int = Form(5)):
            return {"message": "Retrieval endpoint - to be implemented", "top_k": top_k}

        @app.post("/generate")
        async def generate(text: str = Form(None), image: UploadFile = File(None), num_frames: int = Form(16)):
            return {"message": "Generation endpoint - to be implemented"}

        return app
    except ImportError:
        print("FastAPI not installed.")
        return None
