from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import os
from diffusers import DiffusionPipeline
import torch
from PIL import Image
import uuid
import json
import datetime

# Create necessary directories
UPLOAD_DIR = Path("generated_images")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Pixel Art Generator")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe.load_lora_weights("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5")
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

# Define request model
class PromptRequest(BaseModel):
    prompt: str

# Store generated images metadata
generated_images = []

# API endpoints
@app.post("/api/generate")
async def generate_image(request: PromptRequest):
    try:
        # Generate unique filename
        filename = f"{uuid.uuid4()}.png"
        filepath = UPLOAD_DIR / filename

        # Generate image
        image = pipe(request.prompt).images[0]
        
        # Save image
        image.save(filepath)
        
        # Store metadata
        metadata = {
            "id": str(uuid.uuid4()),
            "prompt": request.prompt,
            "filename": filename,
            "timestamp": str(datetime.datetime.now())
        }
        generated_images.append(metadata)
        
        return {"status": "success", "filename": filename, "id": metadata["id"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)

@app.get("/api/images")
async def list_images():
    return generated_images

# Serve static files (the React app)
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pixel Art Generator</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.5/babel.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        function App() {
            const [prompt, setPrompt] = React.useState('');
            const [loading, setLoading] = React.useState(false);
            const [error, setError] = React.useState(null);
            const [generatedImage, setGeneratedImage] = React.useState(null);
            const [images, setImages] = React.useState([]);

            React.useEffect(() => {
                fetchImages();
            }, []);

            const fetchImages = async () => {
                try {
                    const response = await fetch('/api/images');
                    const data = await response.json();
                    setImages(data);
                } catch (error) {
                    console.error('Error fetching images:', error);
                }
            };

            const handleSubmit = async (e) => {
                e.preventDefault();
                setLoading(true);
                setError(null);

                try {
                    const response = await fetch('/api/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ prompt }),
                    });

                    if (!response.ok) {
                        throw new Error('Failed to generate image');
                    }

                    const data = await response.json();
                    setGeneratedImage(data.filename);
                    await fetchImages();
                } catch (error) {
                    setError(error.message);
                } finally {
                    setLoading(false);
                }
            };

            return (
                <div className="container mx-auto px-4 py-8 max-w-4xl">
                    <h1 className="text-3xl font-bold mb-8 text-center">Pixel Art Generator</h1>
                    
                    <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">
                                    Enter your prompt
                                </label>
                                <input
                                    type="text"
                                    value={prompt}
                                    onChange={(e) => setPrompt(e.target.value)}
                                    className="w-full px-4 py-2 border rounded-md focus:ring-blue-500 focus:border-blue-500"
                                    placeholder="e.g., a space miner mining the surface of an asteroid"
                                    required
                                />
                            </div>
                            
                            <button
                                type="submit"
                                disabled={loading}
                                className={`w-full py-2 px-4 rounded-md text-white ${
                                    loading ? 'bg-gray-400' : 'bg-blue-600 hover:bg-blue-700'
                                }`}
                            >
                                {loading ? 'Generating...' : 'Generate Image'}
                            </button>
                        </form>

                        {error && (
                            <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-md">
                                {error}
                            </div>
                        )}

                        {generatedImage && (
                            <div className="mt-6">
                                <h2 className="text-xl font-semibold mb-2">Generated Image:</h2>
                                <img
                                    src={`/api/images/${generatedImage}`}
                                    alt="Generated pixel art"
                                    className="w-full rounded-lg shadow-md"
                                />
                            </div>
                        )}
                    </div>

                    {images.length > 0 && (
                        <div>
                            <h2 className="text-2xl font-bold mb-4">Previous Generations</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {images.map((image) => (
                                    <div key={image.id} className="bg-white rounded-lg shadow p-4">
                                        <img
                                            src={`/api/images/${image.filename}`}
                                            alt={image.prompt}
                                            className="w-full rounded-lg mb-2"
                                        />
                                        <p className="text-sm text-gray-600">{image.prompt}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
</body>
</html>
"""

@app.get("/")
async def serve_app():
    return HTMLResponse(content=HTML_CONTENT)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)