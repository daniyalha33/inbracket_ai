"""
FastAPI Server for 3D Teeth Segmentation
Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Teeth Segmentation API")

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define project paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINT_PATH_FPS = PROJECT_ROOT / "tgnet_fps"
CHECKPOINT_PATH_BDL = PROJECT_ROOT / "tgnet_bdl"
START_TEST_SCRIPT = PROJECT_ROOT / "start_test.py"

# Create necessary directories
CACHE_DIR.mkdir(exist_ok=True)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "3D Teeth Segmentation API",
        "endpoints": {
            "/segment": "POST - Upload lower and upper jaw OBJ files",
            "/outputs/{filename}": "GET - Download output files",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Check if all required files are present"""
    checks = {
        "start_test_exists": START_TEST_SCRIPT.exists(),
        "checkpoint_fps_exists": CHECKPOINT_PATH_FPS.exists(),
        "checkpoint_bdl_exists": CHECKPOINT_PATH_BDL.exists(),
        "cache_dir_exists": CACHE_DIR.exists()
    }
    
    all_healthy = all(checks.values())
    
    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "checks": checks
    }

@app.post("/segment")
async def segment(
    lower: UploadFile = File(..., description="Lower jaw OBJ file"),
    upper: UploadFile = File(..., description="Upper jaw OBJ file")
):
    """
    Segment 3D teeth from lower and upper jaw OBJ files.
    
    Returns paths to segmented OBJ files and their label JSON files.
    """
    
    # Validate file extensions
    if not lower.filename.endswith('.obj'):
        raise HTTPException(status_code=400, detail="Lower jaw file must be .obj format")
    if not upper.filename.endswith('.obj'):
        raise HTTPException(status_code=400, detail="Upper jaw file must be .obj format")
    
    # Define file paths
    lower_path = CACHE_DIR / "lower.obj"
    upper_path = CACHE_DIR / "upper.obj"
    
    try:
        # Save uploaded files
        logger.info(f"Saving uploaded files: {lower.filename}, {upper.filename}")
        
        with open(lower_path, "wb") as f:
            content = await lower.read()
            f.write(content)
            logger.info(f"Saved lower jaw: {len(content)} bytes")
        
        with open(upper_path, "wb") as f:
            content = await upper.read()
            f.write(content)
            logger.info(f"Saved upper jaw: {len(content)} bytes")
        
        # Verify files were saved
        if not lower_path.exists() or not upper_path.exists():
            raise HTTPException(status_code=500, detail="Failed to save uploaded files")
        
        logger.info("✅ Files saved successfully. Starting inference...")
        
        # Run segmentation
        cmd = [
            "python", str(START_TEST_SCRIPT),
            "--input_lower_path", str(lower_path),
            "--input_upper_path", str(upper_path),
            "--cache_path", str(CACHE_DIR),
            "--checkpoint_path", str(CHECKPOINT_PATH_FPS),
            "--checkpoint_path_bdl", str(CHECKPOINT_PATH_BDL)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info(f"Inference stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Inference stderr: {result.stderr}")
        
        logger.info("✅ Inference complete. Checking outputs...")
        
        # Check for output files
        output_files = {
            "lower_obj": CACHE_DIR / "input_lower.obj",
            "upper_obj": CACHE_DIR / "input_upper.obj",
            "lower_labels": CACHE_DIR / "input_lower.json",
            "upper_labels": CACHE_DIR / "input_upper.json"
        }
        
        missing_files = [name for name, path in output_files.items() if not path.exists()]
        
        if missing_files:
            logger.error(f"Missing output files: {missing_files}")
            # List all files in cache directory for debugging
            cache_files = list(CACHE_DIR.glob("*"))
            logger.info(f"Files in cache directory: {[f.name for f in cache_files]}")
            
            raise HTTPException(
                status_code=500,
                detail=f"Inference completed but output files not found: {missing_files}"
            )
        
        logger.info("✅ All output files generated successfully")
        
        return {
            "status": "success",
            "message": "Segmentation completed successfully",
            "outputs": {
                "lower_obj": "/outputs/input_lower.obj",
                "upper_obj": "/outputs/input_upper.obj",
                "lower_labels": "/outputs/input_lower.json",
                "upper_labels": "/outputs/input_upper.json"
            },
            "download_instructions": "Use GET /outputs/{filename} to download each file"
        }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Inference failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {e.stderr or str(e)}"
        )
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/{filename}")
async def get_output(filename: str):
    """Download output files"""
    
    # Security: Only allow specific filenames
    allowed_files = [
        "input_lower.obj",
        "input_upper.obj",
        "input_lower.json",
        "input_upper.json"
    ]
    
    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    file_path = CACHE_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )

@app.delete("/cache")
async def clear_cache():
    """Clear all cached files"""
    try:
        for file in CACHE_DIR.glob("*"):
            if file.is_file():
                file.unlink()
        
        return {
            "status": "success",
            "message": "Cache cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)