"""
FastAPI Server for 3D Teeth Segmentation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Teeth Segmentation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path("/content/inbracket_ai")
CACHE_DIR = Path("/content/inbracket_ai/cache")
UPLOAD_DIR = Path("/content/inbracket_ai/uploads")
OUTPUT_DIR = Path("/content/inbracket_ai/output")
CHECKPOINT_PATH_FPS = Path(os.environ.get("CHECKPOINT_PATH_FPS", "/content/tgnet_fps"))
CHECKPOINT_PATH_BDL = Path(os.environ.get("CHECKPOINT_PATH_BDL", "/content/tgnet_bdl"))
START_TEST_SCRIPT = PROJECT_ROOT / "start_test.py"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def root():
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
    checks = {
        "start_test_exists": START_TEST_SCRIPT.exists(),
        "checkpoint_fps_exists": (Path(str(CHECKPOINT_PATH_FPS) + ".h5")).exists(),
        "checkpoint_bdl_exists": (Path(str(CHECKPOINT_PATH_BDL) + ".h5")).exists(),
        "cache_dir_exists": CACHE_DIR.exists(),
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "output_dir_exists": OUTPUT_DIR.exists(),
    }
    return {
        "status": "healthy" if all(checks.values()) else "unhealthy",
        "checks": checks,
        "resolved_paths": {
            "project_root": str(PROJECT_ROOT),
            "cache_dir": str(CACHE_DIR),
            "upload_dir": str(UPLOAD_DIR),
            "output_dir": str(OUTPUT_DIR),
            "checkpoint_fps": str(CHECKPOINT_PATH_FPS) + ".h5",
            "checkpoint_bdl": str(CHECKPOINT_PATH_BDL) + ".h5",
            "start_test": str(START_TEST_SCRIPT)
        }
    }


@app.post("/segment")
async def segment(
    lower: UploadFile = File(..., description="Lower jaw OBJ file"),
    upper: UploadFile = File(..., description="Upper jaw OBJ file")
):
    if not lower.filename.endswith('.obj'):
        raise HTTPException(status_code=400, detail="Lower jaw file must be .obj format")
    if not upper.filename.endswith('.obj'):
        raise HTTPException(status_code=400, detail="Upper jaw file must be .obj format")

    # ✅ FIX 1: Save to UPLOAD_DIR, not CACHE_DIR (cache gets wiped by start_test.py)
    lower_path = UPLOAD_DIR / "lower.obj"
    upper_path = UPLOAD_DIR / "upper.obj"

    try:
        with open(lower_path, "wb") as f:
            content = await lower.read()
            f.write(content)
            logger.info(f"✅ Saved lower jaw: {len(content)} bytes → {lower_path}")

        with open(upper_path, "wb") as f:
            content = await upper.read()
            f.write(content)
            logger.info(f"✅ Saved upper jaw: {len(content)} bytes → {upper_path}")

        if not lower_path.exists():
            raise HTTPException(status_code=500, detail=f"lower.obj not found at {lower_path}")
        if not upper_path.exists():
            raise HTTPException(status_code=500, detail=f"upper.obj not found at {upper_path}")

        logger.info("Starting inference...")

        # ✅ FIX 2: Correct indentation + output_path added
        cmd = [
            "python", str(START_TEST_SCRIPT),
            "--input_lower_path", str(lower_path),
            "--input_upper_path", str(upper_path),
            "--cache_path", str(CACHE_DIR),
            "--output_path", str(OUTPUT_DIR),
            "--checkpoint_path", str(CHECKPOINT_PATH_FPS),
            "--checkpoint_path_bdl", str(CHECKPOINT_PATH_BDL)
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)

        logger.info(f"Stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")

        # ✅ FIX 3: Correct output filenames and directories
        output_files = {
            "lower_obj": OUTPUT_DIR / "colored_input_lower.obj",
            "upper_obj": OUTPUT_DIR / "colored_input_upper.obj",
            "lower_labels": CACHE_DIR / "input_lower.json",
            "upper_labels": CACHE_DIR / "input_upper.json"
        }

        missing_files = [name for name, path in output_files.items() if not path.exists()]

        if missing_files:
            cache_files = list(CACHE_DIR.glob("*"))
            output_files_list = list(OUTPUT_DIR.glob("*"))
            logger.info(f"Files in cache: {[f.name for f in cache_files]}")
            logger.info(f"Files in output: {[f.name for f in output_files_list]}")
            raise HTTPException(
                status_code=500,
                detail=f"Inference done but output files missing: {missing_files}"
            )

        logger.info("✅ All output files generated successfully")

        # ✅ FIX 4: Return correct filenames
        return {
            "status": "success",
            "message": "Segmentation completed successfully",
            "outputs": {
                "lower_obj": "/outputs/colored_input_lower.obj",
                "upper_obj": "/outputs/colored_input_upper.obj",
                "lower_labels": "/outputs/input_lower.json",
                "upper_labels": "/outputs/input_upper.json"
            },
            "download_instructions": "Use GET /outputs/{filename} to download each file"
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Inference failed stdout: {e.stdout}")
        logger.error(f"❌ Inference failed stderr: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e.stderr or str(e)}")

    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ FIX 5: Correct allowed filenames + search both OUTPUT_DIR and CACHE_DIR
@app.get("/outputs/{filename}")
async def get_output(filename: str):
    allowed_files = [
        "colored_input_lower.obj",
        "colored_input_upper.obj",
        "input_lower.json",
        "input_upper.json"
    ]

    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail="Invalid filename")

    for directory in [OUTPUT_DIR, CACHE_DIR]:
        file_path = directory / filename
        if file_path.exists():
            return FileResponse(path=file_path, filename=filename, media_type="application/octet-stream")

    raise HTTPException(status_code=404, detail="File not found")


@app.delete("/cache")
async def clear_cache():
    try:
        for directory in [CACHE_DIR, UPLOAD_DIR, OUTPUT_DIR]:
            for file in directory.glob("*"):
                if file.is_file():
                    file.unlink()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
