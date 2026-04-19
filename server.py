"""
FastAPI Server for 3D Teeth Segmentation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import sys
import shutil
import time
from pathlib import Path
import logging
import numpy as np

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

PROJECT_ROOT = Path("/workspace/inbracket_ai")
CACHE_DIR = Path("/workspace/inbracket_ai/cache")
UPLOAD_DIR = Path("/workspace/inbracket_ai/uploads")
OUTPUT_DIR = Path("/workspace/inbracket_ai/output")
CHECKPOINT_PATH_FPS = Path(os.environ.get("CHECKPOINT_PATH_FPS", "/tgnet_fps"))
CHECKPOINT_PATH_BDL = Path(os.environ.get("CHECKPOINT_PATH_BDL", "/tgnet_bdl"))
START_TEST_SCRIPT = PROJECT_ROOT / "start_test.py"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(PROJECT_ROOT))


def prepare_output_files(cache_dir: Path, output_dir: Path):
    """
    Process cache files and save clean versions to output dir:
    - Lower: load → dedup → save
    - Upper: load → dedup → flip back (x,z negate) → save
    - JSON:  copy as-is (labels already match deduplicated vertices)
    """
    import gen_utils as gu

    def dedup_and_save(input_path: Path, output_path: Path, flip: bool = False):
        result = gu.read_txt_obj_ls(str(input_path), ret_mesh=True, use_tri_mesh=True)
        mesh = result[1]
        mesh = mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()

        vertices = np.asarray(mesh.vertices).copy()
        faces = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.vertex_normals)

        # Flip upper jaw back to original orientation
        if flip:
            rot_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            vertices = vertices @ rot_mat.T

        # Write clean OBJ manually to preserve exact vertex count
        with open(output_path, 'w') as f:
            f.write("# Processed by InBracket AI\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for vn in normals:
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        v_count = len(vertices)
        logger.info(f"✅ Saved {output_path.name} ({v_count} vertices, flip={flip})")
        return v_count

    # Process lower jaw (dedup only)
    lower_v = dedup_and_save(
        cache_dir / "input_lower.obj",
        output_dir / "lower.obj",
        flip=False
    )

    # Process upper jaw (dedup + flip back to original orientation)
    upper_v = dedup_and_save(
        cache_dir / "input_upper.obj",
        output_dir / "upper.obj",
        flip=True
    )

    # Copy JSON files as-is (labels match deduplicated vertex count)
    shutil.copy(cache_dir / "input_lower.json", output_dir / "lower.json")
    shutil.copy(cache_dir / "input_upper.json", output_dir / "upper.json")
    logger.info("✅ Copied JSON label files")

    return lower_v, upper_v


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

    lower_path = UPLOAD_DIR / "lower.obj"
    upper_path = UPLOAD_DIR / "upper.obj"

    try:
        # ── Total request timer ──────────────────────────────────────────
        total_start = time.time()

        # Save uploaded files
        upload_start = time.time()
        with open(lower_path, "wb") as f:
            content = await lower.read()
            f.write(content)
            logger.info(f"✅ Saved lower jaw: {len(content)} bytes → {lower_path}")

        with open(upper_path, "wb") as f:
            content = await upper.read()
            f.write(content)
            logger.info(f"✅ Saved upper jaw: {len(content)} bytes → {upper_path}")
        upload_time = time.time() - upload_start
        logger.info(f"⏱ File upload+save: {upload_time:.2f}s")

        if not lower_path.exists():
            raise HTTPException(status_code=500, detail=f"lower.obj not found at {lower_path}")
        if not upper_path.exists():
            raise HTTPException(status_code=500, detail=f"upper.obj not found at {upper_path}")

        # ── GPU Inference timer ──────────────────────────────────────────
        logger.info("Starting GPU inference...")
        inference_start = time.time()

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

        inference_time = time.time() - inference_start
        logger.info(f"⏱ GPU inference: {inference_time:.2f}s ({inference_time/60:.2f} min)")

        # Check inference outputs exist in cache
        required_cache_files = {
            "input_lower.obj":  CACHE_DIR / "input_lower.obj",
            "input_upper.obj":  CACHE_DIR / "input_upper.obj",
            "input_lower.json": CACHE_DIR / "input_lower.json",
            "input_upper.json": CACHE_DIR / "input_upper.json",
        }
        missing = [name for name, path in required_cache_files.items() if not path.exists()]
        if missing:
            logger.error(f"Missing cache files: {missing}")
            raise HTTPException(
                status_code=500,
                detail=f"Inference done but cache files missing: {missing}"
            )

        # ── Post-processing timer ────────────────────────────────────────
        logger.info("Processing output files...")
        postprocess_start = time.time()
        lower_v, upper_v = prepare_output_files(CACHE_DIR, OUTPUT_DIR)
        postprocess_time = time.time() - postprocess_start
        logger.info(f"⏱ Post-processing: {postprocess_time:.2f}s")

        # Verify output files exist
        required_output_files = {
            "lower.obj":  OUTPUT_DIR / "lower.obj",
            "upper.obj":  OUTPUT_DIR / "upper.obj",
            "lower.json": OUTPUT_DIR / "lower.json",
            "upper.json": OUTPUT_DIR / "upper.json",
        }
        missing_outputs = [name for name, path in required_output_files.items() if not path.exists()]
        if missing_outputs:
            raise HTTPException(
                status_code=500,
                detail=f"Output processing failed, missing: {missing_outputs}"
            )

        # ── Total time ───────────────────────────────────────────────────
        total_time = time.time() - total_start
        logger.info(f"⏱ TOTAL request time: {total_time:.2f}s ({total_time/60:.2f} min)")
        logger.info("✅ All output files ready")

        return {
            "status": "success",
            "message": "Segmentation completed successfully",
            "outputs": {
                "lower_obj":    "/outputs/lower.obj",
                "upper_obj":    "/outputs/upper.obj",
                "lower_labels": "/outputs/lower.json",
                "upper_labels": "/outputs/upper.json",
            },
            "vertex_counts": {
                "lower": lower_v,
                "upper": upper_v,
            },
            "timing": {
                "upload_seconds":      round(upload_time, 2),
                "inference_seconds":   round(inference_time, 2),
                "inference_minutes":   round(inference_time / 60, 2),
                "postprocess_seconds": round(postprocess_time, 2),
                "total_seconds":       round(total_time, 2),
                "total_minutes":       round(total_time / 60, 2),
            },
            "notes": {
                "lower": "vertex count matches lower.json labels exactly",
                "upper": "vertex count matches upper.json labels exactly, orientation restored",
                "visualization": "map label index to color using FDI palette, 0=gingiva"
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


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    allowed_files = [
        "lower.obj",   # clean deduplicated lower mesh
        "upper.obj",   # clean deduplicated upper mesh (flipped back)
        "lower.json",  # lower labels + instances
        "upper.json",  # upper labels + instances
    ]

    if filename not in allowed_files:
        raise HTTPException(status_code=400, detail=f"Invalid filename. Allowed: {allowed_files}")

    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
        )

    raise HTTPException(status_code=404, detail=f"File not found: {filename}")


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
