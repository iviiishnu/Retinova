import os
import uuid
import json
import subprocess
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials missing")

BUCKET_CVD = "cvd_images"
UPLOAD_DIR = "uploads"
INFER_SCRIPT = "infer.py"

os.makedirs(UPLOAD_DIR, exist_ok=True)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("retinova_cvd")

# -----------------------------
# FASTAPI
# -----------------------------
app = FastAPI(title="Retinova CVD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# 1️⃣ UPLOAD ENDPOINT
# -----------------------------
@app.post("/cvd/upload")
async def upload_cvd_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    user_email: str = Form(...)
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG allowed")

    ext = file.filename.split(".")[-1]
    image_id = str(uuid.uuid4())
    local_path = f"{UPLOAD_DIR}/{image_id}.{ext}"

    # Save file locally
    with open(local_path, "wb") as f:
        f.write(await file.read())

    # Upload to Supabase Storage (CORRECT OPTIONS)
    with open(local_path, "rb") as f:
        supabase.storage.from_(BUCKET_CVD).upload(
            path=f"{image_id}.{ext}",
            file=f,
            file_options={
                "content-type": str(file.content_type),  # MUST be string
                "x-upsert": "True"                         # Boolean is OK here
            }
        )

    # Get public URL
    image_url = supabase.storage.from_(BUCKET_CVD).get_public_url(
        f"{image_id}.{ext}"
    )

    # Insert DB row (MATCHES TABLE EXACTLY)
    supabase.table("cvd_images").insert({
        "image_id": image_id,
        "user_id": user_id,
        "user_email":user_email,
        "image_url": image_url
    }).execute()

    log.info("CVD IMAGE UPLOADED → %s", image_id)

    return {
        "message": "Image uploaded",
        "image_id": image_id,
        "image_url": image_url
    }
# -----------------------------
# 2️⃣ RUN INFERENCE ENDPOINT
# -----------------------------
@app.post("/cvd/run")
async def run_cvd_inference(
    image_id: str = Form(...),
    user_id: str = Form(...),   # ✅ REQUIRED
    age: int = Form(...),
    sex: int = Form(...),       # 0=female, 1=male
    smoker: bool = Form(...),
    diabetes: bool = Form(...),
    sbp: int = Form(...),
    bp_meds: bool = Form(...)
):
    # 1️⃣ Fetch image record
    res = (
        supabase.table("cvd_images")
        .select("image_url")
        .eq("image_id", image_id)
        .limit(1)
        .execute()
    )

    if not res.data:
        raise HTTPException(404, "Image not found")

    image_url = res.data[0]["image_url"]

    # 2️⃣ Download image locally (USING SIGNED URL)
    import requests

    local_path = f"{UPLOAD_DIR}/{image_id}.jpg"

# Generate signed URL (valid for 5 minutes)
    signed = supabase.storage.from_(BUCKET_CVD).create_signed_url(
        f"{image_id}.jpg",
        300
)

    signed_url = signed["signedURL"]

    r = requests.get(signed_url)
    if r.status_code != 200:
        raise HTTPException(500, "Failed to download image")

    with open(local_path, "wb") as f:
        f.write(r.content)

    # 3️⃣ Run CLI inference
    cmd = [
        "python", INFER_SCRIPT,
        "--image", local_path,
        "--age", str(age),
        "--sex", str(sex),
        "--smoker", str(smoker).lower(),
        "--diabetes", str(diabetes).lower(),
        "--sbp", str(sbp),
        "--bp_meds", str(bp_meds).lower()
    ]

    log.info("RUNNING CVD INFERENCE → %s", " ".join(cmd))

    try:
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        output = json.loads(result.decode())
    except subprocess.CalledProcessError as e:
        log.error("INFERENCE FAILED → %s", e.output.decode())
        raise HTTPException(500, "Inference failed")

    # 4️⃣ Save prediction
    supabase.table("cvd_predictions").insert({
    "image_id": image_id,
    "user_id": user_id,
    "age": age,
    "sex": sex,
    "smoker": smoker,
    "diabetes": diabetes,
    "sbp": sbp,
    "bp_meds": bp_meds,
    "base_ml_risk": output["base_ml_risk"],
    "prs_proxy": output["prs_proxy"],
    "final_adjusted_risk": output["final_adjusted_risk"],
    "risk_delta": output["risk_delta"],
    "clinical_explanations": output["clinical_explanations"]
}).execute()

    log.info("CVD RESULT SAVED → %s", image_id)

    return {
        "image_id": image_id,
        "result": output
    }