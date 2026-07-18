from fastapi import FastAPI
from pydantic import BaseModel
import uuid

from vision_mcp import analyze_eye
from medical_mcp import explain_condition
from report_mcp import generate_report

app = FastAPI(
    title="VisionGuard API",
    version="1.0.0"
)


# -------------------------
# Request Models
# -------------------------

class AnalyzeRequest(BaseModel):
    image_file: str


class ExplainRequest(BaseModel):
    diagnosis: str


class ReportRequest(BaseModel):
    diagnosis: str
    angle: float
    prism: float
    confidence: float
    explanation: str


# -------------------------
# Analyze Eye
# -------------------------

@app.post("/analyze")
def analyze_eye_alignment(data: AnalyzeRequest):

    result = analyze_eye(data.image_file)

    return {
        "diagnosis": result["diagnosis"],
        "confidence": result["confidence"],
        "angle": result["angle"],
        "prism": result["prism"],
        "left_offset": result["left_offset"],
        "right_offset": result["right_offset"],
        "difference": result["difference"],
        "debug_image": result["debug_image"]
    }


# -------------------------
# Explain Diagnosis
# -------------------------

@app.post("/explain")
def explain_diagnosis(data: ExplainRequest):

    explanation = explain_condition(data.diagnosis)

    return {
        "diagnosis": data.diagnosis,
        "explanation": explanation
    }


# -------------------------
# Generate PDF Report
# -------------------------

@app.post("/report")
def generate_eye_report(data: ReportRequest):

    pdf_path = generate_report(
        file_name=f"{uuid.uuid4()}.pdf",
        diagnosis=data.diagnosis,
        angle=data.angle,
        prism=data.prism,
        confidence=data.confidence,
        explanation=data.explanation
    )

    return {
        "report_file": pdf_path
    }


# -------------------------
# Full Pipeline
# -------------------------

@app.post("/full-analysis")
def full_eye_analysis(data: AnalyzeRequest):

    result = analyze_eye(data.image_file)

    diagnosis = result["diagnosis"]

    explanation = explain_condition(diagnosis)

    pdf_path = generate_report(
        file_name=f"{uuid.uuid4()}.pdf",
        diagnosis=diagnosis,
        angle=result["angle"],
        prism=result["prism"],
        confidence=result["confidence"],
        explanation=explanation
    )

    return {
        "diagnosis": diagnosis,
        "confidence": result["confidence"],
        "angle": result["angle"],
        "prism": result["prism"],
        "left_offset": result["left_offset"],
        "right_offset": result["right_offset"],
        "difference": result["difference"],
        "explanation": explanation,
        "pdf_report": pdf_path,
        "debug_image": result["debug_image"]
    }


# -------------------------
# Health Check
# -------------------------

@app.get("/")
def health():

    return {
        "status": "running",
        "service": "VisionGuard API"
    }