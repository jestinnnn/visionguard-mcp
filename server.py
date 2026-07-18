from mcp.server.fastmcp import FastMCP
import uuid
from vision_mcp import analyze_eye
from medical_mcp import explain_condition
from report_mcp import generate_report

mcp = FastMCP("VisionGuard")


@mcp.tool()
def analyze_eye_alignment(image_file: str):
    """
    Analyze eye alignment from an uploaded image.
    """

    result = analyze_eye(image_file)

    return {
        "diagnosis": result["diagnosis"],
        "confidence": result["confidence"],
        "angle": result["angle"],
        "prism": result["prism"],
        "left_offset": result["left_offset"],
        "right_offset": result["right_offset"],
        "difference": result["difference"],
        "debug_image": result["debug_image"],
    }


@mcp.tool()
def explain_diagnosis(diagnosis: str):
    """
    Generate medical explanation using Gemini.
    """

    explanation = explain_condition(diagnosis)

    return {"diagnosis": diagnosis, "explanation": explanation}


@mcp.tool()
def generate_eye_report(
    diagnosis: str, angle: float, prism: float, confidence: float, explanation: str
):
    """
    Generate PDF report.
    """

    pdf_path = generate_report(
        file_name=f"{uuid.uuid4()}.pdf",
        diagnosis=diagnosis,
        angle=angle,
        prism=prism,
        confidence=confidence,
        explanation=explanation,
    )

    return {"report_file": pdf_path}


@mcp.tool()
def full_eye_analysis(image_file: str):
    """
    Complete pipeline:
    Image -> Analysis -> Explanation -> PDF
    """

    result = analyze_eye(image_file)

    diagnosis = result["diagnosis"]

    explanation = explain_condition(diagnosis)

    pdf_path = generate_report(
        file_name="report.pdf",
        diagnosis=diagnosis,
        angle=result["angle"],
        prism=result["prism"],
        confidence=result["confidence"],
        explanation=explanation,
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
        "debug_image": result["debug_image"],
    }


if __name__ == "__main__":
    mcp.run()
