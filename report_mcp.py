from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)

def generate_report(
    file_name,
    diagnosis,
    angle,
    prism,
    confidence,
    explanation
):

    pdf = SimpleDocTemplate(file_name)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "VisionGuard AI Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1,12))

    content.append(
        Paragraph(
            f"Diagnosis: {diagnosis}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Angle: {angle} degrees",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Prism Value: {prism} PD",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"Confidence: {confidence}%",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            explanation,
            styles["BodyText"]
        )
    )

    pdf.build(content)

    return file_name