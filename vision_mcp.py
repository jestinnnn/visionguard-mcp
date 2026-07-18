import cv2
import mediapipe as mp
import numpy as np


def analyze_eye(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Image not found")

    # -----------------------------
    # Preprocessing
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    gray = clahe.apply(gray)

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # -----------------------------
    # Padding
    # -----------------------------
    pad = 150

    canvas = (
        np.ones((img.shape[0] + 2 * pad, img.shape[1] + 2 * pad, 3), dtype=np.uint8)
        * 255
    )

    canvas[pad : pad + img.shape[0], pad : pad + img.shape[1]] = img

    h, w = canvas.shape[:2]

    # -----------------------------
    # MediaPipe
    # -----------------------------
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    ) as face_mesh:

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        raise Exception("No face detected")

    face = results.multi_face_landmarks[0]

    # -----------------------------
    # Landmark Helper
    # -----------------------------
    def get_point(index):

        x = int(face.landmark[index].x * w)
        y = int(face.landmark[index].y * h)

        return np.array([x, y])

    # -----------------------------
    # Eye Landmarks
    # -----------------------------
    RE_outer = get_point(33)
    RE_inner = get_point(133)

    LE_outer = get_point(263)
    LE_inner = get_point(362)

    # -----------------------------
    # Iris Centers
    # -----------------------------
    RE_iris = np.mean([get_point(i) for i in [469, 470, 471, 472]], axis=0)

    LE_iris = np.mean([get_point(i) for i in [474, 475, 476, 477]], axis=0)

    # -----------------------------
    # Offset Calculation
    # -----------------------------
    def calculate_offset(outer, inner, iris):

        width = abs(inner[0] - outer[0])

        center = (inner[0] + outer[0]) / 2

        return float((iris[0] - center) / width)

    RE_offset = calculate_offset(RE_outer, RE_inner, RE_iris)

    LE_offset = calculate_offset(LE_outer, LE_inner, LE_iris)

    # -----------------------------
    # Alignment Analysis
    # -----------------------------
    alignment_difference = abs(LE_offset - RE_offset)

    if alignment_difference < 0.13:

        diagnosis = "NORMAL EYE ALIGNMENT"

    elif alignment_difference < 0.17:

        diagnosis = "MILD EYE MISALIGNMENT"

    else:

        diagnosis = "POSSIBLE STRABISMUS (EYE MISALIGNMENT)"

    if diagnosis == "NORMAL EYE ALIGNMENT":
        confidence = round(max(80, 100 - alignment_difference * 150), 1)
    else:
        confidence = round(min(95, 60 + alignment_difference * 200), 1)

    angle = round(alignment_difference * 25, 2)

    prism = round(angle * 1.75, 2)

    # -----------------------------
    # Debug Output
    # -----------------------------
    print("\n-------------------------")

    print("LEFT OFFSET :", LE_offset)
    print("RIGHT OFFSET:", RE_offset)
    print("DIFFERENCE  :", alignment_difference)

    print("DIAGNOSIS   :", diagnosis)
    print("CONFIDENCE  :", confidence)

    print("-------------------------")

    # -----------------------------
    # Debug Image
    # -----------------------------
    debug = canvas.copy()

    cv2.circle(debug, tuple(RE_iris.astype(int)), 8, (0, 0, 255), -1)

    cv2.circle(debug, tuple(LE_iris.astype(int)), 8, (255, 0, 0), -1)

    for p in [RE_outer, RE_inner, LE_outer, LE_inner]:

        cv2.circle(debug, tuple(p.astype(int)), 5, (0, 255, 0), -1)

    cv2.imwrite("debug_eye_analysis.jpg", debug)

    # -----------------------------
    # Return
    # -----------------------------
    return {
        "left_offset": round(LE_offset, 4),
        "right_offset": round(RE_offset, 4),
        "difference": round(alignment_difference, 4),
        "diagnosis": diagnosis,
        "confidence": confidence,
        "angle": angle,
        "prism": prism,
        "debug_image": "debug_eye_analysis.jpg",
    }
