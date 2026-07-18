import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def explain_condition(result):

    prompt = f"""
You are an ophthalmologist.

The following diagnosis refers ONLY to eye alignment and strabismus screening.

Diagnosis:
{result}

Explain:
1. What it means
2. Possible symptoms
3. Possible causes
4. Common treatments

Do not discuss body posture, bones, joints, muscles, or skeletal alignment.
"""

    response = model.generate_content(prompt)

    return response.text
