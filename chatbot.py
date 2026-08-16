import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. Add it to a .env file in the StudyPath folder."
    )

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You are StudyPath Assistant, a friendly and supportive academic coach.

Your role:
- Help students understand concepts in simple, clear language.
- Help students make realistic study plans.
- Create original practice questions, flashcards, and study tips.
- Use the selected course details and retrieved course notes when supplied.
- Be honest when the course notes do not contain enough information.

Academic-integrity rules:
- Do not provide answers for a live graded exam, quiz, homework, or assignment.
- Instead, explain the concept, demonstrate a similar example, or create practice questions.
- Do not claim that you completed work the student should complete themselves.
"""


def get_ai_response(
    user_message,
    course_name="",
    course_context="",
    student_context="",
    chat_history=None
):
    history_text = ""

    if chat_history:
        recent_history = chat_history[-8:]
        history_text = "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in recent_history
        )

    prompt = f"""
Selected course:
{course_name or "No course selected"}

Student context:
{student_context or "No student data available"}

Relevant course notes:
{course_context or "No relevant course notes were found."}

Recent conversation:
{history_text or "No previous messages."}

Current student message:
{user_message}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=prompt
    )

    return response.output_text


def grade_quiz_answer(question, student_answer, course_context):
    prompt = f"""
You are grading a practice answer for StudyPath.

Course-note context:
{course_context}

Question:
{question}

Student answer:
{student_answer}

Give supportive, concise feedback for a student.

Return your response exactly in this format:

Score: X/2
Feedback: Your feedback here.
Suggested answer: A short example of a stronger answer.

Rules:
- Score from 0 to 2 only.
- Give 2 for a correct, clear answer.
- Give 1 for a partially correct answer.
- Give 0 for an incorrect, blank, or unrelated answer.
- Do not pretend this is an official grade.
- Use simple language.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=prompt
    )

    return response.output_text


def generate_flashcards(topic, course_context, number_of_cards=6):
    prompt = f"""
Create exactly {number_of_cards} concise study flashcards about {topic}.

Use only the course-note context below.

Course-note context:
{course_context}

Return only valid JSON in exactly this structure:

[
  {{
    "question": "Short flashcard question",
    "answer": "Short, clear answer"
  }}
]

Rules:
- Return a JSON list only.
- Do not use Markdown or code fences.
- Do not add an introduction, conclusion, labels, or any extra text.
- Create exactly {number_of_cards} flashcards.
- Keep each answer under 45 words.
- Make questions useful for active recall.
- Use only concepts supported by the supplied course notes.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=SYSTEM_PROMPT,
        input=prompt
    )

    cleaned_text = response.output_text.strip()

    if cleaned_text.startswith("```"):
        cleaned_text = (
            cleaned_text
            .replace("```json", "")
            .replace("```JSON", "")
            .replace("```", "")
            .strip()
        )

    start_index = cleaned_text.find("[")
    end_index = cleaned_text.rfind("]")

    if start_index == -1 or end_index == -1:
        raise ValueError(
            "The flashcard generator did not return a valid JSON list."
        )

    cleaned_text = cleaned_text[start_index:end_index + 1]
    flashcards = json.loads(cleaned_text)

    if not isinstance(flashcards, list):
        raise ValueError(
            "The flashcard generator returned data in an invalid format."
        )

    valid_flashcards = []

    for card in flashcards:
        if (
            isinstance(card, dict)
            and card.get("question")
            and card.get("answer")
        ):
            valid_flashcards.append(
                {
                    "question": str(card["question"]).strip(),
                    "answer": str(card["answer"]).strip()
                }
            )

    if not valid_flashcards:
        raise ValueError(
            "No valid flashcards were created. Try another topic."
        )

    return valid_flashcards[:number_of_cards]