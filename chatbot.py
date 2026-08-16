import os
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