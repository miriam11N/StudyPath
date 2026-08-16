import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from chatbot import get_ai_response, grade_quiz_answer

st.set_page_config(
    page_title="StudyPath",
    page_icon="📚",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"

courses = pd.read_csv(DATA_DIR / "courses.csv")
students = pd.read_csv(DATA_DIR / "students.csv")
enrollments = pd.read_csv(DATA_DIR / "student_course_progress.csv")
topic_scores = pd.read_csv(DATA_DIR / "topic_scores.csv")

note_documents = []
note_metadata = []

for note_file in KNOWLEDGE_DIR.glob("*.txt"):
    text = note_file.read_text(encoding="utf-8")

    for sentence in text.split("."):
        sentence = sentence.strip()

        if len(sentence) > 25:
            note_documents.append(sentence)
            note_metadata.append(
                {
                    "source_file": note_file.name,
                    "text": sentence
                }
            )

vectorizer = TfidfVectorizer(stop_words="english")
note_matrix = vectorizer.fit_transform(note_documents)


def search_course_notes(query, course_id=None, top_k=3):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, note_matrix).flatten()

    results = pd.DataFrame(note_metadata)
    results["similarity_score"] = scores

    if course_id:
        results = results[
            results["source_file"].str.startswith(course_id)
        ]

    return results.sort_values(
        "similarity_score",
        ascending=False
    ).head(top_k)


def get_course_context(query, course_id):
    note_results = search_course_notes(
        query=query,
        course_id=course_id,
        top_k=3
    )

    if note_results.empty:
        return "No matching course notes were found."

    return "\n\n".join(
        f"Source: {row['source_file']}\n{row['text']}"
        for _, row in note_results.iterrows()
    )


def create_study_plan(student_id, course_id):
    enrollment = enrollments[
        (enrollments["student_id"] == student_id)
        & (enrollments["course_id"] == course_id)
    ].iloc[0]

    student_topics = topic_scores[
        (topic_scores["student_id"] == student_id)
        & (topic_scores["course_id"] == course_id)
    ].sort_values("topic_quiz_score")

    weak_topics = student_topics[
        student_topics["topic_quiz_score"] < 75
    ]

    if weak_topics.empty:
        weak_topics = student_topics.head(3)

    weekly_minutes = enrollment["weekly_study_hours"] * 60
    sessions_per_week = max(
        3,
        weekly_minutes // enrollment["preferred_session_minutes"]
    )

    plan = []

    for index, (_, topic_row) in enumerate(weak_topics.iterrows()):
        sessions = max(1, sessions_per_week // len(weak_topics))

        plan.append(
            {
                "Priority": index + 1,
                "Topic": topic_row["topic"],
                "Topic Score": topic_row["topic_quiz_score"],
                "Mastery Level": topic_row["mastery_level"],
                "Sessions This Week": sessions,
                "Session Minutes": enrollment["preferred_session_minutes"],
                "Recommended Activity": (
                    "Review course notes and complete practice questions"
                    if topic_row["topic_quiz_score"] < 60
                    else "Practice problems and self-quiz"
                )
            }
        )

    return pd.DataFrame(plan)


def apply_quiz_adaptation(plan, quiz_score, topic):
    adapted_plan = plan.copy()
    topic_mask = adapted_plan["Topic"] == topic

    if quiz_score < 60:
        adapted_plan.loc[
            topic_mask,
            "Sessions This Week"
        ] += 1

        adapted_plan.loc[
            topic_mask,
            "Recommended Activity"
        ] = (
            "Extra review session: review notes, correct quiz mistakes, "
            "and retake a practice quiz"
        )

    elif quiz_score < 80:
        adapted_plan.loc[
            topic_mask,
            "Recommended Activity"
        ] = (
            "Practice problems, self-quiz, and review missed concepts"
        )

    else:
        adapted_plan.loc[
            topic_mask,
            "Recommended Activity"
        ] = (
            "Short review, then move to your next weakest topic"
        )

    return adapted_plan


def get_quiz_adaptation(quiz_score, topic):
    if quiz_score < 60:
        return {
            "message": (
                f"Your practice score for {topic} was {quiz_score}%. "
                f"StudyPath added an extra review session for {topic}. "
                "Review the course notes, correct the difficult answers, "
                "and retake a practice quiz."
            )
        }

    if quiz_score < 80:
        return {
            "message": (
                f"Your practice score for {topic} was {quiz_score}%. "
                "You are making progress. Keep your current plan and complete "
                "one more self-quiz before the exam."
            )
        }

    return {
        "message": (
            f"Your practice score for {topic} was {quiz_score}%. "
            f"You are doing well with {topic}. Keep one short review session, "
            "then move more study time to your next weakest topic."
        )
    }


def generate_practice_quiz(student_id, course_id, topic):
    note_results = search_course_notes(
        query=topic,
        course_id=course_id,
        top_k=1
    )

    if note_results.empty:
        context = "No course note was found for this topic."
    else:
        context = note_results.iloc[0]["text"]

    questions = [
        f"In your own words, explain the main idea of {topic}.",
        f"Give one example of how {topic} could be used in this course.",
        f"What is one common mistake when studying {topic}, and how can it be avoided?"
    ]

    return context, questions


def get_risk_message(score, missed_sessions):
    if score < 60 or missed_sessions >= 3:
        return (
            "High Risk — prioritize weak topics and add extra practice sessions."
        )

    if score < 75 or missed_sessions >= 1:
        return (
            "Needs Support — follow the personalized plan and complete "
            "regular self-quizzes."
        )

    return (
        "On Track — continue reviewing and keep at least one session "
        "for each topic."
    )


st.title("StudyPath")
st.caption(
    "An AI-powered academic coach that creates study plans, finds weak topics, "
    "builds practice quizzes, and adapts recommendations when a student struggles."
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "My Study Plan",
        "Practice Quiz",
        "Dashboard",
        "StudyPath Assistant"
    ]
)

student_options = students[["student_id", "student_name"]].copy()
student_options["label"] = (
    student_options["student_id"]
    + " — "
    + student_options["student_name"]
)

with st.sidebar:
    st.header("Student Profile")

    selected_label = st.selectbox(
        "Choose a student",
        student_options["label"].tolist()
    )

    selected_student_id = selected_label.split(" — ")[0]

    available_courses = enrollments[
        enrollments["student_id"] == selected_student_id
    ][["course_id", "course_name"]].drop_duplicates()

    available_courses["label"] = (
        available_courses["course_id"]
        + " — "
        + available_courses["course_name"]
    )

    selected_course_label = st.selectbox(
        "Choose a course",
        available_courses["label"].tolist()
    )

    selected_course_id = selected_course_label.split(" — ")[0]
    selected_course_name = selected_course_label.split(" — ", 1)[1]

    st.divider()
    st.caption("StudyPath uses synthetic student data for demonstration only.")

selected_enrollment = enrollments[
    (enrollments["student_id"] == selected_student_id)
    & (enrollments["course_id"] == selected_course_id)
].iloc[0]

base_plan = create_study_plan(
    selected_student_id,
    selected_course_id
)

current_plan = base_plan.copy()

if (
    "latest_quiz_score" in st.session_state
    and "quiz_topic" in st.session_state
    and st.session_state.quiz_topic
):
    current_plan = apply_quiz_adaptation(
        plan=base_plan,
        quiz_score=st.session_state.latest_quiz_score,
        topic=st.session_state.quiz_topic
    )

weakest_topic = current_plan.iloc[0]["Topic"]
weakest_score = current_plan.iloc[0]["Topic Score"]

student_context = f"""
Student ID: {selected_student_id}
Selected course: {selected_course_name}
Latest quiz score: {selected_enrollment['latest_quiz_score']}%
Exam in: {selected_enrollment['exam_days_away']} days
Weekly available study time: {selected_enrollment['weekly_study_hours']} hours
Preferred session length: {selected_enrollment['preferred_session_minutes']} minutes
Missed sessions: {selected_enrollment['missed_sessions']}
Risk level: {selected_enrollment['risk_level']}
Weakest topic: {weakest_topic}
Weakest topic score: {weakest_score}%
"""

with tab1:
    st.header("Personalized Study Plan")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Latest Quiz Score",
        f"{selected_enrollment['latest_quiz_score']}%"
    )
    col2.metric(
        "Exam in",
        f"{selected_enrollment['exam_days_away']} days"
    )
    col3.metric(
        "Weekly Study Time",
        f"{selected_enrollment['weekly_study_hours']} hrs"
    )
    col4.metric(
        "Missed Sessions",
        selected_enrollment["missed_sessions"]
    )

    st.info(
        get_risk_message(
            selected_enrollment["latest_quiz_score"],
            selected_enrollment["missed_sessions"]
        )
    )

    if (
        "latest_quiz_score" in st.session_state
        and "quiz_topic" in st.session_state
        and st.session_state.quiz_topic
    ):
        adaptation = get_quiz_adaptation(
            quiz_score=st.session_state.latest_quiz_score,
            topic=st.session_state.quiz_topic
        )

        if st.session_state.latest_quiz_score < 60:
            st.warning(adaptation["message"])
        elif st.session_state.latest_quiz_score < 80:
            st.info(adaptation["message"])
        else:
            st.success(adaptation["message"])

    st.subheader("This Week's Focus")
    st.dataframe(
        current_plan,
        width="stretch",
        hide_index=True
    )

    st.subheader("Agent Recommendation")

    if weakest_score < 60:
        st.warning(
            f"Focus first on **{weakest_topic}**. "
            "Your score shows that this topic needs immediate review, "
            "extra practice, and a self-quiz before moving to the next topic."
        )
    else:
        st.success(
            f"Start with **{weakest_topic}**, then continue through your scheduled "
            "practice sessions to improve confidence before the exam."
        )

with tab2:
    st.header("Interactive Practice Quiz")
    st.caption(
        "Answer the questions, submit your work, and receive AI-generated "
        "feedback based on the selected course notes."
    )

    topic_list = base_plan["Topic"].tolist()

    selected_topic = st.selectbox(
        "Choose a topic to practice",
        topic_list,
        key="quiz_topic_selector"
    )

    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []

    if "quiz_context" not in st.session_state:
        st.session_state.quiz_context = ""

    if "quiz_topic" not in st.session_state:
        st.session_state.quiz_topic = None

    if "quiz_results" not in st.session_state:
        st.session_state.quiz_results = []

    if st.button("Start New Quiz", key="start_new_quiz"):
        context, questions = generate_practice_quiz(
            selected_student_id,
            selected_course_id,
            selected_topic
        )

        st.session_state.quiz_questions = questions
        st.session_state.quiz_context = context
        st.session_state.quiz_topic = selected_topic
        st.session_state.quiz_results = []
        st.session_state.pop("latest_quiz_score", None)

        st.rerun()

    if st.session_state.quiz_questions:
        st.subheader(
            f"Quiz Topic: {st.session_state.quiz_topic}"
        )

        with st.expander("Course note used to create this quiz"):
            st.info(st.session_state.quiz_context)

        with st.form("practice_quiz_form"):
            answers = []

            for index, question in enumerate(
                st.session_state.quiz_questions,
                start=1
            ):
                st.markdown(f"**Question {index}:** {question}")

                answer = st.text_area(
                    "Your answer",
                    key=f"quiz_answer_{index}",
                    height=110
                )

                answers.append(answer)

            submitted = st.form_submit_button("Submit Answers")

        if submitted:
            if not any(answer.strip() for answer in answers):
                st.warning(
                    "Please answer at least one question before submitting."
                )
            else:
                quiz_results = []
                total_score = 0
                max_score = len(st.session_state.quiz_questions) * 2

                with st.spinner("Checking your practice answers..."):
                    for question, answer in zip(
                        st.session_state.quiz_questions,
                        answers
                    ):
                        feedback = grade_quiz_answer(
                            question=question,
                            student_answer=answer,
                            course_context=st.session_state.quiz_context
                        )

                        quiz_results.append(
                            {
                                "question": question,
                                "answer": answer,
                                "feedback": feedback
                            }
                        )

                        try:
                            score_line = feedback.splitlines()[0]
                            score_value = int(
                                score_line.replace("Score:", "")
                                .split("/")[0]
                                .strip()
                            )
                            total_score += score_value
                        except (ValueError, IndexError):
                            pass

                st.session_state.quiz_results = quiz_results
                st.session_state.latest_quiz_score = round(
                    (total_score / max_score) * 100
                )

                st.rerun()

        if st.session_state.quiz_results:
            st.divider()
            st.subheader("Your Results")

            score_percent = st.session_state.latest_quiz_score

            if score_percent >= 80:
                st.success(
                    f"Practice Quiz Score: {score_percent}% — Great work."
                )
            elif score_percent >= 60:
                st.info(
                    f"Practice Quiz Score: {score_percent}% — "
                    "You are making progress. Review the feedback below."
                )
            else:
                st.warning(
                    f"Practice Quiz Score: {score_percent}% — "
                    "Spend another study session reviewing this topic."
                )

            for index, result in enumerate(
                st.session_state.quiz_results,
                start=1
            ):
                with st.expander(f"Question {index} Feedback"):
                    st.markdown(f"**Question:** {result['question']}")
                    st.markdown(f"**Your answer:** {result['answer']}")
                    st.markdown(result["feedback"])

            st.warning(
                "This is practice feedback, not an official grade. "
                "Use it to identify what to review next."
            )

        if st.button("Clear Quiz", key="clear_quiz"):
            st.session_state.quiz_questions = []
            st.session_state.quiz_context = ""
            st.session_state.quiz_topic = None
            st.session_state.quiz_results = []
            st.session_state.pop("latest_quiz_score", None)

            for index in range(1, 4):
                st.session_state.pop(f"quiz_answer_{index}", None)

            st.rerun()

with tab3:
    st.header("Learning Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Students", students["student_id"].nunique())
    col2.metric("Course Enrollments", len(enrollments))
    col3.metric(
        "High-Risk Students",
        (enrollments["risk_level"] == "High Risk").sum()
    )
    col4.metric(
        "Students Needing Support",
        (enrollments["risk_level"] == "Needs Support").sum()
    )

    st.subheader("Student Risk Distribution")

    risk_counts = (
        enrollments["risk_level"]
        .value_counts()
        .reset_index()
    )
    risk_counts.columns = ["Risk Level", "Students"]

    st.bar_chart(risk_counts.set_index("Risk Level"))

    st.subheader("Average Quiz Score by Course")

    average_scores = (
        enrollments.groupby("course_name")["latest_quiz_score"]
        .mean()
        .sort_values()
    )

    st.bar_chart(average_scores)

    st.subheader("Students Who Need Attention")

    attention_students = enrollments[
        enrollments["risk_level"] != "On Track"
    ].sort_values("latest_quiz_score")

    st.dataframe(
        attention_students[
            [
                "student_id",
                "course_name",
                "latest_quiz_score",
                "missed_sessions",
                "risk_level"
            ]
        ],
        width="stretch",
        hide_index=True
    )

    st.subheader("Topic Performance")

    topic_performance = (
        topic_scores.groupby(["course_name", "topic"])["topic_quiz_score"]
        .mean()
        .reset_index()
        .sort_values("topic_quiz_score")
    )

    st.dataframe(
        topic_performance,
        width="stretch",
        hide_index=True
    )

with tab4:
    st.header("StudyPath Assistant")
    st.caption(
        "Ask for help understanding a concept, creating practice questions, "
        "or deciding what to study next."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("Explain my weak topic"):
            st.session_state.pending_prompt = (
                f"Explain my weakest topic, {weakest_topic}, in simple words. "
                "Give one short example and one practice activity."
            )

    with action_col2:
        if st.button("Make a practice quiz"):
            st.session_state.pending_prompt = (
                f"Create five original practice questions about {weakest_topic}. "
                "Do not provide answers until I ask."
            )

    with action_col3:
        if st.button("Update my study plan"):
            st.session_state.pending_prompt = (
                "Review my student data and give me a realistic study plan "
                "for the next seven days."
            )

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_prompt = st.chat_input(
        "Example: I have 30 minutes today. What should I study first?"
    )

    pending_prompt = st.session_state.pop("pending_prompt", None)
    prompt = pending_prompt or typed_prompt

    if prompt:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("StudyPath is thinking..."):
                try:
                    course_context = get_course_context(
                        query=prompt,
                        course_id=selected_course_id
                    )

                    response = get_ai_response(
                        user_message=prompt,
                        course_name=selected_course_name,
                        course_context=course_context,
                        student_context=student_context,
                        chat_history=st.session_state.chat_messages[:-1]
                    )

                    st.markdown(response)

                    with st.expander("Course notes used"):
                        st.write(course_context)

                    st.session_state.chat_messages.append(
                        {
                            "role": "assistant",
                            "content": response
                        }
                    )

                except Exception as error:
                    st.error(
                        "The assistant could not respond. Confirm that your "
                        "OPENAI_API_KEY is correct and that you have internet access."
                    )
                    st.code(str(error))

    if st.button("Clear conversation"):
        st.session_state.chat_messages = []
        st.rerun()