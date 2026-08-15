import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
            note_metadata.append({
                "source_file": note_file.name,
                "text": sentence
            })

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


def create_study_plan(student_id, course_id):
    enrollment = enrollments[
        (enrollments["student_id"] == student_id) &
        (enrollments["course_id"] == course_id)
    ].iloc[0]

    student_topics = topic_scores[
        (topic_scores["student_id"] == student_id) &
        (topic_scores["course_id"] == course_id)
    ].sort_values("topic_quiz_score")

    weak_topics = student_topics[student_topics["topic_quiz_score"] < 75]

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

        plan.append({
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
        })

    return pd.DataFrame(plan)


def generate_practice_quiz(student_id, course_id, topic):
    note_results = search_course_notes(
        query=topic,
        course_id=course_id,
        top_k=1
    )

    context = note_results.iloc[0]["text"]

    questions = [
        f"In your own words, explain the main idea of {topic}.",
        f"Give one example of how {topic} could be used in this course.",
        f"What is one common mistake when studying {topic}, and how can it be avoided?"
    ]

    return context, questions


def get_risk_message(score, missed_sessions):
    if score < 60 or missed_sessions >= 3:
        return "High Risk — prioritize weak topics and add extra practice sessions."
    if score < 75 or missed_sessions >= 1:
        return "Needs Support — follow the personalized plan and complete regular self-quizzes."
    return "On Track — continue reviewing and keep at least one session for each topic."


st.title("StudyPath")
st.caption(
    "An AI-powered academic coach that creates study plans, finds weak topics, "
    "builds practice quizzes, and adapts recommendations when a student struggles."
)

tab1, tab2, tab3 = st.tabs([
    "My Study Plan",
    "Practice Quiz",
    "Dashboard"
])

student_options = students[["student_id", "student_name"]].copy()
student_options["label"] = (
    student_options["student_id"] + " — " + student_options["student_name"]
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
        available_courses["course_id"] + " — " +
        available_courses["course_name"]
    )

    selected_course_label = st.selectbox(
        "Choose a course",
        available_courses["label"].tolist()
    )

    selected_course_id = selected_course_label.split(" — ")[0]

    st.divider()
    st.caption("StudyPath uses synthetic student data for demonstration only.")

selected_enrollment = enrollments[
    (enrollments["student_id"] == selected_student_id) &
    (enrollments["course_id"] == selected_course_id)
].iloc[0]

with tab1:
    st.header("Personalized Study Plan")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Latest Quiz Score", f"{selected_enrollment['latest_quiz_score']}%")
    col2.metric("Exam in", f"{selected_enrollment['exam_days_away']} days")
    col3.metric("Weekly Study Time", f"{selected_enrollment['weekly_study_hours']} hrs")
    col4.metric("Missed Sessions", selected_enrollment["missed_sessions"])

    st.info(
        get_risk_message(
            selected_enrollment["latest_quiz_score"],
            selected_enrollment["missed_sessions"]
        )
    )

    plan = create_study_plan(selected_student_id, selected_course_id)

    st.subheader("This Week's Focus")
    st.dataframe(plan, use_container_width=True, hide_index=True)

    st.subheader("Agent Recommendation")

    weakest_topic = plan.iloc[0]["Topic"]
    weakest_score = plan.iloc[0]["Topic Score"]

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
    st.header("Practice Quiz")

    plan = create_study_plan(selected_student_id, selected_course_id)
    topic_list = plan["Topic"].tolist()

    selected_topic = st.selectbox(
        "Choose a topic to practice",
        topic_list
    )

    if st.button("Generate Practice Quiz"):
        context, questions = generate_practice_quiz(
            selected_student_id,
            selected_course_id,
            selected_topic
        )

        st.subheader("Retrieved Course Note")
        st.info(context)

        st.subheader(f"Practice Questions: {selected_topic}")

        for number, question in enumerate(questions, start=1):
            st.write(f"{number}. {question}")

        st.warning(
            "Academic integrity reminder: These are practice questions for learning. "
            "Do not submit AI-generated answers as graded coursework."
        )

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
        use_container_width=True,
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
        use_container_width=True,
        hide_index=True
    )