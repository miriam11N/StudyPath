StudyPath

StudyPath is a study-support app that helps students figure out what to focus on before an exam. It looks at quiz scores, weak topics, missed study sessions, available study time, and upcoming deadlines to create a simple personalized study plan.

The project uses synthetic student data, so no real student information is included. It is designed as a portfolio and learning project that shows how data, simple recommendation logic, and retrieval-based AI features can support academic success.

What It Does

StudyPath allows users to:

- Select a student and course.
- View latest quiz score, exam timeline, weekly study time, and missed sessions.
- Identify weak or developing topics that need more attention.
- Generate a weekly study plan with recommended session lengths and activities.
- Flag students who may need extra support.
- Generate practice questions based on relevant course notes.
- Retrieve relevant course-note content using a simple RAG-style search process.
- View a dashboard of student risk levels, average scores, topic performance, and students who need attention.

Why I Built It

Students often do not know where to start when they have limited study time. They may spend time reviewing topics they already understand while avoiding areas where they need more practice.

StudyPath makes studying more focused. Instead of giving every student the same plan, it uses performance data to highlight the topics that need the most attention first.

For example, if a student has a low score in Data Visualization and missed several study sessions, StudyPath prioritizes that topic and recommends extra review and practice questions.

Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- TF-IDF and cosine similarity for course-note retrieval
- Jupyter Notebook for synthetic data generation
- Git and GitHub

Dataset

The app uses a synthetic dataset of 100 students. The generated data includes:

- Student IDs and names
- Courses and course topics
- Quiz scores
- Topic-level mastery scores
- Weekly available study hours
- Preferred study session length
- Missed study sessions
- Exam deadlines
- Student risk levels

The project also includes synthetic course notes for:

- Introduction to Python
- Data Science Fundamentals
- College Algebra
- Business Analytics

How the Recommendation Works

StudyPath checks a student’s topic scores and starts with the lowest-performing topics. It then uses the student’s available weekly study time and preferred session length to recommend sessions for the week.

The app flags students as:

- High Risk when they have very low scores or several missed sessions.
- Needs Support when they have moderate scores or some missed sessions.
- On Track when their performance and participation are strong.

Practice Quiz and RAG

The Practice Quiz tab uses a lightweight RAG-style workflow:

1. A student selects a topic.
2. The app searches matching synthetic course notes using TF-IDF similarity.
3. The most relevant course-note section is shown.
4. The app creates practice questions for that topic.

The quiz feature is meant for study and self-review only.

Academic Integrity

StudyPath is a coaching tool, not a homework-answer generator.

The app provides practice questions, study guidance, and course-note retrieval. Students should not submit generated responses as graded coursework and should always follow their instructor’s policy on AI use.

How to Run the App

1. Clone the repository:

```text
git clone https://github.com/miriam11N/StudyPath.git
cd StudyPath
```

2. Create and activate a virtual environment:

```text
python -m venv venv
venv\Scripts\Activate.ps1
```

3. Install the dependencies:

```text
pip install -r requirements.txt
```

4. Run the application:

```text
streamlit run app.py
```

The app will open in your browser at:

```text
http://localhost:8501
```

Future Improvements

- Add more subjects and more course materials.
- Add detailed practice tests with multiple-choice questions and scoring.
- Add a chatbot so students can ask questions about a topic or their study plan.
- Add a student progress log that updates after each quiz.
- Automatically update the study plan when a student misses sessions or gets a low score.
- Add score-improvement charts over time.
- Allow students or instructors to upload their own course notes.
- Deploy the application on Streamlit Community Cloud.

Author

Mihir Bapat

Note

This project is built for demonstration and learning purposes. All student profiles, scores, deadlines, and course notes are synthetic.
