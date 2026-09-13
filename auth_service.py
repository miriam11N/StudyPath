import streamlit as st
from supabase import create_client, Client


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


def sign_up(display_name, email, password):
    try:
        response = supabase.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "display_name": display_name
                    }
                }
            }
        )

        return {
            "success": True,
            "user": response.user
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


def sign_in(email, password):
    try:
        response = supabase.auth.sign_in_with_password(
            {
                "email": email,
                "password": password
            }
        )

        if response.user is None or response.session is None:
            return {
                "success": False,
                "error": "Invalid email or password."
            }

        return {
            "success": True,
            "user": response.user,
            "session": response.session
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


def sign_out():
    try:
        supabase.auth.sign_out()

        return {
            "success": True
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }

def get_profile(user_id):
    try:
        response = (
            supabase.table("profiles")
            .select("*")
            .eq("id", str(user_id))
            .single()
            .execute()
        )

        return {
            "success": True,
            "data": response.data
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


def save_onboarding(
    user_id,
    grade_level,
    school_name,
    exam_date,
    weekly_study_hours,
    study_goals
):
    try:
        response = (
            supabase.table("profiles")
            .update(
                {
                    "grade_level": grade_level,
                    "school_name": school_name,
                    "exam_date": exam_date.isoformat(),
                    "weekly_study_hours": weekly_study_hours,
                    "study_goals": study_goals,
                    "onboarding_complete": True
                }
            )
            .eq("id", str(user_id))
            .execute()
        )

        return {
            "success": True,
            "data": response.data
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }

def get_profile(user_id):
    try:
        response = (
            supabase.table("profiles")
            .select("*")
            .eq("id", str(user_id))
            .single()
            .execute()
        )

        return {
            "success": True,
            "data": response.data
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


def save_onboarding(
    user_id,
    grade_level,
    school_name,
    exam_date,
    weekly_study_hours,
    study_goals
):
    try:
        response = (
            supabase.table("profiles")
            .update(
                {
                    "grade_level": grade_level,
                    "school_name": school_name,
                    "exam_date": exam_date.isoformat(),
                    "weekly_study_hours": weekly_study_hours,
                    "study_goals": study_goals,
                    "onboarding_complete": True
                }
            )
            .eq("id", str(user_id))
            .execute()
        )

        return {
            "success": True,
            "data": response.data
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }

def get_user_courses(user_id):
    try:
        response = (
            supabase.table("user_courses")
            .select("*")
            .eq("user_id", str(user_id))
            .order("created_at")
            .execute()
        )

        return {
            "success": True,
            "data": response.data
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


def add_user_course(
    user_id,
    course_code,
    course_name,
    instructor_name,
    exam_date,
    confidence_level
):
    try:
        response = (
            supabase.table("user_courses")
            .insert(
                {
                    "user_id": str(user_id),
                    "course_code": course_code,
                    "course_name": course_name,
                    "instructor_name": instructor_name or None,
                    "exam_date": (
                        exam_date.isoformat()
                        if exam_date else None
                    ),
                    "confidence_level": confidence_level
                }
            )
            .execute()
        )

        return {
            "success": True,
            "data": response.data
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }


def delete_user_course(course_id, user_id):
    try:
        response = (
            supabase.table("user_courses")
            .delete()
            .eq("id", str(course_id))
            .eq("user_id", str(user_id))
            .execute()
        )

        return {
            "success": True,
            "data": response.data
        }

    except Exception as error:
        return {
            "success": False,
            "error": str(error)
        }