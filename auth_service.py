import streamlit as st
from supabase import create_client, Client

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

def sign_up(name, email, password):
    try:
        response = supabase.auth.sign_up(
            {
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": name
                    }
                }
            }
        )

        if response.user is None:
            return {
                "success": False,
                "error": (
                    "Account was not created. Check the email address "
                    "or try again."
                )
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


def create_or_update_profile(user, full_name=None):
    try:
        metadata = user.user_metadata or {}

        profile_name = (
            full_name
            or metadata.get("full_name")
            or ""
        )

        response = (
            supabase.table("profiles")
            .upsert(
                {
                    "id": str(user.id),
                    "full_name": profile_name
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