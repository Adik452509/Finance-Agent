"""Phase 6.6 Google login: identity rules, the generated secrets file, and the sign-in gate. Offline."""

import tomllib

import pytest

from frontend import auth

GOOGLE = {"google_oauth_client_id": "123-abc.apps.googleusercontent.com",
          "google_oauth_client_secret": 'GOCSPX-se"cret\\x', "auth_cookie_secret": "c" * 64}


@pytest.fixture
def login_on(monkeypatch):
    for key, value in GOOGLE.items():
        monkeypatch.setattr(auth.settings, key, value)
    monkeypatch.setattr(auth.settings, "allowed_emails", "")
    monkeypatch.setattr(auth.settings, "public_app_url", "https://aditya-finance-agent.hf.space/")


def test_signed_out_until_google_says_otherwise(login_on):
    assert auth.resolve_identity({"is_logged_in": False}) == (None, None, "signed_out")
    user = {"is_logged_in": True, "email": "Aditya@Gmail.com", "email_verified": True, "name": "Aditya"}
    assert auth.resolve_identity(user) == ("aditya@gmail.com", "Aditya", None)


def test_unverified_email_is_refused(login_on):
    email, _, problem = auth.resolve_identity({"is_logged_in": True, "email": "x@y.com", "email_verified": False})
    assert email is None and "verified" in problem


def test_the_access_list_is_enforced(login_on, monkeypatch):
    monkeypatch.setattr(auth.settings, "allowed_emails", "aditya@gmail.com, interviewer@company.com")
    ok = {"is_logged_in": True, "email": "interviewer@company.com", "email_verified": True}
    stranger = {"is_logged_in": True, "email": "stranger@gmail.com", "email_verified": True}
    assert auth.resolve_identity(ok)[2] is None
    assert "not on this app's access list" in auth.resolve_identity(stranger)[2]


def test_without_login_dev_uses_the_dev_user_and_production_refuses(monkeypatch):
    monkeypatch.setattr(auth.settings, "google_oauth_client_id", None)
    monkeypatch.setattr(auth.settings, "app_env", "dev")
    assert auth.resolve_identity({"is_logged_in": False})[0] == auth.settings.dev_user_email
    monkeypatch.setattr(auth.settings, "app_env", "production")
    assert auth.resolve_identity({"is_logged_in": False})[2] == "Sign-in is not configured."


def test_production_settings_require_google_login():
    from config.settings import Settings

    with pytest.raises(ValueError, match="needs Google login"):
        Settings(_env_file=None, app_env="production", internal_api_token="t" * 40)
    Settings(_env_file=None, app_env="production", internal_api_token="t" * 40, **GOOGLE)  # complete: fine


def test_secrets_file_is_valid_toml_with_the_right_redirect(login_on, tmp_path):
    path = tmp_path / ".streamlit" / "secrets.toml"
    assert auth.write_secrets(path)
    config = tomllib.loads(path.read_text(encoding="utf-8"))["auth"]
    assert config["redirect_uri"] == "https://aditya-finance-agent.hf.space/oauth2callback"
    assert config["client_secret"] == 'GOCSPX-se"cret\\x'  # quotes and backslashes survive
    assert config["server_metadata_url"].startswith("https://accounts.google.com/")


def test_turning_login_off_removes_old_credentials(tmp_path, monkeypatch):
    path = tmp_path / "secrets.toml"
    path.write_text("[auth]\nclient_secret = 'old'\n", encoding="utf-8")
    monkeypatch.setattr(auth.settings, "google_oauth_client_id", None)
    assert auth.write_secrets(path) is False and not path.exists()


def test_a_different_person_starts_with_a_clean_session(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import frontend.session as session
    from tests.test_frontend import FakeApi

    who = {"email": "first@gmail.com"}
    monkeypatch.setattr(session, "ApiClient", FakeApi)
    monkeypatch.setattr(session, "streamlit_user",
                        lambda: {"is_logged_in": True, "email": who["email"], "email_verified": True})
    for key, value in GOOGLE.items():
        monkeypatch.setattr(auth.settings, key, value)
    app = AppTest.from_file(str(auth.settings.project_root / "frontend" / "app.py"), default_timeout=30)
    app.session_state["conversation_id"] = "c1"
    app.run()
    assert app.session_state["conversation_id"] == "c1"   # the first person keeps their open chat
    who["email"] = "second@gmail.com"                      # sign out, someone else signs in
    app.run()
    assert "conversation_id" not in app.session_state and app.session_state["user_email"] == "second@gmail.com"


def test_the_signin_page_is_shown_when_signed_out(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import frontend.session as session

    monkeypatch.setattr(session, "streamlit_user", lambda: {"is_logged_in": False})
    for key, value in GOOGLE.items():
        monkeypatch.setattr(auth.settings, key, value)
    app = AppTest.from_file(str(auth.settings.project_root / "frontend" / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Research Desk"
    assert [b.label for b in app.button] == ["Sign in with Google"]
    assert not app.sidebar.button  # nothing else is reachable before signing in


def test_an_account_not_on_the_list_gets_the_no_access_page(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import frontend.session as session

    monkeypatch.setattr(session, "streamlit_user",
                        lambda: {"is_logged_in": True, "email": "stranger@gmail.com", "email_verified": True})
    for key, value in GOOGLE.items():
        monkeypatch.setattr(auth.settings, key, value)
    monkeypatch.setattr(auth.settings, "allowed_emails", "aditya@gmail.com")
    app = AppTest.from_file(str(auth.settings.project_root / "frontend" / "app.py"), default_timeout=30)
    app.run()
    assert app.title[0].value == "No access" and [b.label for b in app.button] == ["Sign out"]
