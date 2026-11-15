from app.core.config import Settings, get_settings


def test_settings_load_from_environment():
    settings = get_settings()
    assert settings.APP_NAME == "AtmoGuard API"
    assert settings.APP_ENV == "test"
    assert settings.DATABASE_URL.endswith("atmoguard_test")


def test_cors_origins_list_parses_comma_separated_string():
    settings = Settings(CORS_ORIGINS="http://a.com, http://b.com ,http://c.com")
    assert settings.cors_origins_list == ["http://a.com", "http://b.com", "http://c.com"]


def test_cors_origins_list_handles_single_origin():
    settings = Settings(CORS_ORIGINS="http://localhost:5173")
    assert settings.cors_origins_list == ["http://localhost:5173"]


def test_get_settings_is_cached():
    # Same object identity across calls (lru_cache) — config isn't re-parsed
    # on every request.
    assert get_settings() is get_settings()
