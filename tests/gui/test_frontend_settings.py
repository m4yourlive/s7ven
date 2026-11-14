from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HTML_FILE = (
    PROJECT_ROOT
    / "src"
    / "seven"
    / "gui"
    / "frontend"
    / "index.html"
)


def read_html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


def test_settings_navigation_exists() -> None:
    html = read_html()

    assert 'data-nav-target="settings"' in html
    assert "SETTINGS" in html


def test_settings_page_exists() -> None:
    html = read_html()

    assert 'id="settings-page"' in html
    assert 'data-page="settings"' in html
    assert 'aria-label="Application settings"' in html


def test_new_session_page_is_preserved() -> None:
    html = read_html()

    assert 'id="new-session-page"' in html
    assert 'id="conversation"' in html
    assert 'id="message-form"' in html
    assert 'id="message-input"' in html
    assert 'id="send-button"' in html


def test_general_settings_exist() -> None:
    html = read_html()

    assert 'data-settings-section="general"' in html
    assert 'id="language-setting"' in html
    assert 'id="startup-setting"' in html


def test_appearance_settings_exist() -> None:
    html = read_html()

    assert 'data-settings-section="appearance"' in html
    assert 'id="theme-setting"' in html
    assert 'id="font-size-setting"' in html


def test_system_settings_exist() -> None:
    html = read_html()

    assert 'data-settings-section="system"' in html
    assert 'id="settings-runtime-status"' in html


def test_settings_category_navigation_exists() -> None:
    html = read_html()

    assert 'data-settings-target="general"' in html
    assert 'data-settings-target="appearance"' in html
    assert 'data-settings-target="system"' in html


def test_settings_save_controls_exist() -> None:
    html = read_html()

    assert 'id="settings-save"' in html
    assert 'id="settings-reset"' in html
    assert 'id="settings-save-status"' in html


def test_settings_page_starts_hidden() -> None:
    html = read_html()

    settings_start = html.index('id="settings-page"')
    settings_markup = html[settings_start:settings_start + 300]

    assert "hidden" in settings_markup


def test_settings_sections_have_accessible_labels() -> None:
    html = read_html()

    assert 'for="language-setting"' in html
    assert 'for="startup-setting"' in html
    assert 'for="theme-setting"' in html
    assert 'for="font-size-setting"' in html
    assert 'aria-live="polite"' in html


CSS_FILE = (
    PROJECT_ROOT
    / "src"
    / "seven"
    / "gui"
    / "frontend"
    / "assets"
    / "css"
    / "style.css"
)


def read_css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def test_settings_styles_exist() -> None:
    css = read_css()

    assert ".settings-page" in css
    assert ".settings-header" in css
    assert ".settings-layout" in css
    assert ".settings-content" in css


def test_settings_navigation_styles_exist() -> None:
    css = read_css()

    assert ".settings-navigation" in css
    assert ".settings-nav-item" in css
    assert ".settings-nav-item.active" in css


def test_settings_section_styles_exist() -> None:
    css = read_css()

    assert ".settings-section-heading" in css
    assert ".settings-section-index" in css
    assert ".settings-group" in css
    assert ".setting-item" in css


def test_settings_controls_are_styled() -> None:
    css = read_css()

    assert ".setting-item select" in css
    assert '.setting-item input[type="checkbox"]' in css


def test_settings_action_buttons_are_styled() -> None:
    css = read_css()

    assert ".settings-reset-button" in css
    assert ".settings-save-button" in css
    assert ".settings-save-status" in css


def test_settings_hidden_elements_remain_hidden() -> None:
    css = read_css()

    assert ".page-view[hidden]" in css
    assert ".settings-section[hidden]" in css
    assert "display: none !important" in css


def test_settings_responsive_layout_exists() -> None:
    css = read_css()

    assert "@media (max-width: 760px)" in css
    assert "@media (max-width: 560px)" in css
