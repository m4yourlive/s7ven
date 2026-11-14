from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = (
    PROJECT_ROOT
    / "src"
    / "seven"
    / "gui"
    / "frontend"
)

HTML_FILE = FRONTEND_DIR / "index.html"

CSS_FILE = (
    FRONTEND_DIR
    / "assets"
    / "css"
    / "style.css"
)


def read_html() -> str:
    return HTML_FILE.read_text(encoding="utf-8")


def read_css() -> str:
    return CSS_FILE.read_text(encoding="utf-8")


def test_frontend_html_exists() -> None:
    assert HTML_FILE.exists()


def test_frontend_stylesheet_exists() -> None:
    assert CSS_FILE.exists()


def test_html_contains_application_shell() -> None:
    html = read_html()

    assert 'class="app-container"' in html
    assert 'id="app-shell"' in html


def test_html_contains_main_layout_sections() -> None:
    html = read_html()

    assert "<aside" in html
    assert 'class="sidebar"' in html

    assert "<main" in html
    assert 'class="main-content"' in html

    assert "<header" in html
    assert 'class="topbar"' in html

    assert 'class="workspace"' in html


def test_html_contains_accessible_navigation() -> None:
    html = read_html()

    assert 'aria-label="Main sidebar"' in html
    assert 'aria-label="Main navigation"' in html
    assert 'aria-label="Main workspace"' in html


def test_html_preserves_conversation_elements() -> None:
    html = read_html()

    assert 'id="conversation"' in html
    assert 'id="message-form"' in html
    assert 'id="message-input"' in html
    assert 'id="send-button"' in html


def test_html_preserves_frontend_javascript() -> None:
    html = read_html()

    assert 'src="assets/js/app.js"' in html


def test_css_imports_design_tokens() -> None:
    css = read_css()

    normalized_css = css.lstrip("\ufeff \t\r\n")

    assert normalized_css.startswith(
        '@import url("./tokens.css");'
    )



def test_css_defines_main_layout_components() -> None:
    css = read_css()

    required_selectors = [
        ".app-container",
        ".sidebar",
        ".main-content",
        ".topbar",
        ".workspace",
        ".conversation-area",
        ".composer",
        ".workspace-footer",
    ]

    for selector in required_selectors:
        assert selector in css


def test_css_defines_responsive_tablet_layout() -> None:
    css = read_css()

    assert "@media (max-width: 760px)" in css
    assert "--sidebar-width: 190px" in css


def test_css_defines_responsive_mobile_layout() -> None:
    css = read_css()

    assert "@media (max-width: 560px)" in css
    assert ".sidebar" in css
    assert "display: none" in css


def test_css_supports_dynamic_viewport_height() -> None:
    css = read_css()

    assert "100dvh" in css


def test_css_preserves_accessibility_features() -> None:
    css = read_css()

    assert "::selection" in css
    assert ":focus-visible" in css
    assert "prefers-reduced-motion: reduce" in css
