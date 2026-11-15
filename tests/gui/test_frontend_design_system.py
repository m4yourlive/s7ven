
"""Tests for the S7VEN frontend design system."""

from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = (
    PROJECT_ROOT
    / "src"
    / "seven"
    / "gui"
    / "frontend"
)

CSS_DIR = FRONTEND_DIR / "assets" / "css"


def normalize_css(source: str) -> str:
    """Normalize whitespace for reliable CSS assertions."""
    return re.sub(r"\s+", " ", source).strip()


def test_design_tokens_file_exists() -> None:
    assert (CSS_DIR / "tokens.css").is_file()


def test_main_stylesheet_loads_design_tokens() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(
        encoding="utf-8",
    )

    assert (
        'href="assets/css/tokens.css"' in html
        or 'href="assets\\css\\tokens.css"' in html
    )


def test_design_tokens_define_required_colors() -> None:
    tokens = normalize_css(
        (CSS_DIR / "tokens.css").read_text(encoding="utf-8")
    )

    assert "--color-background: #f3f3f4;" in tokens
    assert "--color-surface: #ffffff;" in tokens
    assert "--color-primary: #141414;" in tokens
    assert "--color-secondary: #666666;" in tokens
    assert "--color-muted: #a3a3a3;" in tokens
    assert "--color-border: #e2e2e4;" in tokens
    assert "--color-accent: #ff2d2d;" in tokens
    assert "--color-accent-hover: #e62626;" in tokens
    assert "--color-surface-dark: #151515;" in tokens


def test_design_tokens_define_typography() -> None:
    tokens = normalize_css(
        (CSS_DIR / "tokens.css").read_text(encoding="utf-8")
    )

    assert '--font-heading: "Press Start 2P", monospace;' in tokens
    assert '--font-display: "Space Mono", monospace;' in tokens
    assert '--font-body: "Share Tech Mono", monospace;' in tokens


def test_design_tokens_define_layout_and_spacing() -> None:
    tokens = normalize_css(
        (CSS_DIR / "tokens.css").read_text(encoding="utf-8")
    )

    assert "--sidebar-width: 230px;" in tokens
    assert "--content-max-width: 1050px;" in tokens
    assert "--dashboard-max-width: 1380px;" in tokens
    assert "--side-panel-width: 340px;" in tokens
    assert "--space-4: 16px;" in tokens
    assert "--space-6: 24px;" in tokens


def test_design_tokens_define_component_radii() -> None:
    tokens = normalize_css(
        (CSS_DIR / "tokens.css").read_text(encoding="utf-8")
    )

    assert "--radius-md: 8px;" in tokens
    assert "--radius-lg: 14px;" in tokens
    assert "--radius-xl: 20px;" in tokens


def test_design_system_supports_accessibility() -> None:
    stylesheet = (
        CSS_DIR / "style.css"
    ).read_text(encoding="utf-8")

    assert ":focus-visible" in stylesheet
    assert "prefers-reduced-motion" in stylesheet
