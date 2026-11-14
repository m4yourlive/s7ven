"""Tests for the S7VEN frontend design system."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = (
    PROJECT_ROOT
    / "src"
    / "seven"
    / "gui"
    / "frontend"
)

CSS_DIR = FRONTEND_DIR / "assets" / "css"


def test_design_tokens_file_exists() -> None:
    assert (CSS_DIR / "tokens.css").is_file()


def test_main_stylesheet_imports_design_tokens() -> None:
    stylesheet = (CSS_DIR / "style.css").read_text(
        encoding="utf-8",
    )

    assert '@import url("./tokens.css");' in stylesheet


def test_design_tokens_define_required_colors() -> None:
    tokens = (CSS_DIR / "tokens.css").read_text(
        encoding="utf-8",
    )

    assert "--color-background: #ffffff;" in tokens
    assert "--color-primary: #171717;" in tokens
    assert "--color-accent: #ff2d2d;" in tokens


def test_design_tokens_define_typography() -> None:
    tokens = (CSS_DIR / "tokens.css").read_text(
        encoding="utf-8",
    )

    assert '--font-heading: "Press Start 2P", monospace;' in tokens
    assert '--font-body: "Share Tech Mono", monospace;' in tokens


def test_design_tokens_define_layout_and_spacing() -> None:
    tokens = (CSS_DIR / "tokens.css").read_text(
        encoding="utf-8",
    )

    assert "--sidebar-width: 250px;" in tokens
    assert "--content-max-width: 1050px;" in tokens
    assert "--space-4: 16px;" in tokens


def test_design_system_supports_accessibility() -> None:
    stylesheet = (CSS_DIR / "style.css").read_text(
        encoding="utf-8",
    )

    assert ":focus-visible" in stylesheet
    assert "prefers-reduced-motion" in stylesheet
