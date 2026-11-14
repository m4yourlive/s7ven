
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
CSS_FILE = FRONTEND_DIR / "assets" / "css" / "style.css"
JS_FILE = FRONTEND_DIR / "assets" / "js" / "app.js"


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_composer_form_exists() -> None:
    html = read_file(HTML_FILE)

    assert 'id="message-form"' in html
    assert 'class="composer"' in html


def test_message_input_configuration() -> None:
    html = read_file(HTML_FILE)

    assert 'id="message-input"' in html
    assert 'rows="1"' in html
    assert 'maxlength="4000"' in html
    assert 'aria-label="Message S7VEN"' in html


def test_composer_keyboard_hint_exists() -> None:
    html = read_file(HTML_FILE)

    assert 'id="composer-hint"' in html
    assert "ENTER TO SEND" in html
    assert "SHIFT + ENTER" in html


def test_character_counter_exists() -> None:
    html = read_file(HTML_FILE)

    assert 'id="character-counter"' in html
    assert 'class="character-counter"' in html
    assert 'aria-live="polite"' in html
    assert "0 / 4000" in html


def test_send_button_configuration() -> None:
    html = read_file(HTML_FILE)

    assert 'id="send-button"' in html
    assert 'type="submit"' in html
    assert 'aria-label="Send message"' in html
    assert "disabled" in html


def test_composer_accessibility_description() -> None:
    html = read_file(HTML_FILE)

    assert 'aria-describedby="composer-hint character-counter"' in html
    assert 'aria-label="Message composer"' in html


def test_composer_maximum_length_constant() -> None:
    js = read_file(JS_FILE)

    assert "const MAX_MESSAGE_LENGTH = 4000;" in js


def test_composer_updates_character_counter() -> None:
    js = read_file(JS_FILE)

    assert "function updateCharacterCounter()" in js
    assert "characterCounter.textContent" in js
    assert "near-limit" in js


def test_composer_resizes_textarea() -> None:
    js = read_file(JS_FILE)

    assert "function resizeMessageInput()" in js
    assert "messageInput.scrollHeight" in js
    assert "MAX_COMPOSER_HEIGHT" in js


def test_composer_validates_message() -> None:
    js = read_file(JS_FILE)

    assert "message.trim().length > 0" in js
    assert "message.length <= MAX_MESSAGE_LENGTH" in js


def test_composer_requires_bridge_connection() -> None:
    js = read_file(JS_FILE)

    assert "bridgeReady" in js
    assert "!bridgeReady" in js
    assert "sendButton.disabled = !canSend" in js


def test_composer_prevents_duplicate_submission() -> None:
    js = read_file(JS_FILE)

    assert "let isProcessing = false;" in js
    assert "!isProcessing" in js
    assert "function setComposerProcessing(processing)" in js


def test_composer_loading_state() -> None:
    js = read_file(JS_FILE)

    assert 'classList.toggle(' in js
    assert '"is-loading"' in js
    assert 'setAttribute(' in js
    assert '"aria-busy"' in js
    assert '"S7VEN is responding"' in js


def test_enter_submits_message() -> None:
    js = read_file(JS_FILE)

    assert 'event.key === "Enter"' in js
    assert "!event.shiftKey" in js
    assert "messageForm.requestSubmit()" in js


def test_shift_enter_preserves_newline() -> None:
    js = read_file(JS_FILE)

    assert "event.shiftKey" in js


def test_ime_composition_is_supported() -> None:
    js = read_file(JS_FILE)

    assert "!event.isComposing" in js


def test_composer_handles_submission_errors() -> None:
    js = read_file(JS_FILE)

    assert "async function handleMessageSubmit(event)" in js
    assert "catch (error)" in js
    assert "finally" in js



def test_composer_restores_input_after_submission() -> None:
    js = read_file(JS_FILE)

    assert "messageInput.disabled" in js
    assert "setComposerProcessing(false)" in js
    assert "messageInput.focus()" in js


def test_composer_css_has_responsive_styles() -> None:
    css = read_file(CSS_FILE)

    assert ".composer-input-wrapper" in css
    assert ".composer-meta" in css
    assert ".character-counter" in css
    assert "@media (max-width: 560px)" in css


def test_composer_css_has_loading_animation() -> None:
    css = read_file(CSS_FILE)

    assert ".send-button.is-loading" in css
    assert "@keyframes composer-loading" in css


def test_composer_respects_reduced_motion() -> None:
    css = read_file(CSS_FILE)

    assert "@media (prefers-reduced-motion: reduce)" in css
