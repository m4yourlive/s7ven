"use strict";

const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const messageForm = document.getElementById("message-form");

function updateComposer() {
    const hasMessage = messageInput.value.trim().length > 0;

    sendButton.disabled = !hasMessage;

    messageInput.style.height = "auto";
    messageInput.style.height = `${Math.min(
        messageInput.scrollHeight,
        150
    )}px`;
}

messageInput.addEventListener("input", updateComposer);

messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();

        if (!sendButton.disabled) {
            messageForm.requestSubmit();
        }
    }
});

messageForm.addEventListener("submit", (event) => {
    event.preventDefault();

    const message = messageInput.value.trim();

    if (!message) {
        return;
    }

    // Message handling will be implemented with the
    // JavaScript-Python Bridge in Module 08.4.
    console.info("Frontend message submitted.");

    messageInput.value = "";
    updateComposer();
});

updateComposer();