"use strict";

const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const messageForm = document.getElementById("message-form");
const conversation = document.getElementById("conversation");
const systemStatus = document.querySelector(".system-status");

let bridgeReady = false;

function updateComposer() {
    const hasMessage = messageInput.value.trim().length > 0;

    sendButton.disabled = !hasMessage || !bridgeReady;

    messageInput.style.height = "auto";
    messageInput.style.height = `${Math.min(
        messageInput.scrollHeight,
        150
    )}px`;
}

function updateSystemStatus(status) {
    systemStatus.lastElementChild.textContent = status;
}

function appendMessage(content, sender) {
    const messageElement = document.createElement("div");
    const labelElement = document.createElement("span");
    const contentElement = document.createElement("p");

    messageElement.className = `message message-${sender}`;
    labelElement.className = "message-label";
    contentElement.className = "message-content";

    labelElement.textContent =
        sender === "user" ? "YOU" : "S7VEN";

    contentElement.textContent = content;

    messageElement.append(labelElement, contentElement);
    conversation.appendChild(messageElement);

    conversation.scrollTop = conversation.scrollHeight;
}

async function initializeBridge() {
    try {
        const api = window.pywebview?.api;

        if (!api) {
            throw new Error("Python bridge is unavailable.");
        }

        const result = await api.ping();

        if (!result.success) {
            throw new Error("Python bridge connection failed.");
        }

        const appInfo = await api.get_app_info();

        bridgeReady = true;
        updateSystemStatus(`${appInfo.name} CONNECTED`);
        updateComposer();

        console.info("S7VEN Python bridge connected.");
    } catch (error) {
        bridgeReady = false;
        updateSystemStatus("BRIDGE UNAVAILABLE");
        updateComposer();

        console.error("Bridge initialization failed:", error);
    }
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

messageForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const message = messageInput.value.trim();

    if (!message || !bridgeReady) {
        return;
    }

    messageInput.value = "";
    updateComposer();

    appendMessage(message, "user");

    sendButton.disabled = true;
    messageInput.disabled = true;

    try {
        const result = await window.pywebview.api.send_message(
            message
        );

        if (!result.success) {
            throw new Error(result.error || "Message submission failed.");
        }

        appendMessage(result.reply, "assistant");
    } catch (error) {
        appendMessage(
            "Unable to communicate with S7VEN Core.",
            "assistant"
        );

        console.error("Message bridge error:", error);
    } finally {
        messageInput.disabled = false;
        updateComposer();
        messageInput.focus();
    }
});

window.addEventListener("pywebviewready", initializeBridge);

updateComposer();