import streamlit as st
import time
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter
import aiml
import os

kernel = aiml.Kernel()

if os.path.exists("bot_brain.brn"):
    kernel.loadBrain("bot_brain.brn")
else:
    kernel.learn("chatbot.aiml")
    kernel.saveBrain("bot_brain.brn")


def stream_response(text):
    placeholder = st.empty()
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(f"""
        <div class="chat-message assistant-message">
            <span class="chat-icon">🎵</span>
            <div class="chat-bubble assistant-bubble">{displayed}▌</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.018)
    placeholder.markdown(f"""
    <div class="chat-message assistant-message">
        <span class="chat-icon">🎵</span>
        <div class="chat-bubble assistant-bubble">{displayed}</div>
    </div>
    """, unsafe_allow_html=True)


def render_chatbot():
    st.markdown("""
    <div class="chat-intro">
        <p><strong>Bine ai venit!</strong> Aici îți poți dezvolta cunoștințele despre instrumentele muzicale.</p>
        <p>Comenzi disponibile:</p>
        <p>🔹 <strong>Buna</strong> — salută botul<br>
        🔹 <strong>Ce este [numele instrumentului]</strong> — află detalii despre un instrument</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .chat-message {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin: 12px 0;
        animation: fadeInMsg 0.3s ease both;
    }
    @keyframes fadeInMsg {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .chat-icon {
        font-size: 1.6rem;
        margin-top: 4px;
        flex-shrink: 0;
    }
    .chat-bubble {
        padding: 0.9rem 1.3rem;
        border-radius: 18px;
        font-family: var(--font-corp);
        font-size: 1.15rem;
        line-height: 1.75;
        max-width: 78%;
        box-shadow: 0 2px 12px rgba(62, 34, 16, 0.08);
    }
    .user-message {
        flex-direction: row-reverse;
    }
    .user-bubble {
        background: linear-gradient(135deg,
            rgba(192, 99, 74, 0.15),
            rgba(192, 99, 74, 0.08));
        border: 1px solid rgba(192, 99, 74, 0.25);
        border-radius: 18px 18px 4px 18px;
        color: var(--text-principal);
    }
    .assistant-bubble {
        background: linear-gradient(135deg,
            rgba(253, 250, 245, 0.95),
            rgba(234, 224, 204, 0.85));
        border: 1px solid rgba(107, 66, 38, 0.15);
        border-radius: 18px 18px 18px 4px;
        color: var(--text-principal);
        backdrop-filter: blur(4px);
    }
    .assistant-message {
        flex-direction: row;
    }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <span class="chat-icon">👤</span>
                <div class="chat-bubble user-bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <span class="chat-icon">🎵</span>
                <div class="chat-bubble assistant-bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

    user_input = st.chat_input("Scrie mesajul tau...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        st.markdown(f"""
        <div class="chat-message user-message">
            <span class="chat-icon">👤</span>
            <div class="chat-bubble user-bubble">{user_input}</div>
        </div>
        """, unsafe_allow_html=True)

        response = kernel.respond(user_input.upper())

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

        stream_response(response)