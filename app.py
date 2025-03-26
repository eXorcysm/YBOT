"""

This is the main module to launch the YBOT app.

"""

### Importing libraries ###

import logging
import sys
import time

from code.bot    import YBOT
from code.prompt import build_system_prompt

import streamlit as st

### Environment settings ###

AI   = "YBOT"
USER = "USER"

logging.basicConfig(
    stream = sys.stdout, level = logging.INFO, format = "%(asctime)s - %(levelname)s - %(message)s"
)

### Module functions ###

def stream_generator(text):
    """
    This generator function is used to simulate streaming text.
    """

    for token in text.split():
        yield token + " "

        time.sleep(0.05)

def launch_ui(ybot):
    """
    Launch chat user interface.
    """

    # Configure interface.
    st.set_page_config(
        page_title = "YBOT Chat 🤖"
    )
    st.markdown(
        "<h2 style='text-align: center; color: #E650D2; font-family: Arial;'>YBOT Chat 🤖</h2>",
        unsafe_allow_html = True
    )
    st.sidebar.header("Introduction")
    st.sidebar.info(
        """
        Hello, I am YBOT, and I'm your new companion. Let's have some fun together!
        """
    )
    st.sidebar.info(
        """
        I'm still under development at the moment. Please be patient with me. I will get better
        with time.
        """
    )

    bot_avatar  = "./data/imgs/autumn.jpg"
    user_avatar = "./data/imgs/user.png"

    _, _, first_msg = build_system_prompt(AI, USER)

    # Initialize chat session history.
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role" : AI, "content" : first_msg}]

    # Display chat messages from session history on app rerun.
    for msg in st.session_state.messages:
        if msg["role"] == AI:
            msg_avatar = bot_avatar
        else:
            msg_avatar = user_avatar

        with st.chat_message(msg["role"], avatar = msg_avatar):
            st.markdown(msg["content"])

    # Handle user input.
    user_msg = st.chat_input("What's on your mind?")

    if user_msg:
        # Add user message to chat session history.
        st.session_state.messages.append({"role" : USER, "content" : user_msg})

        # Display user message in chat message container.
        with st.chat_message("user", avatar = user_avatar):
            st.markdown(user_msg)

        # Generate and display AI response in chat message container.
        with st.chat_message(AI, avatar = bot_avatar):
            response = ybot.chat(query = user_msg)

            # st.markdown(response.content)
            st.write_stream(stream_generator(response))

        # Add AI response to chat session history.
        st.session_state.messages.append({"role" : AI, "content" : response})

def main():
    """
    Activate chatbot and launch application user interface.
    """

    print("\n========== Welcome to YBOT Chat ==========\n")

    ybot = YBOT(AI, USER)

    launch_ui(ybot)

if __name__ == "__main__":
    main()
