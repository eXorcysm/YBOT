# YBOT AI

This project is an offshoot of the [XBOT](https://github.com/exorcysm/XBOT) companion chatbot. YBOT is a large language model (LLM) chatbot built with agentic functions used for storing and retrieving long-term memory. As interaction progresses, YBOT saves chat summarizations in her vector database. As such, her knowledge and awareness of the user will continue to grow with engagement.

## Setup

1. Create new virtual environment:

```bash
conda create -n ybot python=3.12
conda activate ybot
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Install and run [Ollama](https://ollama.com/download) as the local LLM server.

4. Store a `.env` file in the root directory with `PINECONE_API_KEY` set to a valid Pinecone API key.

## Usage

Run application from command line:

```bash
streamlit run app.py
```

After app initialization is complete, browse to the following URL if it is not launched automatically:

[http://localhost:8501](http://localhost:8501)

The character card, which contains the model prompts, is located in the `data/prompts` folder.

### Features

- YBOT is designed to role-play as a loyal companion and helpful assistant.
- The LLM is a quantized version of [Psyonic Cetacean](https://huggingface.co/backyardai/Psyonic-Cetacean-V1-20B-Ultra-Quality-GGUF) -- a model fine-tuned for RP and storytelling.

### Future Improvements

- Enable the user to dynamically perform the following via the UI:
    - change models
    - load custom character cards
    - edit/delete messages
    - upload avatars
- Add multimodal capabilities for image and sound processing.
- Add guardrails.

### References

- [A Long-Term Memory Agent](https://python.langchain.com/docs/versions/migrating_memory/long_term_memory_agent)
- [Build a MemGPT Discord Agent in LangGraph Cloud](https://www.youtube.com/watch?v=ORAecR4hXsQ)
- [Building a Local Chat Application with Streamlit, Ollama and Llama 3.2](https://medium.com/@gelsonm/building-a-local-chat-application-with-streamlit-ollama-and-llama-3-2-8f5b116dd8ee)
- [Building an AI Agent with LangGraph, TypeScript, Next.js, TailwindCSS, and Pinecone](https://dev.to/bobbyhalljr/building-an-ai-agent-with-langgraph-typescript-nextjs-tailwindcss-and-pinecone-3bkb)
- [How to add persistent memory using PostgreSQL to LangGraph ReAct agent](https://www.youtube.com/watch?v=hE8C2M8GRLo)
- [Launching Long-Term Memory Support in LangGraph](https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph)
- [Sesame CSM 1B for Multi-Speaker AI Conversations](https://levelup.gitconnected.com/sesame-csm-1b-for-multi-speaker-ai-conversations-complete-guide-to-installing-and-running-e76b202e5b91)
