# Ollama with Qwen3 8B baked in, for Cloud Run with a GPU.
ARG OLLAMA_VERSION=0.0.0
FROM ollama/ollama:${OLLAMA_VERSION}

ENV OLLAMA_HOST=0.0.0.0:8080 \
    OLLAMA_MODELS=/models \
    OLLAMA_KEEP_ALIVE=-1 \
    LLAMA_ARG_CACHE_RAM=0

# Start a temporary server during the build, pull the model into the image, stop.
RUN ollama serve & sleep 5 && ollama pull qwen3:8b && kill $!

ENTRYPOINT ["ollama", "serve"]