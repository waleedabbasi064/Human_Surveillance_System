FROM python:3.10-slim

# System setup & Streamlit environment routing variables
ENV PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME="/tmp/.cache" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS="0.0.0.0" \
    STREAMLIT_SERVER_HEADLESS="true"

# Install fundamental system tooling for image processing and C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Secure container privileges (Hugging Face user container standard requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Install Python requirements
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application assets and entrypoints using the actual repo layout.
# The repository does not contain a `src/` folder, so do not copy it.
COPY --chown=user config /app/config
COPY --chown=user PoseEstimationModel /app/PoseEstimationModel
COPY --chown=user utils /app/utils
# Copy models code: project has `models.py` (file). Some deploy snapshots lack a `models/` folder.
COPY --chown=user models.py /app/models.py
COPY --chown=user main.py /app/main.py
COPY --chown=user streamlit_pose_app.py /app/streamlit_pose_app.py

# Expose internal Streamlit port
EXPOSE 8501

# Execute main process launcher
CMD ["python", "main.py"]