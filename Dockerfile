FROM python:3.10-slim

# System setup & Streamlit environment routing variables
ENV PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME="/tmp/.cache" \
    # Expose and prefer the HF Spaces default port (7860). Use $PORT so HF can override it.
    PORT=7860

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
RUN useradd -m -u 1000 user && chown -R user:user /app
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
COPY --chown=user args.py /app/args.py
COPY --chown=user dataset.py /app/dataset.py
COPY --chown=user streamlit_pose_app.py /app/streamlit_pose_app.py

# Expose the HF-friendly Streamlit port
EXPOSE 7860

# Force Streamlit to be the main PID (entrypoint). Use ${PORT} so HF can change it if needed.
CMD ["streamlit", "run", "streamlit_pose_app.py", "--server.address=0.0.0.0", "--server.port=${PORT}", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]