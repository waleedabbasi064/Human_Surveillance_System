FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME="/tmp/.cache" \
    PORT=7860

# Install system dependencies required for OpenCV, Mediapipe, and Video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set up a non-root user for security compliance on Hugging Face Spaces
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Install Python dependencies (we explicitly add huggingface_hub to make sure it's present)
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt huggingface_hub

# 1. Create empty target directories for your models
RUN mkdir -p /app/PoseEstimationModel /app/Trained_Models/CHAD

# 2. 🚀 BAKE THE MODEL WEIGHTS IN DURING THE BUILD PHASE
# This downloads your precise 3 files into PoseEstimationModel
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='shahzaib7788/pose-weights', filename='PoseEstimationModel/yolo26n.pt', local_dir='/app'); \
    hf_hub_download(repo_id='shahzaib7788/pose-weights', filename='PoseEstimationModel/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth', local_dir='/app'); \
    hf_hub_download(repo_id='shahzaib7788/pose-weights', filename='PoseEstimationModel/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth', local_dir='/app')"

# 3. 🚀 BAKE THE CHAD FOLDER IN DURING THE BUILD PHASE
# This downloads the entire CHAD directory from your model repo
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='shahzaib7788/pose-weights', allow_patterns='Trained_Models/CHAD/*', local_dir='/app')"

# Copy your code application files (Notice we DO NOT copy local model directories anymore)
COPY --chown=user config /app/config
COPY --chown=user utils /app/utils
COPY --chown=user models.py /app/models.py
COPY --chown=user main.py /app/main.py
COPY --chown=user args.py /app/args.py
COPY --chown=user dataset.py /app/dataset.py
COPY --chown=user streamlit_pose_app.py /app/streamlit_pose_app.py

EXPOSE 7860

# Direct execution array ensures environment flags pass cleanly and instantly
CMD ["streamlit", "run", "streamlit_pose_app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=7860", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]