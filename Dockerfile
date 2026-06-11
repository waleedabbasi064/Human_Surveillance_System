FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME="/tmp/.cache" \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ffmpeg libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user config /app/config
COPY --chown=user PoseEstimationModel /app/PoseEstimationModel
COPY --chown=user utils /app/utils
COPY --chown=user models.py /app/models.py
COPY --chown=user main.py /app/main.py
COPY --chown=user args.py /app/args.py
COPY --chown=user dataset.py /app/dataset.py
COPY --chown=user streamlit_pose_app.py /app/streamlit_pose_app.py

EXPOSE 7860

CMD ["bash", "-c", "streamlit run streamlit_pose_app.py \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false"]