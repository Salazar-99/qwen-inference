FROM adaptfm/adaptfm-base:latest

# Build from this directory:
#   docker build -t my-submission:latest .
#
# Requires qwen-weights/ in this directory (see ../scripts download-weights).

COPY qwen-weights/ /opt/ml/model/
COPY src/qwen_inference/ /opt/program/qwen_inference/
COPY src/qwen_inference/main.py /opt/program/main.py

ENV MODEL_DIR=/opt/ml/model/
ENV PYTHONPATH=/opt/program
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python3", "/opt/program/main.py"]
