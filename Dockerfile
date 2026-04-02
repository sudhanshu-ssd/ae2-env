FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
AutoModel.from_pretrained('distilbert-base-uncased'); \
print('distilbert pre-downloaded')"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]