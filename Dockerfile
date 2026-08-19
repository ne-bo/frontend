FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm install

COPY . .

RUN mkdir -p /app/src/backend
RUN python3 -m venv /app/src/backend/venv

RUN /app/src/backend/venv/bin/pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    python-dotenv \
    langchain-openai \
    langchain-community \
    langchain-text-splitters \
    langchain-classic \
    faiss-cpu \
    pypdf \
    python-docx \
    tiktoken

RUN node -e "const fs=require('fs'); const p=JSON.parse(fs.readFileSync('package.json','utf8')); p.scripts['dev-frontend']='vite --host 0.0.0.0'; fs.writeFileSync('package.json', JSON.stringify(p, null, 2));"

ENV PYTHONUNBUFFERED=1

EXPOSE 5173 8000

CMD ["npm", "run", "dev"]