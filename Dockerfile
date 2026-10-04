FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir '.[api]'
COPY configs ./configs
EXPOSE 8000
CMD ["millst", "serve", "--host", "0.0.0.0"]
