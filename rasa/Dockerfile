FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir rasa==3.6.6 rasa-sdk==3.6.0
EXPOSE 5005
CMD ["rasa", "run", "--enable-api", "--cors", "*", "--port", "5005"]
