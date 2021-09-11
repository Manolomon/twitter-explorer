FROM python:3.9

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

VOLUME [ "/data" ]

VOLUME [ "/output" ]

CMD [ "/bin/bash" ]