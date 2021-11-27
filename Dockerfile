FROM python:3.9

ADD pyproject.toml .
ADD poetry.lock .
RUN pip install -U pip &&\
    pip install poetry &&\
    poetry config virtualenvs.create false --local &&\
    poetry install &&\
    rm -rf ~/.cache

ADD photo2geo photo2geo/
RUN poetry install
ADD scripts scripts/

CMD ["python", "-c", "print('hello')"]
