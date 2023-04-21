# docker build -t atp-webapp -f Dockerfile .
# docker run -it -p 8080:8080 atp-webapp

FROM python:3.10-slim

RUN apt-get update && rm -rf /var/lib/apt/lists /var/cache

RUN useradd -ms /bin/bash atp_user
USER atp_user
WORKDIR /home/atp_user

ENV PATH="/home/atp_user/.local/bin:${PATH}"
ENV PATH_PYTHON_LIBS="/home/atp_user/.local/lib/python3.10/"

# Copy and install custom code developed in the project
COPY --chown=atp_user:atp_user . .

RUN python3 -m pip install ".[app]" --no-cache-dir && \
    find ${PATH_PYTHON_LIBS} -follow -regextype posix-extended -regex '.*\.(txt|a|md|png|jpe?g|js\.map|pyc|c|pxc|pyd)$' -delete && \
    find ${PATH_PYTHON_LIBS} -follow -type d -name 'tests' -exec rm -rf {} + && \
    rm -rf /var/cache/apt root/.cache /home/atp_user/.cache home/atp_user/.local/share/jupyter

# Expose port
EXPOSE 8080

ENTRYPOINT [ \
    "gunicorn", \
    "--name", "ATP_WEBAPP", \
    "--bind", "0.0.0.0:8080", \
    "--pythonpath", "webapp", \
    "index:server" \
    ]
# ENTRYPOINT gunicorn --bind 0.0.0.0:8080 --pythonpath webapp index:server
