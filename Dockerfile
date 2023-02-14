# docker build -t atp-webapp -f Dockerfile .
# docker run -it -p 8080:8080 atp-webapp

FROM python:3.10-slim

RUN apt-get update && rm -rf /var/lib/apt/lists /var/cache

RUN useradd -ms /bin/bash atp_user
USER atp_user
WORKDIR /home/atp_user

ENV PATH="/home/atp_user/.local/bin:${PATH}"

# Copy and install custom code developed in the project
COPY --chown=wp_user:wp_user . .

RUN python3 -m pip install ".[app]" --no-cache-dir && \
    find /usr/local/lib/python3.10/ -follow -type f \
        -name '*.a' -name '*.txt' -name '*.md' -name '*.png' \
        -name '*.jpg' -name '*.jpeg' -name '*.js.map' -name '*.pyc' \
        -name '*.c' -name '*.pxc' -name '*.pyd' \
    -delete && \
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
