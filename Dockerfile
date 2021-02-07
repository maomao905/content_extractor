FROM python:3.7 AS intermediate

ARG SSH_PRIVATE_KEY

WORKDIR /usr/src/app

RUN mkdir /root/.ssh/
RUN echo "${SSH_PRIVATE_KEY}" >> /root/.ssh/id_rsa && chmod 600 /root/.ssh/id_rsa
RUN ssh-keyscan github.com > /root/.ssh/known_hosts

COPY requirements-private.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements-private.txt

FROM python:3.7
COPY --from=intermediate /usr/local/lib/python3.7/site-packages/ /usr/local/lib/python3.7/site-packages/

WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install -r requirements.txt

RUN apt update

COPY . /usr/src/app
