FROM python:3.8.8-buster

ENV PYTHONPYCACHEPREFIX=/root/pycache

RUN apt-get update && apt-get install --no-install-recommends -y \
    vim less bash gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz -O ta-lib-0.4.0-src.tar.gz && \
    tar xvf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    pip install TA-Lib --user && \
    rm -R ta-lib ta-lib-0.4.0-src.tar.gz

RUN pip install --no-cache-dir --upgrade pip setuptools

WORKDIR /root/opt
COPY ./opt .
RUN pip install --no-cache-dir -r requirements.txt
