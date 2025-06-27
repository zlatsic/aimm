FROM python:3.13

WORKDIR /opt/aimm

RUN pip install aimm

ENV PYTHONPATH=/opt/aimm

CMD ["aimm-server", "--conf", "aimm.yaml"]
