FROM python:3.11.7-bookworm
RUN mkdir -p /opt/bout-duration-distributions/Data
RUN mkdir -p /opt/bout-duration-distributions/Figures
ADD . /opt/bout-duration-distributions/code
WORKDIR /opt/bout-duration-distributions/
RUN pip install -r code/requirements.txt