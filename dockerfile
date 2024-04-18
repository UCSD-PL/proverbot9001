FROM rust:1.77-bookworm

ENV APP_HOME=/home/app/web
WORKDIR $APP_HOME
ADD . / $APP_HOME/
SHELL ["/bin/bash", "-c"]
RUN apt update && apt dist-upgrade -y
RUN apt install build-essential curl gawk -y
RUN apt install software-properties-common -y
RUN apt install python3.11 python3-full python3.11-dev python3-pip python3.11-venv -y
RUN apt-get install python3-launchpadlib -y
RUN add-apt-repository ppa:avsm/ppa -y
RUN apt update -y
RUN apt install git opam graphviz libgraphviz-dev -y
RUN python3 -m venv proverbot-env
RUN echo "source proverbot-env/bin/activate" >> /root/.bashrc 
RUN make setup
RUN make download-weights
ENTRYPOINT ["tail", "-f", "/dev/null"]
