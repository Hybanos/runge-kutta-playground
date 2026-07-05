FROM rocm/dev-ubuntu-24.04

RUN apt update -y && apt install -y build-essential cmake git ninja-build

WORKDIR /src
ENTRYPOINT [ "/src/run", "--hip" ]