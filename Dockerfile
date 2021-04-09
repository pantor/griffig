FROM tensorflow/tensorflow:2.4.1-gpu

WORKDIR griffig

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev libglu1-mesa-dev libglew-dev
RUN python3 -m pip install loguru scipy==1.5 Pillow cmake

COPY affx/ affx
COPY griffig/ griffig
COPY include/ include
COPY interfaces/ interfaces
COPY src/ src
COPY CMakeLists.txt .
COPY README.md .
COPY setup.py .

RUN pip install .

EXPOSE 8000

CMD python3 /griffig/interfaces/grpc/grpc.py