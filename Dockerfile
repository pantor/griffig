FROM griffig.xyz/base

WORKDIR griffig

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

RUN bash interfaces/grpc/generate.zsh

EXPOSE 50051

ENTRYPOINT ["python3", "/griffig/interfaces/grpc/server.py"]
