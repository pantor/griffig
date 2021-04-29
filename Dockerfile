FROM griffig.xyz/base

WORKDIR griffig

RUN python3 -m pip install loguru scipy==1.5 Pillow cmake

COPY affx/ affx
COPY griffig/ griffig
COPY include/ include
COPY src/ src
COPY CMakeLists.txt .
COPY README.md .
COPY setup.py .

RUN python3 -m pip install .
