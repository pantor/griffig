# Install OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.5.2
mkdir build && cd build
cmake -DWITH_VTK=OFF -DWITH_GTK=OFF -DWITH_PROTOBUF=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_WEBP=OFF ..
make
make install
cd ../../

# Install PyBind11
git clone https://github.com/pybind/pybind11.git
cd pybind11
git checkout v2.6.2
mkdir build && cd build
cmake -DPYBIND11_TEST=OFF ..
make
make install
cd ../../
