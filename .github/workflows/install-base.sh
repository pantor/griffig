# Fix Cmake find *.so.0
ln -s /usr/lib64/libOpenGL.so.0 /usr/lib64/libOpenGL.so
export LD_LIBRARY_PATH=/usr/local/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Fix CMake find Numpy
python3.9 -m pip install --no-cache-dir numpy
sed -i '3092s/.*/"import sys; import numpy; sys.stdout.write(numpy.get_include())"/' /opt/_internal/tools/lib/python3.9/site-packages/cmake/data/share/cmake-3.21/Modules/FindPython/Support.cmake

# Install Eigen
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout 3.3.9
mkdir build && cd build
cmake ..
make -j2
make install
cd ../../

# Install PyBind11
git clone https://github.com/pybind/pybind11.git
cd pybind11
git checkout v2.6.2
mkdir build && cd build
cmake -DPYBIND11_TEST=OFF -DPYBIND11_PYTHON_VERSION=3.9 ..
make -j2
make install
cd ../../

# Install OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.5.2
mkdir build && cd build
cmake -DWITH_VTK=OFF -DWITH_GTK=OFF -DWITH_PROTOBUF=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_WEBP=OFF -DBUILD_opencv_ml=OFF -DBUILD_opencv_calib3d=OFF -DBUILD_opencv_videoio=OFF -DBUILD_opencv_gapi=OFF -DBUILD_opencv_stitching=OFF -DBUILD_opencv_objdetect=OFF -DBUILD_opencv_flann=OFF -DBUILD_opencv_video=OFF -DBUILD_opencv_features2d=OFF ..
make -j2
make install
cd ../../

# ls /usr/lib
# ls /usr/lib64
# ls /usr/local/lib
ls /usr/local/lib64