# ln -s /usr/lib64/libEGL.so /usr/lib
# ln -s /usr/lib64/libEGL.so.1 /usr/lib
# ln -s /usr/lib64/libEGL.so.1.0.0 /usr/lib
# ln -s /usr/lib64/libGL.so /usr/lib
# ln -s /usr/lib64/libGL.so.1 /usr/lib
# ln -s /usr/lib64/libGL.so.1.2.0 /usr/lib
# ln -s /usr/lib64/libGLU.so /usr/lib
# ln -s /usr/lib64/libGLU.so.1 /usr/lib
# ln -s /usr/lib64/libGLU.so.1.3.1 /usr/lib

# ls /usr/include
# ls /usr/lib
ls /usr/lib64

# # Install OpenCV
# git clone https://github.com/opencv/opencv.git
# cd opencv
# git checkout 4.5.2
# mkdir build && cd build
# cmake -DWITH_VTK=OFF -DWITH_GTK=OFF -DWITH_PROTOBUF=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_WEBP=OFF ..
# make -j2
# make install
# cd ../../

# # Install PyBind11
# git clone https://github.com/pybind/pybind11.git
# cd pybind11
# git checkout v2.6.2
# mkdir build && cd build
# cmake -DPYBIND11_TEST=OFF ..
# make -j2
# make install
# cd ../../
