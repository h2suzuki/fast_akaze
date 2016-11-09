Fast A-KAZE is written in C++11 compliant syntax. Building Fast A-KAZE on Linux platform should be trivial.

# Building Fast A-KAZE on Ubuntu 16.04.1 LTS
This instruction has been tested with AWS t2.micro instance and AMI Ubuntu Server 16.04 LTS (HVM).

## 1. Update OS
Run the following commands from root.

    # apt-get update
    # apt-get -y dist-upgrade
    # apt-get -y build-dep opencv
    
## 2. Install the build-dependencies for OpenCV
Run the following commands from root.

    # apt-get -y build-dep opencv
    # apt-get -y install cmake-curses-gui

## 3. Build OpenCV3.1
OpenCV3.1 can be built by a regular user, but the installation requires root or sudo privilege.

    $ git clone https://github.com/opencv/opencv.git
    $ mkdir opencv.build
    $ cd opencv.build/
    $ cmake -DCMAKE_BUILD_TYPE=Release  \
      -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF -DBUILD_PERF_TESTS=OFF \
      -DWITH_CUDA=OFF -DWITH_CUFFT=OFF -DWITH_OPENMP=ON -DWITH_OPENCL=OFF -DWITH_V4L=ON \
      -DENABLE_SSE=ON -DENABLE_SSE2=ON \
      -DENABLE_SSE41=ON -DENABLE_SSE42=ON -DENABLE_SSSE3=ON -DENABLE_POPCNT=ON \
      -DENABLE_AVX=ON -DENABLE_AVX2=ON ../opencv
    $ make
    $ sudo make install

The cmake options like `-DENABLE_AVX=ON` depends on your CPU capabilities.
Cat /proc/cpuinfo for available capabilities and `ccmake ../opencv` to see
possible customization.

The above commands usually install OpenCV3.1 under /usr/local.

## 4. Build Fast A-KAZE
Fast A-KAZE comes with a simple CMakeLists.txt to compile an OpenCV3.1 program.
So the compilation of our sample program is straight forward.

    $ git clone https://github.com/h2suzuki/fast_akaze.git
    $ cd fast_akaze/fast_akaze/
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/usr/local/share/OpenCV/ ..
    $ make

Now you have "test_fast_akaze" for testing.

Edit `~/fast_akaze/fast_akaze/main.cpp` to do anything you want. :-)
