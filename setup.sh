#!/bin/bash

# Get the ThirdYearProject Path
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============== Install Tensorflow C API =============
mkdir ${BASEDIR}/local/
FILENAME=libtensorflow-cpu-linux-x86_64.tar.gz
wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/versions/2.17.0/${FILENAME}
tar -C ${BASEDIR}/local -xzf ${FILENAME}

# Copy TSL file
cp -r ${BASEDIR}/local/include/external/local_tsl/tsl/ ${BASEDIR}/local/include/tsl/

cd ${BASEDIR}
# =====================================================

# ==================== Install LLVM ===================
git clone https://github.com/chuongg3/llvm-function-merging.git -b LSHfm
mv llvm-function-merging llvm-project
mkdir build
cd build

cmake -G 'Ninja' -DLLVM_ENABLE_PROJECTS='clang;compiler-rt;lld' -DCMAKE_BUILD_TYPE="Release" -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_DUMP=ON -DLLVM_INCLUDE_TESTS=OFF -DLLVM_USE_LINKER=lld -DThirdYearProject_BASE_DIR=${BASEDIR} ../llvm-project/llvm
nice ninja

cd ${BASEDIR}
# =====================================================

# ================== Install IR2Vec ===================
git clone https://github.com/IITH-Compilers/IR2Vec.git
cd IR2Vec

# Apply my changes to IR2Vec
git apply $BASEDIR/Differentials/DemangledName.diff

# Build IR2Vec
mkdir build
cd build

# Set up Eigen build for IR2Vec
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
tar -xvzf eigen-3.3.7.tar.gz
mkdir eigen-build && cd eigen-build
cmake ../eigen-3.3.7 && make

cd ../

# Build IR2Vec
cmake -DLT_LLVM_INSTALL_DIR=/usr/bin/ -DEigen3_DIR=${BASEDIR}/IR2Vec/build/eigen-build/ ..
make && make install

cd ${BASEDIR}
# =====================================================

# ================== Install f3m_Exp ==================
# Convert dropbox URL to direct download URL
F3M_FILE="f3m-cgo22-artifact.v5.tar.xz"
wget --content-disposition -O ${F3M_FILE} "https://www.dropbox.com/scl/fi/lu0tzhiga96fo9q5gse2t/f3m-cgo22-artifact.v5.tar.xz?rlkey=8i7cnghfqusb8m0equw0gucyj&e=1&st=0oj446ul&dl=1"

tar -xf ${F3M_FILE}

# Move f3m_exp to the main director
mv ${BASEDIR}/${F3M_FILE}/f3m_exp/ ${BASEDIR}/f3m_exp/

# Replace original files with my files
cp ${BASEDIR}/f3m_exp_files/config.py ${BASEDIR}/f3m_exp/config.py
cp ${BASEDIR}/f3m_exp_files/flags.py ${BASEDIR}/f3m_exp/flags.py
cp ${BASEDIR}/f3m_exp_files/Makefile.config ${BASEDIR}/f3m_exp/benchmarks/Makefile.config
# =====================================================