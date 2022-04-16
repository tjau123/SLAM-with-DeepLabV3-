
echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../../

echo "Configuring and building libmysegmentation.so ..."

cd Examples/myROS/myORB_SLAM2_PointMap_SegNetM/libsegmentation
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../../../../

echo "Configuring and building libmyORB_SLAM2_PointMap_SegNetM.so ..."

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ..


echo "Configuring and building Executable myTUM ..."

cd Examples/myROS/myORB_SLAM2_PointMap_SegNetM
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../../../



