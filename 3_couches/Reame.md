Write this to create a makefile:

cmake -G "MinGW Makefiles"   -DCMAKE_SH="CMAKE_SH-NOTFOUND"   -DCMAKE_C_COMPILER="C:\msys64\mingw64\bin\gcc.exe"   -DCMAKE_CXX_COMPILER="C:\msys64\mingw64\bin\g++.exe"   -DCMAKE_MAKE_PROGRAM="C:\msys64\mingw64\bin\make.exe" ..
