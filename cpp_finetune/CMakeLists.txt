cmake_minimum_required(VERSION 3.10)
project(LibTorchFineTuningFood101)

set(CMAKE_CXX_STANDARD 17) # Use C++17 for std::filesystem
set(CMAKE_PREFIX_PATH "/path/to/your/libtorch") # <-- IMPORTANT: CHANGE THIS

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(cpp_finetune main.cpp)
target_link_libraries(cpp_finetune PRIVATE torch_cpu torch_cuda c10 Aten ${OpenCV_LIBS})

set_property(TARGET cpp_finetune PROPERTY CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
