find_package(GTest REQUIRED FATAL)
enable_testing()

add_executable(test-ctor test-ctor.cpp)
target_link_libraries(test-ctor toy_gemm gtest gtest_main)
set(CMAKE_CXX_FLAGS "-fPIC -Wall -Wextra -Wpedantic ${CMAKE_CXX_FLAGS}")
gtest_discover_tests(
        test-ctor
)