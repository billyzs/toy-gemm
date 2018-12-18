//
// Created by bzs on 12/17/18.
//
#include <toy-gemm/matrix.hpp>
#include <gtest/gtest.h>

using namespace toy_gemm;
#define TEST_DESCRIPTION(desc) RecordProperty("description", desc)

TEST(toy_gemm_ctor, ctor){
    constexpr Mat<3,3, int> x;
    decltype(x) zeros{0};
    decltype(x) y{0,0,0,0,0,0,0,0,0}; // any other number of params would fail
    ASSERT_EQ(x, zeros);
    ASSERT_EQ(y, zeros);
    Mat<3,2, int> z{{1,2},{3,4},{5,6}};
    constexpr Mat<3,2, int> z2{1,2,3,4,5,6};
    ASSERT_EQ(z, z2);
}

TEST(toy_gemm, accessor) {
    constexpr Mat<3,3> I3 = {1,0,0,0,1,0,0,0,1}; // allowed since ctor is not explicit
    constexpr auto first_row = Vec<int, 3>{1,0,0};
    constexpr auto second_row = Vec<int, 3>{0,1,0};
    ASSERT_EQ(I3.get<0>(), first_row);
    ASSERT_EQ(I3[1], second_row);
    static_assert(I3.get<2,0>() == 0, "test templated get<>()");
    ASSERT_EQ(I3[2][1], 0);
    ASSERT_EQ(I3.at(2,2), 1);
}

TEST(toy_gemm, rows) {
    constexpr Mat<2,3>M23 = {1,2,3,4,5,6};
    auto M23_dup{M23};
    size_t r = 0;
    for (const auto & row : M23.rows()){ // should be compatible with range for
        ASSERT_EQ(row.size(), 3);
        ASSERT_EQ(row, M23_dup[r++]);
    }
}

TEST(toy_gemm_ctor, comparison){
    constexpr Mat<3,3, int> x;
    Mat<3,3, int> y{0,0,0,0,0,0,0,0,1};
    ASSERT_NE(x, y);
    y.get<2,2>() = 0;
    ASSERT_EQ(x, y);
}
