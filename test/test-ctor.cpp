#include <gtest/gtest.h>
#include <toy-gemm/matrix.hpp>

using namespace toy_gemm;

TEST(toy_gemm_ctor, ctor)
{
    constexpr Mat<3, 3, int> x;
    static_assert(!std::is_constructible_v<decltype(x), int, int, int, int, int, int>,
                  "only constructible from either 1 or 9 arguments");
    constexpr decltype(x) zeros{0};
    constexpr decltype(x) y(0, 0, 0, 0, 0, 0, 0, 0, 0);
    ASSERT_EQ(x, zeros);
    ASSERT_EQ(y, zeros);
    Mat<3, 2, int> z{{1, 2}, {3, 4}, {5, 6}};
    constexpr Mat<3, 2, int> z2{1, 2, 3, 4, 5, 6};
    ASSERT_EQ(z, z2);
}

TEST(toy_gemm, accessor)
{
    constexpr Mat<3, 3> I3{1, 0, 0, 0, 1, 0, 0, 0, 1};
    constexpr auto first_row = Vec<int, 3>{1, 0, 0};
    constexpr auto second_row = Vec<int, 3>{0, 1, 0};
    ASSERT_EQ(I3.get<0>(), first_row);
    ASSERT_EQ(I3[1], second_row);
    static_assert(I3.get<2, 0>() == 0, "test templated get<>()");
    ASSERT_EQ(I3[2][1], 0);
    ASSERT_EQ(I3.at(2, 2), 1);
}

TEST(toy_gemm, rows)
{
    const Mat<2, 3, int> M23({1, 2, 3}, {4, 5, 6});
    const Mat<2, 3, int> M23_dup = M23;
    size_t r = 0;
    for (const auto& row : M23.rows()) {  // should be compatible with range for
        ASSERT_EQ(row.size(), 3);
        ASSERT_EQ(row, M23_dup[r++]);
    }
    ASSERT_EQ(r, 2);
}

TEST(toy_gemm_ctor, comparison)
{
    using M33 = Mat<3, 3, int>;
    constexpr M33 x;
    M33 y{0};
    ASSERT_EQ(x, y);
    y.get<2, 2>() = 1;
    ASSERT_NE(x, y);
}
