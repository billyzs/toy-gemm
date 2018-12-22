#include <gtest/gtest.h>
#include <toy-gemm/matrix.hpp>

using namespace toy_gemm;
using M22 = Mat<2, 2>;
using M23 = Mat<2, 3>;
using M32 = Mat<3, 2>;
using M33 = Mat<3>;
using M43 = Mat<4, 3>;
using M34 = Mat<3, 4>;
using M44 = Mat<4>;

TEST(toy_gemm, ctor)
{
    constexpr M33 x;
    static_assert(!std::is_constructible_v<decltype(x), int, int, int, int, int, int>,
                  "only constructible from either 1 or 9 arguments");
    constexpr decltype(x) zeros(0);
    constexpr decltype(x) y(0, 0, 0, 0, 0, 0, 0, 0, 0);
    static_assert(x == zeros);
    static_assert(y == zeros);
    constexpr M32 z{1, 2, 3, 4, 5, 6};
    M32 z_list_ctor{{1, 2}, {3, 4}, {5, 6}};  // sadly unable to constexpr this
    M32 z_copy_ctor(z);
    ASSERT_EQ(z, z_list_ctor);
    ASSERT_EQ(z, z_copy_ctor);
    M22 m22;
    ASSERT_THROW(m22 = M22({1,2}, {3}), std::length_error);
}

TEST(toy_gemm_accessor, get_and_bracket)
{
    constexpr M33 I3{1, 0, 0, 0, 1, 0, 0, 0, 1};
    constexpr auto first_row = Vec<int, 3>{1, 0, 0};
    constexpr auto second_row = Vec<int, 3>{0, 1, 0};
    ASSERT_EQ(I3.get<0>(), first_row);
    ASSERT_EQ(I3[1], second_row);
    static_assert(I3.get<2, 2>() == 1, "test template get<>()");
    ASSERT_EQ(I3[1][1], 1);
    ASSERT_EQ(I3.at(0, 0), 1);
}

TEST(toy_gemm_accessor, rows)
{
    const M23 m23({1, 2, 3}, {4, 5, 6});
    const M23 m23_dup = m23;
    size_t r = 0;
    for (const auto& row : m23.rows()) {  // shall be compatible with range for
        ASSERT_EQ(row.size(), 3);
        ASSERT_EQ(row, m23_dup[r++]);
    }
    ASSERT_EQ(r, 2);
}

TEST(toy_gemm_accessor, col)
{
    constexpr M22 x{1, 2, 3, 4};
    constexpr auto xcol1 = x.get_col<0>();
    constexpr M22::ColType col1{1, 3};
    constexpr auto xcol2 = x.get_col<1>();
    constexpr M22::ColType col2{2, 4};
    ASSERT_EQ(col1, xcol1);
    ASSERT_EQ(col2, xcol2);

    const auto xcolv2 = x.col_view<1>();
    static_assert(std::is_same_v<std::tuple<const int&, const int&>, std::remove_cv_t<decltype(xcolv2)>>);
    const auto [c21, c22] = xcolv2;  // C++17 structured binding
    ASSERT_EQ(c21, 2);
    ASSERT_EQ(c22, 4);

    auto y = x;
    auto ycol1 = y.get_col<0>();
    ASSERT_EQ(col1, ycol1);

    y.col_view<1>() = std::make_tuple(0, 0);
    constexpr M22 yy{1, 0, 3, 0};
    ASSERT_EQ(y, yy);
}

TEST(toy_gemm_ops, comparison)
{
    constexpr M33 x;
    M33 y{0};
    ASSERT_EQ(x, y);
    y.get<2, 2>() = 1;
    ASSERT_NE(x, y);
}

TEST(toy_gemm_ops, transpose)
{
    constexpr M23 m23{1, 2, 3, 4, 5, 6};
    constexpr M32 m23_t{1, 4, 2, 5, 3, 6};
    constexpr auto m32 = m23.transpose();
    static_assert(m32 == m23_t, "value must match");
}

TEST(toy_gemm_ops, multiplication)
{
    constexpr M22 x{1, 2, 3, 4};
    constexpr M22 y{1, 0, 0, 1};
    constexpr M22 z = x * y;
    auto zz = z * y;
    ASSERT_EQ(z, x);
    ASSERT_EQ(y * y, y);
    ASSERT_EQ(z, zz);
    EXPECT_EQ(x * y, y * x);

    const M43 m43({1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12});
    const M34 m34 = m43.transpose();
    const M44 m44({14, 32, 50, 68}, {32, 77, 122, 167}, {50, 122, 194, 266}, {68, 167, 266, 365});
    ASSERT_EQ(m43 * m34, m44);
}

TEST(toy_gemm_ops, special_functions)
{
    constexpr M33 Z3{0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr M33 z3 = M33::zeros();
    static_assert(z3 == Z3);

    constexpr M33 I3{1, 0, 0, 0, 1, 0, 0, 0, 1};
    constexpr M33 i3 = Mat<3>::identity();
    static_assert(i3 == I3);
}
