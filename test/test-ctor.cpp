#include <gtest/gtest.h>
#include <toy-gemm/matrix.hpp>

using namespace toy_gemm;
using M22 = Mat<2, 2>;
using M23 = Mat<2, 3>;
using M32 = Mat<3, 2>;
using M33 = Mat<3>;

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
}

TEST(toy_gemm, accessor)
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

TEST(toy_gemm, rows)
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

TEST(toy_gemm_access, col)
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
    const auto [c1, c2] = xcolv2;  // C++17 structured binding
    ASSERT_EQ(c1, 2);
    ASSERT_EQ(c2, 4);

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
}

TEST(toy_gemm_ops, transpose)
{
    constexpr M23 m23{1, 2, 3, 4, 5, 6};
    constexpr M32 m23_t{1, 4, 2, 5, 3, 6};
    constexpr M32 m32 = m23.transpose();
    static_assert(std::is_same_v<M32, std::remove_cv_t<decltype(m32)>>, "type must match");
    static_assert(m32 == m23_t, "value must match");
}

TEST(toy_gemm, special_functions)
{
    constexpr M33 Z3{0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr M33 z3 = M33::zeros();
    static_assert(z3 == Z3);

    constexpr M33 I3{1, 0, 0, 0, 1, 0, 0, 0, 1};
    constexpr M33 i3 = Mat<3>::identity();
    static_assert(i3 == I3);

    constexpr auto I16 = Mat<16>::identity();
    static_assert(I16.transpose() == I16); // this is can be a bit hard for the compiler
}
