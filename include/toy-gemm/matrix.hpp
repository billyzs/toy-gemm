#ifndef TOY_GEMM_MATRIX_HPP
#define TOY_GEMM_MATRIX_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <initializer_list>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace toy_gemm
{
template <typename T, size_t C>
using Vec = std::array<T, C>;  ///< choosing std::array to represent a 1D vector

template <size_t R, size_t C = R, typename T = int>
class Mat
{
   public:
    using RowType = Vec<T, C>;
    using ColType = Vec<T, R>;
    using ThisType = Mat<R, C, T>;

    using TRef = typename RowType::reference;
    using TCRef = typename RowType::const_reference;
    using StorageType = std::array<RowType, R>;

    constexpr static size_t ELEM_COUNT = R * C;
    constexpr static size_t ROW_COUNT = R;
    constexpr static size_t COL_COUNT = C;

    ~Mat<R, C, T>() = default;

    // construction

    /**
     * @brief default constructor will zero-initialize
     */
    constexpr Mat<R, C, T>() = default;

    constexpr Mat<R, C, T>(const ThisType &) = default;

    constexpr Mat<R, C, T>(ThisType &&) noexcept = default;

    constexpr Mat<R, C, T> &operator=(const ThisType &) = default;

    constexpr Mat<R, C, T> &operator=(ThisType &&) noexcept = default;

    /**
     * @brief performs either element-wise init or uniform init
     * @param e a parameter pack containing either exactly one argument, or exactly ELEM_COUNT arguments
     * when given one argument, uniformly initialize every element of the matrix to e
     * when given ELEM_COUNT arguments, interpret the pack as values of matrix elements at
     * {(0,0)...(0,C-1),(1,0)...(1,C-1), ... (R-1,0), (R-1,C-1)}
     * @note SFINAE to disable the ctor when the number of inputs is not one of {ELEM_COUNT, 1}; this makes
     * @c std::is_constructible_v(Mat<R,C,T>,Args...)
     * evaluate to false if sizeof...(Args) is incorrect
     */
    template <typename... E, std::enable_if_t<ELEM_COUNT == sizeof...(E) || sizeof...(E) == 1, int> = 0>
    explicit constexpr Mat<R, C, T>(E &&... e) noexcept : elems{std::forward<E>(e)...}
    {
        static_assert(ELEM_COUNT == sizeof...(e) || sizeof...(e) == 1,
                      "pass in either exactly one argument, or exactly ELEM_COUNT arguments");
    }

    /**
     * @brief constructor using exactly R initializer_lists
     * It's nice to be able to initialize a Mat like this because this form leaves little room for ambiguity; consider
     * @c Mat<3,2> M{1,2,3,4,5,6};
     * @c Mat<2,3> M{1,2,3,4,5,6};
     * they look very similar, it's not immediately obvious if someone swapped the dimension by mistake; compared with
     * @c Mat<3,2> M{{1,2,3},{4,5,6}};
     * in which the mistake is much more obvious.
     * @note we don't have compile time non-empty initializer_list yet, otherwise we could make the whole thing
     * constexpr
     */
    template <typename... E>
    explicit Mat<R, C, T>(std::initializer_list<E> &&... l)
    {
        static_assert(ROW_COUNT == sizeof...(l));
        const bool every_list_must_have_C_elements = ((COL_COUNT == l.size()) && ...);  // C++17 fold expression
        assert(every_list_must_have_C_elements);
        row_wise_init(std::move(l)...);
    }

    // access (might throw)
    [[nodiscard]] constexpr const RowType &operator[](size_t r) const
    {
        // TODO assert?
        return elems.at(r);
    }

    [[nodiscard]] RowType &operator[](size_t r)
    {
        // TODO assert?
        return elems.at(r);
    }

    [[nodiscard]] constexpr const RowType &at(size_t r) const
    {
        // TODO assert?
        return elems.at(r);
    }

    [[nodiscard]] RowType &at(size_t r) { return elems.at(r); }

    [[nodiscard]] constexpr const T &at(size_t r, size_t c) const
    {
        // TODO assert?
        return elems.at(r).at(c);
    }

    [[nodiscard]] T &at(size_t r, size_t c) { return elems.at(r).at(c); }

    // access (noexcept); prefer these, which gives compile time error if indices are out of range
    template <size_t row>
    [[nodiscard]] constexpr RowType &get() noexcept
    {
        return std::get<row>(elems);
    }

    template <size_t row>
    [[nodiscard]] constexpr const RowType &get() const noexcept
    {
        return std::get<row>(elems);
    }

    template <size_t row, size_t col>
    [[nodiscard]] constexpr T &get() noexcept
    {
        return std::get<col>(get<row>());
    }

    template <size_t row, size_t col>
    [[nodiscard]] constexpr const T &get() const noexcept
    {
        return std::get<col>(get<row>());
    }

    [[nodiscard]] constexpr const StorageType &rows() const noexcept { return elems; }

    /**
     * @brief return a copy of column at Col
     * @tparam Col the column to copy
     * @return copy of a column
     */
    template <size_t Col>
    [[nodiscard]] constexpr ColType get_col() const noexcept
    {
        return GetCol<Col>::impl(elems, std::make_index_sequence<R>());
    }

    /**
     * @brief return a tuple of length R containing references to elements at column Col
     * so far, using tuple of references to represent a "view" of a column has worked well enough; operations such as
     * iterating over the elements in a column, making a copy from the view, or modifying a column as a whole,
     * are supported; see how \ref transpose and \ref operator* uses this
     * @tparam Col the column to view
     * @return a tuple containing R lvalue references to column Col
     */
    template <size_t Col>
    [[nodiscard]] constexpr auto col_view() noexcept
    {
        return GetColView<Col>::impl(elems, std::make_index_sequence<R>());
    }

    /**
     * @brief the const overload of \ref col_view
     */
    template <size_t Col>
    [[nodiscard]] constexpr auto col_view() const noexcept
    {
        // return a tuple of length R containing references to elements at column Col
        return GetColView<Col>::impl(elems, std::make_index_sequence<R>());
    }

    // operators
    [[nodiscard]] constexpr bool operator==(const ThisType &other) const noexcept
    {
        // could do return elems == other.elems but libstdc++ did not implement == for arrays as constexpr :(
        return equal(elems, other.elems, std::make_index_sequence<ROW_COUNT>());
    }

    [[nodiscard]] constexpr bool operator!=(const ThisType &other) const noexcept { return !this->operator==(other); }

    template <size_t OtherC, typename E>
    [[nodiscard]] constexpr auto operator*(const Mat<C, OtherC, E> &other) const noexcept
    {
        // the type of the return element should be the type produced by multiplying an instance of T with an instance
        // of E, taking promotion into account
        using RetElement = decltype(std::declval<E>() * std::declval<T>());
        using RetType = Mat<R, OtherC, RetElement>;

        // using the element-wise initialization overload
        constexpr auto make_ret_mat = [](auto... e) {  // C++17 variadic lambda
            static_assert(ROW_COUNT * OtherC == sizeof...(e), "must be given ROW_COUNT * OtherC elements");
            return RetType{e...};
        };

        // C++17 apply
        return std::apply(make_ret_mat,
                          MulImpl<RetElement, OtherC>::build_mat(elems, other, std::make_index_sequence<R>()));
    }

    /**
     * @return return the transpose of this matrix by value
     */
    [[nodiscard]] constexpr Mat<C, R, T> transpose() const noexcept
    {
        return transpose_impl(std::make_index_sequence<C>());
    }
    // TODO maybe it's also possible to return a view of the transpose of this matrix?

    // special functions; for demo
    static constexpr ThisType zeros() noexcept { return ThisType{0}; }

    static constexpr Mat<R, R, T> identity() noexcept
    {
        static_assert(ROW_COUNT == COL_COUNT, "only defined for square matrices");
        Mat<R, R, T> ret{0};
        ret.fill_diagonal(T{1}, std::make_index_sequence<R>());
        return ret;
    }

   private:
    template <size_t OR, size_t OC, typename OT>
    friend class Mat;  ///< for ease of interoperability with another instance of this class

    StorageType elems{0};  ///< row-major 2D array, defaults to zero-initialized

    /**
     * @brief fill the (main) diagonal with a given value
     * @tparam Rows usually ROW_COUNT
     * @param val the value to fill with
     */
    template <size_t... Rows>
    constexpr void fill_diagonal(T val, std::index_sequence<Rows...>) noexcept
    {
        ((std::get<Rows>(std::get<Rows>(elems)) = val), ...);  // C++17 fold expression
    }

    /**
     * @brief unpacks an initializer list to make a row
     * @tparam Cols usually COL_COUNT
     * @param l an initializer list
     * @return a row with elements from l, in the order they appear
     */
    template <size_t... Cols>
    constexpr RowType make_row(std::initializer_list<T> &&l, std::index_sequence<Cols...>)
    {
        // TODO maybe assert on size of list
        return {(*std::next(std::begin(std::move(l)), Cols))...};
    }

    /**
     * @brief return a tuple containing references to each row
     * @tparam Rows usually ROW_COUNT
     * @return tuple containing lvalue references to rows in elems
     */
    template <size_t... Rows>
    constexpr auto rows(std::index_sequence<Rows...>)
    {
        return std::forward_as_tuple(std::get<Rows>(elems)...);
    }

    template <typename... L>
    constexpr void row_wise_init(std::initializer_list<L> &&... l)
    {
        constexpr auto seq = std::index_sequence_for<L...>();
        rows(seq) = std::forward_as_tuple(make_row(std::move(l), std::make_index_sequence<C>())...);
    }

    template <size_t... idx>
    constexpr Mat<C, R, T> transpose_impl(std::index_sequence<idx...>) const noexcept
    {
        constexpr auto make_transpose_mat = [](auto... e) { return Mat<C, R, T>{e...}; };  // C++17 variadic lambdas
        return std::apply(make_transpose_mat, std::tuple_cat(col_view<idx>()...));
    }

    /**
     * @brief helper struct to get a copy of a column
     * Can't have get_impl take in both a size_t template param and a size_t... param pack, so passing Col via the
     * template params of this struct
     * @tparam Col the column to get
     */
    template <size_t Col>
    struct GetCol final {
        GetCol() = delete;  ///< don't bother generating special functions

        template <typename SType, size_t... idx>
        [[nodiscard]] static constexpr ColType impl(SType &&storage, std::index_sequence<idx...>) noexcept
        {
            static_assert(COL_COUNT == sizeof...(idx), "should be getting exactly COL_COUNT elements");
            return {std::get<Col>(std::get<idx>(std::forward<SType>(storage)))...};
        }
    };

    template <size_t Col>
    struct GetColView final {
        GetColView() = delete;  ///< don't bother generating special functions

        /**
         * @brief return a tuple of references to elements in a column
         * @tparam SType usually StorageType
         * @tparam Rows  usually ROW_COUNT
         * @param storage usually elems
         * @return tuple of references to elements in a column; constness of the references depends on constness of
         * storage
         */
        template <typename SType, size_t... Rows>
        static constexpr auto impl(SType &&storage, std::index_sequence<Rows...>) noexcept
        {
            return std::forward_as_tuple(std::get<Col>(std::get<Rows>(std::forward<SType>(storage)))...);
        }
    };

    template <typename ElemType, size_t OCol>
    struct MulImpl final {
        MulImpl() = delete;  ///< don't bother generating special functions

        /**
         * @brief constexpr dot product of two arrays of same length
         * @tparam OtherCol type of a col from the rhs matrix
         * @tparam Cols param pact of the same length as other_col
         * @param this_row a row form the lhs matrix, should have length C
         * @param other_col a col from the rhs matrix, should have length C
         * @return the inner product of this_row and other_col, with type promotion as necessary
         */
        template <typename OtherCol, size_t... Cols>
        [[nodiscard]] constexpr static ElemType inner_product(const RowType &this_row, const OtherCol &other_col,
                                                              std::index_sequence<Cols...>) noexcept
        {
            constexpr auto accumulate = [](auto... e) -> ElemType {  // C++17 variadic lambda
                return (e + ...);  // sum all elements of a param pack; C++17 fold expression
            };
            return std::apply(accumulate,
                              std::forward_as_tuple(std::get<Cols>(this_row) * std::get<Cols>(other_col)...));
        }

        /**
         * @brief build one row of the output matrix
         * @tparam Row the row to build
         * @tparam OtherMat type of the rhs matrix
         * @tparam OCols number of columns in the output matrix
         * @param this_storage the underlying 2D array in the lhs matrix
         * @param other_matt he underlying 2D array in the rhs matrix
         * @return tuple containing R counts of ElemType
         */
        template <size_t Row, typename OtherMat, size_t... OCols>
        [[nodiscard]] constexpr static auto build_row(const StorageType &this_storage, const OtherMat &other_mat,
                                                      std::index_sequence<OCols...>) noexcept
        {
            return std::make_tuple(inner_product(std::get<Row>(this_storage),
                                                 other_mat.template col_view<OCols>(),  // C++ template disambiguator
                                                 std::make_index_sequence<C>())...);
        }

        /**
         * @brief build a tuple of ElemType, containing R * OCol elements that shall be used to element-wise construct a
         * Mat<Rows, OCol, ElemType>
         * @tparam OtherMat type of the rhs matrix
         * @tparam Rows number of rows in the output matrix; should be R
         * @param this_storage the underlying 2D array in the lhs matrix
         * @param other_mat underlying 2D array in the rhs matrix
         * @return tuple containing R * OCol counts of ElemType
         */
        template <typename OtherMat, size_t... Rows>
        [[nodiscard]] constexpr static auto build_mat(const StorageType &this_storage, const OtherMat &other_mat,
                                                      std::index_sequence<Rows...>) noexcept
        {
            static_assert(COL_COUNT == OtherMat::ROW_COUNT,
                          "the other matrix should have the same number of rows as the columns in this matrix");
            return std::tuple_cat(build_row<Rows>(this_storage, other_mat, std::make_index_sequence<OCol>())...);
        }
    };

    /**
     * @brief the implementation of \ref row_equal
     */
    template <size_t... Cols>
    static constexpr bool row_equal_impl(const RowType &self, const RowType &other,
                                         std::index_sequence<Cols...>) noexcept
    {
        return ((std::get<Cols>(self) == std::get<Cols>(other)) && ...);  // C++17 fold expression
    }

    /**
     * @brief check if two rows are equal;
     * @tparam Row the row to compare
     * @param self storage of the lhs of operator==
     * @param other storage of the rhs of operator==
     * @return true if every element in row Row of self is equal to the element in the corresponding position in other
     */
    template <size_t Row>
    static constexpr bool row_equal(const StorageType &self, const StorageType &other) noexcept
    {
        return row_equal_impl(std::get<Row>(self), std::get<Row>(other), std::make_index_sequence<C>());
    }

    /**
     * @brief check if all the rows in self are equal to all the rows in other
     * @tparam Rows usually ROW_COUNT
     * @param self storage of the lhs of operator==
     * @param other storage of the rhs of operator==
     * @return true if all the rows in self are equal to all the rows in other; false otherwise
     */
    template <size_t... Rows>
    static constexpr bool equal(const StorageType &self, const StorageType &other, std::index_sequence<Rows...>)
    {
        return ((row_equal<Rows>(self, other)) && ...);  // C++17 fold expression
    }
};

}  // namespace toy_gemm

#endif  // TOY_GEMM_MATRIX_HPP
