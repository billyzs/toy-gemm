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
namespace internal
{
struct RowVecTag final {
};
struct ColVecTag final {
};

template <typename RetType, typename ElemType, size_t... idx>
constexpr RetType list_item(std::initializer_list<ElemType> &&l, std::index_sequence<idx...>)
{
    return {(*std::next(std::begin(std::move(l)), idx))...};
}
}  // namespace internal

///< @brief Vec is a thin wrapper around std::array representing a 1D vector
template <typename T, size_t C>
using Vec = std::array<T, C>;

template <size_t R, size_t C, typename T = int>
class Mat
{
   public:
    using RowType = Vec<T, C>;
    using ColType = Vec<T, R>;
    using ThisType = Mat<R, C, T>;

    using TRef = typename RowType::reference;
    using TCRef = typename RowType::const_reference;
    using StorageType = std::array<RowType, R>;
    using RowIterator = typename RowType::iterator;
    using ConstRowIterator = typename RowType::const_iterator;

    constexpr static size_t ELEM_COUNT = R * C;

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
     * it's nice to be able to initialize a Mat like
     * @c Mat<3,2> M{{1,2,3},{4,5,6}}
     * because some compile time dimension checks can be performed
     * @note we don't have compile time non-empty initializer_list yet, otherwise we could make the whole thing
     * constexpr
     */
    template <typename... E>  //, std::enable_if_t<R == sizeof...(E), int> = 0>
    explicit Mat<R, C, T>(std::initializer_list<E> &&... l)
    {
        static_assert(R == sizeof...(l));
        const bool every_list_must_have_C_elements = ((C == l.size()) && ...);
        assert(every_list_must_have_C_elements);
        make(std::move(l)...);
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
    [[nodiscard]] RowType &get() noexcept
    {
        return std::get<row>(elems);
    }

    template <size_t row>
    [[nodiscard]] constexpr const RowType &get() const noexcept
    {
        return std::get<row>(elems);
    }

    template <size_t row, size_t col>
    [[nodiscard]] T &get() noexcept
    {
        return std::get<col>(std::get<row>(elems));
    }

    template <size_t row, size_t col>
    [[nodiscard]] constexpr const T &get() const noexcept
    {
        return std::get<col>(std::get<row>(elems));
    }

    [[nodiscard]] StorageType &rows() noexcept { return elems; }

    [[nodiscard]] constexpr const StorageType &rows() const noexcept { return elems; }

    template <size_t Col>
    [[nodiscard]] constexpr ColType get_col() const noexcept
    {
        return GetCol<Col>::impl(elems, std::make_index_sequence<R>());
    }

    template <size_t Col>
    [[nodiscard]] ColType get_col() noexcept
    {
        return GetCol<Col>::impl(elems, std::make_index_sequence<R>());
    }

    template <size_t Col>
    [[nodiscard]] constexpr auto get_col_view() noexcept
    {
        // return a tuple of length R containing references to elements at column Col
        return GetColView<Col>::impl(elems, std::make_index_sequence<R>());
    }

    template <size_t Col>
    [[nodiscard]] constexpr auto get_col_view() const noexcept
    {
        // return a tuple of length R containing references to elements at column Col
        return GetColView<Col>::impl(elems, std::make_index_sequence<R>());
    }

    // operators
    [[nodiscard]] constexpr bool operator==(const ThisType &other) const noexcept
    {
        // could do return elems == other.elems but libstdc++ did not implement == for arrays as constexpr...
        // return equal(other.elems, std::make_index_sequence<R>());
        return elems == other.elems;
    }

    [[nodiscard]] constexpr bool operator!=(const ThisType &other) const noexcept { return !this->operator==(other); }

    template <size_t OtherC, typename E>
    [[nodiscard]] auto operator*(const Mat<C, OtherC, E> &other) const noexcept
    {
        using RetType = Mat<R, OtherC, std::common_type_t<T, E>>;
        constexpr auto make_ret_mat = [](auto... e) { return RetType{e...}; };
        RetType ret;                          // R*OtherC
        typename RetType::ColType other_col;  // C*1
        for (size_t r = 0; r < R; ++r) {
            for (size_t c = 0; c < OtherC; ++c) {
                for (size_t oc = 0; oc < C; ++oc) {
                    other_col[oc] = other.at(oc).at(c);
                }
                ret[r][c] = std::inner_product(this->at(r).cbegin(), this->at(r).cend(), other_col.cbegin(), T{0});
            }
        }
        return ret;
    }

    [[nodiscard]] constexpr Mat<C, R, T> transpose() const noexcept
    {
        return transpose_impl(std::make_index_sequence<C>());
    }

   private:
    template <size_t OR, size_t OC, typename OT>
    friend class Mat;

    StorageType elems{0};  ///< row-major 2D array, defaults to zero-initialized
    // explicit Mat<R,C,T>(StorageType&& e) : elems(std::move(e)){}

    /**
     * @brief helper struct to get the column
     * Can't have get_impl take in both a size_t tparam and a size_t... param pack, so passing Col via the tparams of
     * this struct
     * @tparam Col the column to get
     */
    template <size_t Col>
    struct GetCol final {
        GetCol() = delete;
        template <size_t... idx>
        static constexpr ColType impl(const StorageType &storage, std::index_sequence<idx...>) noexcept
        {
            return {std::get<Col>(std::get<idx>(storage))...};
        }
        template <size_t... idx>
        static ColType impl(StorageType &storage, std::index_sequence<idx...>) noexcept
        {
            return {std::get<Col>(std::get<idx>(storage))...};  // call get<Col> on every "row" in storage
        }
    };

    // TODO try SFINAE
    constexpr RowType make_row(std::initializer_list<T> &&l)
    {
        return internal::list_item<RowType, T>(std::move(l), std::make_index_sequence<C>());
    }

    template <size_t... idx>
    constexpr auto rows(std::index_sequence<idx...>)
    {
        return std::forward_as_tuple(elems[idx]...);
    }

    template <typename... L>
    constexpr void make(std::initializer_list<L> &&... l)
    {
        constexpr auto seq = std::index_sequence_for<L...>();
        rows(seq) = std::forward_as_tuple(make_row(std::move(l))...);
    }

    template <size_t... idx>
    constexpr Mat<C, R, T> transpose_impl(std::index_sequence<idx...>) const noexcept
    {
        constexpr auto make_transpose_mat = [](auto... e) { return Mat<C, R, T>{e...}; };
        return std::apply(make_transpose_mat, std::tuple_cat(get_col_view<idx>()...));
    }

    template <size_t Col>
    struct GetColView final {
        GetColView() = delete;
        template <typename SType, size_t... Row>
        static constexpr auto impl(SType &&storage, std::index_sequence<Row...>) noexcept
        {
            return std::forward_as_tuple(std::get<Col>(std::get<Row>(std::forward<SType>(storage)))...);
        }
    };

    template <size_t... idx>
    constexpr bool equal(const StorageType &other, std::index_sequence<idx...>) const noexcept
    {
        constexpr auto e = [](const RowType &r1, const RowType &r2, auto... col) {
            return ([&r1, &r2, &col]() { return std::get<col>(r1) == std::get<col>(r2); } && ...);
        };
        return (e(std::get<idx>(elems), std::get<idx>(other)) && ...);
    }
};

}  // namespace toy_gemm

#endif  // TOY_GEMM_MATRIX_HPP
