#ifndef TOY_GEMM_MATRIX_HPP
#define TOY_GEMM_MATRIX_HPP

#include <algorithm>
#include <array>
#include <initializer_list>
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
}  // namespace internal
///< @brief Vec is a thin wrapper around std::array representing a row vector
template <typename T, size_t C>
using Vec = std::array<T, C>;

template <size_t R, size_t C, typename T = int>
class Mat
{
   private:
    using ThisType = Mat<R, C, T>;
    using RowType = Vec<T, C>;
    using TRef = typename RowType::reference;
    using TCRef = typename RowType::const_reference;
    using StorageType = std::array<RowType, R>;

   public:
    using RowIterator = typename RowType::iterator;
    using ConstRowIterator = typename RowType::const_iterator;

    constexpr static size_t ELEM_COUNT = R * C;

    ~Mat<R, C, T>() = default;

    // construction
    constexpr Mat<R, C, T>() = default;
    constexpr Mat<R, C, T>(const ThisType&) = default;
    constexpr Mat<R, C, T>(ThisType&&) noexcept = default;
    constexpr Mat<R, C, T>& operator=(const ThisType&) = default;
    constexpr Mat<R, C, T>& operator=(ThisType&&) noexcept = default;

    ///< @brief performs wither element-wise init or uniform init
    template <typename... U>
    constexpr Mat<R, C, T>(U&&... rows) noexcept : elems{std::forward<U>(rows)...}
    {
        static_assert(ELEM_COUNT == sizeof...(rows) || sizeof...(rows) == 1,
                      "pass in either exactly one argument, or exactly ELEM_COUNT arguments");
    }

    /**
     * @brief constructor using an initializer list
     * it's nice to be able to initialize a Mat like
     * @code Mat<3,2> M{{1,2,3},{4,5,6}} @endcode
     * because compile time dimension checks can then be performed
     */
    explicit Mat<R, C, T>(std::initializer_list<RowType> rows) noexcept
    {
        // TODO can probably use some template techniques to avoid if & loop at runtime
        if (rows.size() == R) {
            size_t r = 0;
            for (auto&& row : std::move(rows)) {
                elems[r++] = std::move(row);
            }
        }
    }

    // access
    [[nodiscard]]
    constexpr const RowType& operator[](size_t r) const
    {
        // TODO assert?
        return elems.at(r);
    }

    [[nodiscard]]
    RowType& operator[](size_t r)
    {
        // TODO assert?
        return elems.at(r);
    }

    [[nodiscard]]
    constexpr const RowType& at(size_t r) const
    {
        // TODO assert?
        return elems.at(r);
    }

    [[nodiscard]]
    RowType& at(size_t r) { return elems.at(r); }

    [[nodiscard]]
    constexpr const T& at(size_t r, size_t c) const
    {
        // TODO assert?
        return elems.at(r).at(c);
    }

    T& at(size_t r, size_t c) { return elems.at(r).at(c); }

    // prefer these, which gives compile time error if indices are out of range
    template <size_t row>
    [[nodiscard]]
    RowType& get()
    {
        return std::get<row>(elems);
    }

    template <size_t row>
    [[nodiscard]]
    constexpr const RowType& get() const
    {
        return std::get<row>(elems);
    }

    template <size_t row, size_t col>
    [[nodiscard]]
    T& get()
    {
        return std::get<col>(std::get<row>(elems));
    }

    template <size_t row, size_t col>
    [[nodiscard]]
    constexpr const T& get() const
    {
        return std::get<col>(std::get<row>(elems));
    }

    [[nodiscard]]
    StorageType& rows() noexcept { return elems; }

    [[nodiscard]]
    constexpr const StorageType& rows() const noexcept { return elems; }

    // operators
    [[nodiscard]] constexpr bool operator==(const ThisType& other) const noexcept { return elems == other.elems; }

    [[nodiscard]] constexpr bool operator!=(const ThisType& other) const noexcept { return !this->operator==(other); }

    //    template <size_t Co, typename U, typename Ret = std::common_type_t<T,U>>
    //    [[nodiscard]]
    //    Mat<R, Co, Ret> operator* (const Mat<C, Co, U>& other) const noexcept{
    //        using RetMat = Mat<R, Co, Ret>;
    //        typename RetMat::StorageType storage;
    //        return RetMat{std::move(storage)};
    //    }

   private:
    StorageType elems{};  ///< row major 2D array
    // explicit Mat<R,C,T>(StorageType&& e) : elems(std::move(e)){}
};

}  // namespace toy_gemm

#endif  // TOY_GEMM_MATRIX_HPP
