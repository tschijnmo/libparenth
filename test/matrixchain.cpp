/** A simple test on a simple matrix chain.
 *
 * Here we have  a very simple matrix chain multiplication problem with three
 * matrices.  In this simple test, we will have three matrices $x$, $y$, and
 * $z$, which are of shapes $1K \times 2K$, $2K \times 3K$, and $3K \times 1K$
 * respectively.
 */

#include <vector>

#include <catch.hpp>

#include <libparenth.hpp>

using namespace libparenth;

TEST_CASE("A simple chain product of three matrices can be parenthesized")
{
    using Dim = size_t;
    std::vector<Dim> dims = { 2000, 3000, 1000, 1000 };
    std::vector<std::vector<size_t>> factors = { { 2, 0 }, { 0, 1 }, { 1, 3 } };

    Parenther<Dim> parenther(
        dims.cbegin(), dims.cend(), 2, factors.cbegin(), factors.cend());
}
