/** A simple test on a simple matrix chain.
 *
 * Here we have a very simple matrix chain multiplication problem with three
 * matrices.  In this simple test, we will have three matrices $x$, $y$, and
 * $z$, which are of shapes $1K \times 2K$, $2K \times 3K$, and $3K \times 1K$
 * respectively.
 *
 * This test aims to give as much coverage as possible with very detailed
 * checking of the results.
 */

#include <initializer_list>
#include <vector>

#include <catch.hpp>

#include <libparenth.hpp>

using namespace libparenth;

TEST_CASE("A simple chain product of three matrices can be parenthesized")
{
    using Dim = size_t;
    std::vector<Dim> dims = { 2000, 3000, 1000, 1000 };
    auto n_dims = dims.size();
    std::vector<std::vector<size_t>> factors = { { 2, 0 }, { 0, 1 }, { 1, 3 } };

    using P = Parenther<Dim>;
    P parenther(
        dims.cbegin(), dims.cend(), 2, factors.cbegin(), factors.cend());

    // Checks the correctness of the optimal evaluation sequence.
    //
    // Here other aspects of the results dependent on the exact optimization
    // arguments will not be checked.

    auto check_opt = [&](const auto& res) {
        // The subproblem keys.
        P::Factor_subset p012(3, true);
        P::Factor_subset p0(3, { 0 });
        P::Factor_subset p1(3, { 1 });
        P::Factor_subset p2(3, { 2 });
        P::Factor_subset p12(3, { 1, 2 });

        auto dims = [n_dims](std::initializer_list<P::Size> idxes) {
            return P::Dim_subset(n_dims, idxes);
        };

        auto it = res.find(p012);
        REQUIRE(it != res.end());
        CHECK(it->second.sums == dims({ 0, 1 }));
        CHECK(it->second.exts == dims({ 2, 3 }));

        const auto& top_eval = it->second.evals.front();
        bool top_eval_ops_good
            = (top_eval.ops.first == p0 && top_eval.ops.second == p12)
            || (top_eval.ops.first == p12 && top_eval.ops.second == p0);
        CHECK(top_eval_ops_good);
        CHECK(top_eval.sums == dims({ 0 }));
        CHECK(top_eval.cost
            == 2ull * 2000 * 1000 * 3000 + 2ull * 1000 * 2000 * 1000);

        it = res.find(p12);
        REQUIRE(it != res.end());
        CHECK(it->second.sums == dims({ 1 }));
        CHECK(it->second.exts == dims({ 3, 0 }));

        CHECK(it->second.evals.size() == 1);
        const auto& eval12 = it->second.evals.front();
        bool ops_12_good = (eval12.ops.first == p1 && eval12.ops.second == p2)
            || (eval12.ops.first == p2 && eval12.ops.second == p1);
        CHECK(ops_12_good);
        CHECK(eval12.sums == it->second.sums);
        CHECK(eval12.cost == 2ull * 2000 * 1000 * 3000);

        // Check the leaves.
        for (const auto& i : { p0, p1, p2 }) {
            it = res.find(i);
            REQUIRE(it != res.end());
            CHECK(it->second.sums.count() == 0);
            const auto& exts = factors[i.find_last()];
            CHECK(it->second.exts
                == P::Dim_subset(n_dims, exts.cbegin(), exts.cend()));

            CHECK(it->second.evals.size() == 1);
            const auto& eval = it->second.evals.front();
            CHECK(eval.ops.first == i);
            CHECK(eval.ops.second.count() == 0);
            CHECK(eval.sums.count() == 0);
            CHECK(eval.cost == 0);
        }
    };

    SECTION("The greedy strategy gives the right answer")
    {
        // For this problem, the greedy strategy happens to give the actual
        // right answer.
        auto res = parenther.opt(Mode::GREEDY, false);
        check_opt(res);

        CHECK(res.size() == 5);
    }

    SECTION("The optimal strategy gives the right answer")
    {
        // Optimal strategy = non-inclusive + normal mode.
        auto res = parenther.opt(Mode::NORMAL, false);
        check_opt(res);

        CHECK(res.size() == 6);
    }

    SECTION("The traversed strategy gives the right answer")
    {
        // Traversed strategy = inclusive + normal mode.
        auto res = parenther.opt(Mode::NORMAL, true);
        check_opt(res);

        CHECK(res.size() == 6);
        auto it = res.find(P::Factor_subset(3, true));
        REQUIRE(it != res.end());
        CHECK(it->second.evals.size() == 2);
    }

    SECTION("The exhaustive mode gives the right answer")
    {
        auto res = parenther.opt(Mode::EXHAUST, true);
        check_opt(res);

        CHECK(res.size() == 7);
        auto it = res.find(P::Factor_subset(3, true));
        REQUIRE(it != res.end());
        CHECK(it->second.evals.size() == 3);
    }
}
