/** The main header for libparenth.
 */

#ifndef LIBPARENTH_LIBPARENTH_HPP
#define LIBPARENTH_LIBPARENTH_HPP

#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fbitset.hpp>

/** The main libparenth namespace.
 *
 * Symbols defined for libparenth all reside in this namespace.
 */

namespace libparenth {

/** The modes for the main loop termination.
 *
 * The greedy mode only find the greedy solution based on the last-step cost.
 * The normal mode finds the optimal solution with some branches pruned if it
 * is safe to do so.  The exhaust mode traverses all problems.
 */

enum class Mode { GREEDY, NORMAL, EXHAUST };

/** The main parenthesization core.
 *
 * An object can be created for each parenthesization problem, then the answer
 * can be queried from the object.
 *
 * @tparam D The type for the sizes of the summations and external indices.  It
 * can be any class supporting the basic arithmetic operations and a total
 * order, in addition to the capability of being able to be initialized by
 * integers 0, 1, and 2.  To distinguish it between the normal sizes, the size
 * of the symbolic ranges are called dimensions.
 *
 * @tparam FS The data type to be used for subsets of factors.  The results
 * will be written in terms of this data type.
 *
 * @tparam DS The data type to be used for for subsets of dimensions.  This is
 * only going to be used internally.
 *
 * @tparam S The data type for the number of sums or factors.
 *
 */

template <typename D, typename FS = fbitset::Fbitset<1>,
    typename DS = fbitset::Fbitset<2>, typename S = fbitset::Size>
class Parenther {
public:
    /** The data type for the sizes of the dimensions.
     */

    using Dim = D;

    /** The data type for the factor subsets.
     */

    using Factor_subset = FS;

    /** The data type for the dimension subsets.
     */

    using Dim_subset = DS;

    /** The data type for sizes of factors and dimensions.
     */

    using Size = S;

    /** Initializes with information about the problem.
     *
     * All dimensions should be given, the first `n_sums` of them are actual
     * summations, with the rest being external indices.
     *
     * All factors should be given as a pair of iterators, which generates an
     * iterable giving the indices of the dimensions involved by each of the
     * factors.
     *
     */

    template <typename D_it, typename F_it>
    Parenther(D_it first_dim, D_it last_dim, Size n_sums, F_it first_factor,
        F_it last_factor)
        : dims_(first_dim, last_dim)
        , n_sums_{ n_sums }
        , dims_on_{}
        , factors_with_{}
    {
        factors_with_.assign(n_dims(), Idxes{});

        for (Size factor_idx = 0; first_factor != last_factor;
             ++first_factor, ++factor_idx) {
            assert(dims_on_.size() == factor_idx);
            dims_on_.emplace_back(n_dims());

            for (Size i : *first_factor) {
                assert(i < n_dims());
                factors_with_[i].push_back(factor_idx);
                dims_on_.back().set(i);
            }
        }
    }

    // Except the above constructor, normally we would put the basic
    // information about factors before that about the summation, which comes
    // before that about external indices.

    /** The total number of summation and external indices.
     */

    size_t n_dims() const noexcept { return dims_.size(); }

    /** The total number of factors.
     */

    size_t n_factors() const noexcept { return dims_on_.size(); }

    /** Some indices.
     *
     * It can be used to give an enumeration of things by listing their indices
     * in another linear container.  For instance, a list of dimensions in the
     * problem.
     */

    using Idxes = std::vector<size_t>;

    /** The two operands of a pairwise contraction.
     *
     * An empty second subset is used to indicate a leaf node.
     */

    using Ops = std::pair<Factor_subset, Factor_subset>;

    /** Information about the evaluation for a given subproblem.
     */

    struct Eval {
        /** The factor subsets for the two operands.
         */

        Ops ops;

        /** The summations to be carried out in the last step.
         */

        Idxes sums;

        /** The external indices for the result.
         */

        Idxes exts;

        /** The aggregate cost for the evaluation.
         */

        Dim cost;
    };

    /** The evaluations of a particular subproblem.
     *
     * Here, we guarantee that the first evaluation is always an optimal
     * evaluation.
     */

    using Evals = std::vector<Eval>;

    /** The memoir for the dynamic programming.
     *
     * It also gives the evaluation of all the intermediate steps.
     */

    using Mem = std::unordered_map<Factor_subset, Evals>;

    /** Finds the optimal parenthesization of the problem.
     *
     * The entire memoir for dynamical programming, where all solutions to all
     * the traversed subproblems can be found, are returned.
     */

    Mem opt(Mode mode, bool if_incl)
    {
        Mem mem{};

        Factor_subset top_probl(n_factors(), true);
        Idxes sums(n_sums_);
        std::iota(sums.begin(), sums.end(), 0);
        Idxes exts(n_dims() - n_sums_);
        std::iota(exts.begin(), exts.end(), n_sums_);

        opt(mem, top_probl, sums, exts, mode, if_incl);

        return mem;
    }

private:
    //
    // Core data types.
    //

    /** All dimension of all symbolic ranges, summation and external.
     */

    using Dims = std::vector<Dim>;

    /** Information about broken summations.
     */

    struct Bsums {
        /** The last step cost.
         */

        Dim lsc;

        /** The summation subset in the current subproblem.
         *
         * Here the bits are set according to the indices in the summation list
         * of the current subproblem.
         */

        Dim_subset curr_sums;

        /** The actual summation subset.
         *
         * Here the bits are set according to the index in the global
         * dimensions list.
         */

        Dim_subset sums;

        /** Makes a less-than comparison.
         *
         * The result actually gives the greater-than relation between the last
         * step cost.  This makes it easy for a min heap based on STL.
         */

        bool operator<(const Bsums& o) const noexcept { return lsc > o.lsc; }
    };

    /** Iteration over broken summations.
     */

    class Bsums_it {
    public:
        Bsums_it(
            const Parenther& parenther, const Idxes& sums, const Idxes& exts)
            : parenther_{ parenther }
            , sums_{ sums }
            , q_{}
        {
            q_.emplace(Bsums{ parenther_.get_tot(exts.cbegin(), exts.cend()),
                Dim_subset(sums.size()), Dim_subset(parenther_.n_dims()) });
        }

        /** If we currently have a value.
         */

        explicit operator bool() const { return !q_.empty(); }

        /** Gets the current broken summations.
         *
         * Note that it gives undefined result for invalid iterators.  And the
         * returned reference could be invalidated after the iterator is moved.
         */

        const Bsums& operator*() const { return q_.front(); }

        /** Increment the iterator.
         */

        Bsums_it& operator++()
        {
            assert(!q_.empty());
            auto bsums = std::move(q_.top());
            q_.pop();

            auto top_idx = bsums.sums.find_last();
            auto next_idx = top_idx + 1;
            if (next_idx < sums_.size()) {
                bsums.lsc *= parenther_.dims_[sums_[next_idx]];
                if (top_idx < 0) {
                    bsums.lsc *= Dim(2);
                }
                bsums.curr_sums.set(next_idx);
                bsums.sums.set(sums_[next_idx]);
                q_.push(bsums);

                if (top_idx >= 0) {
                    const auto& top_dim = parenther_.dims_[sums_[top_idx]];
                    assert(bsums.lsc % top_dim == 0);
                    bsums.lsc /= top_dim;

                    assert(bsums.curr_sums[top_idx]);
                    bsums.curr_sums.flip(top_idx);

                    assert(bsums.sums[sums_[top_idx]]);
                    bsums.sums.flip(sums_[top_idx]);

                    q_.push(bsums);
                }
            }

            return *this;
        }

    private:
        /** The underlying parenthesizer.
         */

        const Parenther& parenther_;

        /** The indices of the summations in the current subproblem.
         */

        const Idxes& sums_;

        /** The core heap queue data structure.
         */

        std::priority_queue<Bsums> q_;
    };

    /** A subset of factors and the dimensions on them.
     */

    struct Subset {
        /** The subset of factors.
         */

        Factor_subset factors;

        /** All the dimensions involved by all the factors.
         */

        Dim_subset dims;

        /** Constructs an empty subset.
         */

        Subset(Size n_factors, Size n_dims)
            : factors{ n_factors }
            , dims{ n_dims }
        {
        }
    };

    /** The indivisible chunks mandated by the unbroken summations.
     */

    using Chunks = std::vector<Subset>;

    /** The current bipartition of the factors.
     */

    using Bipart = std::pair<Subset, Subset>;

    /** Nodes for the disjoint-set forest.
     *
     * In addition to the normal parent and rank, here we also have the subset
     * of factors and dimensions.  These values are valid only at root nodes.
     * Factors outside the current sub-problem can be marked by empty factor
     * subsets.
     */

    struct DSF_node {
        size_t parent;
        size_t rank;
        Subset subset;

        DSF_node(Size n_factors, Size n_dims, size_t i)
            : parent{ i }
            , rank{ 0 }
            , subset(n_factors, n_dims)
        {
        }
    };

    /** The disjoint-set forest data structure.
     */

    struct DSF {
        /** The actual nodes in the DSF.
         */
        std::vector<DSF_node> nodes;

        /** Constructs the DSF for a given subproblem.
         */

        DSF(const Parenther& parenther, const Factor_subset& subproblem)
        {
            auto n_factors = parenther.n_factors();
            nodes.reserve(n_factors);

            for (size_t i = 0; i < n_factors; ++i) {
                nodes.emplace_back(n_factors, parenther.n_dims(), i);
                auto& node = nodes.back();

                if (subproblem[i]) {
                    node.subproblem.factors.set(i);
                    node.subproblem.dims |= parenther.dims_on_[i];
                }
            }
        }

        /** Finds the root of the node at the given index.
         */

        size_t find(size_t i)
        {
            size_t& parent = nodes[i].parent;
            if (parent == i)
                return i;

            auto root = find(parent);
            parent = root;
            return root;
        }

        /** Unions the subset with the nodes at the two given indices.
         */

        void merge(size_t i, size_t j)
        {
            auto root_i = find(i);
            auto root_j = find(j);
            if (root_i == root_j)
                return;

            auto& node_i = nodes[root_i];
            auto& node_j = nodes[root_j];

            if (node_i.rank < node_j.rank) {
                merge_core(root_j, root_i);
            } else {
                merge_core(root_i, root_j);
                if (node_i.rank == node_j.rank) {
                    ++node_i.rank;
                }
            }
        }

    private:
        /** Merges the subset from src_idx into that in dest_idx.
         *
         * The two indices must already be root indices.
         */
        void merge_core(size_t dest_idx, size_t src_idx)
        {
            auto& dest = nodes[dest_idx];
            auto& src = nodes[src_idx];
            assert(dest.parent == dest_idx);
            assert(src.parent == src_idx);

            src.parent = dest_idx;
            dest.subset.factors |= src.subset.factors;
            dest.subset.dims |= src.subset.dims;
        }
    };

    /** Iterator for bipartitions compatible with a broken sums set.
     */

    class Bipart_it {
    public:
        Bipart_it(const Chunks& chunks, const Bsums& bsums, Size n_factors,
            Size n_dims)
            : chunks_{ chunks }
            , broken_{ bsums.sums }
            , curr_{ 0 }
            , bipart_{ Subset(n_factors, n_dims), Subset(n_factors, n_dims) }
        {
            assert(chunks_.size() < std::numeric_limits<size_t>::digits - 1);

            form_bipart();
        }

        /** If the current iterator is a valid one.
         */

        explicit operator bool() const noexcept
        {
            return curr_ < (static_cast<size_t>(1) << chunks_.size());
        }

        /** Gets the current bipartition.
         */

        const Bipart& operator*() const noexcept { return bipart_; }

        /** Increments the bipartition iterator.
         */

        Bipart_it& operator++()
        {
            curr_ += 2;
            form_bipart();
            return *this;
        }

    private:
        /** Forms the bipartition from the current counter.
         *
         * The current counter will be taken as the first candidate counter.
         * It will be bumped until a bipartition is found to be compatible with
         * the broken summations.
         */

        void form_bipart()
        {
            while (*this) {
                bipart_.first.clear();
                bipart_.second.clear();

                for (Size i = 0; i < chunks_.size(); ++i) {
                    Bipart* dest;
                    if (curr_ & (static_cast<size_t>(1) << i)) {
                        dest = &bipart_.first;
                    } else {
                        dest = &bipart_.second;
                    }

                    dest->factors |= chunks_[i].factors;
                    dest->dims |= chunks_[i].dims;
                }

                if ((bipart_.first.dims & broken_) == broken_
                    && (bipart_.second.dims & broken_) == broken_) {
                    break;
                } else {
                    curr_ += 2;
                    continue;
                }
            }
            return;
        }

        /** The indivisible chunks.
         */

        const Chunks& chunks_;

        /** The summations to be broken.
         *
         * These are the summations that are required to be involved by both
         * parts.
         */

        const Dim_subset& broken_;

        /** The chunks in the first part of the bipartition.
         */

        size_t curr_;

        /** The current bipartition.
         */

        Bipart bipart_;
    };

    //
    // Internal functions
    //

    const Dim& opt(Mem& mem, const Factor_subset& subprobl, const Idxes& sums,
        const Idxes& exts, Mode mode, bool if_incl)
    {
        auto mem_entry = mem.find(subprobl);
        if (mem_entry != mem.end()) {
            return mem_entry->second.cost;
        }

        auto mem_stat = mem.emplace(subprobl, Evals{});
        assert(!mem_stat.second);
        auto& evals = mem_stat.first->second;
        assert(evals.empty());

        auto n_factors = subprobl.count();
        if (n_factors < 2) {
            // Leaf problem.
            assert(n_factors == 1);

            evals.emplace_back(
                Eval{ Ops{ subprobl, Factor_subset(n_factors()) }, sums, exts,
                    // Here we ignore the possible internal trace cost, since it
                    // does not differentiate between any of the different
                    // parenthesizations.
                    Dim(0) });
            return evals.front().cost;
        }

        for (Bsums_it bsums_it(dims_, sums); bsums_it; ++bsums_it) {
            bool if_break = false;
            if (mode == Mode::GREEDY) {
                if_break = !evals.empty();
            } else if (mode == Mode::NORMAL) {
                if_break = !evals.empty() && bsums_it->lsc > evals.front().cost;
            }
            if (if_break) {
                break;
            }

            // Form the chunks.
            auto chunks = form_chunks(subprobl, sums, *bsums_it);

            for (Bipart_it bipart_it(chunks, *bsums_it, n_factors(), n_dims());
                 bipart_it; ++bipart_it) {

                const auto& l_factors = bipart_it->first.factors;
                const auto& r_factors = bipart_it->second.factors;
                const auto& l_dims = bipart_it->first.dims;
                const auto& r_dims = bipart_it->second.dims;

                Idxes l_sums{};
                Idxes l_exts{};
                Idxes r_sums{};
                Idxes r_exts{};
                // The summations to be carried out right here for the last
                // step.
                Idxes last_sums{};

                // Here we treat the externals first so that the originally
                // external indices appear before the newly externalized
                // indices.
                for (auto i : exts) {
                    if (l_dims[i]) {
                        l_exts.push_back(i);
                    }
                    if (r_dims[i]) {
                        r_exts.push_back(i);
                    }
                }

                for (auto i : sums) {
                    if (bsums_it->sums[i]) {
                        assert(l_dims[i]);
                        assert(r_dims[i]);
                        l_exts.push_back(i);
                        r_exts.push_back(i);
                        last_sums.push_back(i);
                    } else if (l_dims[i]) {
                        assert(!r_dims[i]);
                        l_sums.push_back(i);
                    } else if (r_dims[i]) {
                        assert(!l_dims[i]);
                        r_sums.push_back(i);
                    }
                }

                const auto& l_cost
                    = opt(mem, l_factors, l_sums, l_exts, mode, if_incl);
                const auto& r_cost
                    = opt(mem, r_factors, r_sums, r_exts, mode, if_incl);

                Dim total_cost = bipart_it->lsc + l_cost + r_cost;

                if (!evals.empty() && !if_incl
                    && total_cost > evals.front().cost) {
                    continue;
                }

                bool new_opt = evals.empty() || total_cost < evals.front().cost;

                if (!if_incl && new_opt) {
                    evals.clear();
                }
                evals.emplace_back(Eval{
                    Ops{ l_factors, r_factors }, last_sums, exts, total_cost });

                // Ensure the optimality of the first evaluation.
                if (new_opt && evals.size() > 1) {
                    std::swap(evals.front(), evals.back());
                }
            }
        }

        return evals.front().cost;
    }

    /** Forms the indivisible chunks mandated by the unbroken summations.
     */

    Chunks form_chunks(
        const Factor_subset& subprobl, const Idxes& sums, const Bsums& bsums)
    {
        // Initialize the DSF.
        DSF dsf(*this, subprobl);

        // Union the DSF into chunks.
        for (auto sum : sums) {
            if (bsums.sums[sum]) {
                // Broken summation does not merge things.
                continue;
            }

            ptrdiff_t base = -1;
            for (auto i : factors_with_[sum]) {
                if (!subprobl[i]) {
                    continue;
                }

                if (base == -1) {
                    base = i;
                } else {
                    dsf.merge(base, i);
                }
            }
        }

        Chunks chunks{};
        for (Size i = 0; i < n_factors(); ++i) {
            if (!subprobl[i] || dsf.nodes[i].parent != i) {
                continue;
            }
            chunks.emplace_back(std::move(dsf.nodes[i].subset));
        }

        return chunks;
    }

    //
    // Small utilities
    //

    /** Gets the total size of the given dimensions.
     *
     * Note that an empty set of dimensions gives a size of unity (rather than
     * zero).
     */

    template <typename It> Dim get_tot(It first, It last) const noexcept
    {
        Dim res{ 1 };
        while (first != last) {
            res *= dims_[*first];
            ++first;
        }
        return res;
    }

    //
    // Data fields
    //

    /** The dimensions.
     */

    Dims dims_;

    /** The number of original summation indices.
     */
    Size n_sums_;

    /** The factors with each of the dimensions.
     */

    std::vector<Idxes> factors_with_;

    /** The dimensions on each of the factors.
     */

    std::vector<Dim_subset> dims_on_;
};
}

#endif
