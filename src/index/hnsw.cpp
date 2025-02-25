
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hnsw.h"

#include <fmt/format-inl.h>

#include <cstdint>
#include <exception>
#include <new>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "algorithm/hnswlib/hnswlib.h"
#include "common.h"
#include "data_cell/flatten_datacell.h"
#include "data_cell/graph_datacell_parameter.h"
#include "empty_index_binary_set.h"
#include "index/hnsw_zparameters.h"
#include "io/memory_block_io_parameter.h"
#include "quantization/fp32_quantizer_parameter.h"
#include "safe_allocator.h"
#include "utils/slow_task_timer.h"
#include "utils/timer.h"
#include "vsag/binaryset.h"
#include "vsag/constants.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"

namespace vsag {

const static int64_t EXPANSION_NUM = 1000000;
const static int64_t DEFAULT_MAX_ELEMENT = 1;
const static int MINIMAL_M = 8;
const static int MAXIMAL_M = 64;
const static uint32_t GENERATE_SEARCH_K = 50;
const static uint32_t UPDATE_CHECK_SEARCH_K = 10;
const static uint32_t GENERATE_SEARCH_L = 400;
const static uint32_t UPDATE_CHECK_SEARCH_L = 100;
const static float GENERATE_OMEGA = 0.51;

HNSW::HNSW(HnswParameters hnsw_params, const IndexCommonParam& index_common_param)
    : space_(std::move(hnsw_params.space)),
      use_static_(hnsw_params.use_static),
      use_conjugate_graph_(hnsw_params.use_conjugate_graph),
      use_reversed_edges_(hnsw_params.use_reversed_edges),
      type_(hnsw_params.type),
      max_degree_(hnsw_params.max_degree),
      dim_(index_common_param.dim_),
      index_common_param_(index_common_param) {
    auto M = std::min(  // NOLINT(readability-identifier-naming)
        std::max((int)hnsw_params.max_degree, MINIMAL_M),
        MAXIMAL_M);

    if (hnsw_params.ef_construction <= 0) {
        throw std::runtime_error(MESSAGE_PARAMETER);
    }

    allocator_ = index_common_param.allocator_;

    if (hnsw_params.use_conjugate_graph) {
        conjugate_graph_ = std::make_shared<ConjugateGraph>(allocator_.get());
    }

    if (!use_static_) {
        alg_hnsw_ =
            std::make_shared<hnswlib::HierarchicalNSW>(space_.get(),
                                                       DEFAULT_MAX_ELEMENT,
                                                       allocator_.get(),
                                                       M,
                                                       hnsw_params.ef_construction,
                                                       use_reversed_edges_,
                                                       hnsw_params.normalize,
                                                       Options::Instance().block_size_limit());
    } else {
        if (dim_ % 4 != 0) {
            // FIXME(wxyu): remove throw stmt from construct function
            throw std::runtime_error("cannot build static hnsw while dim % 4 != 0");
        }
        alg_hnsw_ = std::make_shared<hnswlib::StaticHierarchicalNSW>(
            space_.get(),
            DEFAULT_MAX_ELEMENT,
            allocator_.get(),
            M,
            hnsw_params.ef_construction,
            Options::Instance().block_size_limit());
    }

    this->init_feature_list();
}

tl::expected<std::vector<int64_t>, Error>
HNSW::build(const DatasetPtr& base) {
    try {
        if (base->GetNumElements() == 0) {
            empty_index_ = true;
            return std::vector<int64_t>();
        }

        logger::debug("index.dim={}, base.dim={}", this->dim_, base->GetDim());

        auto base_dim = base->GetDim();
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));

        int64_t num_elements = base->GetNumElements();

        std::unique_lock lock(rw_mutex_);

        const auto* ids = base->GetIds();
        void* vectors = nullptr;
        size_t data_size = 0;
        get_vectors(base, &vectors, &data_size);
        std::vector<int64_t> failed_ids;
        {
            SlowTaskTimer t("hnsw graph");
            for (int64_t i = 0; i < num_elements; ++i) {
                // noexcept runtime
                if (!alg_hnsw_->addPoint((const void*)((char*)vectors + data_size * i), ids[i])) {
                    logger::debug("duplicate point: {}", ids[i]);
                    failed_ids.emplace_back(ids[i]);
                }
            }
        }

        if (use_static_) {
            SlowTaskTimer t("hnsw pq", 1000);
            auto* hnsw = static_cast<hnswlib::StaticHierarchicalNSW*>(alg_hnsw_.get());
            hnsw->encode_hnsw_data();
        }

        return failed_ids;
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to build(invalid argument): ", e.what());
    }
}

tl::expected<std::vector<int64_t>, Error>
HNSW::add(const DatasetPtr& base) {
#ifndef ENABLE_TESTS
    SlowTaskTimer t("hnsw add", 20);
#endif
    if (use_static_) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                              "static index does not support add");
    }
    try {
        auto base_dim = base->GetDim();
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));

        int64_t num_elements = base->GetNumElements();
        const auto* ids = base->GetIds();
        void* vectors = nullptr;
        size_t data_size = 0;
        get_vectors(base, &vectors, &data_size);
        std::vector<int64_t> failed_ids;
        for (int64_t i = 0; i < num_elements; ++i) {
            // noexcept runtime
            std::shared_lock lock(rw_mutex_);
            if (!alg_hnsw_->addPoint((const void*)((char*)vectors + data_size * i), ids[i])) {
                logger::debug("duplicate point: {}", i);
                failed_ids.push_back(ids[i]);
            }
        }

        return failed_ids;
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to add(invalid argument): ", e.what());
    }
}

template <typename FilterType>
tl::expected<DatasetPtr, Error>
HNSW::knn_search_internal(const DatasetPtr& query,
                          int64_t k,
                          const std::string& parameters,
                          const FilterType& filter_obj) const {
    if (filter_obj) {
        auto filter = std::make_shared<UniqueFilter>(filter_obj);
        return this->knn_search(query, k, parameters, filter);
    }
    return this->knn_search(query, k, parameters, nullptr);
};

tl::expected<DatasetPtr, Error>
HNSW::knn_search(const DatasetPtr& query,
                 int64_t k,
                 const std::string& parameters,
                 const FilterPtr filter_ptr) const {
#ifndef ENABLE_TESTS
    SlowTaskTimer t_total("hnsw knnsearch", 20);
#endif
    try {
        // cannot perform search on empty index
        if (empty_index_) {
            auto ret = Dataset::Make();
            ret->Dim(0)->NumElements(1);
            return ret;
        }

        // check query vector
        CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");
        void* vector = nullptr;
        size_t data_size = 0;
        get_vectors(query, &vector, &data_size);
        int64_t query_dim = query->GetDim();
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));

        // check k
        CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k))
        k = std::min(k, GetNumElements());

        std::shared_lock lock_global(rw_mutex_);

        // check search parameters
        auto params = HnswSearchParameters::FromJson(parameters);

        // perform search
        int64_t original_k = k;
        std::priority_queue<std::pair<float, LabelType>> results;
        double time_cost;
        try {
            Timer t(time_cost);
            if (use_conjugate_graph_ and params.use_conjugate_graph_search) {
                k = std::max(k, LOOK_AT_K);
            }
            results = alg_hnsw_->searchKnn((const void*)(vector),
                                           k,
                                           std::max(params.ef_search, k),
                                           filter_ptr,
                                           params.skip_ratio);
        } catch (const std::runtime_error& e) {
            LOG_ERROR_AND_RETURNS(ErrorType::INTERNAL_ERROR,
                                  "failed to perofrm knn_search(internalError): ",
                                  e.what());
        }

        // update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            result_queues_[STATSTIC_KNN_TIME].Push(static_cast<float>(time_cost));
        }

        // return result
        auto result = Dataset::Make();
        if (results.empty()) {
            result->Dim(0)->NumElements(1);
            return result;
        }

        // perform conjugate graph enhancement
        if (use_conjugate_graph_ and params.use_conjugate_graph_search) {
            std::shared_lock lock(rw_mutex_);
            time_cost = 0;
            Timer t(time_cost);

            auto func = [this, vector](int64_t label) {
                return this->alg_hnsw_->getDistanceByLabel(label, vector);
            };
            conjugate_graph_->EnhanceResult(results, func);
            k = original_k;
        }

        // return result
        while (results.size() > k) {
            results.pop();
        }

        result->Dim(static_cast<int64_t>(results.size()))
            ->NumElements(1)
            ->Owner(true, allocator_.get());

        auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * results.size());
        result->Ids(ids);
        auto* dists = (float*)allocator_->Allocate(sizeof(float) * results.size());
        result->Distances(dists);

        for (auto j = static_cast<int64_t>(results.size() - 1); j >= 0; --j) {
            dists[j] = results.top().first;
            ids[j] = results.top().second;
            results.pop();
        }

        return std::move(result);
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to perform knn_search(invalid argument): ",
                              e.what());
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY,
                              "failed to perform knn_search(not enough memory): ",
                              e.what());
    }
}

template <typename FilterType>
tl::expected<DatasetPtr, Error>
HNSW::range_search_internal(const DatasetPtr& query,
                            float radius,
                            const std::string& parameters,
                            const FilterType& filter_obj,
                            int64_t limited_size) const {
    if (filter_obj) {
        auto filter = std::make_shared<UniqueFilter>(filter_obj);
        return this->range_search(query, radius, parameters, filter, limited_size);
    }
    return this->range_search(query, radius, parameters, nullptr, limited_size);
};

tl::expected<DatasetPtr, Error>
HNSW::range_search(const DatasetPtr& query,
                   float radius,
                   const std::string& parameters,
                   const FilterPtr filter_ptr,
                   int64_t limited_size) const {
#ifndef ENABLE_TESTS
    SlowTaskTimer t("hnsw rangesearch", 20);
#endif
    try {
        // cannot perform search on empty index
        if (empty_index_) {
            auto ret = Dataset::Make();
            ret->Dim(0)->NumElements(1);
            return ret;
        }

        if (use_static_) {
            LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                  "static index does not support rangesearch");
        }

        // check query vector
        CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");
        void* vector = nullptr;
        size_t data_size = 0;
        get_vectors(query, &vector, &data_size);
        int64_t query_dim = query->GetDim();
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));

        // check radius
        CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))

        // check limited_size
        CHECK_ARGUMENT(limited_size != 0,
                       fmt::format("limited_size({}) must not be equal to 0", limited_size))

        // check search parameters
        auto params = HnswSearchParameters::FromJson(parameters);

        // perform search
        std::priority_queue<std::pair<float, LabelType>> results;
        double time_cost;
        try {
            std::shared_lock lock(rw_mutex_);
            Timer timer(time_cost);
            results =
                alg_hnsw_->searchRange((const void*)(vector), radius, params.ef_search, filter_ptr);
        } catch (std::runtime_error& e) {
            LOG_ERROR_AND_RETURNS(ErrorType::INTERNAL_ERROR,
                                  "failed to perofrm range_search(internalError): ",
                                  e.what());
        }

        // update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            result_queues_[STATSTIC_KNN_TIME].Push(static_cast<float>(time_cost));
        }

        // return result
        auto result = Dataset::Make();
        size_t target_size = results.size();
        if (results.empty()) {
            result->Dim(0)->NumElements(1);
            return result;
        }
        if (limited_size >= 1) {
            target_size = std::min((size_t)limited_size, target_size);
        }
        result->Dim(static_cast<int64_t>(target_size))
            ->NumElements(1)
            ->Owner(true, allocator_.get());
        auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * target_size);
        result->Ids(ids);
        auto* dists = (float*)allocator_->Allocate(sizeof(float) * target_size);
        result->Distances(dists);
        for (auto j = static_cast<int64_t>(results.size() - 1); j >= 0; --j) {
            if (j < target_size) {
                dists[j] = results.top().first;
                ids[j] = results.top().second;
            }
            results.pop();
        }

        return std::move(result);
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to perform range_search(invalid argument): ",
                              e.what());
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY,
                              "failed to perform range_search(not enough memory): ",
                              e.what());
    }
}

tl::expected<BinarySet, Error>
HNSW::serialize() const {
    if (GetNumElements() == 0) {
        // return a special binaryset means empty
        return EmptyIndexBinarySet::Make("EMPTY_HNSW");
    }

    SlowTaskTimer t("hnsw serialize");
    size_t num_bytes = alg_hnsw_->calcSerializeSize();
    try {
        std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
        std::shared_lock lock(rw_mutex_);
        alg_hnsw_->saveIndex(bin.get());
        Binary b{
            .data = bin,
            .size = num_bytes,
        };
        BinarySet bs;
        bs.Set(HNSW_DATA, b);

        if (use_conjugate_graph_) {
            Binary b_cg = *conjugate_graph_->Serialize();
            bs.Set(CONJUGATE_GRAPH_DATA, b_cg);
        }

        return bs;
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to serialize(bad alloc): ", e.what());
    }
}

tl::expected<void, Error>
HNSW::serialize(std::ostream& out_stream) {
    if (GetNumElements() == 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_EMPTY, "failed to serialize: hnsw index is empty");

        // FIXME(wxyu): cannot support serialize empty index by stream
        // auto bs = empty_binaryset();
        // for (const auto& key : bs.GetKeys()) {
        //     auto b = bs.Get(key);
        //     out_stream.write((char*)b.data.get(), b.size);
        // }
        // return {};
    }

    SlowTaskTimer t("hnsw serialize");

    // no expected exception
    std::shared_lock lock(rw_mutex_);
    alg_hnsw_->saveIndex(out_stream);

    if (use_conjugate_graph_) {
        conjugate_graph_->Serialize(out_stream);
    }

    return {};
}

tl::expected<void, Error>
HNSW::deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw_->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    // check if binaryset is a empty index
    if (binary_set.Contains(BLANK_INDEX)) {
        empty_index_ = true;
        return {};
    }

    Binary b = binary_set.Get(HNSW_DATA);
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        if (len + offset > b.size) {
            throw std::runtime_error(
                fmt::format("offset({}) + len({}) > size({})", offset, len, b.size));
        }
        std::memcpy(dest, b.data.get() + offset, len);
    };

    try {
        std::unique_lock lock(rw_mutex_);
        int64_t cursor = 0;
        ReadFuncStreamReader reader(func, cursor);
        BufferStreamReader buffer_reader(&reader, b.size, allocator_.get());
        alg_hnsw_->loadIndex(buffer_reader, this->space_.get());
        if (use_conjugate_graph_) {
            Binary b_cg = binary_set.Get(CONJUGATE_GRAPH_DATA);
            if (not conjugate_graph_->Deserialize(b_cg).has_value()) {
                throw std::runtime_error("error in deserialize conjugate graph");
            }
        }
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    } catch (const std::out_of_range& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, Error>
HNSW::deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw_->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    // check if readerset is a empty index
    if (reader_set.Contains(BLANK_INDEX)) {
        empty_index_ = true;
        return {};
    }

    const auto& hnsw_data = reader_set.Get(HNSW_DATA);

    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        if (len + offset > hnsw_data->Size()) {
            throw std::runtime_error(
                fmt::format("offset({}) + len({}) > size({})", offset, len, hnsw_data->Size()));
        }
        hnsw_data->Read(offset, len, dest);
    };

    try {
        std::unique_lock lock(rw_mutex_);

        int64_t cursor = 0;
        ReadFuncStreamReader reader(func, cursor);
        BufferStreamReader buffer_reader(&reader, hnsw_data->Size(), allocator_.get());
        alg_hnsw_->loadIndex(buffer_reader, this->space_.get());
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, Error>
HNSW::deserialize(std::istream& in_stream) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw_->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    try {
        std::unique_lock lock(rw_mutex_);

        IOStreamReader reader(in_stream);
        alg_hnsw_->loadIndex(reader, this->space_.get());
        if (use_conjugate_graph_ and not conjugate_graph_->Deserialize(reader).has_value()) {
            throw std::runtime_error("error in deserialize conjugate graph");
        }
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

std::string
HNSW::GetStats() const {
    JsonType j;
    j[STATSTIC_DATA_NUM] = GetNumElements();
    j[STATSTIC_INDEX_NAME] = INDEX_HNSW;
    j[STATSTIC_MEMORY] = GetMemoryUsage();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        for (auto& item : result_queues_) {
            j[item.first] = item.second.GetAvgResult();
        }
    }
    return j.dump();
}

tl::expected<bool, Error>
HNSW::update_id(int64_t old_id, int64_t new_id) {
    if (use_static_) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                              "static hnsw does not support update");
    }

    try {
        if (old_id == new_id) {
            return true;
        }

        // note that the validation of old_id is handled within updateLabel.
        std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)->updateLabel(old_id,
                                                                                        new_id);
        if (use_conjugate_graph_) {
            std::unique_lock lock(rw_mutex_);
            conjugate_graph_->UpdateId(old_id, new_id);
        }
    } catch (const std::runtime_error& e) {
#ifndef ENABLE_TESTS
        logger::warn(
            "update error for replace old_id {} to new_id {}: {}", old_id, new_id, e.what());
#endif
        return false;
    }

    return true;
}

tl::expected<bool, Error>
HNSW::update_vector(int64_t id, const DatasetPtr& new_base, bool force_update) {
    if (use_static_) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                              "static hnsw does not support update");
    }

    try {
        // the validation of the new vector
        void* new_base_vec = nullptr;
        size_t data_size = 0;
        get_vectors(new_base, &new_base_vec, &data_size);

        if (not force_update) {
            std::shared_ptr<int8_t[]> base_data(new int8_t[data_size]);
            auto base = Dataset::Make();

            // check if id exists and get copied base data
            std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)->copyDataByLabel(
                id, base_data.get());
            set_dataset(base, base_data.get(), 1);

            // search neighbors
            auto neighbors = *this->knn_search(base,
                                               vsag::UPDATE_CHECK_SEARCH_K,
                                               fmt::format(R"({{
                                                                "hnsw":
                                                                    {{
                                                                        "ef_search": {}
                                                                    }}
                                                            }})",
                                                           vsag::UPDATE_CHECK_SEARCH_L),
                                               nullptr);

            // check whether the neighborhood relationship is same
            float self_dist = std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)
                                  ->getDistanceByLabel(id, new_base_vec);
            for (int i = 0; i < neighbors->GetDim(); i++) {
                float neighbor_dist =
                    std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)
                        ->getDistanceByLabel(neighbors->GetIds()[i], new_base_vec);
                if (neighbor_dist < self_dist) {
                    return false;
                }
            }
        }

        // note that the validation of old_id is handled within updatePoint.
        std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)->updateVector(
            id, new_base_vec);
    } catch (const std::runtime_error& e) {
#ifndef ENABLE_TESTS
        logger::warn("update error for replace vector of id {}: {}", id, e.what());
#endif
        return false;
    }

    return true;
}

tl::expected<bool, Error>
HNSW::remove(int64_t id) {
    if (use_static_) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                              "static hnsw does not support remove");
    }

    try {
        std::unique_lock lock(rw_mutex_);
        if (use_reversed_edges_) {
            std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)->removePoint(id);
        } else {
            std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)->markDelete(id);
        }
    } catch (const std::runtime_error& e) {
        logger::warn("mark delete error for id {}: {}", id, e.what());
        return false;
    }

    return true;
}

tl::expected<uint32_t, Error>
HNSW::feedback(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               int64_t global_optimum_tag_id) {
    if (not use_conjugate_graph_) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                              "no conjugate graph used for feedback");
    }
    if (empty_index_) {
        return 0;
    }

    if (global_optimum_tag_id == std::numeric_limits<int64_t>::max()) {
        auto exact_result = this->brute_force(query, k);
        if (exact_result.has_value()) {
            global_optimum_tag_id = exact_result.value()->GetIds()[0];
        } else {
            LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                                  "failed to feedback(invalid argument): ",
                                  exact_result.error().message);
        }
    }

    auto result = this->knn_search(query, k, parameters, nullptr);
    if (not result.has_value()) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to feedback(invalid argument): ",
                              result.error().message);
    }
    std::unique_lock lock(rw_mutex_);
    return this->feedback(*result, global_optimum_tag_id, k);
}

tl::expected<uint32_t, Error>
HNSW::feedback(const DatasetPtr& result, int64_t global_optimum_tag_id, int64_t k) {
    if (not alg_hnsw_->isValidLabel(global_optimum_tag_id)) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT,
            "failed to feedback(invalid argument): global optimum tag id doesn't belong to index");
    }

    const auto* tag_ids = result->GetIds();
    k = std::min(k, result->GetDim());
    uint32_t successfully_feedback = 0;

    for (int i = 0; i < k; i++) {
        if (not alg_hnsw_->isValidLabel(tag_ids[i])) {
            LOG_ERROR_AND_RETURNS(
                ErrorType::INVALID_ARGUMENT,
                "failed to feedback(invalid argument): input result don't belong to index");
        }
        if (*conjugate_graph_->AddNeighbor(tag_ids[i], global_optimum_tag_id)) {
            successfully_feedback++;
        }
    }

    return successfully_feedback;
}

tl::expected<DatasetPtr, Error>
HNSW::brute_force(const DatasetPtr& query, int64_t k) {
    try {
        CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
        CHECK_ARGUMENT(query->GetNumElements() == 1,
                       fmt::format("query num({}) must equal to 1", query->GetNumElements()));
        CHECK_ARGUMENT(
            query->GetDim() == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query->GetDim(), dim_));

        auto result = Dataset::Make();
        result->NumElements(k)->Owner(true, allocator_.get());
        auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * k);
        result->Ids(ids);
        auto* dists = (float*)allocator_->Allocate(sizeof(float) * k);
        result->Distances(dists);

        void* vector = nullptr;
        size_t data_size = 0;
        get_vectors(query, &vector, &data_size);

        std::shared_lock lock(rw_mutex_);
        std::priority_queue<std::pair<float, LabelType>> bf_result =
            alg_hnsw_->bruteForce(vector, k);
        result->Dim(std::min(k, (int64_t)bf_result.size()));

        for (auto i = static_cast<int32_t>(result->GetDim() - 1); i >= 0; i--) {
            ids[i] = bf_result.top().second;
            dists[i] = bf_result.top().first;
            bf_result.pop();
        }

        return std::move(result);
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to perform brute force search(invalid argument): ",
                              e.what());
    }
}

bool
HNSW::CheckGraphIntegrity() const {
    auto* hnsw = static_cast<hnswlib::HierarchicalNSW*>(alg_hnsw_.get());
    return hnsw->checkReverseConnection();
}

tl::expected<uint32_t, Error>
HNSW::pretrain(const std::vector<int64_t>& base_tag_ids,
               uint32_t k,
               const std::string& parameters) {
    if (not use_conjugate_graph_) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                              "no conjugate graph used for pretrain");
    }
    if (empty_index_) {
        return 0;
    }

    uint32_t data_size = 0;
    uint32_t add_edges = 0;
    int64_t topk_neighbor_tag_id;
    auto base = Dataset::Make();
    auto generated_query = Dataset::Make();
    if (type_ == DataTypes::DATA_TYPE_INT8) {
        data_size = dim_;
    } else {
        data_size = dim_ * 4;
    }
    std::shared_ptr<int8_t[]> base_data(new int8_t[data_size]);
    std::shared_ptr<int8_t[]> topk_data(new int8_t[data_size]);

    std::shared_ptr<int8_t[]> generated_data(new int8_t[data_size]);
    set_dataset(generated_query, generated_data.get(), 1);

    for (const int64_t& base_tag_id : base_tag_ids) {
        try {
            std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)->copyDataByLabel(
                base_tag_id, base_data.get());
            set_dataset(base, base_data.get(), 1);
        } catch (const std::runtime_error& e) {
            LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                                  fmt::format("failed to pretrain(invalid argument): base tag id "
                                              "({}) doesn't belong to index",
                                              base_tag_id));
        }

        auto result = this->knn_search(base,
                                       vsag::GENERATE_SEARCH_K,
                                       fmt::format(R"(
                                        {{
                                            "hnsw": {{
                                                "ef_search": {},
                                                "use_conjugate_graph": true
                                            }}
                                        }})",
                                                   vsag::GENERATE_SEARCH_L),
                                       nullptr);

        for (int i = 0; i < result.value()->GetDim(); i++) {
            topk_neighbor_tag_id = result.value()->GetIds()[i];
            if (topk_neighbor_tag_id == base_tag_id) {
                continue;
            }

            std::reinterpret_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_)->copyDataByLabel(
                topk_neighbor_tag_id, topk_data.get());

            for (int d = 0; d < dim_; d++) {
                if (type_ == DataTypes::DATA_TYPE_INT8) {
                    generated_data.get()[d] =
                        vsag::GENERATE_OMEGA * (float)(base_data[d]) +  // NOLINT
                        (1 - vsag::GENERATE_OMEGA) * (float)(topk_data[d]);
                } else {
                    ((float*)generated_data.get())[d] =
                        vsag::GENERATE_OMEGA * ((float*)base_data.get())[d] +
                        (1 - vsag::GENERATE_OMEGA) * ((float*)topk_data.get())[d];
                }
            }

            auto feedback_result = this->Feedback(generated_query, k, parameters, base_tag_id);
            if (feedback_result.has_value()) {
                add_edges += *feedback_result;
            } else {
                LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                                      "failed to feedback(invalid argument): ",
                                      feedback_result.error().message);
            }
        }
    }

    return add_edges;
}

tl::expected<bool, Error>
HNSW::InitMemorySpace() {
    if (is_init_memory_) {
        return true;
    }
    try {
        alg_hnsw_->init_memory_space();
    } catch (std::runtime_error& r) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY, "allocate memory failed:", r.what());
    }
    is_init_memory_ = true;
    return true;
}

void
HNSW::get_vectors(const vsag::DatasetPtr& base, void** vectors_ptr, size_t* data_size_ptr) const {
    if (type_ == DataTypes::DATA_TYPE_FLOAT) {
        *vectors_ptr = (void*)base->GetFloat32Vectors();
        *data_size_ptr = dim_ * sizeof(float);
    } else if (type_ == DataTypes::DATA_TYPE_INT8) {
        *vectors_ptr = (void*)base->GetInt8Vectors();
        *data_size_ptr = dim_ * sizeof(int8_t);
    } else {
        throw std::invalid_argument(fmt::format("no support for this metric: {}", (int)type_));
    }
}

void
HNSW::set_dataset(const DatasetPtr& base, const void* vectors_ptr, uint32_t num_element) const {
    if (type_ == DataTypes::DATA_TYPE_FLOAT) {
        base->Float32Vectors((float*)vectors_ptr)
            ->Dim(dim_)
            ->Owner(false)
            ->NumElements(num_element);
    } else if (type_ == DataTypes::DATA_TYPE_INT8) {
        base->Int8Vectors((int8_t*)vectors_ptr)->Dim(dim_)->Owner(false)->NumElements(num_element);
    } else {
        throw std::invalid_argument(fmt::format("no support for this type: {}", (int)type_));
    }
}
bool
HNSW::CheckFeature(IndexFeature feature) const {
    return this->feature_list_.CheckFeature(feature);
}

void
HNSW::init_feature_list() {
    // Add & Build
    feature_list_.SetFeatures({IndexFeature::SUPPORT_BUILD,
                               IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
                               IndexFeature::SUPPORT_ADD_AFTER_BUILD,
                               IndexFeature::SUPPORT_ADD_FROM_EMPTY});
    // Search
    feature_list_.SetFeatures({IndexFeature::SUPPORT_KNN_SEARCH,
                               IndexFeature::SUPPORT_RANGE_SEARCH,
                               IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
                               IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER});
    // concurrency
    feature_list_.SetFeatures({IndexFeature::SUPPORT_SEARCH_CONCURRENT,
                               IndexFeature::SUPPORT_ADD_SEARCH_CONCURRENT,
                               IndexFeature::SUPPORT_ADD_CONCURRENT,
                               IndexFeature::SUPPORT_UPDATE_ID_CONCURRENT,
                               IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT});
    // serialize
    feature_list_.SetFeatures({IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
                               IndexFeature::SUPPORT_DESERIALIZE_FILE,
                               IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
                               IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
                               IndexFeature::SUPPORT_SERIALIZE_FILE});
    // other
    feature_list_.SetFeatures({IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID,
                               IndexFeature::SUPPORT_CHECK_ID_EXIST,
                               IndexFeature::SUPPORT_MERGE_INDEX});
}

bool
HNSW::ExtractDataAndGraph(FlattenInterfacePtr& data,
                          GraphInterfacePtr& graph,
                          Vector<LabelType>& ids,
                          IdMapFunction func,
                          Allocator* allocator) {
    if (use_static_) {
        return false;
    }
    auto hnsw = std::static_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_);
    auto cur_element_count = hnsw->getCurrentElementCount();
    int64_t origin_data_num = data->total_count_;
    int64_t valid_id_count = 0;
    BitsetPtr bitset = std::make_shared<BitsetImpl>();
    for (auto i = 0; i < cur_element_count; ++i) {
        int64_t id = hnsw->getExternalLabel(i);
        auto [is_exist, new_id] = func(id);
        if (not is_exist) {
            bitset->Set(i);
        }
        auto offset = valid_id_count + origin_data_num;
        char* vector_data = hnsw->getDataByInternalId(i);
        data->InsertVector(reinterpret_cast<float*>(vector_data));
        int* link_data = (int*)hnsw->getLinklistAtLevel(i, 0);
        size_t size = hnsw->getListCount((unsigned int*)link_data);
        Vector<InnerIdType> edge(allocator);
        for (int j = 0; j < size; ++j) {
            if (not bitset->Test(*(link_data + 1 + j))) {
                edge.push_back(origin_data_num + *(link_data + 1 + j));
            }
        }
        graph->InsertNeighborsById(offset, edge);
        ids.push_back(new_id);
        valid_id_count++;
    }
    return true;
}

bool
HNSW::SetDataAndGraph(FlattenInterfacePtr& data, GraphInterfacePtr& graph, Vector<LabelType>& ids) {
    if (use_static_) {
        return false;
    }
    auto hnsw = std::static_pointer_cast<hnswlib::HierarchicalNSW>(alg_hnsw_);
    hnsw->setDataAndGraph(data, graph, ids);
    return true;
}

void
extract_data_and_graph(const std::vector<MergeUnit>& merge_units,
                       FlattenInterfacePtr& data,
                       GraphInterfacePtr& graph,
                       Vector<LabelType>& ids,
                       Allocator* allocator) {
    for (const auto& merge_unit : merge_units) {
        auto stat_string = merge_unit.index->GetStats();
        auto stats = JsonType::parse(stat_string);
        std::string index_name = stats[STATSTIC_INDEX_NAME];
        auto hnsw = std::dynamic_pointer_cast<HNSW>(merge_unit.index);
        hnsw->ExtractDataAndGraph(data, graph, ids, merge_unit.id_map_func, allocator);
    }
}

tl::expected<void, Error>
HNSW::merge(const std::vector<MergeUnit>& merge_units) {
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->io_parameter_ = std::make_shared<MemoryBlockIOParameter>();
    param->quantizer_parameter_ = std::make_shared<FP32QuantizerParameter>();
    GraphDataCellParamPtr graph_param_ptr = std::make_shared<GraphDataCellParameter>();
    graph_param_ptr->io_parameter_ = std::make_shared<vsag::MemoryBlockIOParameter>();
    graph_param_ptr->max_degree_ = max_degree_ * 2;

    FlattenInterfacePtr flatten_interface =
        FlattenInterface::MakeInstance(param, index_common_param_);
    GraphInterfacePtr graph_interface =
        GraphInterface::MakeInstance(graph_param_ptr, index_common_param_, false);
    Vector<LabelType> ids(allocator_.get());
    // extract data and graph
    IdMapFunction id_map = [](int64_t id) -> std::tuple<bool, int64_t> {
        return std::make_tuple(true, id);
    };
    this->ExtractDataAndGraph(flatten_interface, graph_interface, ids, id_map, allocator_.get());
    extract_data_and_graph(merge_units, flatten_interface, graph_interface, ids, allocator_.get());
    // TODO(inabao): merge graph
    // set graph
    SetDataAndGraph(flatten_interface, graph_interface, ids);
    return {};
}

template tl::expected<DatasetPtr, Error>
HNSW::knn_search_internal<BitsetPtr>(const DatasetPtr& query,
                                     int64_t k,
                                     const std::string& parameters,
                                     const BitsetPtr& filter_obj) const;

template tl::expected<DatasetPtr, Error>
HNSW::knn_search_internal<std::function<bool(int64_t)>>(
    const DatasetPtr& query,
    int64_t k,
    const std::string& parameters,
    const std::function<bool(int64_t)>& filter_obj) const;

}  // namespace vsag
