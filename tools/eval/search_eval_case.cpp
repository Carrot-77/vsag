
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

#include "search_eval_case.h"

#include <fstream>
#include <iostream>

#include "monitor/latency_monitor.h"
#include "monitor/memory_peak_monitor.h"
#include "monitor/recall_monitor.h"

namespace vsag::eval {

SearchEvalCase::SearchEvalCase(const std::string& dataset_path,
                               const std::string& index_path,
                               vsag::IndexPtr index,
                               EvalConfig config)
    : EvalCase(dataset_path, index_path, index), config_(std::move(config)) {
    auto search_mode = config.search_mode;
    if (search_mode == "knn") {
        this->search_type_ = SearchType::KNN;
    } else if (search_mode == "range") {
        this->search_type_ = SearchType::RANGE;
    } else if (search_mode == "knn_filter") {
        this->search_type_ = SearchType::KNN_FILTER;
    } else if (search_mode == "range_filter") {
        this->search_type_ = SearchType::RANGE_FILTER;
    }
    this->init_monitor();
}

void
SearchEvalCase::init_monitor() {
    this->init_latency_monitor();
    this->init_recall_monitor();
    this->init_memory_monitor();
}

void
SearchEvalCase::init_latency_monitor() {
    if (config_.enable_latency or config_.enable_tps or config_.enable_percent_latency) {
        auto latency_monitor =
            std::make_shared<LatencyMonitor>(this->dataset_ptr_->GetNumberOfQuery());
        if (config_.enable_qps) {
            latency_monitor->SetMetrics("qps");
        }
        if (config_.enable_latency) {
            latency_monitor->SetMetrics("avg_latency");
        }
        if (config_.enable_percent_latency) {
            latency_monitor->SetMetrics("percent_latency");
        }
        this->monitors_.emplace_back(std::move(latency_monitor));
    }
}

void
SearchEvalCase::init_recall_monitor() {
    if (config_.enable_recall or config_.enable_percent_recall) {
        auto recall_monitor =
            std::make_shared<RecallMonitor>(this->dataset_ptr_->GetNumberOfQuery());
        if (config_.enable_recall) {
            recall_monitor->SetMetrics("avg_recall");
        }
        if (config_.enable_percent_recall) {
            recall_monitor->SetMetrics("percent_recall");
        }
        this->monitors_.emplace_back(std::move(recall_monitor));
    }
}

void
SearchEvalCase::init_memory_monitor() {
    if (config_.enable_memory) {
        auto memory_peak_monitor = std::make_shared<MemoryPeakMonitor>();
        this->monitors_.emplace_back(std::move(memory_peak_monitor));
    }
}

void
SearchEvalCase::Run() {
    this->deserialize();
    switch (this->search_type_) {
        case KNN:
            this->do_knn_search();
            break;
        case RANGE:
            this->do_range_search();
            break;
        case KNN_FILTER:
            this->do_knn_filter_search();
            break;
        case RANGE_FILTER:
            this->do_range_filter_search();
            break;
    }
    auto result = this->process_result();
    eval::SearchEvalCase::PrintResult(result);
}
void
SearchEvalCase::deserialize() {
    std::ifstream infile(this->index_path_, std::ios::binary);
    this->index_->Deserialize(infile);
}
void
SearchEvalCase::do_knn_search() {
    uint64_t topk = config_.top_k;
    auto query_count = this->dataset_ptr_->GetNumberOfQuery();
    this->logger_->Debug("query count is " + std::to_string(query_count));
    for (auto& monitor : this->monitors_) {
        monitor->Start();
        for (int64_t i = 0; i < query_count; ++i) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(this->dataset_ptr_->GetDim())->Owner(false);
            if (this->dataset_ptr_->GetTestDataType() == vsag::DATATYPE_FLOAT32) {
                query->Float32Vectors((const float*)this->dataset_ptr_->GetOneTest(i));
            } else if (this->dataset_ptr_->GetTestDataType() == vsag::DATATYPE_INT8) {
                query->Int8Vectors((const int8_t*)this->dataset_ptr_->GetOneTest(i));
            }
            auto result = this->index_->KnnSearch(query, topk, config_.search_param);
            if (not result.has_value()) {
                std::cerr << "query error: " << result.error().message << std::endl;
                exit(-1);
            }
            int64_t* neighbors = dataset_ptr_->GetNeighbors(i);
            const int64_t* ground_truth = result.value()->GetIds();
            auto record = std::make_tuple(neighbors, ground_truth, topk);
            monitor->Record(&record);
        }
        monitor->Stop();
    }
}
void
SearchEvalCase::do_range_search() {
}
void
SearchEvalCase::do_knn_filter_search() {
}
void
SearchEvalCase::do_range_filter_search() {
}

SearchEvalCase::JsonType
SearchEvalCase::process_result() {
    JsonType result;
    for (auto& monitor : this->monitors_) {
        const auto& one_result = monitor->GetResult();
        EvalCase::MergeJsonType(one_result, result);
    }
    return result;
}

}  // namespace vsag::eval