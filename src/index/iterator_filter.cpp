
// Copyright 2024-present the vsag project
//
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

#include "iterator_filter.h"

#include "../logger.h"

namespace vsag {

IteratorFilterContext::~IteratorFilterContext() {
    allocator_->Deallocate(list_);
    allocator_->Deallocate(visited_time_);
}

tl::expected<void, Error>
IteratorFilterContext::init(InnerIdType max_size,
                            int64_t ef_search,
                            Allocator* allocator) {
    if (ef_search == 0 || max_size == 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to serialize: hnsw index is empty");
    }
    try {
        ef_search_ = ef_search;
        allocator_ = allocator;
        max_size_ = max_size;
        discard_ = std::make_unique<std::priority_queue<std::pair<float, InnerIdType>>>();
        inner_distance_ = std::make_unique<std::unordered_map<InnerIdType, float>>();
        list_ = reinterpret_cast<VisitedListType*>(
            allocator_->Allocate((uint64_t)max_size * sizeof(VisitedListType)));
        memset(list_, 0, max_size * sizeof(VisitedListType));
        visited_time_ = reinterpret_cast<VisitedListType*>(
            allocator_->Allocate((uint64_t)max_size * sizeof(VisitedListType)));
        memset(visited_time_, 0, max_size * sizeof(VisitedListType));
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY,
                              "failed to init iterator filter(not enough memory): ",
                              e.what());
    }
    return {};
}

void
IteratorFilterContext::AddDiscardNode(float dis, uint32_t id) {
    if (discard_->size() >= ef_search_*2) {
        if (discard_->top().first > dis) {
            discard_->pop();
            discard_->emplace(dis, id);
        }
    } else {
        discard_->emplace(dis, id);
    }
}

int64_t
IteratorFilterContext::GetTopID() {
    return discard_->top().second;
}

float
IteratorFilterContext::GetTopDist() {
    return discard_->top().first;
}

void
IteratorFilterContext::PopDiscard() {
    discard_->pop();
}

bool
IteratorFilterContext::Empty() {
    return discard_->empty();
}

bool
IteratorFilterContext::IsFirstUsed() {
    return is_first_used_;
}
void
IteratorFilterContext::SetOFFFirstUsed() {
    is_first_used_ = false;
}

void
IteratorFilterContext::SetPoint(InnerIdType id) {
    list_[id] = 1;
}

bool
IteratorFilterContext::CheckPoint(InnerIdType id) {
    return list_[id] == 0;
}

void
IteratorFilterContext::SetVisited(InnerIdType id) {
    visited_time_[id] += 1;
}

int64_t IteratorFilterContext::GetDiscardElementNum() {
    return discard_->size();
}

void IteratorFilterContext::PrintVisited()
{
    for (int i = 0; i < max_size_; i++) {
        if (visited_time_[i] > 0) {
            logger::info("{} visited {}", i, visited_time_[i]);
        }
    }

}

void IteratorFilterContext::SetDistance(uint32_t id, float distance)
{
    (*inner_distance_)[id] = distance;
}

float IteratorFilterContext::GetDistance(uint32_t id) 
{
    auto search = inner_distance_->find(id);
    return search == inner_distance_->end() ? -1.0 : search->second;;
}

};  // namespace vsag
