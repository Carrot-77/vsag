
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

#pragma once
#include <queue>

#include "../src/utils/visited_list.h"

namespace vsag {

class DiscardNode {
public:
  virtual int AddDiscardNode(float dis, int64_t label) = 0;
  virtual int64_t GetTopID() = 0;
  virtual float GetTopDist() = 0;
  virtual void PopDiscard() = 0;
  virtual bool Empty() = 0;

};

using DiscardPtr = std::shared_ptr<DiscardNode>;
};