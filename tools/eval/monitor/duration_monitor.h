
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

#include <unistd.h>

#include <chrono>
#include <fstream>

#include "monitor.h"

namespace vsag::eval {

class DurationMonitor : public Monitor {
public:
    explicit DurationMonitor();

    ~DurationMonitor() override = default;

    void
    Start() override;

    void
    Stop() override;

    JsonType
    GetResult() override;

private:
    double duration_{0};

    using Clock = std::chrono::high_resolution_clock;
    decltype(Clock::now()) cur_time_{};
};

}  // namespace vsag::eval