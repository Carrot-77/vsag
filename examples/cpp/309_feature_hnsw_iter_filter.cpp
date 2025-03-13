
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

#include <vsag/vsag.h>

#include <iostream>

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    // Transfer the ownership of the data (ids, vectors) to the base.
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

    /******************* Create HNSW Index *****************/
    // hnsw_build_parameters is the configuration for building an HNSW index.
    // The "dtype" specifies the data type, which supports float32 and int8.
    // The "metric_type" indicates the distance metric type (e.g., cosine, inner product, and L2).
    // The "dim" represents the dimensionality of the vectors, indicating the number of features for each data point.
    // The "hnsw" section contains parameters specific to HNSW:
    // - "max_degree": The maximum number of connections for each node in the graph.
    // - "ef_construction": The size used for nearest neighbor search during graph construction, which affects both speed and the quality of the graph.
    auto hnsw_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();

    /******************* Build HNSW Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HNSW contains: " << index->GetNumElements() << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        exit(-1);
    }

    /******************* KnnSearch For HNSW Index *****************/
    auto query_vector = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    // hnsw_search_parameters is the configuration for searching in an HNSW index.
    // The "hnsw" section contains parameters specific to the search operation:
    // - "ef_search": The size of the dynamic list used for nearest neighbor search, which influences both recall and search speed.
    auto hnsw_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);

    /******************* Prepare Filter Object *****************/
    class MyFilter : public vsag::Filter {
    public:
        bool
        CheckValid(int64_t id) const override {
            return id % 2;
        }

        float
        ValidRatio() const override {
            return 0.618f;
        }
    };
    auto filter_object = std::make_shared<MyFilter>();

    /******************* Print Search Result All topK * 3 *****************/
    auto time7 = std::chrono::steady_clock::now();
    auto knn_result0 = index->KnnSearch(query, topk * 3, hnsw_search_parameters, filter_object);
    auto time8 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration0 = time8 - time7;
    std::cout << "knn_result0: " << duration0.count() << std::endl;
    if (knn_result0.has_value()) {
        auto result = knn_result0.value();
        std::cout << "results0: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result0.error().message << std::endl;
    }

    /******************* Print Search Result All topK *****************/
    auto time9 = std::chrono::steady_clock::now();
    auto knn_result0_1 = index->KnnSearch(query, topk, hnsw_search_parameters, filter_object);
    auto time10 = std::chrono::steady_clock::now();
    duration0 = time10 - time9;
    std::cout << "knn_result0_1: " << duration0.count() << std::endl;
    if (knn_result0_1.has_value()) {
        auto result = knn_result0_1.value();
        std::cout << "results0: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result0_1.error().message << std::endl;
    }

    auto time_start = std::chrono::steady_clock::now();
    //for (int i = 0; i < 10000; i++) {
    vsag::IteratorContextPtr filter_ctx = nullptr;
    /******************* Search And Print Result1 *****************/
    auto time0 = std::chrono::steady_clock::now();
    auto knn_result =
        index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx);
    auto time1 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = time1 - time0;
    std::cout << "knn_result1: " << duration.count() << std::endl;

    if (knn_result.has_value()) {
        auto result = knn_result.value();
        std::cout << "results1: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result.error().message << std::endl;
    }

    /******************* Search And Print Result2 *****************/
    auto time3 = std::chrono::steady_clock::now();
    auto knn_result2 =
        index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx);

    auto time4 = std::chrono::steady_clock::now();
    duration = time4 - time3;
    std::cout << "knn_result2: " << duration.count() << std::endl;

    if (knn_result2.has_value()) {
        auto result = knn_result2.value();
        std::cout << "results2: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result2.error().message << std::endl;
    }

    /******************* Search And Print Result3 *****************/
    auto time5 = std::chrono::steady_clock::now();
    auto knn_result3 =
        index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx);
    
    auto time6 = std::chrono::steady_clock::now();
    duration = time6 - time5;
    std::cout << "knn_result3: " << duration.count() << std::endl;

    if (knn_result3.has_value()) {
        auto result = knn_result3.value();
        std::cout << "results3: " << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
        }
    } else {
        std::cerr << "Search Error: " << knn_result3.error().message << std::endl;
    }
    //}
    auto time_end = std::chrono::steady_clock::now();
    std::cout << "knn_result2: " << (time_start - time_end).count() << std::endl;
    //filter_ctx->PrintVisited();

    return 0;
}

// int
// main(int argc, char** argv) {
//     /******************* Prepare Base Dataset *****************/
//     int64_t num_vectors = 100000;
//     int64_t dim = 128;
//     auto ids = new int64_t[num_vectors];
//     auto vectors = new float[dim * num_vectors];

//     std::mt19937 rng;
//     rng.seed(47);
//     std::uniform_real_distribution<> distrib_real;
//     for (int64_t i = 0; i < num_vectors; ++i) {
//         ids[i] = i;
//     }
//     for (int64_t i = 0; i < dim * num_vectors; ++i) {
//         vectors[i] = distrib_real(rng);
//     }
//     auto base = vsag::Dataset::Make();
//     // Transfer the ownership of the data (ids, vectors) to the base.
//     base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

//     /******************* Create HNSW Index *****************/
//     // hnsw_build_parameters is the configuration for building an HNSW index.
//     // The "dtype" specifies the data type, which supports float32 and int8.
//     // The "metric_type" indicates the distance metric type (e.g., cosine, inner product, and L2).
//     // The "dim" represents the dimensionality of the vectors, indicating the number of features for each data point.
//     // The "hnsw" section contains parameters specific to HNSW:
//     // - "max_degree": The maximum number of connections for each node in the graph.
//     // - "ef_construction": The size used for nearest neighbor search during graph construction, which affects both speed and the quality of the graph.
//     auto hnsw_build_paramesters = R"(
//     {
//         "dtype": "float32",
//         "metric_type": "l2",
//         "dim": 128,
//         "hnsw": {
//             "max_degree": 16,
//             "ef_construction": 100
//         }
//     }
//     )";
//     auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();

//     /******************* Build HNSW Index *****************/
//     if (auto build_result = index->Build(base); build_result.has_value()) {
//         std::cout << "After Build(), Index HNSW contains: " << index->GetNumElements() << std::endl;
//     } else {
//         std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
//         exit(-1);
//     }

//     /******************* KnnSearch For HNSW Index *****************/
//     auto query_vector = new float[dim];
//     for (int64_t i = 0; i < dim; ++i) {
//         query_vector[i] = distrib_real(rng);
//     }

//     // hnsw_search_parameters is the configuration for searching in an HNSW index.
//     // The "hnsw" section contains parameters specific to the search operation:
//     // - "ef_search": The size of the dynamic list used for nearest neighbor search, which influences both recall and search speed.
//     auto hnsw_search_parameters = R"(
//     {
//         "hnsw": {
//             "ef_search": 100
//         }
//     }
//     )";
//     int64_t topk = 10;
//     auto query = vsag::Dataset::Make();
//     query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(true);

//     /******************* Prepare Filter Object *****************/
//     class MyFilter : public vsag::Filter {
//     public:
//         bool
//         CheckValid(int64_t id) const override {
//             return id % 2;
//         }

//         float
//         ValidRatio() const override {
//             return 0.618f;
//         }
//     };
//     auto filter_object = std::make_shared<MyFilter>();

//     /******************* Print Search Result All topK * 3 *****************/
//     auto time7 = std::chrono::steady_clock::now();
//     auto knn_result0 = index->KnnSearch(query, topk * 3, hnsw_search_parameters, filter_object);
//     auto time8 = std::chrono::steady_clock::now();
//     std::chrono::duration<double, std::milli> duration0 = time8 - time7;
//     std::cout << "knn_result0: " << duration0.count() << std::endl;
//     if (knn_result0.has_value()) {
//         auto result = knn_result0.value();
//         std::cout << "results0: " << std::endl;
//         for (int64_t i = 0; i < result->GetDim(); ++i) {
//             std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
//         }
//     } else {
//         std::cerr << "Search Error: " << knn_result0.error().message << std::endl;
//     }

//     /******************* Print Search Result All topK *****************/
//     auto time9 = std::chrono::steady_clock::now();
//     auto knn_result0_1 = index->KnnSearch(query, topk, hnsw_search_parameters, filter_object);
//     auto time10 = std::chrono::steady_clock::now();
//     duration0 = time10 - time9;
//     std::cout << "knn_result0_1: " << duration0.count() << std::endl;
//     if (knn_result0_1.has_value()) {
//         auto result = knn_result0_1.value();
//         std::cout << "results0: " << std::endl;
//         for (int64_t i = 0; i < result->GetDim(); ++i) {
//             std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
//         }
//     } else {
//         std::cerr << "Search Error: " << knn_result0_1.error().message << std::endl;
//     }

//     auto time_start = std::chrono::steady_clock::now();
//     for (int i = 0; i < 10000; i++) {
//       vsag::IteratorContextPtr filter_ctx = nullptr;
//       /******************* Search And Print Result1 *****************/
//       auto knn_result =
//           index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx);
//       /******************* Search And Print Result2 *****************/
//       auto knn_result2 =
//           index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx);
  
//       /******************* Search And Print Result3 *****************/
//     //   auto knn_result3 =
//     //       index->KnnSearch(query, topk, hnsw_search_parameters, filter_object, &filter_ctx);
      
//     }
//     auto time_end = std::chrono::steady_clock::now();
//     std::cout << "knn_result2: " << (time_start - time_end).count() << std::endl;
//     //filter_ctx->PrintVisited();

//     return 0;
// }