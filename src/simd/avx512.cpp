
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
#if defined(ENABLE_AVX512)
#include <immintrin.h>
#endif

#include <cmath>

#include "simd.h"

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

namespace vsag::avx512 {
float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* float_vec1 = (float*)pVect1v;
    auto* float_vec2 = (float*)pVect2v;
    size_t dim = *((size_t*)qty_ptr);
    return avx512::FP32ComputeL2Sqr(float_vec1, float_vec2, dim);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    auto* float_vec1 = (float*)pVect1v;
    auto* float_vec2 = (float*)pVect2v;
    size_t dim = *((size_t*)qty_ptr);
    return avx512::FP32ComputeIP(float_vec1, float_vec2, dim);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - avx512::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
#if defined(ENABLE_AVX512)
    __mmask32 mask = 0xFFFFFFFF;

    auto qty = *((size_t*)qty_ptr);
    const uint32_t n = (qty >> 5);
    if (n == 0) {
        return avx2::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
    }

    auto* pVect1 = (int8_t*)pVect1v;
    auto* pVect2 = (int8_t*)pVect2v;

    __m512i sum512 = _mm512_set1_epi32(0);
    for (uint32_t i = 0; i < n; ++i) {
        __m256i v1 = _mm256_maskz_loadu_epi8(mask, pVect1 + (i << 5));
        __m512i v1_512 = _mm512_cvtepi8_epi16(v1);
        __m256i v2 = _mm256_maskz_loadu_epi8(mask, pVect2 + (i << 5));
        __m512i v2_512 = _mm512_cvtepi8_epi16(v2);
        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v1_512, v2_512));
    }
    auto res = static_cast<float>(_mm512_reduce_add_epi32(sum512));
    uint64_t new_dim = qty & (0x1F);
    res += avx2::INT8InnerProduct(pVect1 + (n << 5), pVect2 + (n << 5), &new_dim);
    return res;
#else
    return avx2::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
#endif
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -avx512::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    return avx2::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
}

float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    const int n = dim / 16;
    if (n == 0) {
        return avx2::FP32ComputeIP(query, codes, dim);
    }
    // process 16 floats at a time
    __m512 sum = _mm512_setzero_ps();  // initialize to 0
    for (int i = 0; i < n; ++i) {
        __m512 a = _mm512_loadu_ps(query + i * 16);  // load 16 floats from memory
        __m512 b = _mm512_loadu_ps(codes + i * 16);  // load 16 floats from memory
        sum = _mm512_fmadd_ps(a, b, sum);            // accumulate the product
    }
    float ip = _mm512_reduce_add_ps(sum);
    ip += avx2::FP32ComputeIP(query + n * 16, codes + n * 16, dim - n * 16);
    return ip;
#else
    return avx2::FP32ComputeIP(query, codes, dim);
#endif
}

float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    const int n = dim / 16;
    if (n == 0) {
        return avx2::FP32ComputeL2Sqr(query, codes, dim);
    }
    // process 16 floats at a time
    __m512 sum = _mm512_setzero_ps();  // initialize to 0
    for (int i = 0; i < n; ++i) {
        __m512 a = _mm512_loadu_ps(query + i * 16);  // load 16 floats from memory
        __m512 b = _mm512_loadu_ps(codes + i * 16);  // load 16 floats from memory
        __m512 diff = _mm512_sub_ps(a, b);           // calculate the difference
        sum = _mm512_fmadd_ps(diff, diff, sum);      // accumulate the squared difference
    }
    float l2 = _mm512_reduce_add_ps(sum);
    l2 += avx2::FP32ComputeL2Sqr(query + n * 16, codes + n * 16, dim - n * 16);
    return l2;
#else
    return avx2::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             uint64_t dim) {
#if defined(ENABLE_AVX512)
    // Initialize the sum to 0
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m128i code_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
        __m512i codes_512 = _mm512_cvtepu8_epi32(code_values);
        __m512 code_floats = _mm512_cvtepi32_ps(codes_512);
        __m512 query_values = _mm512_loadu_ps(query + i);
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lower_bound_values = _mm512_loadu_ps(lower_bound + i);

        // Perform calculations
        __m512 scaled_codes =
            _mm512_mul_ps(_mm512_div_ps(code_floats, _mm512_set1_ps(255.0f)), diff_values);
        __m512 adjusted_codes = _mm512_add_ps(scaled_codes, lower_bound_values);
        __m512 val = _mm512_mul_ps(query_values, adjusted_codes);
        sum = _mm512_add_ps(sum, val);
    }
    // Horizontal addition
    float finalResult = _mm512_reduce_add_ps(sum);
    // Process the remaining elements recursively
    finalResult += avx2::SQ8ComputeIP(query + i, codes + i, lower_bound + i, diff + i, dim - i);
    return finalResult;
#else
    return avx2::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                uint64_t dim) {
#if defined(ENABLE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m128i code_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
        __m512 diff_values = _mm512_loadu_ps(diff + i);

        __m512i codes_512 = _mm512_cvtepu8_epi32(code_values);
        __m512 code_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes_512), _mm512_set1_ps(255.0f));
        __m512 lower_bound_values = _mm512_loadu_ps(lower_bound + i);
        __m512 query_values = _mm512_loadu_ps(query + i);

        // Perform calculations
        __m512 scaled_codes = _mm512_mul_ps(code_floats, diff_values);
        scaled_codes = _mm512_add_ps(scaled_codes, lower_bound_values);
        __m512 val = _mm512_sub_ps(query_values, scaled_codes);
        val = _mm512_mul_ps(val, val);
        sum = _mm512_add_ps(sum, val);
    }

    // Horizontal addition
    float result = _mm512_reduce_add_ps(sum);
    // Process the remaining elements
    result += avx2::SQ8ComputeL2Sqr(query + i, codes + i, lower_bound + i, diff + i, dim - i);
    return result;
#else
    return avx2::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        // Load data into registers
        __m128i code1_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes1 + i));
        __m128i code2_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes2 + i));
        __m512i codes1_512 = _mm512_cvtepu8_epi32(code1_values);
        __m512i codes2_512 = _mm512_cvtepu8_epi32(code2_values);
        __m512 code1_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes1_512), _mm512_set1_ps(255.0f));
        __m512 code2_floats = _mm512_div_ps(_mm512_cvtepi32_ps(codes2_512), _mm512_set1_ps(255.0f));
        __m512 diff_values = _mm512_loadu_ps(diff + i);
        __m512 lower_bound_values = _mm512_loadu_ps(lower_bound + i);

        // Perform calculations
        __m512 scaled_codes1 = _mm512_fmadd_ps(code1_floats, diff_values, lower_bound_values);
        __m512 scaled_codes2 = _mm512_fmadd_ps(code2_floats, diff_values, lower_bound_values);
        __m512 val = _mm512_mul_ps(scaled_codes1, scaled_codes2);
        sum = _mm512_add_ps(sum, val);
    }
    // Horizontal addition
    float result = _mm512_reduce_add_ps(sum);
    // Process the remaining elements
    result += avx2::SQ8ComputeCodesIP(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
    return result;
#else
    return avx2::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     uint64_t dim) {
#if defined(ENABLE_AVX512)
    __m512 sum = _mm512_setzero_ps();
    uint64_t i = 0;

    for (; i + 15 < dim; i += 16) {
        __m128i code1_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes1 + i));
        __m128i code2_values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes2 + i));
        __m512 diff_values = _mm512_loadu_ps(diff + i);

        __m512i codes1_512 = _mm512_cvtepu8_epi32(code1_values);
        __m512i codes2_512 = _mm512_cvtepu8_epi32(code2_values);
        __m512 sub = _mm512_cvtepi32_ps(_mm512_sub_epi32(codes1_512, codes2_512));
        __m512 scaled = _mm512_mul_ps(sub, _mm512_set1_ps(1.0 / 255.0f));
        __m512 val = _mm512_mul_ps(scaled, diff_values);
        sum = _mm512_fmadd_ps(val, val, sum);
    }

    // Horizontal addition
    float result = _mm512_reduce_add_ps(sum);
    // Process the remaining elements
    result +=
        avx2::SQ8ComputeCodesL2Sqr(codes1 + i, codes2 + i, lower_bound + i, diff + i, dim - i);
    return result;
#else
    return avx2::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
#endif
}

float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lower_bound,
             const float* diff,
             uint64_t dim) {
    return avx2::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const float* lower_bound,
                const float* diff,
                uint64_t dim) {
    return avx2::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lower_bound,
                  const float* diff,
                  uint64_t dim) {
    return avx2::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* codes1,
                     const uint8_t* codes2,
                     const float* lower_bound,
                     const float* diff,
                     uint64_t dim) {
    return avx2::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim == 0) {
        return 0;
    }
    alignas(512) int16_t temp[32];
    int32_t result = 0;
    uint64_t d = 0;
    __m512i sum = _mm512_setzero_si512();
    __m512i mask = _mm512_set1_epi8(0xf);
    for (; d + 127 < dim; d += 128) {
        auto xx = _mm512_loadu_si512((__m512i*)(codes1 + (d >> 1)));
        auto yy = _mm512_loadu_si512((__m512i*)(codes2 + (d >> 1)));
        auto xx1 = _mm512_and_si512(xx, mask);                        // 64 * 8bits
        auto xx2 = _mm512_and_si512(_mm512_srli_epi16(xx, 4), mask);  // 64 * 8bits
        auto yy1 = _mm512_and_si512(yy, mask);
        auto yy2 = _mm512_and_si512(_mm512_srli_epi16(yy, 4), mask);

        sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(xx1, yy1));
        sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(xx2, yy2));
    }
    _mm512_store_si512((__m512i*)temp, sum);
    for (int i = 0; i < 32; ++i) {
        result += temp[i];
    }
    result += avx2::SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    return result;
#else
    return avx2::SQ4UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

float
SQ8UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
#if defined(ENABLE_AVX512)
    if (dim == 0) {
        return 0.0f;
    }

    alignas(64) int32_t temp[16];
    int32_t result = 0;
    uint64_t d = 0;
    __m512i sum = _mm512_setzero_si512();
    __m512i mask = _mm512_set1_epi16(0xff);
    for (; d + 63 < dim; d += 64) {
        auto xx = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes1 + d));
        auto yy = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(codes2 + d));

        auto xx1 = _mm512_and_si512(xx, mask);
        auto xx2 = _mm512_srli_epi16(xx, 8);
        auto yy1 = _mm512_and_si512(yy, mask);
        auto yy2 = _mm512_srli_epi16(yy, 8);

        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(xx1, yy1));
        sum = _mm512_add_epi32(sum, _mm512_madd_epi16(xx2, yy2));
    }
    _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum);
    for (int i = 0; i < 16; ++i) {
        result += temp[i];
    }
    result += static_cast<int32_t>(avx2::SQ8UniformComputeCodesIP(codes1 + d, codes2 + d, dim - d));
    return static_cast<float>(result);
#else
    return avx2::SQ8UniformComputeCodesIP(codes1, codes2, dim);
#endif
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
#if defined(ENABLE_AVX2)
    if (dim == 0) {
        return;
    }
    if (scalar == 0) {
        scalar = 1.0f;  // TODO(LHT): logger?
    }
    int i = 0;
    __m512 scalarVec = _mm512_set1_ps(scalar);
    for (; i + 15 < dim; i += 16) {
        __m512 vec = _mm512_loadu_ps(from + i);
        vec = _mm512_div_ps(vec, scalarVec);
        _mm512_storeu_ps(to + i, vec);
    }
    avx2::DivScalar(from + i, to + i, dim - i, scalar);
#else
    avx2::DivScalar(from, to, dim, scalar);
#endif
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(FP32ComputeIP(from, from, dim));
    avx512::DivScalar(from, to, dim, norm);
    return norm;
}

}  // namespace vsag::avx512
