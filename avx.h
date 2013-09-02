
// Vector of 8x int32 samples:
typedef __m256i vec8_i32;
// Vector of 4x int32 samples:
typedef __m128i vec4_i32;
// Vector of float64 samples:
typedef __m256d vec4_d64;

// Vector of decibal (dB) values:
typedef __m256d vec4_dB;
// Vector of decibal relative to full scale (dBFS) values:
typedef __m256d vec4_dBFS;

// Vector of linear scalar values (likely converted from dB):
typedef __m256d vec4_scalar;
// Vector of msec values (for input):
typedef __m256d vec4_msec;

#ifndef __AVX2__
// Emulations of AVX2 instrincs for AVX1-only processors, using SSE2 intrinsics.

static __forceinline __m256i mm256_srli_epi64(const __m256i x, const int s)
{
    // Split into two 128-bit registers and perform the operation on each:
    const auto v0 = _mm256_extractf128_si256(x, 0);
    const auto v1 = _mm256_extractf128_si256(x, 1);
    const auto r0 = _mm_srli_epi64(v0, s);
    const auto r1 = _mm_srli_epi64(v1, s);
    const auto r = _mm256_setr_m128i(r0, r1);
    return r;
}

static __forceinline __m256i mm256_sub_epi64(const __m256i a, const __m256i b)
{
    // Split into two 128-bit registers and perform the operation on each:
    const auto a0 = _mm256_extractf128_si256(a, 0);
    const auto a1 = _mm256_extractf128_si256(a, 1);
    const auto b0 = _mm256_extractf128_si256(b, 0);
    const auto b1 = _mm256_extractf128_si256(b, 1);
    const auto r0 = _mm_sub_epi64(a0, b0);
    const auto r1 = _mm_sub_epi64(a1, b1);
    const auto r = _mm256_setr_m128i(r0, r1);
    return r;
}

#else
// Use the native AVX2 intrinsics:

#  define mm256_srli_epi64 _mm256_srli_epi64
#  define mm256_sub_epi64 _mm256_sub_epi64
#endif

// There appears to be a lack of abs(x) for AVX and doubles.
static __forceinline __m256d mm256_abs_pd(const __m256d x)
{
    return _mm256_and_pd(x, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)));
}

// DC offset used to avoid denormal floats:
static const vec4_d64 DC_OFFSET = _mm256_set1_pd(1.0E-25);
static const vec4_d64 DB_2_LOG = _mm256_set1_pd(0.11512925464970228420089957273422);

// log10(x) implementation - simple/stupid for now.
// TODO(jsd): implement log10(x) with AVX intrinsics, hopefully for performance improvement. OK to sacrifice 
static __forceinline vec4_d64 mm256_log10_pd(const vec4_d64 &in)
{
    __declspec(align(32)) double ind[4];
    __declspec(align(32)) double outd[4];
    _mm256_stream_pd(ind, in);
    for (int i = 0; i < 4; ++i)
        outd[i] = log10(ind[i]);
    return _mm256_load_pd(outd);
}

static __forceinline vec4_d64 mm256_exp_pd(const vec4_d64 &in)
{
    __declspec(align(32)) double ind[4];
    __declspec(align(32)) double outd[4];
    _mm256_stream_pd(ind, in);
    for (int i = 0; i < 4; ++i)
        outd[i] = exp(ind[i]);
    return _mm256_load_pd(outd);
}

// Calculate dBFS values for linear samples:
static __forceinline vec4_d64 scalar_to_dBFS(const vec4_d64 &in)
{
    vec4_d64 l10 = mm256_log10_pd(mm256_abs_pd(in));
    return _mm256_mul_pd(l10, _mm256_set1_pd(20.0));
}

// Calculate dBFS values for linear samples + DC_OFFSET:
static __forceinline vec4_d64 scalar_to_dBFS_offs(const vec4_d64 &in)
{
    vec4_d64 adj = _mm256_add_pd(mm256_abs_pd(in), DC_OFFSET);
    vec4_d64 l10 = mm256_log10_pd(adj);
    return _mm256_mul_pd(l10, _mm256_set1_pd(20.0));
}

static __forceinline vec4_scalar dB_to_scalar(const vec4_dB &in)
{
    vec4_d64 t = _mm256_mul_pd(in, DB_2_LOG);
    return mm256_exp_pd(t);
}

static __forceinline vec4_d64 mm256_if_then_else(const vec4_d64 &mask, const vec4_d64 &a, const vec4_d64 &b)
{
    return _mm256_or_pd(_mm256_andnot_pd(mask, b), _mm256_and_pd(a, mask));
}
