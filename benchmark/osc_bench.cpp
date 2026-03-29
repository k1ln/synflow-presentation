// Synflow C++ DSP Benchmark v3
// Uses volatile sink + DoNotOptimize to prevent dead-code elimination.
// Voice counts match JS benchmark (100..50000) so results are directly comparable.
//
// clang++ -O3 -std=c++17 -march=native -o osc_bench osc_bench.cpp
// ./osc_bench 2>/dev/null > cpp_results.json
#include <arm_neon.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static constexpr int    BS   = 128;
static constexpr int    TS   = 512;
static constexpr double SR   = 44100.0;
static constexpr double FREQ = 440.0;
static constexpr double BUD  = (BS / SR) * 1000.0;   // 2.9025 ms

// Same voice counts as JS benchmark for apples-to-apples comparison
static const std::array<int,9> VC = {100,250,500,1000,2500,5000,10000,20000,50000};

using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t0){
    return std::chrono::duration<double,std::milli>(clk::now()-t0).count();
}

// Prevent dead-code elimination of output buffer
static volatile float g_sink = 0.0f;
static void do_not_optimize(const std::vector<float>& b){
    g_sink = b[0] + b[BS-1];
}

// Adaptive iters: target ~0.5 s per row after warmup
static int niters(int v){
    // On M4 with -O3: sinf ≈ 2.5 ns/sample, so V*BS samples ≈ V*BS*2.5 ns per block
    long long est_ns = (long long)v * BS * 3;  // conservative 3 ns/sample
    long long n = (long long)500000000LL / std::max(1LL, est_ns);
    return (int)std::max(200LL, std::min(5000LL, n));
}

// ── 1. scalar sinf ──────────────────────────────────────────────────────────
static double bench_sinf(int V, int N){
    std::vector<float> b(BS, 0.0f);
    float ph = 0.0f;
    const float inc = (float)(2.0 * M_PI * FREQ / SR);
    for(int w=0; w<300; ++w){
        for(int v=0; v<V; ++v)
            for(int i=0; i<BS; ++i){ b[i] = sinf(ph); ph += inc; }
        do_not_optimize(b);
    }
    auto t = clk::now();
    for(int n=0; n<N; ++n){
        for(int v=0; v<V; ++v)
            for(int i=0; i<BS; ++i){ b[i] = sinf(ph); ph += inc; }
        do_not_optimize(b);
    }
    return ms_since(t) / N;
}

// ── 2. 512-entry wavetable + integer phase ───────────────────────────────────
static double bench_table(int V, int N){
    static float T[TS]; static bool ok = false;
    if(!ok){ for(int i=0;i<TS;++i) T[i]=sinf(2.0f*(float)M_PI*i/TS); ok=true; }
    std::vector<float> b(BS, 0.0f);
    uint32_t ph = 0;
    const uint32_t inc = (uint32_t)(FREQ/SR * TS + 0.5);
    for(int w=0; w<300; ++w){
        for(int v=0; v<V; ++v)
            for(int i=0; i<BS; ++i){ b[i]=T[ph&(TS-1)]; ph=(ph+inc)&0xFFFFFF; }
        do_not_optimize(b);
    }
    auto t = clk::now();
    for(int n=0; n<N; ++n){
        for(int v=0; v<V; ++v)
            for(int i=0; i<BS; ++i){ b[i]=T[ph&(TS-1)]; ph=(ph+inc)&0xFFFFFF; }
        do_not_optimize(b);
    }
    return ms_since(t) / N;
}

// ── 3. NEON 4-wide polynomial sine (arm64) ───────────────────────────────────
static inline float32x4_t nsin4(float32x4_t x){
    const float32x4_t IPI = vdupq_n_f32(1.0f/(float)M_PI);
    const float32x4_t PI  = vdupq_n_f32((float)M_PI);
    float32x4_t n  = vrndnq_f32(vmulq_f32(x, IPI));
    float32x4_t xr = vmlsq_f32(x, n, PI);
    uint32x4_t sgn = vshlq_n_u32(
        vandq_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)), vdupq_n_u32(1)), 31);
    float32x4_t x2 = vmulq_f32(xr, xr), p;
    p = vmlaq_f32(vdupq_n_f32( 8.33174401e-03f), vdupq_n_f32(-1.95152968e-04f), x2);
    p = vmlaq_f32(vdupq_n_f32(-1.66665673e-01f), p, x2);
    p = vmlaq_f32(vdupq_n_f32( 9.99999702e-01f), p, x2);
    return vreinterpretq_f32_u32(
        veorq_u32(vreinterpretq_u32_f32(vmulq_f32(p, xr)), sgn));
}
static double bench_neon(int V, int N){
    std::vector<float> b(BS, 0.0f);
    float ph = 0.0f;
    const float inc  = (float)(2.0 * M_PI * FREQ / SR);
    const float inc4 = inc * 4.0f;
    for(int w=0; w<300; ++w){
        for(int v=0; v<V; ++v){
            float p = ph;
            for(int j=0; j<BS/4; ++j){
                float32x4_t q = {p, p+inc, p+inc*2, p+inc*3};
                vst1q_f32(&b[j*4], nsin4(q));
                p += inc4;
            }
        }
        do_not_optimize(b);
    }
    auto t = clk::now();
    for(int n=0; n<N; ++n){
        for(int v=0; v<V; ++v){
            float p = ph;
            for(int j=0; j<BS/4; ++j){
                float32x4_t q = {p, p+inc, p+inc*2, p+inc*3};
                vst1q_f32(&b[j*4], nsin4(q));
                p += inc4;
            }
        }
        do_not_optimize(b);
    }
    return ms_since(t) / N;
}

// ── helpers ───────────────────────────────────────────────────────────────────
static std::string jrow(int v, double ms){
    std::ostringstream s;
    s << std::fixed << std::setprecision(6);
    s << "    {\"voices\":" << v << ",\"ms\":" << ms
      << ",\"over\":" << (ms > BUD ? "true" : "false") << "}";
    return s.str();
}
static std::string joinv(const std::vector<std::string>& v){
    std::string s = "[\n";
    for(size_t i=0; i<v.size(); ++i){ s += v[i]; if(i+1<v.size()) s+=","; s+="\n"; }
    return s + "  ]";
}
static std::string first_over(const std::vector<std::string>& rows){
    for(auto& r : rows)
        if(r.find("\"over\":true") != std::string::npos){
            auto p = r.find("\"voices\":")+9, q = r.find(",",p);
            return r.substr(p,q-p) + " voices";
        }
    return "none at tested counts";
}

int main(){
    std::cerr << "\n── C++ DSP v3 (clang++ -O3 -march=native, arm64 NEON) ──────\n";
    std::cerr << "   Budget: " << std::fixed << std::setprecision(4) << BUD
              << " ms  (" << BS << " samples @ " << (int)SR << " Hz)\n";
    std::cerr << "   NOTE: do_not_optimize() prevents dead-code elimination.\n";
    std::cerr << std::left
              << std::setw(9)  << "voices"
              << std::setw(14) << "sinf"
              << std::setw(14) << "wavetable"
              << "NEON SIMD\n"
              << std::string(50,'-') << "\n";

    std::vector<std::string> sr, tr, nr;
    for(int v : VC){
        int it = niters(v);
        double sm = bench_sinf(v, it);
        double tm = bench_table(v, it);
        double nm = bench_neon(v, it);
        auto fl  = [](double m)->const char*{ return m>BUD*2?"[X]":m>BUD?"[!]":"[OK]"; };
        auto fmt = [](double m)->std::string{
            std::ostringstream s;
            if(m < 0.0001)      s << std::fixed << std::setprecision(4) << (m*1e6)  << "ns";
            else if(m < 0.1)    s << std::fixed << std::setprecision(4) << (m*1000) << "us";
            else                s << std::fixed << std::setprecision(4) <<  m        << "ms";
            return s.str();
        };
        std::cerr << std::left
                  << std::setw(9)  << v
                  << std::setw(14) << (fmt(sm)+" "+fl(sm))
                  << std::setw(14) << (fmt(tm)+" "+fl(tm))
                  <<               (fmt(nm)+" "+fl(nm)) << "\n";
        sr.push_back(jrow(v,sm));
        tr.push_back(jrow(v,tm));
        nr.push_back(jrow(v,nm));
    }
    std::cerr << "\nFirst underrun:\n"
              << "  sinf      : " << first_over(sr) << "\n"
              << "  wavetable : " << first_over(tr) << "\n"
              << "  NEON SIMD : " << first_over(nr) << "\n\n";

    std::cout << "{\n"
              << "  \"generator\":\"osc_bench.cpp v3\",\n"
              << "  \"compiler\":\"clang++ -O3 -march=native\",\n"
              << "  \"arch\":\"arm64\",\n"
              << "  \"bufferSize\":" << BS << ",\n"
              << "  \"sampleRate\":" << (int)SR << ",\n"
              << "  \"budgetMs\":"   << std::fixed << std::setprecision(6) << BUD << ",\n"
              << "  \"voiceCounts\":[100,250,500,1000,2500,5000,10000,20000,50000],\n"
              << "  \"results\":{\n"
              << "    \"sinf\":      " << joinv(sr) << ",\n"
              << "    \"wavetable\": " << joinv(tr) << ",\n"
              << "    \"neon\":      " << joinv(nr) << "\n"
              << "  }\n}\n";
}
