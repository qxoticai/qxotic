/* jam_metal.mm - Apple GPU backend (chosen by JAM_ISA_AUTO on Apple silicon). A different executor from the CPU
 * row-range kernels: jam_mm routes dense weights and n >= 16 quant prefill here before the pool
 * (small-n quant decode stays on the CPU GEMVs - see jam.c). Two kernel families:
 *  - block-quant weights (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, canonical or JAM_PACKED): one simdgroup-MMA
 *    template, llama.cpp's kernel_mul_mm structure (ggml-metal, MIT). A format only contributes a
 *    loader that decodes 16 weights to half; accumulation is float.
 *  - dense weights and MXFP4: a scalar tiled kernel on the exact F32 activations (each thread
 *    decodes a weight block once and reuses it across TN columns).
 * MSL is compiled at runtime (no .metallib). Output is token-major C[j*ldc+i].
 *
 * Zero-copy over unified memory: W/A/C are borrowed per call through page-rounded
 * newBufferWithBytesNoCopy views (nil deallocator - the caller owns every byte) and released after
 * the synchronous wait. No uploads, no result copies, no persistent state; strides (ldw/ldb/ldc)
 * ride in mm_params, so strided views are consumed directly instead of packed. APPLE-only TU. */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "jam_internal.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <mach/mach_time.h>
#include <os/lock.h>

#define JAM_MTN 8   /* output columns per GPU thread - the one tunable; injected into the MSL as TN */

static const char* JAM_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant float MXFP4_LUT[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };  /* ggml kvalues_mxfp4 */

struct mm_params { int m, n, k, ldw, ldb, ldc; };

/* ---- scalar tiled kernels: dense weights and MXFP4 ---------------------------------------- */

/* thread (gid.x, gid.y) owns weight row i = gid.y and the TN columns [j0, j0+TN), j0 = gid.x*TN. */
#define JAM_TILE_PROLOGUE \
    int m=p.m, n=p.n, k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc; \
    int i=int(gid.y), j0=int(gid.x)*TN; \
    if (i>=m || j0>=n) return; \
    float acc[TN]; for (int c=0;c<TN;++c) acc[c]=0.0f;
#define JAM_TILE_EPILOGUE \
    for (int c=0;c<TN;++c){ int j=j0+c; if (j<n) C[(size_t)j*ldc+i]=acc[c]; }

/* MXFP4 block = { uchar e; uchar qs[16] } = 17 B. value = (0.5·2^(e-127))·LUT[nibble]; lo -> e, hi -> e+16. */
kernel void mxfp4_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int nb=k/32;
    device const uchar* wrow = A + (size_t)i*(ldw/32)*17;
    for (int b=0;b<nb;++b){
        device const uchar* blk = wrow + (size_t)b*17;
        float dh = 0.5f * exp2(float(blk[0]) - 127.0f);
        device const uchar* qs = blk+1;
        float wq[32]; for (int e=0;e<16;++e){ wq[e]=dh*MXFP4_LUT[qs[e]&0xF]; wq[e+16]=dh*MXFP4_LUT[qs[e]>>4]; }
        for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
            device const float* bb = B + (size_t)j*ldb + b*32;
            float s=0.0f; for (int e=0;e<32;++e) s += wq[e]*bb[e];
            acc[c] += s;
        }
    }
    JAM_TILE_EPILOGUE
}


/* Dense weights (F32/F16/BF16) @ F32. Same TN-column tiling for a uniform grid; the weight row is streamed
 * per column (too big to cache), so the win here is output grouping, not dequant amortization. */
kernel void f32_mm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    device const float* arow = A + (size_t)i*ldw;
    for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
        device const float* bb = B + (size_t)j*ldb;
        float s=0.0f; for (int t=0;t<k;++t) s += arow[t]*bb[t];
        acc[c]=s;
    }
    JAM_TILE_EPILOGUE
}

kernel void f16_mm(device const half* A [[buffer(0)]], device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    device const half* arow = A + (size_t)i*ldw;
    for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
        device const float* bb = B + (size_t)j*ldb;
        float s=0.0f; for (int t=0;t<k;++t) s += float(arow[t])*bb[t];
        acc[c]=s;
    }
    JAM_TILE_EPILOGUE
}

kernel void bf16_mm(device const ushort* A [[buffer(0)]], device const float* B [[buffer(1)]],
                    device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    device const ushort* arow = A + (size_t)i*ldw;
    for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
        device const float* bb = B + (size_t)j*ldb;
        float s=0.0f; for (int t=0;t<k;++t){ uint bits=uint(arow[t])<<16; s += as_type<float>(bits)*bb[t]; }
        acc[c]=s;
    }
    JAM_TILE_EPILOGUE
}

/* ---- block-quant prefill: simdgroup MMA ---------------------------------------------------- *
 * A threadgroup of 128 threads owns a 64-row x 32-token output tile. Per 32-wide K step it stages
 * the weight slice (64x32) and the activation slice (32x32) as half in threadgroup memory, in 8x8
 * blocks, and each of its 4 simdgroups multiplies a 32-row x 16-token sub-tile into float
 * accumulators. Thread lid decodes weight row lid/2, values [ah, ah+16) of each 32-block (ah =
 * 16*(lid%2)) through the format's loader, and converts 8 activations of token lid/4.
 *
 * Rows and tokens past the edge of the matrix are clamped to the last real one: they compute
 * finite values that are never stored, so one kernel serves aligned and edge tiles alike. */

/* Q4_0 stores element e in the low nibble and e+16 in the high nibble. Blocks are 18 bytes, hence
 * every block and its q payload remain ushort-aligned. Match llama.cpp's dequantize_q4_0 trick:
 * extract two nibbles at once with 16-bit masks and fold their shifts into d1/d2, avoiding a byte
 * vector's mask/shift/unpack sequence. The float intermediates are converted once to the MMA half. */
static half4 q4_0_vec4(device const uchar* q, int off, bool high, half d) {
    device const ushort* words=(device const ushort*)(q+off);
    float d1=high ? float(d)/16.0f : float(d);
    float d2=d1/256.0f, md=-8.0f*float(d);
    ushort mask0=high ? 0x00f0 : 0x000f, mask1=mask0<<8;
    ushort a=words[0], b=words[1];
    return half4(float4(d1*float(a&mask0)+md, d2*float(a&mask1)+md,
                        d1*float(b&mask0)+md, d2*float(b&mask1)+md));
}

/* The 6-bit (scale, min) of sub-block s of a Q4_K/Q5_K super-block (ggml packing); sp = scales[12]. */
static float2 kq_scale_min(device const uchar* sp, int s) {
    if (s<4) return float2(sp[s]&63u, sp[s+4]&63u);
    return float2((sp[s+4]&0xFu)|((sp[s-4]>>6)<<4), (sp[s+4]>>4)|((sp[s]>>6)<<4));
}

/* Canonical wire layouts: a loader holds its row's first block; load(b, ah, aq) decodes values
 * [ah, ah+16) of 32-block b. */

/* Q8_0 block = { half d; char qs[32] } = 34 B. */
struct q8_0_rows {
    device const uchar* row;
    q8_0_rows(device const uchar* A, constant mm_params& p, int r) : row(A+(size_t)r*(p.ldw/32)*34) {}
    void load(int b, int ah, thread half4* aq) const {
        device const uchar* blk=row+(size_t)b*34;
        half d=*(device const half*)blk;
        device const packed_char4* q=(device const packed_char4*)(blk+2+ah);
        for (int t=0;t<4;++t) aq[t]=d*half4(char4(q[t]));
    }
};

/* Q4_0 block = { half d; uchar qs[16] } = 18 B. value = d·(nibble-8); lo nibble -> e, hi -> e+16. */
struct q4_0_rows {
    device const uchar* row;
    q4_0_rows(device const uchar* A, constant mm_params& p, int r) : row(A+(size_t)r*(p.ldw/32)*18) {}
    void load(int b, int ah, thread half4* aq) const {
        device const uchar* blk=row+(size_t)b*18;
        half d=*(device const half*)blk;
        for (int t=0;t<4;++t) aq[t]=q4_0_vec4(blk+2,4*t,ah!=0,d);
    }
};

/* Q4_K super-block = { half d; half dmin; uchar scales[12]; uchar qs[128] } = 144 B, 8 sub-blocks
 * of 32; sub-block s reads nibbles qs[(s/2)*32 + e] (low if s even, else high). */
struct q4_k_rows {
    device const uchar* row;
    q4_k_rows(device const uchar* A, constant mm_params& p, int r) : row(A+(size_t)r*(p.ldw/256)*144) {}
    void load(int b, int ah, thread half4* aq) const {
        device const uchar* w=row+(size_t)(b>>3)*144;
        int s=b&7;
        float2 sm=kq_scale_min(w+4,s)*float2(*(device const half*)w, *(device const half*)(w+2));
        device const packed_uchar4* q=(device const packed_uchar4*)(w+16+(s>>1)*32+ah);
        for (int t=0;t<4;++t) {
            uchar4 raw=uchar4(q[t]);
            aq[t]=half4(float4((s&1) ? raw>>4 : raw&0xF)*sm.x-sm.y);
        }
    }
};

/* Q5_K = { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128] } = 176 B. The value is
 * the Q4_K nibble | (bit s of qh[e]) << 4. */
struct q5_k_rows {
    device const uchar* row;
    q5_k_rows(device const uchar* A, constant mm_params& p, int r) : row(A+(size_t)r*(p.ldw/256)*176) {}
    void load(int b, int ah, thread half4* aq) const {
        device const uchar* w=row+(size_t)(b>>3)*176;
        int s=b&7;
        float2 sm=kq_scale_min(w+4,s)*float2(*(device const half*)w, *(device const half*)(w+2));
        device const packed_uchar4* q=(device const packed_uchar4*)(w+48+(s>>1)*32+ah);
        device const packed_uchar4* qh=(device const packed_uchar4*)(w+16+ah);
        for (int t=0;t<4;++t) {
            uchar4 raw=uchar4(q[t]), hb=uchar4(qh[t]);
            uchar4 v=((s&1) ? raw>>4 : raw&0xF) | (((hb>>uchar(s))&1)<<4);
            aq[t]=half4(float4(v)*sm.x-sm.y);
        }
    }
};

/* Q6_K = { uchar ql[128]; uchar qh[64]; char scales[16]; half d } = 210 B. value = d·sc·(q-32), q
 * 6-bit (ql nibble | qh 2-bit << 4), an int8 scale per 16 values: exactly one per loader call. */
struct q6_k_rows {
    device const uchar* row;
    q6_k_rows(device const uchar* A, constant mm_params& p, int r) : row(A+(size_t)r*(p.ldw/256)*210) {}
    void load(int b, int ah, thread half4* aq) const {
        device const uchar* w=row+(size_t)(b>>3)*210;
        int h=(b&7)>>2, g=b&3;
        float scale=float(*(device const half*)(w+208))*float(((device const char*)(w+192))[h*8+g*2+(ah>>4)]);
        device const packed_uchar4* ql=(device const packed_uchar4*)(w+h*64+(g&1)*32+ah);
        device const packed_uchar4* qh=(device const packed_uchar4*)(w+128+h*32+ah);
        for (int t=0;t<4;++t) {
            uchar4 lo=uchar4(ql[t]), hi=uchar4(qh[t]);
            uchar4 v=(g>=2 ? lo>>4 : lo&0xF) | (((hi>>uchar(2*g))&3)<<4);
            aq[t]=half4((float4(v)-32.0f)*scale);
        }
    }
};

/* Packed layouts (jam.h JAM_PACK_ABI 1): caller-packed 4-row groups, the same single copy the CPU
 * decode GEMVs read. A loader holds its row's group and lane (row % 4). The expanded int8 layouts
 * make these cheaper than the canonical k-quant loaders: no bit surgery, flat f32 scales. */

/* Q4_0 group = [nb][4 rows][16 B nibbles], then [nb][4] f32 scales: 80 B per block. */
struct q4_0_packed {
    device const uchar* g; int nb, lane;
    q4_0_packed(device const uchar* A, constant mm_params& p, int r)
        : g(A+(size_t)(r>>2)*((size_t)(p.k/32)*80)), nb(p.k/32), lane(r&3) {}
    void load(int b, int ah, thread half4* aq) const {
        device const packed_uchar4* q=(device const packed_uchar4*)(g+(size_t)b*64+lane*16);
        float s=((device const float*)(g+(size_t)nb*64))[b*4+lane];
        for (int t=0;t<4;++t) {
            uchar4 raw=uchar4(q[t]);
            aq[t]=half4((float4(ah ? raw>>4 : raw&0xF)-8.0f)*s);
        }
    }
};

/* Q4_K group = [nb][4][16 B nibbles], [nb][8] u8 (scale, min) per lane, then per super-block 8
 * f32 (4 scale multipliers, 4 min multipliers). */
struct q4_k_packed {
    device const uchar* g; int nb, lane;
    q4_k_packed(device const uchar* A, constant mm_params& p, int r)
        : g(A+(size_t)(r>>2)*((size_t)(p.k/32)*72+(size_t)(p.k/256)*32)), nb(p.k/32), lane(r&3) {}
    void load(int b, int ah, thread half4* aq) const {
        device const packed_uchar4* q=(device const packed_uchar4*)(g+(size_t)b*64+lane*16);
        device const uchar* sm=g+(size_t)nb*64+b*8;
        device const float* dd=(device const float*)(g+(size_t)nb*72)+(b>>3)*8;
        float fs=dd[lane]*float(sm[lane]), fm=dd[4+lane]*float(sm[4+lane]);
        for (int t=0;t<4;++t) {
            uchar4 raw=uchar4(q[t]);
            aq[t]=half4(float4(ah ? raw>>4 : raw&0xF)*fs-fm);
        }
    }
};

/* Q5_K group = [nb][4][32 int8 values], [nb][8] u8 (scale, min), then 8 f32 per super-block. */
struct q5_k_packed {
    device const uchar* g; int nb, lane;
    q5_k_packed(device const uchar* A, constant mm_params& p, int r)
        : g(A+(size_t)(r>>2)*((size_t)(p.k/32)*136+(size_t)(p.k/256)*32)), nb(p.k/32), lane(r&3) {}
    void load(int b, int ah, thread half4* aq) const {
        device const packed_char4* q=(device const packed_char4*)(g+(size_t)b*128+lane*32+ah);
        device const uchar* sm=g+(size_t)nb*128+b*8;
        device const float* dd=(device const float*)(g+(size_t)nb*136)+(b>>3)*8;
        float fs=dd[lane]*float(sm[lane]), fm=dd[4+lane]*float(sm[4+lane]);
        for (int t=0;t<4;++t) aq[t]=half4(float4(char4(q[t]))*fs-fm);
    }
};

/* Q6_K group = [nb][4][32 int8 values], [nb][2 halves][4] int8 scales, then 4 f32 per super-block. */
struct q6_k_packed {
    device const uchar* g; int nb, lane;
    q6_k_packed(device const uchar* A, constant mm_params& p, int r)
        : g(A+(size_t)(r>>2)*((size_t)(p.k/32)*136+(size_t)(p.k/256)*16)), nb(p.k/32), lane(r&3) {}
    void load(int b, int ah, thread half4* aq) const {
        device const packed_char4* q=(device const packed_char4*)(g+(size_t)b*128+lane*32+ah);
        device const char* sc=(device const char*)(g+(size_t)nb*128);
        device const float* dd=(device const float*)(g+(size_t)nb*136);
        float s=dd[(b>>3)*4+lane]*float(sc[b*8+(ah>>4)*4+lane]);
        for (int t=0;t<4;++t) aq[t]=half4(float4(char4(q[t]))*s);
    }
};

/* Threadgroup memory (8 KiB, set by the host): sa = 64 rows x 32 k (4 KiB), sb = 32 tokens x 32 k
 * (2 KiB), both as 8x8 blocks; an edge tile reuses all of it as its 32x64 float output. */
template <typename W>
kernel void mma(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                threadgroup char* shmem [[threadgroup(0)]],
                uint2 tg [[threadgroup_position_in_grid]],
                ushort lid [[thread_index_in_threadgroup]],
                ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half* sa=(threadgroup half*)shmem;          /* block (k/8, row/8) = [k%8][row%8] */
    threadgroup half* sb=(threadgroup half*)(shmem+4096);   /* block (k/8, token/8) = [token%8][k%8] */
    const int r0=int(tg.y)*64, t0=int(tg.x)*32;
    const short row=lid/2, ah=16*(lid%2), tok=lid/4, kb=8*(lid%4);
    const W w(A,p,min(r0+row,p.m-1));
    device const float* y=B+(size_t)min(t0+tok,p.n-1)*p.ldb+kb;
    threadgroup half4* yb=(threadgroup half4*)(sb+64*(4*(kb/8)+tok/8)+8*(tok%8));
    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];   /* mc[i] = tokens 8*(i/4) x rows 8*(i%4) of this simdgroup's sub-tile */
    for (short i=0;i<8;++i) mc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);

    for (int b=0, nb=p.k/32; b<nb; ++b, y+=32) {
        half4 aq[4];
        w.load(b,ah,aq);   /* the global loads issue before the wait on the previous step's readers */
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (short i=0;i<16;++i)
            sa[64*(8*((ah+i)/8)+row/8)+8*(i%8)+row%8]=aq[i/4][i%4];
        yb[0]=half4(*(device const float4*)y);
        yb[1]=half4(*(device const float4*)(y+4));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* la=sa+4*64*(sg%2);
        threadgroup const half* lb=sb+2*64*(sg/2);
        #pragma clang loop unroll(full)
        for (short kk=0;kk<4;++kk, la+=8*64, lb+=4*64) {
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i=0;i<4;++i) simdgroup_load(ma[i],la+64*i,8,0,false);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i=0;i<2;++i) simdgroup_load(mb[i],lb+64*i,8,0,false);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (short i=0;i<8;++i) simdgroup_multiply_accumulate(mc[i],mb[i/4],ma[i%4],mc[i]);
        }
    }

    const int sr=32*(sg%2), st=16*(sg/2);   /* this simdgroup's sub-tile: first row, first token */
    if (r0+64<=p.m && t0+32<=p.n) {
        device float* c=C+(size_t)(t0+st)*p.ldc+r0+sr;
        for (short i=0;i<8;++i) simdgroup_store(mc[i],c+8*(i%4)+(size_t)8*p.ldc*(i/4),p.ldc);
        return;
    }
    threadgroup float* ct=(threadgroup float*)shmem;   /* [token][row] */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short i=0;i<8;++i) simdgroup_store(mc[i],ct+(st+8*(i/4))*64+sr+8*(i%4),64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e=lid;e<32*64;e+=128) {
        int t=e/64, r=e%64;
        if (t0+t<p.n && r0+r<p.m) C[(size_t)(t0+t)*p.ldc+r0+r]=ct[e];
    }
}

typedef decltype(mma<q8_0_rows>) mma_t;
template [[host_name("q8_0_mma")]]  kernel mma_t mma<q8_0_rows>;
template [[host_name("q4_0_mma")]]  kernel mma_t mma<q4_0_rows>;
template [[host_name("q4k_mma")]]   kernel mma_t mma<q4_k_rows>;
template [[host_name("q5k_mma")]]   kernel mma_t mma<q5_k_rows>;
template [[host_name("q6k_mma")]]   kernel mma_t mma<q6_k_rows>;
template [[host_name("q4_0p_mma")]] kernel mma_t mma<q4_0_packed>;
template [[host_name("q4kp_mma")]]  kernel mma_t mma<q4_k_packed>;
template [[host_name("q5kp_mma")]]  kernel mma_t mma<q5_k_packed>;
template [[host_name("q6kp_mma")]]  kernel mma_t mma<q6_k_packed>;
)MSL";

/* Per weight dtype: its tiled kernel (dense, MXFP4), MMA kernel (block quants), packed-layout MMA
 * kernel, and the block geometry K and the weight stride must divide. */
typedef struct {
    jam_dtype dtype;
    const char *tiled, *mma, *packed;
    int blk;        /* elements per block */
    size_t bpb;     /* bytes per block */
} jam_metal_kind;

static const jam_metal_kind JAM_METAL_KINDS[] = {
    {JAM_F32,   "f32_mm",   NULL,       NULL,        1,   4},
    {JAM_F16,   "f16_mm",   NULL,       NULL,        1,   2},
    {JAM_BF16,  "bf16_mm",  NULL,       NULL,        1,   2},
    {JAM_MXFP4, "mxfp4_mm", NULL,       NULL,        32,  17},
    {JAM_Q8_0,  NULL,       "q8_0_mma", NULL,        32,  34},
    {JAM_Q4_0,  NULL,       "q4_0_mma", "q4_0p_mma", 32,  18},
    {JAM_Q4_K,  NULL,       "q4k_mma",  "q4kp_mma",  256, 144},
    {JAM_Q5_K,  NULL,       "q5k_mma",  "q5kp_mma",  256, 176},
    {JAM_Q6_K,  NULL,       "q6k_mma",  "q6kp_mma",  256, 210},
};
#define JAM_METAL_NKINDS ((int)(sizeof JAM_METAL_KINDS / sizeof *JAM_METAL_KINDS))
#define JAM_MMA_SHMEM 8192   /* the mma kernel's threadgroup memory, see the MSL */

/* One jam_share: [lo, hi) exactly as declared, viewed by a long-lived buffer from the page at base. */
typedef struct { uintptr_t lo, hi, base; id<MTLBuffer> buffer; } jam_metal_share_entry;

struct jam_metal {
    id<MTLDevice>               dev;
    id<MTLCommandQueue>         queue;
    MTLCommandBufferDescriptor* command_desc;
    id<MTLComputePipelineState> tiled[JAM_METAL_NKINDS], mma[JAM_METAL_NKINDS], packed[JAM_METAL_NKINDS];
    int profile;
    uint64_t profile_calls, profile_mma, profile_tiled, profile_failures;
    double profile_encode_s, profile_submit_s, profile_wait_s, profile_gpu_s, profile_total_s;
    os_unfair_lock share_lock;              /* guards the jam_share table below */
    jam_metal_share_entry* shares;
    int nshares, share_cap;
};

/* A per-call Metal view over caller-owned VM pages: page-rounded MTLBuffer + the offset of the
 * tensor inside it. Released before the synchronous jam_mm returns; newBufferWithBytesNoCopy with
 * a nil deallocator never owns or frees the caller's bytes. */
typedef struct {
    id<MTLBuffer> buffer;
    size_t offset;
} jam_bound_buffer;

typedef struct {
    jam_bound_buffer weight;
    jam_bound_buffer activation;
    jam_bound_buffer result;
} jam_metal_resources;

typedef struct {
    int m, n, k, ldw, ldb, ldc;
} jam_metal_mm_params;

static_assert(sizeof(jam_metal_mm_params)==6*sizeof(int),"Metal mm_params layout drift");

typedef struct {
    id<MTLComputePipelineState> pipeline;
    jam_metal_mm_params params;
    MTLSize grid;
    MTLSize threads;
    int mma;   /* threadgroup dispatch with JAM_MMA_SHMEM; else a thread grid */
} jam_metal_plan;

/* The page-rounded NoCopy view over [addr, addr+bytes), or nil when Metal can't map it. It never
 * owns or frees the caller's memory, and rounding does not authorize GPU access outside the bytes. */
static id<MTLBuffer> jam_wrap(jam_metal* m, uintptr_t addr, size_t bytes, uintptr_t* base) {
    size_t page=(size_t)getpagesize();
    uintptr_t wrap_base=addr & ~(uintptr_t)(page-1);
    size_t poff=(size_t)(addr-wrap_base);
    if (bytes > SIZE_MAX-poff-(page-1)) return nil;
    size_t wrap_size=(poff+bytes+page-1) & ~(page-1);
    if (wrap_size > (size_t)m->dev.maxBufferLength) return nil;
    *base=wrap_base;
    return [m->dev newBufferWithBytesNoCopy:(void*)wrap_base
            length:wrap_size options:MTLResourceStorageModeShared deallocator:nil];
}

/* ponytail: linear scan under one lock - 150-300 per-tensor shares measured 50-160 ns per
 * lookup against ~1 ms calls; sort the table only if a profile ever shows it. */
static id<MTLBuffer> jam_shared_buffer(jam_metal* m, uintptr_t addr, size_t bytes, size_t* offset) {
    id<MTLBuffer> buffer=nil;
    os_unfair_lock_lock(&m->share_lock);
    for (int i=0;i<m->nshares;i++) {
        const jam_metal_share_entry* s=&m->shares[i];
        if (s->lo <= addr && addr < s->hi && bytes <= s->hi-addr) {
            buffer=[s->buffer retain];   /* the call's own reference: a concurrent unshare is safe */
            *offset=(size_t)(addr-s->base);
            break;
        }
    }
    os_unfair_lock_unlock(&m->share_lock);
    return buffer;
}

/* An operand's view: its share's long-lived buffer when it lies wholly inside one, else a view made
 * for this call (M3 Pro: ~30 us per MB, 1-1.6 ms per prefill matmul - what jam_share removes).
 * Either way the call holds one reference, released after the synchronous wait. */
static id<MTLBuffer> jam_host_buffer(jam_metal* m, const void* tensor, size_t bytes, size_t* offset) {
    uintptr_t addr=(uintptr_t)tensor;
    if (!tensor || !bytes) return nil;
    id<MTLBuffer> shared=jam_shared_buffer(m,addr,bytes,offset);
    if (shared) return shared;
    uintptr_t base=0;
    id<MTLBuffer> buffer=jam_wrap(m,addr,bytes,&base);
    if (buffer) *offset=(size_t)(addr-base);
    return buffer;
}

extern "C" jam_status jam_metal_share(jam_metal* m, const void* p, size_t bytes) {
    const uintptr_t lo=(uintptr_t)p;
    if (bytes > UINTPTR_MAX-lo) return JAM_EINVAL;
    const uintptr_t hi=lo+bytes;
    uintptr_t base=0;
    id<MTLBuffer> buffer;
    @autoreleasepool { buffer=jam_wrap(m,lo,bytes,&base); }
    if (!buffer) return JAM_EUNSUPPORTED;
    jam_status status=JAM_OK;
    os_unfair_lock_lock(&m->share_lock);
    for (int i=0;i<m->nshares && status == JAM_OK;i++)
        if (lo < m->shares[i].hi && m->shares[i].lo < hi) status=JAM_EINVAL;
    if (status == JAM_OK && m->nshares == m->share_cap) {
        int cap=m->share_cap ? 2*m->share_cap : 64;
        void* grown=realloc(m->shares,(size_t)cap*sizeof *m->shares);
        if (grown) { m->shares=(jam_metal_share_entry*)grown; m->share_cap=cap; }
        else status=JAM_ENOMEM;
    }
    if (status == JAM_OK) {
        jam_metal_share_entry* e=&m->shares[m->nshares++];
        e->lo=lo; e->hi=hi; e->base=base; e->buffer=buffer;
    }
    os_unfair_lock_unlock(&m->share_lock);
    if (status != JAM_OK) [buffer release];
    return status;
}

extern "C" void jam_metal_unshare(jam_metal* m, const void* p) {
    id<MTLBuffer> gone=nil;
    os_unfair_lock_lock(&m->share_lock);
    for (int i=0;i<m->nshares;i++)
        if (m->shares[i].lo == (uintptr_t)p) {
            gone=m->shares[i].buffer;
            m->shares[i]=m->shares[--m->nshares];
            break;
        }
    os_unfair_lock_unlock(&m->share_lock);
    [gone release];
}

static void jam_metal_resources_release(jam_metal_resources* resources) {
    [resources->weight.buffer release];
    [resources->activation.buffer release];
    [resources->result.buffer release];
    memset(resources,0,sizeof *resources);
}

static int jam_command_ok(id<MTLCommandBuffer> cb) {
    if (cb.status == MTLCommandBufferStatusCompleted) return 1;
    NSError* err=cb.error;
    fprintf(stderr,"[jam] metal command failed: %s\n",
            err ? err.localizedDescription.UTF8String : "(unknown)");
    return 0;
}

static double jam_metal_time_s(uint64_t ticks) {
    static mach_timebase_info_data_t info={0,0};
    if (!info.denom) mach_timebase_info(&info);
    return (double)ticks*(double)info.numer/(double)info.denom*1e-9;
}

static int jam_metal_profile_enabled(void) {
    const char* value=getenv("JAM_METAL_PROFILE");
    return value && *value && strcmp(value,"0") && strcmp(value,"false") && strcmp(value,"no");
}

static id<MTLComputePipelineState> jam_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, const char* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        fprintf(stderr, "[jam] metal function not found: %s\n", name);
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!p)
        fprintf(stderr, "[jam] metal pipeline %s failed: %s\n", name,
                err ? err.localizedDescription.UTF8String : "(unknown)");
    [fn release];
    return p;
}

static jam_metal* g_profile_atexit_ctx;
static void jam_metal_profile_atexit(void);

extern "C" jam_metal* jam_metal_create(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return NULL;
        NSError* err = nil;
        /* inject TN from the single host-side JAM_MTN so the tile width can't drift between C and MSL. */
        NSString* src = [NSString stringWithFormat:@"#define TN %d\n%s", JAM_MTN, JAM_MSL];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "[jam] metal MSL compile failed: %s\n",
                    err ? err.localizedDescription.UTF8String : "(unknown)");
            [dev release]; return NULL;
        }
        jam_metal* m = (jam_metal*) calloc(1, sizeof(jam_metal));
        m->dev   = dev;                                  /* +1 from Create, owned */
        m->queue = [dev newCommandQueue];
        m->command_desc = [MTLCommandBufferDescriptor new];
        m->command_desc.retainedReferences = NO;         /* synchronous call owns every resource until wait */
        m->profile = jam_metal_profile_enabled();
        if (m->profile && !g_profile_atexit_ctx) {
            g_profile_atexit_ctx = m;
            atexit(jam_metal_profile_atexit);
        }
        int ok = m->queue && m->command_desc;
        for (int i = 0; i < JAM_METAL_NKINDS; i++) {
            const jam_metal_kind* kind = &JAM_METAL_KINDS[i];
            if (kind->tiled)  ok &= (m->tiled[i]  = jam_pipe(dev, lib, kind->tiled))  != nil;
            if (kind->mma)    ok &= (m->mma[i]    = jam_pipe(dev, lib, kind->mma))    != nil;
            if (kind->packed) ok &= (m->packed[i] = jam_pipe(dev, lib, kind->packed)) != nil;
        }
        [lib release];
        if (!ok) { jam_metal_destroy(m); return NULL; }
        return m;
    }
}

static void jam_metal_profile_print(jam_metal* m) {
    if (!m || !m->profile || !m->profile_calls) return;
    double calls=(double)m->profile_calls;
    double gpu=m->profile_gpu_s/calls*1e6;
    fprintf(stderr,"[jam] metal profile: calls=%llu mma=%llu tiled=%llu failures=%llu\n",
            (unsigned long long)m->profile_calls,(unsigned long long)m->profile_mma,
            (unsigned long long)m->profile_tiled,(unsigned long long)m->profile_failures);
    fprintf(stderr,"[jam] metal profile: per-call encode=%.2fus submit=%.2f wait=%.2f"
                   " gpu=%.2f queue+completion=%.2fus; total_gpu_s=%.3f total_wait_s=%.3f\n",
            m->profile_encode_s/calls*1e6,
            m->profile_submit_s/calls*1e6,m->profile_wait_s/calls*1e6,gpu,
            m->profile_wait_s/calls*1e6-gpu,m->profile_gpu_s,m->profile_total_s);
}

/* Hosts that never destroy their context (the JVM) still get the numbers at process exit. */
static void jam_metal_profile_atexit(void) { jam_metal_profile_print(g_profile_atexit_ctx); }

extern "C" void jam_metal_destroy(jam_metal* m) {
    if (!m) return;
    jam_metal_profile_print(m);
    if (m == g_profile_atexit_ctx) g_profile_atexit_ctx = NULL;
    for (int i = 0; i < JAM_METAL_NKINDS; i++) {
        [m->tiled[i] release]; [m->mma[i] release]; [m->packed[i] release];
    }
    for (int i = 0; i < m->nshares; i++) [m->shares[i].buffer release];
    free(m->shares);
    [m->command_desc release];
    [m->queue release]; [m->dev release];
    free(m);
}

static jam_status jam_metal_prepare_mm(jam_metal* m, jam_metal_resources* resources,
                                       jam_metal_plan* plan, const void* a,
                                       jam_dtype at, int lda,
                                       const void* b, jam_dtype bt, int ldb,
                                       void* c, jam_dtype ct, int ldc,
                                       int M, int N, int K) {
    if (bt != JAM_F32 || ct != JAM_F32) return JAM_EUNSUPPORTED;
    if (lda < K || ldb < K)             return JAM_EINVAL;
    const jam_dtype base = (jam_dtype)(at & ~JAM_PACKED);
    int i = 0;
    while (i < JAM_METAL_NKINDS && JAM_METAL_KINDS[i].dtype != base) i++;
    if (i == JAM_METAL_NKINDS) return JAM_EUNSUPPORTED;
    const jam_metal_kind* kind = &JAM_METAL_KINDS[i];
    /* Match CPU dispatch: a quantized shape outside its block granularity is a valid call for which
     * no kernel exists, not a malformed leading dimension. */
    if (K % kind->blk || lda % kind->blk) return JAM_EUNSUPPORTED;

    size_t wbytes;
    if (at & JAM_PACKED) {
        /* Packed rows are dense 4-row groups (bytes mirror jam_pack_group_bytes), so lda == K; a
         * layout without a Metal kernel falls back to the packed CPU kernels. */
        if (!m->packed[i] || lda != K) return JAM_EUNSUPPORTED;
        plan->pipeline = m->packed[i];
        wbytes = (size_t)(M/4)*jam_pack_group_bytes(base, K);
    } else {
        plan->pipeline = m->mma[i] ? m->mma[i] : m->tiled[i];
        wbytes = (size_t)(M-1)*(lda/kind->blk)*kind->bpb + (size_t)(K/kind->blk)*kind->bpb;
    }
    resources->weight.buffer=jam_host_buffer(m,a,wbytes,&resources->weight.offset);
    resources->activation.buffer=jam_host_buffer(m,b,((size_t)(N-1)*ldb+K)*sizeof(float),
                                                 &resources->activation.offset);
    resources->result.buffer=jam_host_buffer(m,c,((size_t)(N-1)*ldc+M)*sizeof(float),
                                             &resources->result.offset);
    if (!resources->weight.buffer || !resources->activation.buffer || !resources->result.buffer)
        return JAM_EUNSUPPORTED;

    plan->params=(jam_metal_mm_params){M,N,K,lda,ldb,ldc};
    plan->mma = plan->pipeline != m->tiled[i];
    if (plan->mma) {
        plan->grid=MTLSizeMake((N+31)/32,(M+63)/64,1);
        plan->threads=MTLSizeMake(128,1,1);
        ++m->profile_mma;
    } else {
        plan->grid=MTLSizeMake((N+JAM_MTN-1)/JAM_MTN,M,1);
        plan->threads=MTLSizeMake(8,16,1);
        ++m->profile_tiled;
    }
    return JAM_OK;
}

static void jam_metal_encode_mm(id<MTLComputeCommandEncoder> encoder,
                                const jam_metal_resources* resources,
                                const jam_metal_plan* plan) {
    id<MTLBuffer> bound[3]={resources->weight.buffer,resources->activation.buffer,
                            resources->result.buffer};
    NSUInteger offsets[3]={resources->weight.offset,resources->activation.offset,
                           resources->result.offset};
    [encoder setComputePipelineState:plan->pipeline];
    [encoder setBuffers:bound offsets:offsets withRange:NSMakeRange(0,3)];
    [encoder setBytes:&plan->params length:sizeof plan->params atIndex:3];
    if (plan->mma) {
        [encoder setThreadgroupMemoryLength:JAM_MMA_SHMEM atIndex:0];
        [encoder dispatchThreadgroups:plan->grid threadsPerThreadgroup:plan->threads];
    } else {
        [encoder dispatchThreads:plan->grid threadsPerThreadgroup:plan->threads];
    }
}

extern "C" jam_status jam_metal_mm(jam_metal* m, const void* a, jam_dtype at, int lda,
                                   const void* b, jam_dtype bt, int ldb,
                                   void* c, jam_dtype ct, int ldc,
                                   int M, int N, int K) {
    if (!m) return JAM_EINVAL;
    @autoreleasepool {
        const uint64_t t0=mach_absolute_time();
        jam_metal_resources resources={};
        jam_metal_plan plan={};
        jam_status status=jam_metal_prepare_mm(m,&resources,&plan,a,at,lda,b,bt,ldb,c,ct,ldc,M,N,K);
        if (status == JAM_OK) {
            /* One synchronous command buffer per call: encode, submit, wait. The pool releases
             * cb/enc; the operand views are released below, after the GPU is done with them. */
            id<MTLCommandBuffer> cb=[m->queue commandBufferWithDescriptor:m->command_desc];
            id<MTLComputeCommandEncoder> enc=[cb computeCommandEncoder];
            if (!enc) status=JAM_EUNSUPPORTED;
            else {
                jam_metal_encode_mm(enc,&resources,&plan);
                [enc endEncoding];
                const uint64_t t_encoded=mach_absolute_time();
                [cb commit];
                const uint64_t t_submitted=mach_absolute_time();
                [cb waitUntilCompleted];
                const uint64_t t_done=mach_absolute_time();
                status=jam_command_ok(cb) ? JAM_OK : JAM_EUNSUPPORTED;
                if (m->profile) {
                    ++m->profile_calls;
                    if (status != JAM_OK) ++m->profile_failures;
                    m->profile_encode_s+=jam_metal_time_s(t_encoded-t0);
                    m->profile_submit_s+=jam_metal_time_s(t_submitted-t_encoded);
                    m->profile_wait_s+=jam_metal_time_s(t_done-t_submitted);
                    m->profile_total_s+=jam_metal_time_s(t_done-t_encoded);
                    if (cb.GPUEndTime >= cb.GPUStartTime)
                        m->profile_gpu_s+=cb.GPUEndTime-cb.GPUStartTime;
                }
            }
        }
        jam_metal_resources_release(&resources);
        return status;
    }
}
