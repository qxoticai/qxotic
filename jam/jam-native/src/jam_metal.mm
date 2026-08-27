/* jam_metal.mm — Apple GPU backend (opt-in via JAM_ISA=metal). A different executor from the CPU
 * row-range kernels: jam_mm routes supported dtypes here before the pool (except small-n quant
 * decode, which stays on the CPU GEMVs — see jam.c). Quant weights are decoded on-GPU; block-quant
 * prefill (n >= JAM_MMA_MIN_N, canonical AND packed layouts) stages half operands and float
 * accumulators through simdgroup matrices and threadgroup tiles, while dense dtypes and remaining
 * shapes dot the exact F32 activation with a scalar tiled path (each thread decodes a weight block
 * once and reuses it across TN columns). MSL is compiled at runtime (no .metallib). Output is
 * token-major C[j*ldc+i].
 *
 * Zero-copy over unified memory: W/A/C are borrowed per call through page-rounded
 * newBufferWithBytesNoCopy views (nil deallocator — the caller owns every byte) and released after
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

#define JAM_MTN 8   /* output columns per GPU thread — the one tunable; injected into the MSL as TN */
#define JAM_MMA_MIN_N 16

static const char* JAM_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant float MXFP4_LUT[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };  /* ggml kvalues_mxfp4 */

struct mm_params { int m, n, k, ldw, ldb, ldc; };

/* thread (gid.x, gid.y) owns weight row i = gid.y and the TN columns [j0, j0+TN), j0 = gid.x*TN. */
#define JAM_TILE_PROLOGUE \
    int m=p.m, n=p.n, k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc; \
    int i=int(gid.y), j0=int(gid.x)*TN; \
    if (i>=m || j0>=n) return; \
    float acc[TN]; for (int c=0;c<TN;++c) acc[c]=0.0f;
#define JAM_TILE_EPILOGUE \
    for (int c=0;c<TN;++c){ int j=j0+c; if (j<n) C[(size_t)j*ldc+i]=acc[c]; }

/* The 8 6-bit (scale,min) pairs of a Q4_K/Q5_K super-block (ggml packing); sb = scales[12]. */
static void kq_scales_mins(device const uchar* sb, thread uchar* sc, thread uchar* mn) {
    for (int t=0;t<4;++t){
        sc[t]=sb[t]&63; mn[t]=sb[t+4]&63;
        sc[t+4]=(sb[t+8]&0xF)|((sb[t]>>6)<<4);
        mn[t+4]=(sb[t+8]>>4)|((sb[t+4]>>6)<<4);
    }
}

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

/* Q8_0 block = { half d; char qs[32] } = 34 B. */
kernel void q8_0_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                    device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int nb=k/32;
    device const uchar* wrow = A + (size_t)i*(ldw/32)*34;
    for (int b=0;b<nb;++b){
        device const uchar* blk = wrow + (size_t)b*34;
        float d = float(*(device const half*)blk);
        device const char* qs = (device const char*)(blk+2);
        char wq[32]; for (int e=0;e<32;++e) wq[e]=qs[e];           /* decode ONCE */
        for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
            device const float* bb = B + (size_t)j*ldb + b*32;
            float s=0.0f; for (int e=0;e<32;++e) s += float(wq[e])*bb[e];
            acc[c] += d*s;
        }
    }
    JAM_TILE_EPILOGUE
}

/* Q4_0 block = { half d; uchar qs[16] } = 18 B. value = d·(nibble-8); lo nibble -> e, hi -> e+16. */
kernel void q4_0_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                    device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int nb=k/32;
    device const uchar* wrow = A + (size_t)i*(ldw/32)*18;
    for (int b=0;b<nb;++b){
        device const uchar* blk = wrow + (size_t)b*18;
        float d = float(*(device const half*)blk);
        device const uchar* qs = blk+2;
        float wq[32]; for (int e=0;e<16;++e){ wq[e]=float(qs[e]&0xF)-8.0f; wq[e+16]=float(qs[e]>>4)-8.0f; }
        for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
            device const float* bb = B + (size_t)j*ldb + b*32;
            float s=0.0f; for (int e=0;e<32;++e) s += wq[e]*bb[e];
            acc[c] += d*s;
        }
    }
    JAM_TILE_EPILOGUE
}

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

/* Q4_K super-block = { half d; half dmin; uchar scales[12]; uchar qs[128] } = 144 B, 256 vals, 8 sub-blocks
 * of 32; sub-block s -> elements s*32, nibbles q[(s/2)*32 + e] (low if s even, else high), scale sc[s]. */
kernel void q4k_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int ns=k/256;
    device const uchar* wrow = A + (size_t)i*(ldw/256)*144;
    for (int B2=0;B2<ns;++B2){
        device const uchar* w = wrow + (size_t)B2*144;
        float d=float(*(device const half*)w), dmin=float(*(device const half*)(w+2));
        uchar sc[8], mn[8]; kq_scales_mins(w+4, sc, mn);
        device const uchar* q = w+16;
        for (int s=0;s<8;++s){
            int g=s/2; device const uchar* qb = q + g*32;
            uchar wq[32];
            if ((s&1)==0) for (int e=0;e<32;++e) wq[e]=qb[e]&0xF; else for (int e=0;e<32;++e) wq[e]=qb[e]>>4;
            float dl=d*float(sc[s]), ml=dmin*float(mn[s]);
            for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
                device const float* x = B + (size_t)j*ldb + (size_t)B2*256 + s*32;
                float sWB=0.0f, sB=0.0f; for (int e=0;e<32;++e){ float bv=x[e]; sWB += float(wq[e])*bv; sB += bv; }
                acc[c] += dl*sWB - ml*sB;
            }
        }
    }
    JAM_TILE_EPILOGUE
}

/* Q5_K = { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128] } = 176 B. q5 = nibble |
 * (bit s of qh[e] << 4); sub-block s like Q4_K. */
kernel void q5k_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int ns=k/256;
    device const uchar* wrow = A + (size_t)i*(ldw/256)*176;
    for (int B2=0;B2<ns;++B2){
        device const uchar* w = wrow + (size_t)B2*176;
        float d=float(*(device const half*)w), dmin=float(*(device const half*)(w+2));
        uchar sc[8], mn[8]; kq_scales_mins(w+4, sc, mn);
        device const uchar* qh = w+16; device const uchar* qs = w+48;
        for (int s=0;s<8;++s){
            int g=s/2; device const uchar* qb = qs + g*32;
            uchar wq[32];
            if ((s&1)==0) for (int e=0;e<32;++e) wq[e]=(qb[e]&0xF)|(((qh[e]>>s)&1)<<4);
            else          for (int e=0;e<32;++e) wq[e]=(qb[e]>>4) |(((qh[e]>>s)&1)<<4);
            float dl=d*float(sc[s]), ml=dmin*float(mn[s]);
            for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
                device const float* x = B + (size_t)j*ldb + (size_t)B2*256 + s*32;
                float sWB=0.0f, sB=0.0f; for (int e=0;e<32;++e){ float bv=x[e]; sWB += float(wq[e])*bv; sB += bv; }
                acc[c] += dl*sWB - ml*sB;
            }
        }
    }
    JAM_TILE_EPILOGUE
}

/* Q6_K = { uchar ql[128]; uchar qh[64]; char scales[16]; half d } = 210 B. value = d·sc·(qv-32), qv 6-bit
 * (ql nibble | qh 2-bit << 4); int8 scale per 16 elements. Sub-block hg = h*4+g -> elements hg*32. */
kernel void q6k_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int ns=k/256;
    device const uchar* wrow = A + (size_t)i*(ldw/256)*210;
    for (int B2=0;B2<ns;++B2){
        device const uchar* w = wrow + (size_t)B2*210;
        device const uchar* ql = w; device const uchar* qh = w+128;
        device const char* sc = (device const char*)(w+192);
        float d = float(*(device const half*)(w+208));
        for (int h=0;h<2;++h){
            device const uchar* qlb = ql + h*64; device const uchar* qhb = qh + h*32;
            for (int g=0;g<4;++g){
                device const uchar* qlp = qlb + (g&1)*32;
                char wq[32];                                          /* qv-32, signed */
                for (int l=0;l<32;++l){
                    int qv = (g<2) ? (qlp[l]&0xF) : (qlp[l]>>4);
                    qv |= ((qhb[l]>>(2*g)) & 3)<<4;
                    wq[l] = char(qv-32);
                }
                float s0=d*float(sc[h*8+g*2]), s1=d*float(sc[h*8+g*2+1]);   /* scale per 16 */
                int hg = h*4+g;
                for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
                    device const float* x = B + (size_t)j*ldb + (size_t)B2*256 + hg*32;
                    float d0=0.0f, d1=0.0f;
                    for (int l=0;l<16;++l)  d0 += float(wq[l])*x[l];
                    for (int l=16;l<32;++l) d1 += float(wq[l])*x[l];
                    acc[c] += s0*d0 + s1*d1;
                }
            }
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

/* Aligned Q8_0 prefill fast path. Production projections are multiples of the 64-row x 32-token
 * output tile, so keep a separate kernel with no edge predicates and only the 6 KiB needed for its
 * half input tiles. Each loader prefetches the next Q8/B slice into registers before waiting for the
 * previous MMA readers, moving global-memory latency across the threadgroup-memory reuse barrier. */
kernel void q8_0_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    device const uchar* block=A+(size_t)arow*(ldw/32)*34;
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4];
    half d=*(device const half*)block;
    device const uchar* q=block+2+ah;
    aq[0]=d*half4(*(device const packed_char4*)(q+0));
    aq[1]=d*half4(*(device const packed_char4*)(q+4));
    aq[2]=d*half4(*(device const packed_char4*)(q+8));
    aq[3]=d*half4(*(device const packed_char4*)(q+12));
    half4 bv0=half4(*(device const float4*)(bp+0));
    half4 bv1=half4(*(device const float4*)(bp+4));

    for (int b=0;;) {
        /* On iterations after the first, A/B are already in registers. Waiting here protects the
         * old staging tile without placing the global loads themselves behind the barrier. */
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }

        if (++b==nb) break;
        block+=34;
        bp+=32;
        d=*(device const half*)block;
        q=block+2+ah;
        aq[0]=d*half4(*(device const packed_char4*)(q+0));
        aq[1]=d*half4(*(device const packed_char4*)(q+4));
        aq[2]=d*half4(*(device const packed_char4*)(q+8));
        aq[3]=d*half4(*(device const packed_char4*)(q+12));
        bv0=half4(*(device const float4*)(bp+0));
        bv1=half4(*(device const float4*)(bp+4));
    }

    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}

/* Edge-capable Q8_0 path. The 8 KiB slab is aliased as half input tiles during MMA and then as
 * float output scratch for a partial tile. Fully aligned calls use q8_0_mma_full above. */
kernel void q8_0_mma(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    int m=p.m, n=p.n, k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup float tile[64*32];
    threadgroup half* As=(threadgroup half*)tile;
    threadgroup half* Bs=As+64*32;
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    for (int b=0;b<nb;++b) {
        if (arow<m) {
            device const uchar* block=A+(size_t)arow*(ldw/32)*34+(size_t)b*34;
            half d=*(device const half*)block;
            device const char* q=(device const char*)(block+2);
            for (int e=0;e<16;++e) {
                int kk=ah+e;
                As[((ar>>3)*4+(kk>>3))*64+(ar&7)*8+(kk&7)]=d*half(q[kk]);
            }
        } else {
            for (int e=0;e<16;++e) {
                int kk=ah+e;
                As[((ar>>3)*4+(kk>>3))*64+(ar&7)*8+(kk&7)]=0.0f;
            }
        }
        half4 bv0=half4(0.0h), bv1=half4(0.0h);
        if (j0+bt<n) {
            device const float* bp=B+(size_t)(j0+bt)*ldb+b*32+bk;
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        /* B is token x K in memory, so keep each vector load contiguous in the tile too. This also
         * makes the MMA accumulator token x weight-row and the full-tile output directly storable. */
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /* Every production LFM projection is tile-aligned: transpose-store its row x token accumulator
     * directly into token-major C. Only a true edge tile pays for materialization and scalar bounds. */
    if (i0+64<=m && j0+32<=n) {
        int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
        for (int tt=0;tt<2;++tt)
            for (int ww=0;ww<4;++ww)
                simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                                ldc,ulong2(0,0),false);
        return;
    }
    threadgroup float* Cs=tile;
    int wi=(int(sg)&1)*32, tj=(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],Cs+(size_t)(wi+ww*8)*32+tj+tt*8,
                            32,ulong2(0,0),true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e=int(lid);e<64*32;e+=128) {
        int row=e/32, col=e%32;
        if (i0+row<m && j0+col<n) C[(size_t)(j0+col)*ldc+i0+row]=Cs[e];
    }
}

/* Aligned Q4_0 prefill. The MMA and staging layout match Q8_0; only the packed-weight loader is
 * different. Two threads own each row/block: ah==0 expands the low nibbles, ah==16 the high ones. */
kernel void q4_0_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    device const uchar* block=A+(size_t)arow*(ldw/32)*18;
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4];
    half d=*(device const half*)block;
    device const uchar* q=block+2;
    bool high=ah!=0;
    aq[0]=q4_0_vec4(q,0,high,d);
    aq[1]=q4_0_vec4(q,4,high,d);
    aq[2]=q4_0_vec4(q,8,high,d);
    aq[3]=q4_0_vec4(q,12,high,d);
    half4 bv0=half4(*(device const float4*)(bp+0));
    half4 bv1=half4(*(device const float4*)(bp+4));

    for (int b=0;;) {
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }

        if (++b==nb) break;
        block+=18;
        bp+=32;
        d=*(device const half*)block;
        q=block+2;
        aq[0]=q4_0_vec4(q,0,high,d);
        aq[1]=q4_0_vec4(q,4,high,d);
        aq[2]=q4_0_vec4(q,8,high,d);
        aq[3]=q4_0_vec4(q,12,high,d);
        bv0=half4(*(device const float4*)(bp+0));
        bv1=half4(*(device const float4*)(bp+4));
    }

    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}

/* Partial-tile Q4_0 path. It retains the same vector unpack but allocates the 8 KiB alias needed
 * to materialize a bounded output tile. */
kernel void q4_0_mma(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    int m=p.m, n=p.n, k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup float tile[64*32];
    threadgroup half* As=(threadgroup half*)tile;
    threadgroup half* Bs=As+64*32;
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    bool high=ah!=0;
    for (int b=0;b<nb;++b) {
        half4 aq0=half4(0.0h), aq1=half4(0.0h), aq2=half4(0.0h), aq3=half4(0.0h);
        if (arow<m) {
            device const uchar* block=A+(size_t)arow*(ldw/32)*18+(size_t)b*18;
            half d=*(device const half*)block;
            device const uchar* q=block+2;
            aq0=q4_0_vec4(q,0,high,d);
            aq1=q4_0_vec4(q,4,high,d);
            aq2=q4_0_vec4(q,8,high,d);
            aq3=q4_0_vec4(q,12,high,d);
        }
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq0;
        *(threadgroup half4*)(As+abase+4)=aq1;
        *(threadgroup half4*)(As+abase+64)=aq2;
        *(threadgroup half4*)(As+abase+68)=aq3;

        half4 bv0=half4(0.0h), bv1=half4(0.0h);
        if (j0+bt<n) {
            device const float* bp=B+(size_t)(j0+bt)*ldb+b*32+bk;
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i0+64<=m && j0+32<=n) {
        int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
        for (int tt=0;tt<2;++tt)
            for (int ww=0;ww<4;++ww)
                simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                                ldc,ulong2(0,0),false);
        return;
    }
    threadgroup float* Cs=tile;
    int wi=(int(sg)&1)*32, tj=(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],Cs+(size_t)(wi+ww*8)*32+tj+tt*8,
                            32,ulong2(0,0),true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e=int(lid);e<64*32;e+=128) {
        int row=e/32, col=e%32;
        if (i0+row<m && j0+col<n) C[(size_t)(j0+col)*ldc+i0+row]=Cs[e];
    }
}

/* K-quant MMA prefill over the CANONICAL wire layouts (aligned fast paths: m%64==0, n%32==0,
 * k%256==0; other shapes keep the tiled kernels). These coexist with the packed-layout MMA
 * kernels below because unpacked tensors still reach Metal prefill: row-read tensors are never
 * packed (a tied Q6_K LM head is the largest single matmul of a gemma prefill pass), and callers
 * without a packer get the wire dtypes' full speed. Identical staging/MMA structure to
 * q8_0_mma_full - only the loader's dequant-to-half differs. Scale unpacking runs in the loader
 * phase, where ALU hides under the global loads. A thread's 16-element slice matches Q6_K's
 * per-16 scale granularity exactly. */
kernel void q4k_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    device const uchar* wrow=A+(size_t)arow*(ldw/256)*144;
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4]; half4 bv0, bv1;
    for (int b=0;;) {
        {   int sb=b>>3, blk=b&7;
            device const uchar* w=wrow+(size_t)sb*144;
            float d=float(*(device const half*)w), dmin=float(*(device const half*)(w+2));
            device const uchar* sp=w+4;
            uint sc, mn;
            if (blk<4) { sc=sp[blk]&63u; mn=sp[blk+4]&63u; }
            else { sc=(sp[blk+4]&0xFu)|((sp[blk-4]>>6)<<4); mn=(sp[blk+4]>>4)|((sp[blk]>>6)<<4); }
            float fs=d*float(sc), fm=dmin*float(mn);
            device const uchar* qq=w+16+(blk>>1)*32+ah;
            bool hi=(blk&1)!=0;
            for (int t=0;t<4;++t) {
                uchar4 raw=*(device const packed_uchar4*)(qq+t*4);
                uchar4 nib=hi ? (raw>>4) : (raw&0xF);
                aq[t]=half4(float4(nib)*fs-fm);
            }
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        if (++b==nb) break;
        bp+=32;
    }
    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}

kernel void q5k_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    device const uchar* wrow=A+(size_t)arow*(ldw/256)*176;
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4]; half4 bv0, bv1;
    for (int b=0;;) {
        {   int sb=b>>3, blk=b&7;
            device const uchar* w=wrow+(size_t)sb*176;
            float d=float(*(device const half*)w), dmin=float(*(device const half*)(w+2));
            device const uchar* sp=w+4;
            uint sc, mn;
            if (blk<4) { sc=sp[blk]&63u; mn=sp[blk+4]&63u; }
            else { sc=(sp[blk+4]&0xFu)|((sp[blk-4]>>6)<<4); mn=(sp[blk+4]>>4)|((sp[blk]>>6)<<4); }
            float fs=d*float(sc), fm=dmin*float(mn);
            device const uchar* qq=w+48+(blk>>1)*32+ah;
            device const uchar* qh=w+16+ah;
            bool hi=(blk&1)!=0;
            uint shift=uint(blk);
            for (int t=0;t<4;++t) {
                uchar4 raw=*(device const packed_uchar4*)(qq+t*4);
                uchar4 hb=*(device const packed_uchar4*)(qh+t*4);
                uchar4 nib=(hi ? (raw>>4) : (raw&0xF)) | (((hb>>shift)&1)<<4);
                aq[t]=half4(float4(nib)*fs-fm);
            }
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        if (++b==nb) break;
        bp+=32;
    }
    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}

kernel void q6k_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldw=p.ldw, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    device const uchar* wrow=A+(size_t)arow*(ldw/256)*210;
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4]; half4 bv0, bv1;
    for (int b=0;;) {
        {   int sb=b>>3, blk=b&7, h=blk>>2, g=blk&3;
            device const uchar* w=wrow+(size_t)sb*210;
            float d=float(*(device const half*)(w+208));
            device const char* sc=(device const char*)(w+192);
            float fs=d*float(sc[h*8+g*2+(ah>>4)]);
            device const uchar* qlb=w+h*64+(g&1)*32+ah;
            device const uchar* qhb=w+128+h*32+ah;
            bool hi=(g>=2);
            uint shift=uint(2*g);
            for (int t=0;t<4;++t) {
                uchar4 ql=*(device const packed_uchar4*)(qlb+t*4);
                uchar4 qh=*(device const packed_uchar4*)(qhb+t*4);
                uchar4 q6=(hi ? (ql>>4) : (ql&0xF)) | (((qh>>shift)&3)<<4);
                aq[t]=half4((float4(q6)-32.0f)*fs);
            }
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        if (++b==nb) break;
        bp+=32;
    }
    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}

/* Packed-layout MMA prefill (jam.h JAM_PACK_ABI 1: caller-packed per-4-row-group weights, the
 * SAME single copy the CPU decode GEMVs read). Aligned shapes only (m%64, n%32, k%blk); other
 * shapes fall back to the packed CPU kernels. Same staging/MMA structure as q8_0_mma_full -
 * the loaders differ, and the expanded-int8 layouts make them CHEAPER than the canonical
 * k-quant loaders above (no bit surgery, flat f32 scales). */
kernel void q4_0p_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                           device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    int wr_=arow&3;
    device const uchar* G=A+(size_t)(arow>>2)*((size_t)nb*80);
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4]; half4 bv0, bv1;
    for (int b=0;;) {
        {
            device const uchar* qq=G+(size_t)b*64+wr_*16;
            float s=((device const float*)(G+(size_t)nb*64))[b*4+wr_];
            for (int t=0;t<4;++t) {
                uchar4 raw=*(device const packed_uchar4*)(qq+t*4);
                uchar4 nib=ah ? (raw>>4) : (raw&0xF);
                aq[t]=half4((float4(nib)-8.0f)*s);
            }
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        if (++b==nb) break;
        bp+=32;
    }
    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}

kernel void q4kp_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    int wr_=arow&3;
    device const uchar* G=A+(size_t)(arow>>2)*((size_t)nb*72+(size_t)(nb/8)*32);
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4]; half4 bv0, bv1;
    for (int b=0;;) {
        {
            device const uchar* qq=G+(size_t)b*64+wr_*16;
            device const uchar* sm=G+(size_t)nb*64+b*8;
            device const float* dd=(device const float*)(G+(size_t)nb*72);
            float fs=dd[(b>>3)*8+wr_]*float(sm[wr_]);
            float fm=dd[(b>>3)*8+4+wr_]*float(sm[4+wr_]);
            for (int t=0;t<4;++t) {
                uchar4 raw=*(device const packed_uchar4*)(qq+t*4);
                uchar4 nib=ah ? (raw>>4) : (raw&0xF);
                aq[t]=half4(float4(nib)*fs-fm);
            }
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        if (++b==nb) break;
        bp+=32;
    }
    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}

kernel void q5kp_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    int wr_=arow&3;
    device const uchar* G=A+(size_t)(arow>>2)*((size_t)nb*136+(size_t)(nb/8)*32);
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4]; half4 bv0, bv1;
    for (int b=0;;) {
        {
            device const char* q=(device const char*)(G+(size_t)b*128+wr_*32+ah);
            device const uchar* sm=G+(size_t)nb*128+b*8;
            device const float* dd=(device const float*)(G+(size_t)nb*136);
            float fs=dd[(b>>3)*8+wr_]*float(sm[wr_]);
            float fm=dd[(b>>3)*8+4+wr_]*float(sm[4+wr_]);
            for (int t=0;t<4;++t)
                aq[t]=half4(float4(*(device const packed_char4*)(q+t*4))*fs-fm);
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        if (++b==nb) break;
        bp+=32;
    }
    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}

kernel void q6kp_mma_full(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                          device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    int k=p.k, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup half As[64*32];
    threadgroup half Bs[32*32];
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    int wr_=arow&3;
    device const uchar* G=A+(size_t)(arow>>2)*((size_t)nb*136+(size_t)(nb/8)*16);
    device const float* bp=B+(size_t)(j0+bt)*ldb+bk;
    half4 aq[4]; half4 bv0, bv1;
    for (int b=0;;) {
        {
            device const char* q=(device const char*)(G+(size_t)b*128+wr_*32+ah);
            device const char* sc=(device const char*)(G+(size_t)nb*128);
            device const float* dd=(device const float*)(G+(size_t)nb*136);
            float fs=dd[(b>>3)*4+wr_]*float(sc[b*8+(ah>>4)*4+wr_]);
            for (int t=0;t<4;++t)
                aq[t]=half4(float4(*(device const packed_char4*)(q+t*4))*fs);
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        if (b) threadgroup_barrier(mem_flags::mem_threadgroup);
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        if (++b==nb) break;
        bp+=32;
    }
    int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                            ldc,ulong2(0,0),false);
}


/* Packed Q4_0 edge MMA: any M / N >= 16 (MoE expert slices land here - their token counts are
 * rarely multiples of 32). Same guarded structure as q4_0_mma; the packed loader replaces the
 * canonical block decode. Packed k-quant edge shapes stay on the CPU kernels until a model
 * needs them. */
kernel void q4_0p_mma(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                      device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    int m=p.m, n=p.n, k=p.k, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup float tile[64*32];
    threadgroup half* As=(threadgroup half*)tile;
    threadgroup half* Bs=As+64*32;
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    for (int b=0;b<nb;++b) {
        half4 aq0=half4(0.0h), aq1=half4(0.0h), aq2=half4(0.0h), aq3=half4(0.0h);
        if (arow<m) {
            device const uchar* G=A+(size_t)(arow>>2)*((size_t)nb*80);
            device const uchar* qq=G+(size_t)b*64+(arow&3)*16;
            float sc=((device const float*)(G+(size_t)nb*64))[b*4+(arow&3)];
            uchar4 r0=*(device const packed_uchar4*)(qq+0);
            uchar4 r1=*(device const packed_uchar4*)(qq+4);
            uchar4 r2=*(device const packed_uchar4*)(qq+8);
            uchar4 r3=*(device const packed_uchar4*)(qq+12);
            if (ah) { r0>>=4; r1>>=4; r2>>=4; r3>>=4; }
            else { r0&=0xF; r1&=0xF; r2&=0xF; r3&=0xF; }
            aq0=half4((float4(r0)-8.0f)*sc);
            aq1=half4((float4(r1)-8.0f)*sc);
            aq2=half4((float4(r2)-8.0f)*sc);
            aq3=half4((float4(r3)-8.0f)*sc);
        }
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq0;
        *(threadgroup half4*)(As+abase+4)=aq1;
        *(threadgroup half4*)(As+abase+64)=aq2;
        *(threadgroup half4*)(As+abase+68)=aq3;

        half4 bv0=half4(0.0h), bv1=half4(0.0h);
        if (j0+bt<n) {
            device const float* bp=B+(size_t)(j0+bt)*ldb+b*32+bk;
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i0+64<=m && j0+32<=n) {
        int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
        for (int tt=0;tt<2;++tt)
            for (int ww=0;ww<4;++ww)
                simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                                ldc,ulong2(0,0),false);
        return;
    }
    threadgroup float* Cs=tile;
    int wi=(int(sg)&1)*32, tj=(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],Cs+(size_t)(wi+ww*8)*32+tj+tt*8,
                            32,ulong2(0,0),true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e=int(lid);e<64*32;e+=128) {
        int row=e/32, col=e%32;
        if (i0+row<m && j0+col<n) C[(size_t)(j0+col)*ldc+i0+row]=Cs[e];
    }
}


/* Packed k-quant edge MMA: any M / N >= 16, completing the packed kernel matrix (every dtype has
 * full + edge on both engines). Same guarded skeleton as q4_0p_mma; loaders are the _full ones. */
kernel void q4kp_mma(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    int m=p.m, n=p.n, k=p.k, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup float tile[64*32];
    threadgroup half* As=(threadgroup half*)tile;
    threadgroup half* Bs=As+64*32;
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    int wr_=arow&3;
    device const uchar* G=A+(size_t)(arow>>2)*((size_t)nb*72+(size_t)(nb/8)*32);
    for (int b=0;b<nb;++b) {
        half4 aq[4]={half4(0.0h),half4(0.0h),half4(0.0h),half4(0.0h)};
        if (arow<m) {
            device const uchar* qq=G+(size_t)b*64+wr_*16;
            device const uchar* sm=G+(size_t)nb*64+b*8;
            device const float* dd=(device const float*)(G+(size_t)nb*72);
            float fs=dd[(b>>3)*8+wr_]*float(sm[wr_]);
            float fm=dd[(b>>3)*8+4+wr_]*float(sm[4+wr_]);
            for (int t=0;t<4;++t) {
                uchar4 raw=*(device const packed_uchar4*)(qq+t*4);
                uchar4 nib=ah ? (raw>>4) : (raw&0xF);
                aq[t]=half4(float4(nib)*fs-fm);
            }
        }
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];

        half4 bv0=half4(0.0h), bv1=half4(0.0h);
        if (j0+bt<n) {
            device const float* bp=B+(size_t)(j0+bt)*ldb+b*32+bk;
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i0+64<=m && j0+32<=n) {
        int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
        for (int tt=0;tt<2;++tt)
            for (int ww=0;ww<4;++ww)
                simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                                ldc,ulong2(0,0),false);
        return;
    }
    threadgroup float* Cs=tile;
    int wi=(int(sg)&1)*32, tj=(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],Cs+(size_t)(wi+ww*8)*32+tj+tt*8,
                            32,ulong2(0,0),true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e=int(lid);e<64*32;e+=128) {
        int row=e/32, col=e%32;
        if (i0+row<m && j0+col<n) C[(size_t)(j0+col)*ldc+i0+row]=Cs[e];
    }
}

kernel void q5kp_mma(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    int m=p.m, n=p.n, k=p.k, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup float tile[64*32];
    threadgroup half* As=(threadgroup half*)tile;
    threadgroup half* Bs=As+64*32;
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    int wr_=arow&3;
    device const uchar* G=A+(size_t)(arow>>2)*((size_t)nb*136+(size_t)(nb/8)*32);
    for (int b=0;b<nb;++b) {
        half4 aq[4]={half4(0.0h),half4(0.0h),half4(0.0h),half4(0.0h)};
        if (arow<m) {
            device const char* q=(device const char*)(G+(size_t)b*128+wr_*32+ah);
            device const uchar* sm=G+(size_t)nb*128+b*8;
            device const float* dd=(device const float*)(G+(size_t)nb*136);
            float fs=dd[(b>>3)*8+wr_]*float(sm[wr_]);
            float fm=dd[(b>>3)*8+4+wr_]*float(sm[4+wr_]);
            for (int t=0;t<4;++t)
                aq[t]=half4(float4(*(device const packed_char4*)(q+t*4))*fs-fm);
        }
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];

        half4 bv0=half4(0.0h), bv1=half4(0.0h);
        if (j0+bt<n) {
            device const float* bp=B+(size_t)(j0+bt)*ldb+b*32+bk;
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i0+64<=m && j0+32<=n) {
        int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
        for (int tt=0;tt<2;++tt)
            for (int ww=0;ww<4;++ww)
                simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                                ldc,ulong2(0,0),false);
        return;
    }
    threadgroup float* Cs=tile;
    int wi=(int(sg)&1)*32, tj=(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],Cs+(size_t)(wi+ww*8)*32+tj+tt*8,
                            32,ulong2(0,0),true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e=int(lid);e<64*32;e+=128) {
        int row=e/32, col=e%32;
        if (i0+row<m && j0+col<n) C[(size_t)(j0+col)*ldc+i0+row]=Cs[e];
    }
}

kernel void q6kp_mma(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant mm_params& p [[buffer(3)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    int m=p.m, n=p.n, k=p.k, ldb=p.ldb, ldc=p.ldc, nb=k/32;
    int i0=int(tg.x)*64, j0=int(tg.y)*32;
    threadgroup float tile[64*32];
    threadgroup half* As=(threadgroup half*)tile;
    threadgroup half* Bs=As+64*32;
    simdgroup_matrix<float,8,8> acc[2][4];
    for (int r=0;r<2;++r)
        for (int c=0;c<4;++c)
            acc[r][c]=make_filled_simdgroup_matrix<float,8,8>(0.0f);

    int ar=int(lid)/2, ah=(int(lid)%2)*16, arow=i0+ar;
    int bt=int(lid)/4, bk=(int(lid)%4)*8;
    int wr_=arow&3;
    device const uchar* G=A+(size_t)(arow>>2)*((size_t)nb*136+(size_t)(nb/8)*16);
    for (int b=0;b<nb;++b) {
        half4 aq[4]={half4(0.0h),half4(0.0h),half4(0.0h),half4(0.0h)};
        if (arow<m) {
            device const char* q=(device const char*)(G+(size_t)b*128+wr_*32+ah);
            device const char* sc=(device const char*)(G+(size_t)nb*128);
            device const float* dd=(device const float*)(G+(size_t)nb*136);
            float fs=dd[(b>>3)*4+wr_]*float(sc[b*8+(ah>>4)*4+wr_]);
            for (int t=0;t<4;++t)
                aq[t]=half4(float4(*(device const packed_char4*)(q+t*4))*fs);
        }
        int abase=((ar>>3)*4+(ah>>3))*64+(ar&7)*8;
        *(threadgroup half4*)(As+abase+0)=aq[0];
        *(threadgroup half4*)(As+abase+4)=aq[1];
        *(threadgroup half4*)(As+abase+64)=aq[2];
        *(threadgroup half4*)(As+abase+68)=aq[3];

        half4 bv0=half4(0.0h), bv1=half4(0.0h);
        if (j0+bt<n) {
            device const float* bp=B+(size_t)(j0+bt)*ldb+b*32+bk;
            bv0=half4(*(device const float4*)(bp+0));
            bv1=half4(*(device const float4*)(bp+4));
        }
        threadgroup half* bdst=Bs+((bt>>3)*4+(bk>>3))*64+(bt&7)*8;
        *(threadgroup half4*)(bdst+0)=bv0;
        *(threadgroup half4*)(bdst+4)=bv1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk=0;kk<32;kk+=8) {
            int k8=kk>>3;
            int wr=(int(sg)&1)*4, jt=(int(sg)>>1)*2;
            simdgroup_matrix<half,8,8> w0,w1,w2,w3,x0,x1;
            simdgroup_load(w0,As+(size_t)((wr+0)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w1,As+(size_t)((wr+1)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w2,As+(size_t)((wr+2)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(w3,As+(size_t)((wr+3)*4+k8)*64,8,ulong2(0,0),true);
            simdgroup_load(x0,Bs+(size_t)((jt+0)*4+k8)*64,8);
            simdgroup_load(x1,Bs+(size_t)((jt+1)*4+k8)*64,8);
            simdgroup_multiply_accumulate(acc[0][0],x0,w0,acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1],x0,w1,acc[0][1]);
            simdgroup_multiply_accumulate(acc[0][2],x0,w2,acc[0][2]);
            simdgroup_multiply_accumulate(acc[0][3],x0,w3,acc[0][3]);
            simdgroup_multiply_accumulate(acc[1][0],x1,w0,acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1],x1,w1,acc[1][1]);
            simdgroup_multiply_accumulate(acc[1][2],x1,w2,acc[1][2]);
            simdgroup_multiply_accumulate(acc[1][3],x1,w3,acc[1][3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (i0+64<=m && j0+32<=n) {
        int wi=i0+(int(sg)&1)*32, tj=j0+(int(sg)>>1)*16;
        for (int tt=0;tt<2;++tt)
            for (int ww=0;ww<4;++ww)
                simdgroup_store(acc[tt][ww],C+(size_t)(tj+tt*8)*ldc+wi+ww*8,
                                ldc,ulong2(0,0),false);
        return;
    }
    threadgroup float* Cs=tile;
    int wi=(int(sg)&1)*32, tj=(int(sg)>>1)*16;
    for (int tt=0;tt<2;++tt)
        for (int ww=0;ww<4;++ww)
            simdgroup_store(acc[tt][ww],Cs+(size_t)(wi+ww*8)*32+tj+tt*8,
                            32,ulong2(0,0),true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e=int(lid);e<64*32;e+=128) {
        int row=e/32, col=e%32;
        if (i0+row<m && j0+col<n) C[(size_t)(j0+col)*ldc+i0+row]=Cs[e];
    }
}

)MSL";

struct jam_metal {
    id<MTLDevice>               dev;
    id<MTLCommandQueue>         queue;
    MTLCommandBufferDescriptor* command_desc;
    id<MTLComputePipelineState> f32, f16, bf16, q8, q4_0, mxfp4, q4k, q5k, q6k;
    id<MTLComputePipelineState> q8_mma, q8_mma_full, q4_0_mma, q4_0_mma_full;
    id<MTLComputePipelineState> q4k_mma_full, q5k_mma_full, q6k_mma_full;
    id<MTLComputePipelineState> q4_0p_mma_full, q4kp_mma_full, q5kp_mma_full, q6kp_mma_full;
    id<MTLComputePipelineState> q4_0p_mma, q4kp_mma, q5kp_mma, q6kp_mma;
    int profile;
    uint64_t profile_calls, profile_mma, profile_tiled, profile_failures;
    double profile_encode_s, profile_submit_s, profile_wait_s, profile_gpu_s, profile_total_s;
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
    int threadgroups;
} jam_metal_plan;

/* Borrow the VM pages containing a host tensor. The MTLBuffer view is released before the synchronous
 * jam_mm call returns; it never owns or deallocates the caller's memory. Rounding does not authorize
 * GPU access outside tensor bytes. */
static id<MTLBuffer> jam_host_buffer(jam_metal* m, const void* tensor, size_t bytes, size_t* offset) {
    uintptr_t addr=(uintptr_t)tensor;
    size_t page=(size_t)getpagesize();
    if (!tensor || !bytes) return nil;
    uintptr_t wrap_base=addr & ~(uintptr_t)(page-1);
    size_t poff=(size_t)(addr-wrap_base);
    if (bytes > SIZE_MAX-poff-(page-1)) return nil;
    size_t wrap_size=(poff+bytes+page-1) & ~(page-1);
    if (wrap_size > (size_t)m->dev.maxBufferLength) return nil;
    id<MTLBuffer> buffer=[m->dev newBufferWithBytesNoCopy:(void*)wrap_base
            length:wrap_size options:MTLResourceStorageModeShared deallocator:nil];
    if (buffer) *offset=poff;
    return buffer;
}

/* Every operand view is temporary: created by jam_host_buffer in prepare, released after the
 * flush waits (measured ~2 us per wrap against ms-scale prefill work - a persistent cache is
 * not worth its hidden state; ponytail: revisit only if profiling says otherwise). */
static void jam_metal_resources_release(jam_metal* m, jam_metal_resources* resources) {
    (void) m;
    if (!resources) return;
    [resources->weight.buffer release];
    [resources->activation.buffer release];
    [resources->result.buffer release];
    memset(resources,0,sizeof *resources);
}

static void jam_metal_set_arguments(id<MTLComputeCommandEncoder> encoder,
                                    id<MTLComputePipelineState> pipeline,
                                    const jam_metal_resources* resources,
                                    const jam_metal_mm_params* params) {
    id<MTLBuffer> bound[3]={resources->weight.buffer,resources->activation.buffer,
                            resources->result.buffer};
    NSUInteger offsets[3]={resources->weight.offset,resources->activation.offset,
                           resources->result.offset};
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffers:bound offsets:offsets withRange:NSMakeRange(0,3)];
    [encoder setBytes:params length:sizeof *params atIndex:3];
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
        m->f32   = jam_pipe(dev, lib, "f32_mm");
        m->f16   = jam_pipe(dev, lib, "f16_mm");
        m->bf16  = jam_pipe(dev, lib, "bf16_mm");
        m->q8    = jam_pipe(dev, lib, "q8_0_mm");
        m->q4_0  = jam_pipe(dev, lib, "q4_0_mm");
        m->mxfp4 = jam_pipe(dev, lib, "mxfp4_mm");
        m->q4k   = jam_pipe(dev, lib, "q4k_mm");
        m->q5k   = jam_pipe(dev, lib, "q5k_mm");
        m->q6k   = jam_pipe(dev, lib, "q6k_mm");
        m->q8_mma = jam_pipe(dev, lib, "q8_0_mma");
        m->q8_mma_full = jam_pipe(dev, lib, "q8_0_mma_full");
        m->q4_0_mma = jam_pipe(dev, lib, "q4_0_mma");
        m->q4_0_mma_full = jam_pipe(dev, lib, "q4_0_mma_full");
        m->q4k_mma_full = jam_pipe(dev, lib, "q4k_mma_full");
        m->q5k_mma_full = jam_pipe(dev, lib, "q5k_mma_full");
        m->q6k_mma_full = jam_pipe(dev, lib, "q6k_mma_full");
        m->q4_0p_mma_full = jam_pipe(dev, lib, "q4_0p_mma_full");
        m->q4_0p_mma = jam_pipe(dev, lib, "q4_0p_mma");
        m->q4kp_mma = jam_pipe(dev, lib, "q4kp_mma");
        m->q5kp_mma = jam_pipe(dev, lib, "q5kp_mma");
        m->q6kp_mma = jam_pipe(dev, lib, "q6kp_mma");
        m->q4kp_mma_full = jam_pipe(dev, lib, "q4kp_mma_full");
        m->q5kp_mma_full = jam_pipe(dev, lib, "q5kp_mma_full");
        m->q6kp_mma_full = jam_pipe(dev, lib, "q6kp_mma_full");
        [lib release];
        if (!m->queue || !m->f32 || !m->f16 || !m->bf16 || !m->q8 || !m->q4_0 || !m->mxfp4 ||
            !m->q4k || !m->q5k || !m->q6k || !m->q8_mma || !m->q8_mma_full ||
            !m->q4_0_mma || !m->q4_0_mma_full ||
            !m->q4k_mma_full || !m->q5k_mma_full || !m->q6k_mma_full ||
            !m->q4_0p_mma_full || !m->q4kp_mma_full || !m->q5kp_mma_full || !m->q6kp_mma_full ||
            !m->q4_0p_mma || !m->q4kp_mma || !m->q5kp_mma || !m->q6kp_mma) {
            jam_metal_destroy(m); return NULL;
        }
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
    [m->f32 release]; [m->f16 release]; [m->bf16 release]; [m->q8 release]; [m->q4_0 release];
    [m->mxfp4 release]; [m->q4k release]; [m->q5k release]; [m->q6k release];
    [m->q8_mma release]; [m->q8_mma_full release];
    [m->q4_0_mma release]; [m->q4_0_mma_full release];
    [m->q4k_mma_full release]; [m->q5k_mma_full release]; [m->q6k_mma_full release];
    [m->q4_0p_mma_full release]; [m->q4_0p_mma release]; [m->q4kp_mma release];
    [m->q5kp_mma release]; [m->q6kp_mma release]; [m->q4kp_mma_full release];
    [m->q5kp_mma_full release]; [m->q6kp_mma_full release];
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
    if (!m || !resources || !plan) return JAM_EINVAL;
    if (bt != JAM_F32 || ct != JAM_F32) return JAM_EUNSUPPORTED;
    if (lda < K || ldb < K)             return JAM_EINVAL;

    /* Packed weights (jam.h JAM_PACK_ABI): MMA-aligned prefill shapes only; anything else falls
     * back to the packed CPU kernels (EUNSUPPORTED here is routing, not failure). Group bytes
     * mirror jam_pack_group_bytes; packed rows are dense, so lda must be exactly K. */
    if (at & JAM_PACKED) {
        size_t gb = jam_pack_group_bytes((jam_dtype)(at & ~JAM_PACKED), K);
        id<MTLComputePipelineState> ppipe = nil; int pblk = 256;
        switch (at & ~JAM_PACKED) {
            case JAM_Q4_0: ppipe = m->q4_0p_mma_full; pblk = 32; break;
            case JAM_Q4_K: ppipe = m->q4kp_mma_full;  break;
            case JAM_Q5_K: ppipe = m->q5kp_mma_full;  break;
            case JAM_Q6_K: ppipe = m->q6kp_mma_full;  break;
            default: return JAM_EUNSUPPORTED;
        }
        if (!ppipe || N < JAM_MMA_MIN_N || K % pblk || lda != K) return JAM_EUNSUPPORTED;
        if (M % 64 || N % 32) {
            switch (at & ~JAM_PACKED) {   /* guarded edge variants (e.g. MoE expert slices) */
                case JAM_Q4_0: ppipe = m->q4_0p_mma; break;
                case JAM_Q4_K: ppipe = m->q4kp_mma;  break;
                case JAM_Q5_K: ppipe = m->q5kp_mma;  break;
                default:       ppipe = m->q6kp_mma;  break;
            }
        }
        size_t arow_ = (size_t) K * sizeof(float), asrc_ = (size_t) ldb * sizeof(float);
        resources->weight.buffer = jam_host_buffer(m, a, (size_t) (M / 4) * gb, &resources->weight.offset);
        resources->activation.buffer = jam_host_buffer(m, b, (size_t) (N - 1) * asrc_ + arow_, &resources->activation.offset);
        resources->result.buffer = jam_host_buffer(m, c, ((size_t) (N - 1) * ldc + M) * sizeof(float), &resources->result.offset);
        if (!resources->weight.buffer || !resources->activation.buffer || !resources->result.buffer)
            return JAM_EUNSUPPORTED;
        plan->params = (jam_metal_mm_params){M, N, K, lda, ldb, ldc};
        plan->pipeline = ppipe;
        plan->grid = MTLSizeMake((M + 63) / 64, (N + 31) / 32, 1);
        plan->threads = MTLSizeMake(128, 1, 1);
        plan->threadgroups = 1;
        ++m->profile_mma;
        return JAM_OK;
    }

    /* dtype -> (pipeline, bytes-per-block, element granularity that K + the weight stride must divide). */
    id<MTLComputePipelineState> pipe = nil; int blk = 1; size_t bpb = 0;
    switch (at) {
        case JAM_F32:   pipe=m->f32;   bpb=4;   blk=1;   break;
        case JAM_F16:   pipe=m->f16;   bpb=2;   blk=1;   break;
        case JAM_BF16:  pipe=m->bf16;  bpb=2;   blk=1;   break;
        case JAM_Q8_0:  pipe=m->q8;    bpb=34;  blk=32;  break;
        case JAM_Q4_0:  pipe=m->q4_0;  bpb=18;  blk=32;  break;
        case JAM_MXFP4: pipe=m->mxfp4; bpb=17;  blk=32;  break;
        case JAM_Q4_K:  pipe=m->q4k;   bpb=144; blk=256; break;
        case JAM_Q5_K:  pipe=m->q5k;   bpb=176; blk=256; break;
        case JAM_Q6_K:  pipe=m->q6k;   bpb=210; blk=256; break;
        default: return JAM_EUNSUPPORTED;
    }
    if (!pipe)                     return JAM_EUNSUPPORTED;
    /* Match CPU dispatch: a quantized shape outside its block granularity is a valid call for which
     * no kernel exists, not a malformed leading dimension. jam_mm already rejected lda < K above. */
    if (K % blk || lda % blk)      return JAM_EUNSUPPORTED;

    size_t wrow=(size_t)(K/blk)*bpb, wsrc=(size_t)(lda/blk)*bpb;
    size_t arow=(size_t)K*sizeof(float), asrc=(size_t)ldb*sizeof(float);
    size_t asz=(size_t)(M-1)*wsrc+wrow;
    size_t bsz=(size_t)(N-1)*asrc+arow;
    size_t csz=((size_t)(N-1)*ldc+M)*sizeof(float);
    /* Temporary page-views over the caller's W/A/C — the zero-copy path. Measured on M3 Pro:
     * ~2 us per wrap against ms-scale prefill GPU work; a persistent-buffer cache is NOT worth
     * its hidden state at these numbers (ponytail: revisit only if profiling says otherwise). */
    resources->weight.buffer=jam_host_buffer(m,a,asz,&resources->weight.offset);
    resources->activation.buffer=jam_host_buffer(m,b,bsz,&resources->activation.offset);
    resources->result.buffer=jam_host_buffer(m,c,csz,&resources->result.offset);
    if (!resources->weight.buffer || !resources->activation.buffer || !resources->result.buffer)
        return JAM_EUNSUPPORTED;

    plan->params=(jam_metal_mm_params){M,N,K,lda,ldb,ldc};
    const bool full=(M%64)==0 && (N%32)==0;
    /* K-quants have no edge MMA variant: unaligned shapes stay on the tiled kernel. */
    id<MTLComputePipelineState> kq_mma=nil;
    if (full && N >= JAM_MMA_MIN_N) {
        switch (at) {
            case JAM_Q4_K: kq_mma=m->q4k_mma_full; break;
            case JAM_Q5_K: kq_mma=m->q5k_mma_full; break;
            case JAM_Q6_K: kq_mma=m->q6k_mma_full; break;
            default: break;
        }
    }
    if ((at == JAM_Q8_0 || at == JAM_Q4_0) && N >= JAM_MMA_MIN_N) {
        plan->pipeline = at == JAM_Q8_0
                ? (full ? m->q8_mma_full : m->q8_mma)
                : (full ? m->q4_0_mma_full : m->q4_0_mma);
        plan->grid=MTLSizeMake((M+63)/64,(N+31)/32,1);
        plan->threads=MTLSizeMake(128,1,1);
        plan->threadgroups=1;
        ++m->profile_mma;
    } else if (kq_mma) {
        plan->pipeline=kq_mma;
        plan->grid=MTLSizeMake(M/64,N/32,1);
        plan->threads=MTLSizeMake(128,1,1);
        plan->threadgroups=1;
        ++m->profile_mma;
    } else {
        plan->pipeline=pipe;
        plan->grid=MTLSizeMake((N+JAM_MTN-1)/JAM_MTN,M,1);
        plan->threads=MTLSizeMake(8,16,1);
        plan->threadgroups=0;
        ++m->profile_tiled;
    }
    return JAM_OK;
}

static void jam_metal_encode_mm(id<MTLComputeCommandEncoder> encoder,
                                const jam_metal_resources* resources,
                                const jam_metal_plan* plan) {
    jam_metal_set_arguments(encoder,plan->pipeline,resources,&plan->params);
    if (plan->threadgroups)
        [encoder dispatchThreadgroups:plan->grid threadsPerThreadgroup:plan->threads];
    else
        [encoder dispatchThreads:plan->grid threadsPerThreadgroup:plan->threads];
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
            id<MTLCommandBuffer> cb=m->command_desc
                    ? [m->queue commandBufferWithDescriptor:m->command_desc]
                    : [m->queue commandBuffer];
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
        jam_metal_resources_release(m,&resources);
        return status;
    }
}
