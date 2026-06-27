/* jam_metal.mm — Apple GPU backend (opt-in via JAM_ISA=metal). A different executor from the CPU
 * row-range kernels: jam_mm routes supported dtypes here before the pool. Every quant kernel DEQUANTIZES
 * the weight on-GPU and dots the EXACT F32 B (so it matches the exact-B reference); decode mirrors
 * jam_kernels_generic.c. MSL compiled at runtime (no .metallib). Output is token-major C[j*m+i].
 *
 * FAST: each thread computes TN output columns for one weight row, decoding each weight block/sub-block
 * ONCE into registers and reusing it across all TN columns — so the (expensive) dequant is amortized TN×
 * instead of redone per output. Register-only (no threadgroup barriers). TN is the one tunable; bump it on
 * the Mac if occupancy allows. Threadgroup B-caching / simdgroup_matrix are the next step. APPLE-only TU. */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "jam_internal.h"
#include <string.h>
#include <stdlib.h>

#define JAM_MTN 8   /* output columns per GPU thread — the one tunable; injected into the MSL as TN */

static const char* JAM_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant float MXFP4_LUT[16] = { 0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12 };  /* ggml kvalues_mxfp4 */

/* thread (gid.x, gid.y) owns weight row i = gid.y and the TN columns [j0, j0+TN), j0 = gid.x*TN. */
#define JAM_TILE_PROLOGUE \
    int m=dim.x, n=dim.y, k=dim.z, i=int(gid.y), j0=int(gid.x)*TN; \
    if (i>=m || j0>=n) return; \
    float acc[TN]; for (int c=0;c<TN;++c) acc[c]=0.0f;
#define JAM_TILE_EPILOGUE \
    for (int c=0;c<TN;++c){ int j=j0+c; if (j<n) C[(size_t)j*m+i]=acc[c]; }

/* The 8 6-bit (scale,min) pairs of a Q4_K/Q5_K super-block (ggml packing); sb = scales[12]. */
static void kq_scales_mins(device const uchar* sb, thread uchar* sc, thread uchar* mn) {
    for (int t=0;t<4;++t){
        sc[t]=sb[t]&63; mn[t]=sb[t+4]&63;
        sc[t+4]=(sb[t+8]&0xF)|((sb[t]>>6)<<4);
        mn[t+4]=(sb[t+8]>>4)|((sb[t+4]>>6)<<4);
    }
}

/* Q8_0 block = { half d; char qs[32] } = 34 B. */
kernel void q8_0_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                    device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int nb=k/32;
    device const uchar* wrow = A + (size_t)i*nb*34;
    for (int b=0;b<nb;++b){
        device const uchar* blk = wrow + (size_t)b*34;
        float d = float(*(device const half*)blk);
        device const char* qs = (device const char*)(blk+2);
        char wq[32]; for (int e=0;e<32;++e) wq[e]=qs[e];           /* decode ONCE */
        for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
            device const float* bb = B + (size_t)j*k + b*32;
            float s=0.0f; for (int e=0;e<32;++e) s += float(wq[e])*bb[e];
            acc[c] += d*s;
        }
    }
    JAM_TILE_EPILOGUE
}

/* Q4_0 block = { half d; uchar qs[16] } = 18 B. value = d·(nibble-8); lo nibble -> e, hi -> e+16. */
kernel void q4_0_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                    device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int nb=k/32;
    device const uchar* wrow = A + (size_t)i*nb*18;
    for (int b=0;b<nb;++b){
        device const uchar* blk = wrow + (size_t)b*18;
        float d = float(*(device const half*)blk);
        device const uchar* qs = blk+2;
        float wq[32]; for (int e=0;e<16;++e){ wq[e]=float(qs[e]&0xF)-8.0f; wq[e+16]=float(qs[e]>>4)-8.0f; }
        for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
            device const float* bb = B + (size_t)j*k + b*32;
            float s=0.0f; for (int e=0;e<32;++e) s += wq[e]*bb[e];
            acc[c] += d*s;
        }
    }
    JAM_TILE_EPILOGUE
}

/* MXFP4 block = { uchar e; uchar qs[16] } = 17 B. value = (0.5·2^(e-127))·LUT[nibble]; lo -> e, hi -> e+16. */
kernel void mxfp4_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int nb=k/32;
    device const uchar* wrow = A + (size_t)i*nb*17;
    for (int b=0;b<nb;++b){
        device const uchar* blk = wrow + (size_t)b*17;
        float dh = 0.5f * exp2(float(blk[0]) - 127.0f);
        device const uchar* qs = blk+1;
        float wq[32]; for (int e=0;e<16;++e){ wq[e]=dh*MXFP4_LUT[qs[e]&0xF]; wq[e+16]=dh*MXFP4_LUT[qs[e]>>4]; }
        for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
            device const float* bb = B + (size_t)j*k + b*32;
            float s=0.0f; for (int e=0;e<32;++e) s += wq[e]*bb[e];
            acc[c] += s;
        }
    }
    JAM_TILE_EPILOGUE
}

/* Q4_K super-block = { half d; half dmin; uchar scales[12]; uchar qs[128] } = 144 B, 256 vals, 8 sub-blocks
 * of 32; sub-block s -> elements s*32, nibbles q[(s/2)*32 + e] (low if s even, else high), scale sc[s]. */
kernel void q4k_mm(device const uchar* A [[buffer(0)]], device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int ns=k/256;
    device const uchar* wrow = A + (size_t)i*ns*144;
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
                device const float* x = B + (size_t)j*k + (size_t)B2*256 + s*32;
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
                   device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int ns=k/256;
    device const uchar* wrow = A + (size_t)i*ns*176;
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
                device const float* x = B + (size_t)j*k + (size_t)B2*256 + s*32;
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
                   device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    int ns=k/256;
    device const uchar* wrow = A + (size_t)i*ns*210;
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
                    device const float* x = B + (size_t)j*k + (size_t)B2*256 + hg*32;
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
                   device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    device const float* arow = A + (size_t)i*k;
    for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
        device const float* bb = B + (size_t)j*k;
        float s=0.0f; for (int t=0;t<k;++t) s += arow[t]*bb[t];
        acc[c]=s;
    }
    JAM_TILE_EPILOGUE
}

kernel void f16_mm(device const half* A [[buffer(0)]], device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    device const half* arow = A + (size_t)i*k;
    for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
        device const float* bb = B + (size_t)j*k;
        float s=0.0f; for (int t=0;t<k;++t) s += float(arow[t])*bb[t];
        acc[c]=s;
    }
    JAM_TILE_EPILOGUE
}

kernel void bf16_mm(device const ushort* A [[buffer(0)]], device const float* B [[buffer(1)]],
                    device float* C [[buffer(2)]], constant int4& dim [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    JAM_TILE_PROLOGUE
    device const ushort* arow = A + (size_t)i*k;
    for (int c=0;c<TN;++c){ int j=j0+c; if (j>=n) break;
        device const float* bb = B + (size_t)j*k;
        float s=0.0f; for (int t=0;t<k;++t){ uint bits=uint(arow[t])<<16; s += as_type<float>(bits)*bb[t]; }
        acc[c]=s;
    }
    JAM_TILE_EPILOGUE
}
)MSL";

struct jam_metal {
    id<MTLDevice>               dev;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> f32, f16, bf16, q8, q4_0, mxfp4, q4k, q5k, q6k;
};

static id<MTLComputePipelineState> jam_pipe(id<MTLDevice> dev, id<MTLLibrary> lib, const char* name) {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) return nil;
    NSError* err = nil;
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    return p;
}

extern "C" jam_metal* jam_metal_create(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return NULL;
        NSError* err = nil;
        /* inject TN from the single host-side JAM_MTN so the tile width can't drift between C and MSL. */
        NSString* src = [NSString stringWithFormat:@"#define TN %d\n%s", JAM_MTN, JAM_MSL];
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
        if (!lib) { [dev release]; return NULL; }
        jam_metal* m = (jam_metal*) calloc(1, sizeof(jam_metal));
        m->dev   = dev;                                  /* +1 from Create, owned */
        m->queue = [dev newCommandQueue];
        m->f32   = jam_pipe(dev, lib, "f32_mm");
        m->f16   = jam_pipe(dev, lib, "f16_mm");
        m->bf16  = jam_pipe(dev, lib, "bf16_mm");
        m->q8    = jam_pipe(dev, lib, "q8_0_mm");
        m->q4_0  = jam_pipe(dev, lib, "q4_0_mm");
        m->mxfp4 = jam_pipe(dev, lib, "mxfp4_mm");
        m->q4k   = jam_pipe(dev, lib, "q4k_mm");
        m->q5k   = jam_pipe(dev, lib, "q5k_mm");
        m->q6k   = jam_pipe(dev, lib, "q6k_mm");
        [lib release];
        if (!m->queue || !m->f32 || !m->f16 || !m->bf16 || !m->q8 || !m->q4_0 || !m->mxfp4 ||
            !m->q4k || !m->q5k || !m->q6k) { jam_metal_destroy(m); return NULL; }
        return m;
    }
}

extern "C" void jam_metal_destroy(jam_metal* m) {
    if (!m) return;
    [m->f32 release]; [m->f16 release]; [m->bf16 release]; [m->q8 release]; [m->q4_0 release];
    [m->mxfp4 release]; [m->q4k release]; [m->q5k release]; [m->q6k release];
    [m->queue release]; [m->dev release];
    free(m);
}

extern "C" jam_status jam_metal_mm(jam_metal* m, const void* a, jam_dtype at, int lda,
                                   const void* b, jam_dtype bt, int ldb, void* c, jam_dtype ct, int ldc,
                                   int M, int N, int K) {
    if (bt != JAM_F32 || ct != JAM_F32) return JAM_EUNSUPPORTED;
    if (lda != K || ldb != K)           return JAM_EUNSUPPORTED;   /* kernels assume contiguous A,B -> CPU */

    /* dtype -> (pipeline, weight bytes, element granularity that K must divide). */
    id<MTLComputePipelineState> pipe = nil; size_t asz = 0; int blk = 1;
    switch (at) {
        case JAM_F32:   pipe=m->f32;   asz=(size_t)M*K*4;          blk=1;   break;
        case JAM_F16:   pipe=m->f16;   asz=(size_t)M*K*2;          blk=1;   break;
        case JAM_BF16:  pipe=m->bf16;  asz=(size_t)M*K*2;          blk=1;   break;
        case JAM_Q8_0:  pipe=m->q8;    asz=(size_t)M*(K/32)*34;    blk=32;  break;
        case JAM_Q4_0:  pipe=m->q4_0;  asz=(size_t)M*(K/32)*18;    blk=32;  break;
        case JAM_MXFP4: pipe=m->mxfp4; asz=(size_t)M*(K/32)*17;    blk=32;  break;
        case JAM_Q4_K:  pipe=m->q4k;   asz=(size_t)M*(K/256)*144;  blk=256; break;
        case JAM_Q5_K:  pipe=m->q5k;   asz=(size_t)M*(K/256)*176;  blk=256; break;
        case JAM_Q6_K:  pipe=m->q6k;   asz=(size_t)M*(K/256)*210;  blk=256; break;
        default: return JAM_EUNSUPPORTED;
    }
    if (!pipe)        return JAM_EUNSUPPORTED;
    if (K % blk != 0) return JAM_EINVAL;

    @autoreleasepool {
        size_t bsz = (size_t)N*K*sizeof(float), csz = (size_t)M*N*sizeof(float);
        id<MTLBuffer> ba = [m->dev newBufferWithBytes:a length:asz options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [m->dev newBufferWithBytes:b length:bsz options:MTLResourceStorageModeShared];
        id<MTLBuffer> bc = [m->dev newBufferWithLength:csz options:MTLResourceStorageModeShared];

        int dim[4] = { M, N, K, 0 };
        int gx = (N + JAM_MTN - 1) / JAM_MTN;   /* each thread does JAM_MTN columns */
        id<MTLCommandBuffer>         cb  = [m->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:ba offset:0 atIndex:0];
        [enc setBuffer:bb offset:0 atIndex:1];
        [enc setBuffer:bc offset:0 atIndex:2];
        [enc setBytes:dim length:sizeof(dim) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(gx, M, 1) threadsPerThreadgroup:MTLSizeMake(8, 16, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        const float* res = (const float*)[bc contents];
        float* dst = (float*) c;
        if (ldc == M) memcpy(dst, res, (size_t)M*N*sizeof(float));    /* contiguous: one shot */
        else for (int j=0;j<N;j++) memcpy(dst + (size_t)j*ldc, res + (size_t)j*M, (size_t)M*sizeof(float));

        [ba release]; [bb release]; [bc release];
    }
    return JAM_OK;
}
