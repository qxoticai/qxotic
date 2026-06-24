// Standalone parity check: chunked gated-DeltaNet vs the sequential recurrence (the oracle).
// No deps. javac bench/DeltaNetParity.java -d /tmp/dn && java -cp /tmp/dn DeltaNetParity
import java.util.Random;
import jdk.incubator.vector.*;

public class DeltaNetParity {
    static final VectorSpecies<Float> SP = FloatVector.SPECIES_PREFERRED;

    // Lever 1+2: same recurrence as seqOracle, but float[] state (no segment) + vectorized inner ops.
    // This is the per-head body that goes inside parallelFor(heads) in the model (heads independent).
    static float[] flippedVec(float[] K, float[] V, float[] Q, float[] beta, float[] a, float[] S0, float[] outO) {
        int Uw = SP.length();
        float[] S = S0.clone();
        float[] sk = new float[d], dd = new float[d];
        for (int t = 0; t < L; t++) {
            int ro = t*d; float at = a[t], bt = beta[t];
            for (int idx = 0; idx < d*d; idx += Uw)                       // decay: S *= a
                FloatVector.fromArray(SP, S, idx).mul(at).intoArray(S, idx);
            for (int j = 0; j < d; j++) {                                 // sk[j] = dot(S row j, k)
                var acc = FloatVector.zero(SP);
                for (int i = 0; i < d; i += Uw)
                    acc = FloatVector.fromArray(SP, S, j*d+i).fma(FloatVector.fromArray(SP, K, ro+i), acc);
                sk[j] = acc.reduceLanes(VectorOperators.ADD);
            }
            for (int i = 0; i < d; i++) dd[i] = (V[ro+i]-sk[i])*bt;       // d = beta(v - sk)
            for (int j = 0; j < d; j++) {                                 // S row j += dd[j]*k
                var djv = FloatVector.broadcast(SP, dd[j]);
                for (int i = 0; i < d; i += Uw)
                    FloatVector.fromArray(SP, K, ro+i).fma(djv, FloatVector.fromArray(SP, S, j*d+i)).intoArray(S, j*d+i);
            }
            for (int j = 0; j < d; j++) {                                 // out[j] = dot(S row j, q)
                var acc = FloatVector.zero(SP);
                for (int i = 0; i < d; i += Uw)
                    acc = FloatVector.fromArray(SP, S, j*d+i).fma(FloatVector.fromArray(SP, Q, ro+i), acc);
                outO[ro+j] = acc.reduceLanes(VectorOperators.ADD);
            }
        }
        return S;
    }

    static final int d = 128;   // headVDim
    static final int L = 512;   // seqLen
    static final int C = 64;    // chunk size

    // S indexed S[j*d+i] (row j, col i). Sequential oracle == exactly the Qwen35 inner loop.
    static float[] seqOracle(float[] K, float[] V, float[] Q, float[] beta, float[] a,
                             float[] S0, float[] outO) {
        float[] S = S0.clone();
        float[] sk = new float[d], dd = new float[d];
        for (int t = 0; t < L; t++) {
            float at = a[t], bt = beta[t];
            int ro = t * d;
            for (int idx = 0; idx < d * d; idx++) S[idx] *= at;        // decay
            for (int j = 0; j < d; j++) {                              // sk = S k
                float s = 0; for (int i = 0; i < d; i++) s += S[j*d+i]*K[ro+i]; sk[j] = s;
            }
            for (int i = 0; i < d; i++) dd[i] = (V[ro+i] - sk[i]) * bt; // d = beta(v - sk)
            for (int i = 0; i < d; i++) { float ki = K[ro+i];          // S += d k^T
                for (int j = 0; j < d; j++) S[j*d+i] += ki * dd[j];
            }
            for (int j = 0; j < d; j++) {                              // out = S q
                float s = 0; for (int i = 0; i < d; i++) s += S[j*d+i]*Q[ro+i]; outO[ro+j] = s;
            }
        }
        return S;
    }

    // Chunked gated form. Scalar matmuls (correctness first). Returns final state.
    static float[] chunked(float[] K, float[] V, float[] Q, float[] beta, float[] a,
                           float[] S0, float[] outO) {
        float[] S = S0.clone();
        for (int c0 = 0; c0 < L; c0 += C) {
            int n = Math.min(C, L - c0);
            // cumulative decay within chunk (relative to chunk start): gamma[t] = prod_{s<=t} a
            float[] g = new float[n];
            float acc = 1f;
            for (int t = 0; t < n; t++) { acc *= a[c0+t]; g[t] = acc; }
            // Vtilde = V/g ; M = beta*(Vtilde - K S0^T)
            float[] M = new float[n*d];
            for (int t = 0; t < n; t++) {
                int ro = (c0+t)*d; float bt = beta[c0+t], ig = 1f/g[t];
                for (int j = 0; j < d; j++) {
                    float ks0 = 0; for (int i = 0; i < d; i++) ks0 += S[j*d+i]*K[ro+i]; // (S0 k_t)[j]
                    M[t*d+j] = bt * (V[ro+j]*ig - ks0);
                }
            }
            // A[t,r] = beta_t (k_r . k_t), strictly lower; solve (I+A) U = M  (forward subst)
            float[] U = new float[n*d];
            for (int t = 0; t < n; t++) {
                int rt = (c0+t)*d; float bt = beta[c0+t];
                for (int j = 0; j < d; j++) U[t*d+j] = M[t*d+j];
                for (int r = 0; r < t; r++) {
                    float kk = 0; int rr = (c0+r)*d;
                    for (int i = 0; i < d; i++) kk += K[rr+i]*K[rt+i];
                    float coef = bt * kk;
                    for (int j = 0; j < d; j++) U[t*d+j] -= coef * U[r*d+j];
                }
            }
            // O = diag(g)[ Q S0^T + tril(Q K^T) U ]
            for (int t = 0; t < n; t++) {
                int rt = (c0+t)*d; float gt = g[t];
                for (int j = 0; j < d; j++) {
                    float s0q = 0; for (int i = 0; i < d; i++) s0q += S[j*d+i]*Q[rt+i]; // (S0 q_t)[j]
                    float intra = 0;
                    for (int r = 0; r <= t; r++) {
                        float qk = 0; int rr = (c0+r)*d;
                        for (int i = 0; i < d; i++) qk += Q[rt+i]*K[rr+i];
                        intra += qk * U[r*d+j];
                    }
                    outO[rt+j] = gt * (s0q + intra);
                }
            }
            // S_L = g_L ( S0 + U^T K )
            float gL = g[n-1];
            float[] Snew = new float[d*d];
            for (int j = 0; j < d; j++) for (int i = 0; i < d; i++) {
                float utk = 0; for (int t = 0; t < n; t++) utk += U[t*d+j]*K[(c0+t)*d+i];
                Snew[j*d+i] = gL * (S[j*d+i] + utk);
            }
            S = Snew;
        }
        return S;
    }

    // --- vector primitives over the contiguous d (headVDim) dimension ---
    static float vdot(float[] A, int ao, float[] B, int bo, int n) {
        var acc = FloatVector.zero(SP); int i = 0, U = SP.length();
        for (; i + U <= n; i += U) acc = FloatVector.fromArray(SP, A, ao+i).fma(FloatVector.fromArray(SP, B, bo+i), acc);
        float s = acc.reduceLanes(VectorOperators.ADD);
        for (; i < n; i++) s += A[ao+i]*B[bo+i];
        return s;
    }
    static void vaxpy(float[] Y, int yo, float s, float[] X, int xo, int n) {   // Y += s*X
        var sv = FloatVector.broadcast(SP, s); int i = 0, U = SP.length();
        for (; i + U <= n; i += U) FloatVector.fromArray(SP, X, xo+i).fma(sv, FloatVector.fromArray(SP, Y, yo+i)).intoArray(Y, yo+i);
        for (; i < n; i++) Y[yo+i] += s*X[xo+i];
    }
    static void vscale(float[] Y, int yo, float s, int n) {
        var sv = FloatVector.broadcast(SP, s); int i = 0, U = SP.length();
        for (; i + U <= n; i += U) FloatVector.fromArray(SP, Y, yo+i).mul(sv).intoArray(Y, yo+i);
        for (; i < n; i++) Y[yo+i] *= s;
    }

    // Chunked gated DeltaNet, Vector API. Same math as chunked(); all d-loops vectorized.
    static float[] chunkedVec(float[] K, float[] V, float[] Q, float[] beta, float[] a, float[] S0, float[] outO) {
        float[] S = S0.clone();
        float[] g = new float[C], M = new float[C*d], Uu = new float[C*d];
        for (int c0 = 0; c0 < L; c0 += C) {
            int n = Math.min(C, L - c0);
            float acc = 1f; for (int t = 0; t < n; t++) { acc *= a[c0+t]; g[t] = acc; }
            for (int t = 0; t < n; t++) {                                   // M = beta(V/g - K S0^T)
                int rt = (c0+t)*d; float bt = beta[c0+t], ig = 1f/g[t];
                for (int j = 0; j < d; j++) M[t*d+j] = bt*(V[rt+j]*ig - vdot(K, rt, S, j*d, d));
            }
            for (int t = 0; t < n; t++) {                                   // solve (I+A)U=M, forward subst
                int rt = (c0+t)*d; float bt = beta[c0+t];
                System.arraycopy(M, t*d, Uu, t*d, d);
                for (int r = 0; r < t; r++) vaxpy(Uu, t*d, -bt*vdot(K, rt, K, (c0+r)*d, d), Uu, r*d, d);
            }
            for (int t = 0; t < n; t++) {                                   // O = g(Q S0^T + tril(QK^T)U)
                int rt = (c0+t)*d; float gt = g[t];
                for (int j = 0; j < d; j++) outO[rt+j] = vdot(Q, rt, S, j*d, d);
                for (int r = 0; r <= t; r++) vaxpy(outO, rt, vdot(Q, rt, K, (c0+r)*d, d), Uu, r*d, d);
                vscale(outO, rt, gt, d);
            }
            float gL = g[n-1];                                              // S_L = gL(S0 + U^T K)
            for (int j = 0; j < d; j++) {
                int row = j*d;
                for (int t = 0; t < n; t++) vaxpy(S, row, Uu[t*d+j], K, (c0+t)*d, d);
                vscale(S, row, gL, d);
            }
        }
        return S;
    }

    // Register-tiled C[i,j] = sum_k A[aB + i*lda + k] * B[bB + j*ldb + k]  (contract over contiguous k=K).
    // 4x4 tile: 16 vector accumulators, each loaded A/B vector reused 4x (AI ~2 vs vdot's ~1). Scalar tails.
    static void abt(float[] A, int aB, int lda, float[] B, int bB, int ldb, float[] C, int cB, int ldc, int I, int J, int K) {
        int U = SP.length(), i = 0;
        for (; i + 4 <= I; i += 4) {
            int ai0=aB+i*lda, ai1=ai0+lda, ai2=ai1+lda, ai3=ai2+lda, j = 0;
            for (; j + 4 <= J; j += 4) {
                int bj0=bB+j*ldb, bj1=bj0+ldb, bj2=bj1+ldb, bj3=bj2+ldb;
                var c00=FloatVector.zero(SP);var c01=c00;var c02=c00;var c03=c00;
                var c10=c00;var c11=c00;var c12=c00;var c13=c00;
                var c20=c00;var c21=c00;var c22=c00;var c23=c00;
                var c30=c00;var c31=c00;var c32=c00;var c33=c00;
                for (int k = 0; k + U <= K; k += U) {
                    var a0=FloatVector.fromArray(SP,A,ai0+k);var a1=FloatVector.fromArray(SP,A,ai1+k);
                    var a2=FloatVector.fromArray(SP,A,ai2+k);var a3=FloatVector.fromArray(SP,A,ai3+k);
                    var b0=FloatVector.fromArray(SP,B,bj0+k);var b1=FloatVector.fromArray(SP,B,bj1+k);
                    var b2=FloatVector.fromArray(SP,B,bj2+k);var b3=FloatVector.fromArray(SP,B,bj3+k);
                    c00=a0.fma(b0,c00);c01=a0.fma(b1,c01);c02=a0.fma(b2,c02);c03=a0.fma(b3,c03);
                    c10=a1.fma(b0,c10);c11=a1.fma(b1,c11);c12=a1.fma(b2,c12);c13=a1.fma(b3,c13);
                    c20=a2.fma(b0,c20);c21=a2.fma(b1,c21);c22=a2.fma(b2,c22);c23=a2.fma(b3,c23);
                    c30=a3.fma(b0,c30);c31=a3.fma(b1,c31);c32=a3.fma(b2,c32);c33=a3.fma(b3,c33);
                }
                int r0=cB+i*ldc+j, r1=r0+ldc, r2=r1+ldc, r3=r2+ldc;
                C[r0]=c00.reduceLanes(VectorOperators.ADD);C[r0+1]=c01.reduceLanes(VectorOperators.ADD);C[r0+2]=c02.reduceLanes(VectorOperators.ADD);C[r0+3]=c03.reduceLanes(VectorOperators.ADD);
                C[r1]=c10.reduceLanes(VectorOperators.ADD);C[r1+1]=c11.reduceLanes(VectorOperators.ADD);C[r1+2]=c12.reduceLanes(VectorOperators.ADD);C[r1+3]=c13.reduceLanes(VectorOperators.ADD);
                C[r2]=c20.reduceLanes(VectorOperators.ADD);C[r2+1]=c21.reduceLanes(VectorOperators.ADD);C[r2+2]=c22.reduceLanes(VectorOperators.ADD);C[r2+3]=c23.reduceLanes(VectorOperators.ADD);
                C[r3]=c30.reduceLanes(VectorOperators.ADD);C[r3+1]=c31.reduceLanes(VectorOperators.ADD);C[r3+2]=c32.reduceLanes(VectorOperators.ADD);C[r3+3]=c33.reduceLanes(VectorOperators.ADD);
            }
            for (; j < J; j++) for (int ii = 0; ii < 4; ii++) C[cB+(i+ii)*ldc+j] = vdot(A, aB+(i+ii)*lda, B, bB+j*ldb, K);
        }
        for (; i < I; i++) for (int j = 0; j < J; j++) C[cB+i*ldc+j] = vdot(A, aB+i*lda, B, bB+j*ldb, K);
    }

    // vdot-loop equivalent (the current chunked approach) for the same C = A B^T.
    static void abtNaive(float[] A, int aB, int lda, float[] B, int bB, int ldb, float[] C, int cB, int ldc, int I, int J, int K) {
        for (int i = 0; i < I; i++) for (int j = 0; j < J; j++) C[cB+i*ldc+j] = vdot(A, aB+i*lda, B, bB+j*ldb, K);
    }

    // i-tiled broadcast GEMM: C[i,j]=sum_k A[i,k]*Bt[k,j]. 4 output rows x 4 j-vectors = 16 accumulators;
    // Bt[k,:] loaded ONCE and reused across the 4 rows (AI ~ 4). Zero horizontal reduces. (I,J mult of 4/4U.)
    static void abMul(float[] A, int aB, int lda, float[] Bt, int btB, int ldbt, float[] C, int cB, int ldc, int I, int J, int K) {
        int U = SP.length(), JV = 4*U, i = 0;
        for (; i + 4 <= I; i += 4) {
            int a0=aB+i*lda, a1=a0+lda, a2=a1+lda, a3=a2+lda;
            int c0=cB+i*ldc, c1=c0+ldc, c2=c1+ldc, c3=c2+ldc, j = 0;
            for (; j + JV <= J; j += JV) {
                var x00=FloatVector.zero(SP);var x01=x00;var x02=x00;var x03=x00;
                var x10=x00;var x11=x00;var x12=x00;var x13=x00;
                var x20=x00;var x21=x00;var x22=x00;var x23=x00;
                var x30=x00;var x31=x00;var x32=x00;var x33=x00;
                for (int k = 0; k < K; k++) {
                    int bk = btB + k*ldbt + j;
                    var b0=FloatVector.fromArray(SP,Bt,bk);var b1=FloatVector.fromArray(SP,Bt,bk+U);
                    var b2=FloatVector.fromArray(SP,Bt,bk+2*U);var b3=FloatVector.fromArray(SP,Bt,bk+3*U);
                    var v=FloatVector.broadcast(SP,A[a0+k]); x00=v.fma(b0,x00);x01=v.fma(b1,x01);x02=v.fma(b2,x02);x03=v.fma(b3,x03);
                    v=FloatVector.broadcast(SP,A[a1+k]); x10=v.fma(b0,x10);x11=v.fma(b1,x11);x12=v.fma(b2,x12);x13=v.fma(b3,x13);
                    v=FloatVector.broadcast(SP,A[a2+k]); x20=v.fma(b0,x20);x21=v.fma(b1,x21);x22=v.fma(b2,x22);x23=v.fma(b3,x23);
                    v=FloatVector.broadcast(SP,A[a3+k]); x30=v.fma(b0,x30);x31=v.fma(b1,x31);x32=v.fma(b2,x32);x33=v.fma(b3,x33);
                }
                x00.intoArray(C,c0+j);x01.intoArray(C,c0+j+U);x02.intoArray(C,c0+j+2*U);x03.intoArray(C,c0+j+3*U);
                x10.intoArray(C,c1+j);x11.intoArray(C,c1+j+U);x12.intoArray(C,c1+j+2*U);x13.intoArray(C,c1+j+3*U);
                x20.intoArray(C,c2+j);x21.intoArray(C,c2+j+U);x22.intoArray(C,c2+j+2*U);x23.intoArray(C,c2+j+3*U);
                x30.intoArray(C,c3+j);x31.intoArray(C,c3+j+U);x32.intoArray(C,c3+j+2*U);x33.intoArray(C,c3+j+3*U);
            }
            for (; j < J; j++) for (int ii=0;ii<4;ii++){ float s=0; for(int k=0;k<K;k++) s+=A[aB+(i+ii)*lda+k]*Bt[btB+k*ldbt+j]; C[cB+(i+ii)*ldc+j]=s; }
        }
        for (; i < I; i++) for (int j=0;j<J;j++){ float s=0; for(int k=0;k<K;k++) s+=A[aB+i*lda+k]*Bt[btB+k*ldbt+j]; C[cB+i*ldc+j]=s; }
    }
    static void transpose(float[] B, int J, int K, float[] Bt) {   // [j][k] -> [k][j], blocked (cache-friendly)
        final int BL = 16;
        for (int j0=0;j0<J;j0+=BL) for (int k0=0;k0<K;k0+=BL) {
            int jE=Math.min(j0+BL,J), kE=Math.min(k0+BL,K);
            for (int j=j0;j<jE;j++) for (int k=k0;k<kE;k++) Bt[k*J+j]=B[j*K+k];
        }
    }

    static void microbench() {
        int I = 64, J = 128, K = 128;   // K S0^T shape: n x d, contract d
        Random r = new Random(1);
        float[] A = new float[I*K], B = new float[J*K], Bt = new float[K*J], C1 = new float[I*J], C2 = new float[I*J];
        for (int i=0;i<A.length;i++) A[i]=(float)r.nextGaussian();
        for (int i=0;i<B.length;i++) B[i]=(float)r.nextGaussian();
        transpose(B,J,K,Bt);
        abtNaive(A,0,K,B,0,K,C1,0,J,I,J,K); abMul(A,0,K,Bt,0,J,C2,0,J,I,J,K);
        double md=0; for(int i=0;i<C1.length;i++) md=Math.max(md,Math.abs(C1[i]-C2[i]));
        long iters=200000;
        for (long w=0; w<5000; w++){ abtNaive(A,0,K,B,0,K,C1,0,J,I,J,K); abMul(A,0,K,Bt,0,J,C2,0,J,I,J,K); transpose(B,J,K,Bt); }
        long t0=System.nanoTime(); for(long it=0;it<iters;it++) abtNaive(A,0,K,B,0,K,C1,0,J,I,J,K); long t1=System.nanoTime();
        for(long it=0;it<iters;it++) abMul(A,0,K,Bt,0,J,C2,0,J,I,J,K); long t2=System.nanoTime();   // GEMM only, B pre-transposed
        for(long it=0;it<iters;it++) transpose(B,J,K,Bt); long t3=System.nanoTime();
        double pk = 2.0*I*J*K;
        System.out.printf("microbench %dx%dx%d  corr maxDiff=%.2e%n", I,J,K, md);
        System.out.printf("  vdot-loop   %.0f ns  (%.1f GFLOP/s)%n", (t1-t0)/(double)iters, pk/((t1-t0)/(double)iters));
        System.out.printf("  tiled-bcast %.0f ns  (%.1f GFLOP/s)  speedup %.2fx%n", (t2-t1)/(double)iters, pk/((t2-t1)/(double)iters), (t1-t0)/(double)(t2-t1));
        System.out.printf("  transpose   %.0f ns  (amortized over M+O per chunk)%n", (t3-t2)/(double)iters);
        System.out.println("  JVM: " + System.getProperty("java.vm.name") + " / " + System.getProperty("java.vm.version"));
    }

    // contract-n broadcast: C[j,i] = sum_t U[t,j]*K[t,i]  (vectorize i, broadcast U[t,j]; tile 4 j-rows x 4 i-vec).
    static void utk(float[] U, int n, int d, float[] K, int kB, int kld, float[] C, int cB, int cld) {
        int Uw = SP.length(), IV = 4*Uw, j = 0;
        for (; j + 4 <= d; j += 4) {
            int i = 0;
            for (; i + IV <= d; i += IV) {
                var c00=FloatVector.zero(SP);var c01=c00;var c02=c00;var c03=c00;
                var c10=c00;var c11=c00;var c12=c00;var c13=c00;
                var c20=c00;var c21=c00;var c22=c00;var c23=c00;
                var c30=c00;var c31=c00;var c32=c00;var c33=c00;
                for (int t = 0; t < n; t++) {
                    int kt = kB + t*kld + i, ut = t*d + j;
                    var k0=FloatVector.fromArray(SP,K,kt);var k1=FloatVector.fromArray(SP,K,kt+Uw);
                    var k2=FloatVector.fromArray(SP,K,kt+2*Uw);var k3=FloatVector.fromArray(SP,K,kt+3*Uw);
                    var u=FloatVector.broadcast(SP,U[ut]);   c00=u.fma(k0,c00);c01=u.fma(k1,c01);c02=u.fma(k2,c02);c03=u.fma(k3,c03);
                    u=FloatVector.broadcast(SP,U[ut+1]); c10=u.fma(k0,c10);c11=u.fma(k1,c11);c12=u.fma(k2,c12);c13=u.fma(k3,c13);
                    u=FloatVector.broadcast(SP,U[ut+2]); c20=u.fma(k0,c20);c21=u.fma(k1,c21);c22=u.fma(k2,c22);c23=u.fma(k3,c23);
                    u=FloatVector.broadcast(SP,U[ut+3]); c30=u.fma(k0,c30);c31=u.fma(k1,c31);c32=u.fma(k2,c32);c33=u.fma(k3,c33);
                }
                int r0=cB+j*cld+i,r1=r0+cld,r2=r1+cld,r3=r2+cld;
                c00.intoArray(C,r0);c01.intoArray(C,r0+Uw);c02.intoArray(C,r0+2*Uw);c03.intoArray(C,r0+3*Uw);
                c10.intoArray(C,r1);c11.intoArray(C,r1+Uw);c12.intoArray(C,r1+2*Uw);c13.intoArray(C,r1+3*Uw);
                c20.intoArray(C,r2);c21.intoArray(C,r2+Uw);c22.intoArray(C,r2+2*Uw);c23.intoArray(C,r2+3*Uw);
                c30.intoArray(C,r3);c31.intoArray(C,r3+Uw);c32.intoArray(C,r3+2*Uw);c33.intoArray(C,r3+3*Uw);
            }
            for (; i < d; i++) for (int jj=0;jj<4;jj++){ float s=0; for(int t=0;t<n;t++) s+=U[t*d+j+jj]*K[kB+t*kld+i]; C[cB+(j+jj)*cld+i]=s; }
        }
    }

    // Phase 1: KS0^T, QS0^T via abMul(+S0T transpose); U^T K via utk; the n^2 parts stay vdot/vaxpy.
    static float[] chunkedTiled(float[] K, float[] V, float[] Q, float[] beta, float[] a, float[] S0, float[] outO) {
        float[] S = S0.clone();
        float[] g = new float[C], M = new float[C*d], S0T = new float[d*d], KS0 = new float[C*d], QS0 = new float[C*d], UtK = new float[d*d];
        for (int c0 = 0; c0 < L; c0 += C) {
            int n = Math.min(C, L - c0);
            float acc = 1f; for (int t=0;t<n;t++){ acc*=a[c0+t]; g[t]=acc; }
            transpose(S, d, d, S0T);                                   // S0T[i,j] = S[j,i]
            abMul(K, c0*d, d, S0T, 0, d, KS0, 0, d, n, d, d);          // KS0[t,j] = sum_i K[t,i] S0T[i,j]
            for (int t=0;t<n;t++){ float bt=beta[c0+t], ig=1f/g[t]; for(int j=0;j<d;j++) M[t*d+j]=bt*(V[(c0+t)*d+j]*ig - KS0[t*d+j]); }
            for (int t=0;t<n;t++){ int rt=(c0+t)*d; float bt=beta[c0+t]; for(int r=0;r<t;r++) vaxpy(M,t*d,-bt*vdot(K,rt,K,(c0+r)*d,d),M,r*d,d); }
            abMul(Q, c0*d, d, S0T, 0, d, QS0, 0, d, n, d, d);          // QS0[t,j] = sum_i Q[t,i] S0T[i,j]
            for (int t=0;t<n;t++){ int rt=(c0+t)*d; float gt=g[t];
                for(int j=0;j<d;j++) outO[rt+j]=QS0[t*d+j];
                for(int r=0;r<=t;r++) vaxpy(outO,rt,vdot(Q,rt,K,(c0+r)*d,d),M,r*d,d);
                vscale(outO,rt,gt,d); }
            utk(M, n, d, K, c0*d, d, UtK, 0, d);                       // UtK[j,i] = sum_t U[t,j] K[t,i]
            float gL=g[n-1];
            for(int j=0;j<d;j++) for(int i=0;i<d;i++) S[j*d+i]=gL*(S[j*d+i]+UtK[j*d+i]);
        }
        return S;
    }

    // Numerically-stable chunked form (the llama.cpp factoring): substitute W_t = gamma_t * U_t so the
    // unstable V/gamma (1/gamma overflows when gamma underflows) never appears. All decay factors are
    // gamma_t = prod_{s<=t} a (<=1) or ratios gamma_t/gamma_r = prod_{s=r+1..t} a (<=1, r<=t), built as
    // direct running products. Same tiled kernels (abMul/utk) as chunkedTiled; only decays change.
    //   W_t = beta(V_t - gamma_t (S0 k_t)); solve W_t -= beta(k_t.k_r)(gamma_t/gamma_r)W_r;
    //   O_t = gamma_t (S0 q_t) + sum_{r<=t}(q_t.k_r)(gamma_t/gamma_r)W_r; S = gamma_L S0 + sum_t (gamma_L/gamma_t)W_t k_t^T.
    static float[] chunkedStable(float[] K, float[] V, float[] Q, float[] beta, float[] a, float[] S0, float[] outO) {
        float[] S = S0.clone();
        float[] g = new float[C], D = new float[C*C], M = new float[C*d];
        float[] S0T = new float[d*d], KS0 = new float[C*d], QS0 = new float[C*d], UtK = new float[d*d];
        for (int c0 = 0; c0 < L; c0 += C) {
            int n = Math.min(C, L - c0);
            float acc = 1f; for (int t=0;t<n;t++){ acc*=a[c0+t]; g[t]=acc; }     // gamma_t (cumulative, <=1)
            for (int t=0;t<n;t++){ D[t*C+t]=1f; float p=1f;                       // D[t,r]=prod_{s=r+1..t}a (<=1)
                for (int r=t-1;r>=0;r--){ p*=a[c0+r+1]; D[t*C+r]=p; } }
            transpose(S, d, d, S0T);
            abMul(K, c0*d, d, S0T, 0, d, KS0, 0, d, n, d, d);                     // KS0[t,j]=(S0 k_t)[j]
            for (int t=0;t<n;t++){ float bt=beta[c0+t], gt=g[t];                  // W init: beta(V - gamma KS0)
                for (int j=0;j<d;j++) M[t*d+j]=bt*(V[(c0+t)*d+j] - gt*KS0[t*d+j]); }
            for (int t=0;t<n;t++){ int rt=(c0+t)*d; float bt=beta[c0+t];          // solve (decayed)
                for (int r=0;r<t;r++) vaxpy(M,t*d,-bt*vdot(K,rt,K,(c0+r)*d,d)*D[t*C+r],M,r*d,d); }
            abMul(Q, c0*d, d, S0T, 0, d, QS0, 0, d, n, d, d);                     // QS0[t,j]=(S0 q_t)[j]
            for (int t=0;t<n;t++){ int rt=(c0+t)*d; float gt=g[t];                // O = gamma QS0 + tril decayed
                for (int j=0;j<d;j++) outO[rt+j]=gt*QS0[t*d+j];
                for (int r=0;r<=t;r++) vaxpy(outO,rt,vdot(Q,rt,K,(c0+r)*d,d)*D[t*C+r],M,r*d,d); }
            for (int t=0;t<n;t++) vscale(M,t*d,D[(n-1)*C+t],d);                    // W -> (gamma_L/gamma_t)W
            utk(M, n, d, K, c0*d, d, UtK, 0, d);                                  // UtK[j,i]=sum_t Wd[t,j]K[t,i]
            float gL=g[n-1];
            for (int j=0;j<d;j++) for (int i=0;i<d;i++) S[j*d+i]=gL*S[j*d+i]+UtK[j*d+i];
        }
        return S;
    }

    // Phase 2: the n^2 dot products (k_t.k_r, q_t.k_r) are precomputed ONCE per chunk as matrices KK, QK
    // via the broadcast GEMM abMul (no horizontal reduces), instead of a reduce-bound vdot per (t,r) pair.
    // The solve and output then read matrix elements; only the vaxpy accumulations remain in the n^2 loops.
    static float[] chunkedPhase2(float[] K, float[] V, float[] Q, float[] beta, float[] a, float[] S0, float[] outO) {
        float[] S = S0.clone();
        float[] g = new float[C], D = new float[C*C], M = new float[C*d];
        float[] S0T = new float[d*d], KS0 = new float[C*d], QS0 = new float[C*d], UtK = new float[d*d];
        float[] Kt = new float[d*C], KK = new float[C*C], QK = new float[C*C];
        for (int c0 = 0; c0 < L; c0 += C) {
            int n = Math.min(C, L - c0);
            float acc = 1f; for (int t=0;t<n;t++){ acc*=a[c0+t]; g[t]=acc; }     // gamma_t
            for (int t=0;t<n;t++){ D[t*C+t]=1f; float p=1f;                       // D[t,r]=prod_{s=r+1..t}a
                for (int r=t-1;r>=0;r--){ p*=a[c0+r+1]; D[t*C+r]=p; } }
            transpose(S, d, d, S0T);
            for (int i=0;i<d;i++) for (int r=0;r<n;r++) Kt[i*n+r] = K[(c0+r)*d+i];  // Kt[i,r]=k_r[i] ([d x n])
            abMul(K, c0*d, d, S0T, 0, d, KS0, 0, d, n, d, d);                     // KS0[t,j]=(S0 k_t)[j]
            abMul(Q, c0*d, d, S0T, 0, d, QS0, 0, d, n, d, d);                     // QS0[t,j]=(S0 q_t)[j]
            abMul(K, c0*d, d, Kt,  0, n, KK,  0, n, n, n, d);                     // KK[t,r]=k_t.k_r
            abMul(Q, c0*d, d, Kt,  0, n, QK,  0, n, n, n, d);                     // QK[t,r]=q_t.k_r
            for (int t=0;t<n;t++){ float bt=beta[c0+t], gt=g[t];                  // W = beta(V - gamma KS0)
                for (int j=0;j<d;j++) M[t*d+j]=bt*(V[(c0+t)*d+j] - gt*KS0[t*d+j]); }
            for (int t=0;t<n;t++){ float bt=beta[c0+t];                           // solve (decayed), KK precomputed
                for (int r=0;r<t;r++) vaxpy(M,t*d,-bt*KK[t*n+r]*D[t*C+r],M,r*d,d); }
            for (int t=0;t<n;t++){ int rt=(c0+t)*d; float gt=g[t];                // O = gamma QS0 + tril decayed
                for (int j=0;j<d;j++) outO[rt+j]=gt*QS0[t*d+j];
                for (int r=0;r<=t;r++) vaxpy(outO,rt,QK[t*n+r]*D[t*C+r],M,r*d,d); }
            for (int t=0;t<n;t++) vscale(M,t*d,D[(n-1)*C+t],d);                    // W -> (gamma_L/gamma_t)W
            utk(M, n, d, K, c0*d, d, UtK, 0, d);
            float gL=g[n-1];
            for (int j=0;j<d;j++) for (int i=0;i<d;i++) S[j*d+i]=gL*S[j*d+i]+UtK[j*d+i];
        }
        return S;
    }

    // C[i,j] -= sum_k A[aB+i*lda+k]*B[bB+k*ldb+j]  (accumulate-subtract GEMM, broadcast form, scalar tail-free for our sizes)
    static void abMulSub(float[] A, int aB, int lda, float[] B, int bB, int ldb, float[] C, int cB, int ldc, int I, int J, int Kk) {
        int U = SP.length(), JV = 4*U, i = 0;
        for (; i + 4 <= I; i += 4) {
            int a0=aB+i*lda,a1=a0+lda,a2=a1+lda,a3=a2+lda, r0=cB+i*ldc,r1=r0+ldc,r2=r1+ldc,r3=r2+ldc, j = 0;
            for (; j + JV <= J; j += JV) {
                var x00=FloatVector.zero(SP);var x01=x00;var x02=x00;var x03=x00;var x10=x00;var x11=x00;var x12=x00;var x13=x00;
                var x20=x00;var x21=x00;var x22=x00;var x23=x00;var x30=x00;var x31=x00;var x32=x00;var x33=x00;
                for (int k = 0; k < Kk; k++) {
                    int bk = bB + k*ldb + j;
                    var b0=FloatVector.fromArray(SP,B,bk);var b1=FloatVector.fromArray(SP,B,bk+U);var b2=FloatVector.fromArray(SP,B,bk+2*U);var b3=FloatVector.fromArray(SP,B,bk+3*U);
                    var v=FloatVector.broadcast(SP,A[a0+k]);x00=v.fma(b0,x00);x01=v.fma(b1,x01);x02=v.fma(b2,x02);x03=v.fma(b3,x03);
                    v=FloatVector.broadcast(SP,A[a1+k]);x10=v.fma(b0,x10);x11=v.fma(b1,x11);x12=v.fma(b2,x12);x13=v.fma(b3,x13);
                    v=FloatVector.broadcast(SP,A[a2+k]);x20=v.fma(b0,x20);x21=v.fma(b1,x21);x22=v.fma(b2,x22);x23=v.fma(b3,x23);
                    v=FloatVector.broadcast(SP,A[a3+k]);x30=v.fma(b0,x30);x31=v.fma(b1,x31);x32=v.fma(b2,x32);x33=v.fma(b3,x33);
                }
                FloatVector.fromArray(SP,C,r0+j).sub(x00).intoArray(C,r0+j);FloatVector.fromArray(SP,C,r0+j+U).sub(x01).intoArray(C,r0+j+U);FloatVector.fromArray(SP,C,r0+j+2*U).sub(x02).intoArray(C,r0+j+2*U);FloatVector.fromArray(SP,C,r0+j+3*U).sub(x03).intoArray(C,r0+j+3*U);
                FloatVector.fromArray(SP,C,r1+j).sub(x10).intoArray(C,r1+j);FloatVector.fromArray(SP,C,r1+j+U).sub(x11).intoArray(C,r1+j+U);FloatVector.fromArray(SP,C,r1+j+2*U).sub(x12).intoArray(C,r1+j+2*U);FloatVector.fromArray(SP,C,r1+j+3*U).sub(x13).intoArray(C,r1+j+3*U);
                FloatVector.fromArray(SP,C,r2+j).sub(x20).intoArray(C,r2+j);FloatVector.fromArray(SP,C,r2+j+U).sub(x21).intoArray(C,r2+j+U);FloatVector.fromArray(SP,C,r2+j+2*U).sub(x22).intoArray(C,r2+j+2*U);FloatVector.fromArray(SP,C,r2+j+3*U).sub(x23).intoArray(C,r2+j+3*U);
                FloatVector.fromArray(SP,C,r3+j).sub(x30).intoArray(C,r3+j);FloatVector.fromArray(SP,C,r3+j+U).sub(x31).intoArray(C,r3+j+U);FloatVector.fromArray(SP,C,r3+j+2*U).sub(x32).intoArray(C,r3+j+2*U);FloatVector.fromArray(SP,C,r3+j+3*U).sub(x33).intoArray(C,r3+j+3*U);
            }
        }
        for (; i < I; i++) for (int j=0;j<J;j++){ float s=0; for(int k=0;k<Kk;k++) s+=A[aB+i*lda+k]*B[bB+k*ldb+j]; C[cB+i*ldc+j]-=s; }
    }

    // Phase 3: matmul-ize the n^2 accumulations. Solve via BLOCKED forward substitution (intra-block forward
    // subst, inter-block A_sub @ U_sub as GEMM -> shorter dependency chain); output intra term O = QKd @ U as a
    // GEMM. KK/QK precomputed (Phase 2). BL must divide C. Numerically identical to chunkedStable.
    static final int BL3 = 16;
    static float[] chunkedPhase3(float[] K, float[] V, float[] Q, float[] beta, float[] a, float[] S0, float[] outO) {
        float[] S = S0.clone();
        float[] g = new float[C], D = new float[C*C], M = new float[C*d];
        float[] S0T = new float[d*d], KS0 = new float[C*d], QS0 = new float[C*d], UtK = new float[d*d];
        float[] Kt = new float[d*C], KK = new float[C*C], QK = new float[C*C], Amat = new float[C*C], QKd = new float[C*C], Otmp = new float[C*d];
        for (int c0 = 0; c0 < L; c0 += C) {
            int n = Math.min(C, L - c0);
            float acc = 1f; for (int t=0;t<n;t++){ acc*=a[c0+t]; g[t]=acc; }
            for (int t=0;t<n;t++){ D[t*C+t]=1f; float p=1f; for (int r=t-1;r>=0;r--){ p*=a[c0+r+1]; D[t*C+r]=p; } }
            transpose(S, d, d, S0T);
            for (int i=0;i<d;i++) for (int r=0;r<n;r++) Kt[i*n+r] = K[(c0+r)*d+i];
            abMul(K, c0*d, d, S0T, 0, d, KS0, 0, d, n, d, d);
            abMul(Q, c0*d, d, S0T, 0, d, QS0, 0, d, n, d, d);
            abMul(K, c0*d, d, Kt,  0, n, KK,  0, n, n, n, d);
            abMul(Q, c0*d, d, Kt,  0, n, QK,  0, n, n, n, d);
            // A[t,r] = beta_t KK[t,r] D[t,r] (strictly lower, else 0); QKd[t,r] = QK D (lower incl diag, else 0)
            for (int t=0;t<n;t++){ float bt=beta[c0+t];
                for (int r=0;r<t;r++){ Amat[t*n+r]=bt*KK[t*n+r]*D[t*C+r]; QKd[t*n+r]=QK[t*n+r]*D[t*C+r]; }
                Amat[t*n+t]=0f; QKd[t*n+t]=QK[t*n+t];
                for (int r=t+1;r<n;r++){ Amat[t*n+r]=0f; QKd[t*n+r]=0f; } }
            for (int t=0;t<n;t++){ float bt=beta[c0+t], gt=g[t];                  // W = beta(V - gamma KS0)
                for (int j=0;j<d;j++) M[t*d+j]=bt*(V[(c0+t)*d+j] - gt*KS0[t*d+j]); }
            // blocked forward solve (I+A)U=M in place; BL3-row blocks; inter-block via abMulSub
            for (int bi=0; bi<n; bi+=BL3) {
                int bsz = Math.min(BL3, n-bi);
                for (int bj=0; bj<bi; bj+=BL3) abMulSub(Amat, bi*n+bj, n, M, bj*d, d, M, bi*d, d, bsz, d, BL3);
                for (int t=bi; t<bi+bsz; t++) for (int r=bi; r<t; r++) vaxpy(M, t*d, -Amat[t*n+r], M, r*d, d);
            }
            // O = gamma QS0 + QKd @ U   (GEMM)
            abMul(QKd, 0, n, M, 0, d, Otmp, 0, d, n, d, n);
            for (int t=0;t<n;t++){ int rt=(c0+t)*d; float gt=g[t]; for (int j=0;j<d;j++) outO[rt+j]=gt*QS0[t*d+j]+Otmp[t*d+j]; }
            for (int t=0;t<n;t++) vscale(M,t*d,D[(n-1)*C+t],d);
            utk(M, n, d, K, c0*d, d, UtK, 0, d);
            float gL=g[n-1];
            for (int j=0;j<d;j++) for (int i=0;i<d;i++) S[j*d+i]=gL*S[j*d+i]+UtK[j*d+i];
        }
        return S;
    }

    static void scanBench() {
        Random rng = new Random(7);
        float[] K=new float[L*d],V=new float[L*d],Q=new float[L*d],beta=new float[L],a=new float[L],S0=new float[d*d];
        for(int i=0;i<L*d;i++){K[i]=(float)rng.nextGaussian();V[i]=(float)rng.nextGaussian();Q[i]=(float)rng.nextGaussian();}
        float sc=(float)(1.0/Math.sqrt(d));
        for(int t=0;t<L;t++){float kn=0,qn=0;int ro=t*d;for(int i=0;i<d;i++){kn+=K[ro+i]*K[ro+i];qn+=Q[ro+i]*Q[ro+i];}float ki=(float)(1/Math.sqrt(kn+1e-6)),qi=(float)(1/Math.sqrt(qn+1e-6))*sc;for(int i=0;i<d;i++){K[ro+i]*=ki;Q[ro+i]*=qi;}}
        for(int t=0;t<L;t++){beta[t]=(float)rng.nextDouble();a[t]=(float)Math.exp(-0.05*rng.nextDouble());}
        for(int i=0;i<d*d;i++)S0[i]=(float)rng.nextGaussian()*0.1f;
        float[] o1=new float[L*d],o2=new float[L*d],o3=new float[L*d];
        float[] s1=seqOracle(K,V,Q,beta,a,S0,o1);
        float[] s2=chunkedStable(K,V,Q,beta,a,S0,o2);
        float[] s3=flippedVec(K,V,Q,beta,a,S0,o3);                  // sequential recurrence (llama.cpp CPU style)
        report("chunkedStable vs seq", o1,o2,s1,s2);
        report("flippedVec   vs seq", o1,o3,s1,s3);
        long it=4000;
        for(long w=0;w<400;w++){ chunkedStable(K,V,Q,beta,a,S0,o2); flippedVec(K,V,Q,beta,a,S0,o3); }
        long t0=System.nanoTime(); for(long x=0;x<it;x++) chunkedStable(K,V,Q,beta,a,S0,o2); long t1=System.nanoTime();
        for(long x=0;x<it;x++) flippedVec(K,V,Q,beta,a,S0,o3); long t2=System.nanoTime();
        System.out.printf("per-head scan: chunkedStable %.1f us   sequential(flippedVec) %.1f us   speedup %.2fx%n",
            (t1-t0)/1e3/it, (t2-t1)/1e3/it, (t1-t0)/(double)(t2-t1));
    }

    public static void main(String[] args) {
        if (args.length > 0 && args[0].equals("micro")) { microbench(); return; }
        if (args.length > 0 && args[0].equals("scan")) { scanBench(); return; }
        Random rng = new Random(42);
        float[] K = new float[L*d], V = new float[L*d], Q = new float[L*d];
        float[] beta = new float[L], a = new float[L], S0 = new float[d*d];
        for (int i = 0; i < L*d; i++) { K[i]=(float)rng.nextGaussian(); V[i]=(float)rng.nextGaussian(); Q[i]=(float)rng.nextGaussian(); }
        // model L2-normalizes k_t and q_t per head (qInv/kInv); replicate so the delta rule is contractive
        float scale = (float)(1.0/Math.sqrt(d));
        for (int t = 0; t < L; t++) {
            float kn=0, qn=0; int ro=t*d;
            for (int i=0;i<d;i++){ kn+=K[ro+i]*K[ro+i]; qn+=Q[ro+i]*Q[ro+i]; }
            float ki=(float)(1.0/Math.sqrt(kn+1e-6)), qi=(float)(1.0/Math.sqrt(qn+1e-6))*scale;
            for (int i=0;i<d;i++){ K[ro+i]*=ki; Q[ro+i]*=qi; }
        }
        for (int t = 0; t < L; t++) { beta[t]=(float)rng.nextDouble(); a[t]=(float)Math.exp(-0.05*rng.nextDouble()); } // a in (~0.95,1)
        for (int i = 0; i < d*d; i++) S0[i]=(float)rng.nextGaussian()*0.1f;

        float[] o1 = new float[L*d], o2 = new float[L*d], o3 = new float[L*d];
        float[] s1 = seqOracle(K,V,Q,beta,a,S0,o1);
        float[] s2 = chunked(K,V,Q,beta,a,S0,o2);
        float[] s3 = flippedVec(K,V,Q,beta,a,S0,o3);
        System.out.printf("d=%d L=%d C=%d  vec=%d-bit%n", d, L, C, SP.vectorBitSize());
        float[] o4 = new float[L*d];
        float[] s4 = chunkedVec(K,V,Q,beta,a,S0,o4);
        float[] o5 = new float[L*d], o6 = new float[L*d], o7 = new float[L*d];
        float[] s5 = chunkedStable(K,V,Q,beta,a,S0,o5);
        float[] s6 = chunkedPhase2(K,V,Q,beta,a,S0,o6);
        float[] s7 = chunkedPhase3(K,V,Q,beta,a,S0,o7);
        report("chunked     vs seq", o1,o2,s1,s2);
        report("flippedVec  vs seq", o1,o3,s1,s3);
        report("chunkedVec  vs seq", o1,o4,s1,s4);
        report("chunkedStable vs seq", o1,o5,s1,s5);
        report("chunkedPhase2 vs seq", o1,o6,s1,s6);
        report("chunkedPhase3 vs seq", o1,o7,s1,s7);

        // Extreme-gate case: strong decays (a in ~[e^-8, 1]) make gamma underflow within a chunk, so the
        // unstable chunked forms (V/gamma) overflow to NaN. The stable form must still match the oracle.
        Random eg = new Random(99);
        for (int t = 0; t < L; t++) a[t] = (float)Math.exp(-8.0*eg.nextDouble());   // log-gate in [-8,0]
        float[] eo1=new float[L*d], eo2=new float[L*d], eo3=new float[L*d];
        float[] eo4=new float[L*d];
        float[] es1=seqOracle(K,V,Q,beta,a,S0,eo1);
        float[] es2=chunkedTiled(K,V,Q,beta,a,S0,eo2);
        float[] es3=chunkedStable(K,V,Q,beta,a,S0,eo3);
        float[] es4=chunkedPhase2(K,V,Q,beta,a,S0,eo4);
        System.out.println("--- extreme gates (a in [e^-8,1], gamma underflows) ---");
        report("chunkedTiled  vs seq", eo1,eo2,es1,es2);   // expected: FAIL (NaN)
        report("chunkedStable vs seq", eo1,eo3,es1,es3);   // expected: PASS
        report("chunkedPhase2 vs seq", eo1,eo4,es1,es4);   // expected: PASS
    }
    static void report(String tag, float[] o1, float[] o2, float[] s1, float[] s2) {
        double maxO=0,sumO=0,maxS=0,refO=0;
        for (int i=0;i<o1.length;i++){ double e=Math.abs(o1[i]-o2[i]); maxO=Math.max(maxO,e); sumO+=e; refO+=Math.abs(o1[i]); }
        for (int i=0;i<s1.length;i++) maxS=Math.max(maxS,Math.abs(s1[i]-s2[i]));
        System.out.printf("%-20s output maxDiff=%.3e mean=%.3e (|ref|=%.3e)  state maxDiff=%.3e  %s%n",
            tag, maxO, sumO/o1.length, refO/o1.length, maxS, maxO<1e-2?"PASS":"FAIL");
    }
}
