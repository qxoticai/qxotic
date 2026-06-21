import org.graalvm.nativeimage.hosted.Feature;
import com.oracle.svm.core.jdk.PlatformNativeLibrarySupport;
import com.oracle.svm.hosted.FeatureImpl;
import com.oracle.svm.hosted.c.NativeLibraries;

/**
 * Statically links libjinferjni.a (AVX-512 GEMM) into the native image: native methods in
 * com.qxotic.jinfer.* bind to the direct JNI-mangled exports at image link time — no System.loadLibrary.
 * Build-time only; uses unexported SVM internals (see oracle/graal#3359 for the missing public API).
 * Pass -Dllama.staticGemm=true to the image build so Q8_0FloatTensor enables the native path.
 */
public class LFM25StaticGemmFeature implements Feature {
    @Override
    public void duringSetup(DuringSetupAccess access) {
        PlatformNativeLibrarySupport.singleton().addBuiltinPkgNativePrefix("com_qxotic_jinfer");
    }

    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        NativeLibraries nl = ((FeatureImpl.BeforeAnalysisAccessImpl) access).getNativeLibraries();
        nl.addStaticJniLibrary("jinferjni");
    }
}
