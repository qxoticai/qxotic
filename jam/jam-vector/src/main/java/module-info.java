/** The Java Vector API provider for JAM. */
module com.qxotic.jam.vector {
    requires com.qxotic.jam;
    requires static com.qxotic.graalvm.annotations;
    requires static java.management; // JIT detection (Graal vs C2) for tile-shape selection
    requires jdk.incubator.vector;

    provides com.qxotic.jam.JAM.Provider with
            com.qxotic.jam.vector.VectorJAMProvider;
}
