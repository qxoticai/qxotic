/** The Java Vector API provider for JAM. */
module com.qxotic.jam.vector {
    requires com.qxotic.jam;
    requires static com.qxotic.graalvm.annotations;
    requires jdk.incubator.vector;

    provides com.qxotic.jam.JAM.Provider with
            com.qxotic.jam.vector.VectorJAMProvider;
}
