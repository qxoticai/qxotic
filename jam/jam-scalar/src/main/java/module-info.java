/** The portable scalar reference provider for JAM. */
module com.qxotic.jam.scalar {
    requires com.qxotic.jam;
    requires static com.qxotic.graalvm.annotations;

    provides com.qxotic.jam.JAM.Provider with
            com.qxotic.jam.scalar.ScalarJAMProvider;
}
