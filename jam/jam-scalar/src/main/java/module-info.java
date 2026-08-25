/** The portable scalar reference provider for JAM. */
module com.qxotic.jam.scalar {
    requires com.qxotic.jam;

    provides com.qxotic.jam.JAM.Provider with
            com.qxotic.jam.scalar.ScalarJAMProvider;
}
