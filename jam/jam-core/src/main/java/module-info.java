/** JAM's matrix multiplication API and provider SPI. */
@SuppressWarnings("module")
module com.qxotic.jam {
    exports com.qxotic.jam;
    exports com.qxotic.jam.internal to
            com.qxotic.jam.libjam,
            com.qxotic.jam.scalar,
            com.qxotic.jam.vector;

    uses com.qxotic.jam.JAM.Provider;
}
