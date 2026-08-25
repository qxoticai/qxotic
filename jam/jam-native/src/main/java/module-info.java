/** The libjam native provider for JAM. */
module com.qxotic.jam.libjam {
    requires com.qxotic.jam;

    provides com.qxotic.jam.JAM.Provider with
            com.qxotic.jam.libjam.NativeJAMProvider;
}
