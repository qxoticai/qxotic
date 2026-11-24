package com.qxotic.jota.stablehlo.attr;

/** StableHLO enum {@code #stablehlo<fft_type ...>}. */
public enum FftType implements StableHloEnum {
    FFT,
    IFFT,
    RFFT,
    IRFFT;

    public static final String KIND = "fft_type";

    @Override
    public String kind() {
        return KIND;
    }

    public static FftType from(Attribute attribute) {
        return StableHloEnum.decode(FftType.class, KIND, attribute);
    }
}
