package com.qxotic.toknroll;

record TokenizerSourceStats(int exact, int checked, int encodeErrors, double exactRatio) {
    @Override
    public String toString() {
        return "exact=" + exact + "/" + checked + " err=" + encodeErrors + " ratio=" + exactRatio;
    }
}
