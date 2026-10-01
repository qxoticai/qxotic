module com.qxotic.toknroll {
    exports com.qxotic.toknroll;
    exports com.qxotic.toknroll.spi;
    exports com.qxotic.toknroll.impl to
            com.qxotic.toknroll.gguf,
            com.qxotic.toknroll.hf;

    uses com.qxotic.toknroll.spi.TiktokenEngine;
}
