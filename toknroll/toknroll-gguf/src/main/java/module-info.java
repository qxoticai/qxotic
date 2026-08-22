module com.qxotic.toknroll.gguf {
    requires transitive com.qxotic.toknroll;
    requires transitive com.qxotic.format.gguf;
    requires java.net.http;

    exports com.qxotic.toknroll.gguf;
}
