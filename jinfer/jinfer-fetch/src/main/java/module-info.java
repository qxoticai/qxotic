/** One HTTP GET: the JDK client, or curl as a process. */
module com.qxotic.jinfer.fetch {
    requires java.net.http;

    exports com.qxotic.jinfer.fetch;

    uses com.qxotic.jinfer.fetch.Transport;

    provides com.qxotic.jinfer.fetch.Transport with
            com.qxotic.jinfer.fetch.JdkTransport;
}
