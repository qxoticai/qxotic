/** One HTTP GET: the JDK client, or curl as a process. */
module com.qxotic.fetch {
    requires java.net.http;

    exports com.qxotic.fetch;

    uses com.qxotic.fetch.Transport;

    provides com.qxotic.fetch.Transport with
            com.qxotic.fetch.JdkTransport;
}
