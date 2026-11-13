/** Model reference resolution and downloads. */
module com.qxotic.jinfer.hub {
    requires com.qxotic.format.json;
    requires com.qxotic.jinfer.fetch;

    exports com.qxotic.jinfer.hub;

    uses com.qxotic.jinfer.hub.Console;

    provides com.qxotic.jinfer.hub.Console with
            com.qxotic.jinfer.hub.FfmConsole;
}
