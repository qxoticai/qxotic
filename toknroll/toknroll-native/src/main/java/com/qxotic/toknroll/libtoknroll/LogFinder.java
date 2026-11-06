package com.qxotic.toknroll.libtoknroll;

/**
 * The image's System.LoggerFinder, the JDK's own seam for a logging backend: registered as a
 * service, so every System.getLogger in the image gets a logger of {@link Log}, and
 * java.util.logging, the JDK's default backend, is never chosen. Public with a no-argument
 * constructor because it is a service provider.
 */
public final class LogFinder extends System.LoggerFinder {
    public LogFinder() {}

    @Override
    public System.Logger getLogger(String name, Module module) {
        return Log.logger(name);
    }
}
