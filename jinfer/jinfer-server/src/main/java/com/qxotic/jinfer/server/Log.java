package com.qxotic.jinfer.server;

/**
 * The server's log. {@link System.Logger} is the JDK's own facade (JEP 264): no dependency to add,
 * and an embedder that runs a logging backend picks these records up through its {@code
 * LoggerFinder} without this module knowing the backend exists. With no backend the JDK's default
 * routes INFO and above to stderr - the behaviour the raw {@code System.err} calls had, now with a
 * level and a name to filter on ({@code jinfer.server}, beside {@code jinfer.leaks}).
 *
 * <p>Serving logs; it never prints. Presentation - the startup banner, the endpoint, generated
 * tokens - belongs to whoever owns the process's stdout, which for the CLI is {@link Main}.
 */
final class Log {

    static final System.Logger LOG = System.getLogger("jinfer.server");

    private Log() {}
}
