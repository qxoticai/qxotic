package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.OptionalLong;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class ProcessRssTest {

    @Test
    void linuxMacAndWindowsReportAResidentSetAndEverywhereElseIsEmptyWithAReason() {
        OptionalLong rss = ProcessRss.kilobytes();
        if (ProcessRss.supported()) {
            assertTrue(rss.isPresent() && rss.getAsLong() > 10_000, "kB: " + rss);
        } else {
            assertFalse(rss.isPresent());
            assertTrue(ProcessRss.unsupportedReason().contains("Linux"));
        }
    }

    @Test
    void aCommandThatOutlivesTheTimeoutIsAbandonedNotAwaited() throws Exception {
        Assumptions.assumeTrue(
                Files.isExecutable(Path.of("/bin/sleep"))
                        && Files.isExecutable(Path.of("/bin/echo")),
                "needs /bin/sleep and /bin/echo");
        long start = System.nanoTime();
        assertNull(ProcessRss.run(Duration.ofMillis(200), "/bin/sleep", "30"));
        assertTrue(
                System.nanoTime() - start < Duration.ofSeconds(10).toNanos(),
                "waited for the command instead of the timeout");
        assertEquals("hi", ProcessRss.run(Duration.ofSeconds(10), "/bin/echo", "hi"));
    }
}
