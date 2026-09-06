package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.OptionalLong;
import org.junit.jupiter.api.Test;

class ProcessRssTest {

    @Test
    void linuxAndMacReportAResidentSetAndEverywhereElseIsEmptyWithAReason() {
        OptionalLong rss = ProcessRss.kilobytes();
        if (ProcessRss.supported()) {
            assertTrue(rss.isPresent() && rss.getAsLong() > 10_000, "kB: " + rss);
        } else {
            assertFalse(rss.isPresent());
            assertTrue(ProcessRss.unsupportedReason().contains("Linux"));
        }
    }
}
