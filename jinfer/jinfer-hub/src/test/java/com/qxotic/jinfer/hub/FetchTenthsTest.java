package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Off a terminal, a download's plain log gets one line at each tenth of its size. */
class FetchTenthsTest {
    private static List<String> tenths(Fetch.Progress progress, long from, long to) {
        List<String> lines = new ArrayList<>();
        for (long written = from; written <= to; written++) {
            String line = progress.tenth(written);
            if (line != null) lines.add(line);
        }
        return lines;
    }

    @Test
    void aLineAtEachTenthNoneAtTheEndAndNoneWhileVerifying() {
        Fetch.Progress progress = new Fetch.Progress("model.gguf", 1000);
        progress.start(0);
        List<String> lines = tenths(progress, 1, 1000);
        assertEquals(9, lines.size(), lines.toString());
        for (int i = 0; i < 9; i++) {
            String line = lines.get(i);
            assertTrue(line.startsWith("model.gguf  " + (i + 1) * 10 + "%  "), line);
            assertTrue(line.contains("/1000 B ("), line);
        }
        progress.verifying();
        assertEquals(List.of(), tenths(progress, 1, 1000), "the read-back is not progress");
    }

    @Test
    void aResumedTransferStartsAtItsOwnTenth() {
        Fetch.Progress progress = new Fetch.Progress("model.gguf", 1000);
        progress.start(450);
        List<String> lines = tenths(progress, 451, 1000);
        assertEquals(5, lines.size(), lines.toString());
        assertTrue(lines.get(0).startsWith("model.gguf  50%  "), lines.get(0));
    }

    @Test
    void anUnsizedTransferHasNoTenths() {
        Fetch.Progress progress = new Fetch.Progress("listing", 0);
        progress.start(0);
        assertNull(progress.tenth(500));
    }
}
