package com.qxotic.jinfer.examples.inflect2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import org.junit.jupiter.api.Test;

class PlayerTest {

    @Test
    void aDeadPlayerReportsItsStatusNotAStreamClosed() throws IOException {
        // `false` exits at once, so its stdin is a dead pipe by the time we close it. The bytes
        // that close() flushes into it used to throw "Stream closed" out of --play, naming our
        // own pipe for a failure one process away; the player's exit status is the diagnosis.
        Player player = new Player("false");
        player.offer(new byte[4096]); // may or may not reach the pipe: either is fine

        Player.Failed failed = assertThrows(Player.Failed.class, player::close);
        assertEquals(1, failed.status, "the player's status, so it can become ours");
        assertFalse(failed.getMessage().contains("Stream closed"), failed.getMessage());
    }

    @Test
    void aPlayerThatSucceedsClosesQuietly() throws IOException {
        Player player = new Player("true");
        player.offer(new byte[4096]);
        player.close();
    }
}
