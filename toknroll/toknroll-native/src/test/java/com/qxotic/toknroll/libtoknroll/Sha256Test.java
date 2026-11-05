package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.security.MessageDigest;
import java.util.Random;
import org.junit.jupiter.api.Test;

class Sha256Test {
    @Test
    void agreesWithTheJdkOnEveryLengthAndChunking() throws Exception {
        MessageDigest jdk = MessageDigest.getInstance("SHA-256");
        Random random = new Random(7);
        for (int length : new int[] {0, 1, 3, 55, 56, 63, 64, 65, 119, 120, 1000, 65537}) {
            byte[] data = new byte[length];
            random.nextBytes(data);
            Sha256 own = new Sha256();
            for (int at = 0; at < length; ) { // fed in uneven pieces, a byte at a time among them
                int n = Math.min(length - at, random.nextInt(70));
                if (n == 0) own.update(data[at++]);
                else own.update(data, at, n);
                at += n;
            }
            assertArrayEquals(jdk.digest(data), own.digest(), "length " + length);
            assertArrayEquals(jdk.digest(data), own.digest(data), "reset, then whole");
        }
    }
}
