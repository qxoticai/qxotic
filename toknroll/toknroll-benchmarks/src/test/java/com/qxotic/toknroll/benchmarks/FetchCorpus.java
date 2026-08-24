package com.qxotic.toknroll.benchmarks;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Fetches the benchmark corpora into the toknroll cache. Corpora are expensive (enwik9 is a 322 MB
 * zip), so fetching is this explicit command, never a side effect of a benchmark run:
 *
 * <pre>
 * make toknroll-fixtures                     # enwik8 and enwik9
 * make toknroll-fixtures FIXTURES="enwik8"   # just one
 * </pre>
 *
 * Already-cached corpora are size-checked and skipped; a size mismatch is refetched.
 */
public final class FetchCorpus {

    public static void main(String[] args) throws Exception {
        List<String> names =
                args.length == 0
                        ? WikiCorpusPaths.corpusNames()
                        : Arrays.stream(args).filter(a -> !a.isBlank()).toList();
        if (names.isEmpty()) {
            names = WikiCorpusPaths.corpusNames();
        }
        for (String name : names) {
            long expected = WikiCorpusPaths.expectedBytes(name);
            Path target = WikiCorpusPaths.downloadTarget(name);
            if (Files.exists(target)) {
                long size = Files.size(target);
                if (size == expected) {
                    System.out.printf("%s already cached (%,d bytes): %s%n", name, size, target);
                    continue;
                }
                System.out.printf(
                        "%s has %,d bytes, expected %,d - refetching%n", name, size, expected);
            }
            try {
                WikiCorpusPaths.download(name, target);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw e;
            }
            System.out.printf("%s -> %s%n", name, target);
        }
    }
}
