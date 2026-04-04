package com.qxotic.toknroll.impl;

/**
 * Prototype Aho-Corasick model for byte-level BPE vocab matching.
 *
 * <p>This model performs greedy longest-match token counting in O(n) with a failureless transition
 * table. It is intentionally minimal and is meant as a stepping stone for full AC+backtracking
 * integration.
 */
public final class AcBpeModel {

    private static final int ALPHABET = 256;

    final int stateCount;
    final int[] next;
    final int[] outHead;
    final int[] outNext;
    final int[] outToken;
    final short[] outLen;
    final int[] singleByteTokenId;
    final int[] nextPrefix;
    final short[] tokenLen;
    final int maxTokenLen;

    AcBpeModel(
            int stateCount,
            int[] next,
            int[] outHead,
            int[] outNext,
            int[] outToken,
            short[] outLen,
            int[] singleByteTokenId,
            int[] nextPrefix,
            short[] tokenLen,
            int maxTokenLen) {
        this.stateCount = stateCount;
        this.next = next;
        this.outHead = outHead;
        this.outNext = outNext;
        this.outToken = outToken;
        this.outLen = outLen;
        this.singleByteTokenId = singleByteTokenId;
        this.nextPrefix = nextPrefix;
        this.tokenLen = tokenLen;
        this.maxTokenLen = maxTokenLen;
    }

    public int nextMatch(byte[] bytes, int start, int endExclusive) {
        int bestLen = 0;
        int bestToken = -1;

        int state = 0;
        int i = start;
        int end = Math.min(endExclusive, start + maxTokenLen);
        while (i < end) {
            state = next[(state << 8) | (bytes[i] & 0xFF)];

            int out = outHead[state];
            while (out != -1) {
                int len = outLen[out] & 0xFFFF;
                int token = outToken[out];
                if (len > bestLen || (len == bestLen && (bestToken < 0 || token < bestToken))) {
                    bestLen = len;
                    bestToken = token;
                }
                out = outNext[out];
            }

            i++;
            if (state == 0 && bestLen == 0) {
                break;
            }
        }
        return bestToken;
    }

    public int nextPrefix(int token) {
        if (token < 0 || token >= nextPrefix.length) {
            return -1;
        }
        return nextPrefix[token];
    }

    public int tokenLen(int token) {
        if (token < 0 || token >= tokenLen.length) {
            return 0;
        }
        return tokenLen[token] & 0xFFFF;
    }

    public int singleByteToken(byte b) {
        return singleByteTokenId[b & 0xFF];
    }

    /**
     * Greedy longest-match token count (no backtracking).
     *
     * <p>For equal lengths, lower token id (rank) wins.
     */
    public int greedyCount(byte[] bytes, int byteLength) {
        int count = 0;
        int pos = 0;

        while (pos < byteLength) {
            int bestLen = 0;
            int bestToken = Integer.MAX_VALUE;

            int state = 0;
            int i = pos;
            int end = Math.min(byteLength, pos + maxTokenLen);
            while (i < end) {
                state = next[(state << 8) | (bytes[i] & 0xFF)];

                int out = outHead[state];
                while (out != -1) {
                    int len = outLen[out] & 0xFFFF;
                    int token = outToken[out];
                    if (len > bestLen || (len == bestLen && token < bestToken)) {
                        bestLen = len;
                        bestToken = token;
                    }
                    out = outNext[out];
                }

                i++;
                if (state == 0 && bestLen == 0) {
                    break;
                }
            }

            pos += (bestLen > 0) ? bestLen : 1;
            count++;
        }

        return count;
    }

    /**
     * Greedy longest-match segmentation.
     *
     * @return number of produced tokens
     */
    public int greedySegment(
            byte[] bytes, int byteLength, int[] outTokens, int[] outStarts, int[] outEnds) {
        int produced = 0;
        int pos = 0;

        while (pos < byteLength) {
            int bestLen = 0;
            int bestToken = Integer.MAX_VALUE;

            int state = 0;
            int i = pos;
            int end = Math.min(byteLength, pos + maxTokenLen);
            while (i < end) {
                state = next[(state << 8) | (bytes[i] & 0xFF)];

                int out = outHead[state];
                while (out != -1) {
                    int len = outLen[out] & 0xFFFF;
                    int token = outToken[out];
                    if (len > bestLen || (len == bestLen && token < bestToken)) {
                        bestLen = len;
                        bestToken = token;
                    }
                    out = outNext[out];
                }

                i++;
                if (state == 0 && bestLen == 0) {
                    break;
                }
            }

            int len = (bestLen > 0) ? bestLen : 1;
            int token = (bestLen > 0) ? bestToken : singleByteTokenId[bytes[pos] & 0xFF];
            outTokens[produced] = token;
            outStarts[produced] = pos;
            pos += len;
            outEnds[produced] = pos;
            produced++;
        }

        return produced;
    }
}
