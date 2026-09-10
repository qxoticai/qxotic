package com.qxotic.jinfer.testkit;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContextConfiguration;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.stream.IntStream;

/** A weightless, cacheable model that always emits 'x'; protocol tests need no learned weights. */
public final class TestLanguageModel
        implements LanguageModel<TestLanguageModel.Configuration, Void, TestLanguageModel.State> {

    public record Configuration(int vocabularySize, int maxContextLength)
            implements ContextConfiguration {}

    public static final Tokenizer TOKENIZER =
            new Tokenizer() {
                private final Vocabulary vocabulary =
                        new Vocabulary() {
                            public int size() {
                                return 256;
                            }

                            public boolean contains(int id) {
                                return id >= 0 && id < size();
                            }

                            public boolean contains(String text) {
                                return text.length() == 1 && text.charAt(0) < 256;
                            }

                            public String token(int id) {
                                if (!contains(id)) throw new NoSuchElementException("token " + id);
                                return Character.toString((char) id);
                            }

                            public int id(String text) {
                                if (!contains(text)) throw new NoSuchElementException(text);
                                return text.charAt(0);
                            }

                            public boolean isTokenOfType(int id, TokenType type) {
                                if (!contains(id)) throw new NoSuchElementException("token " + id);
                                return type == StandardTokenType.NORMAL;
                            }

                            public Iterator<Map.Entry<String, Integer>> iterator() {
                                return IntStream.range(0, size())
                                        .mapToObj(i -> Map.entry(token(i), i))
                                        .iterator();
                            }
                        };

                public Vocabulary vocabulary() {
                    return vocabulary;
                }

                public void encodeInto(
                        CharSequence text, int from, int to, IntSequence.Builder out) {
                    for (byte b :
                            text.subSequence(from, to).toString().getBytes(StandardCharsets.UTF_8))
                        out.add(b & 255);
                }

                public int countTokens(CharSequence text, int from, int to) {
                    return text.subSequence(from, to)
                            .toString()
                            .getBytes(StandardCharsets.UTF_8)
                            .length;
                }

                public int decodeBytesInto(IntSequence tokens, int from, ByteBuffer out) {
                    int start = from;
                    while (from < tokens.length() && out.hasRemaining())
                        out.put((byte) tokens.intAt(from++));
                    return from - start;
                }
            };

    public static final class State extends ContextState {
        private final MemorySegment tokens;
        private final MemoryView<MemorySegment> logits;

        State(int context, int batch, MemoryArena<MemorySegment> memory, boolean owns) {
            super(context, batch, memory, owns);
            tokens = memory.allocateMemory((long) context * Integer.BYTES, 64).base();
            logits = Views.allocateF32(memory, 256);
            logits.memory().base().setAtIndex(ValueLayout.JAVA_FLOAT, 'x', 1f);
        }

        protected void clearHistory() {
            tokens.fill((byte) 0);
        }

        void ingest(Batch batch) {
            int[] ids = ((Batch.Input.Tokens) batch.input()).ids();
            for (int i = 0; i < ids.length; i++)
                tokens.setAtIndex(ValueLayout.JAVA_INT, position() + i, ids[i]);
            advanceContext(batch.count(), batch.outputs());
        }
    }

    public Configuration configuration() {
        return new Configuration(256, 16384);
    }

    public Void weights() {
        return null;
    }

    public State newState(int context, int batch, MemoryArena<MemorySegment> memory) {
        return new State(context, batch, memory, false);
    }

    public State newState(int context, int batch) {
        return new State(context, batch, Arenas.newCrossThreadMemoryArena(), true);
    }

    public void ingest(State state, Batch batch) {
        state.exclusively(() -> state.ingest(batch));
    }

    public MemoryView<?> logits(State state, int output) {
        return state.exclusively(() -> state.logits);
    }

    public Optional<CheckpointCodec<State>> checkpointCodec() {
        return Optional.of(
                new CheckpointCodec<>() {
                    protected long sizeOf(int positions) {
                        return (long) positions * Integer.BYTES;
                    }

                    protected void transfer(
                            State state, int from, int to, MemorySegment memory, boolean capture) {
                        MemorySegment rows =
                                state.tokens.asSlice(
                                        (long) from * Integer.BYTES, memory.byteSize());
                        if (capture) memory.copyFrom(rows);
                        else rows.copyFrom(memory);
                    }
                });
    }
}
