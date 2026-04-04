package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.toknroll.advanced.Splitter;
import org.junit.jupiter.api.Test;

class SplitterContractTest {

    @Test
    void identityRejectsInvalidRange() {
        Splitter splitter = Splitter.identity();

        assertThrows(
                IndexOutOfBoundsException.class,
                () -> splitter.splitAll("abc", -1, 2, (source, start, end) -> {}));
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> splitter.splitAll("abc", 1, 4, (source, start, end) -> {}));
        assertThrows(
                IndexOutOfBoundsException.class,
                () -> splitter.splitAll("abc", 2, 1, (source, start, end) -> {}));
    }

    @Test
    void sequenceRejectsStageThatEmitsDifferentSource() {
        Splitter bad =
                (text, start, end, consumer) ->
                        consumer.accept(text.subSequence(start, end - 1), 0, end - start - 1);

        Splitter splitter = Splitter.sequence(Splitter.identity(), bad);

        assertThrows(
                AssertionError.class, () -> splitter.splitAll("abc", 0, 3, (source, s, e) -> {}));
    }

    @Test
    void sequenceRejectsOutOfBoundsStageRange() {
        Splitter bad = (text, start, end, consumer) -> consumer.accept(text, start, end + 1);

        Splitter splitter = Splitter.sequence(Splitter.identity(), bad);

        assertThrows(
                AssertionError.class, () -> splitter.splitAll("abc", 0, 3, (source, s, e) -> {}));
    }

    @Test
    void sequenceRejectsHoleInStageOutput() {
        Splitter bad =
                (text, start, end, consumer) -> {
                    consumer.accept(text, start, start + 1);
                    consumer.accept(text, start + 2, end);
                };

        Splitter splitter = Splitter.sequence(Splitter.identity(), bad);

        assertThrows(
                AssertionError.class, () -> splitter.splitAll("abcd", 0, 4, (source, s, e) -> {}));
    }

    @Test
    void sequenceRejectsOverlapInStageOutput() {
        Splitter bad =
                (text, start, end, consumer) -> {
                    consumer.accept(text, start, start + 2);
                    consumer.accept(text, start + 1, end);
                };

        Splitter splitter = Splitter.sequence(Splitter.identity(), bad);

        assertThrows(
                AssertionError.class, () -> splitter.splitAll("abcd", 0, 4, (source, s, e) -> {}));
    }
}
