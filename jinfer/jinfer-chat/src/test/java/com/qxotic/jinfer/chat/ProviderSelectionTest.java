package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Duplicate-architecture resolution: highest priority wins; equal priorities resolve by class name
 * (deterministic, never classpath order). The warning path is exercised but not asserted - stderr
 * text is presentation, the selection is the contract.
 */
final class ProviderSelectionTest {

    // distinct classes so the class-name tie-break has names to compare
    static final class A extends Base {}

    static final class B extends Base {}

    abstract static class Base implements ModelProvider {
        public boolean supports(String architecture) {
            return "dup".equals(architecture);
        }
    }

    @Test
    void higherPriorityWins() {
        ModelProvider low =
                new ModelProvider() {
                    public boolean supports(String a) {
                        return "dup".equals(a);
                    }
                };
        ModelProvider high =
                new ModelProvider() {
                    public boolean supports(String a) {
                        return "dup".equals(a);
                    }

                    public int priority() {
                        return 10;
                    }
                };
        assertSame(high, Models.select(List.<ModelProvider>of(low, high), "dup"));
        assertSame(
                high, Models.select(List.<ModelProvider>of(high, low), "dup")); // order-independent
    }

    @Test
    void equalPriorityIsDeterministicByClassName() {
        ModelProvider a = new A();
        ModelProvider b = new B();
        // A sorts before B whichever way the "classpath" lists them
        assertSame(a, Models.select(List.<ModelProvider>of(a, b), "dup"));
        assertSame(a, Models.select(List.<ModelProvider>of(b, a), "dup"));
    }

    @Test
    void noClaimIsNull() {
        assertNull(Models.select(List.<ModelProvider>of(new A()), "other"));
    }
}
