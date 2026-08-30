package com.qxotic.jinfer.models.inflect2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.util.Set;
import org.junit.jupiter.api.Test;

final class Inflect2ProviderTest {

    @Test
    void claimsOnlyTheArchitectureItCanLoad() {
        Inflect2Provider provider = new Inflect2Provider();
        assertEquals(Set.of("inflect-v2"), provider.architectures());
        assertFalse(provider.supports("inflect-v3"));
        assertFalse(provider.supports("inflection"));
    }
}
