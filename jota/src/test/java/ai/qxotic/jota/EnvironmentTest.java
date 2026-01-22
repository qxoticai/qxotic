package ai.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class EnvironmentTest {

    @Test
    void currentDefaultsToGlobal() {
        assertSame(Environment.global(), Environment.current());
    }

    @Test
    void scopedEnvironmentOverridesDefaults() {
        Environment env = new Environment(Device.NATIVE, DataType.FP64, DeviceRegistry.global());

        Environment.with(
                env,
                () -> {
                    assertEquals(Device.NATIVE, Device.defaultDevice());
                    assertEquals(DataType.FP64, DataType.defaultFloat());
                    return null;
                });
    }

    @Test
    void configureGlobalOnlyOnce() {
        Environment env = Environment.global();
        Environment.configureGlobal(env);

        assertSame(env, Environment.global());
        assertThrows(IllegalStateException.class, () -> Environment.configureGlobal(env));
    }
}
