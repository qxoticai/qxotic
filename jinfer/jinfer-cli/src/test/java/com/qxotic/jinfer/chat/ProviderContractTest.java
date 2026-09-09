package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Every port on the classpath honours the {@link ModelProvider} contract: it names its
 * architectures, offers only well-formed companions, and the kinds it does not implement refuse by
 * name instead of failing some other way.
 */
class ProviderContractTest {

    private static final Set<String> KINDS =
            Set.of("loadLanguage", "loadEmbedder", "loadReranker", "loadSpeech");

    static Stream<ModelProvider> providers() {
        return ServiceLoader.load(ModelProvider.class).stream().map(ServiceLoader.Provider::get);
    }

    @ParameterizedTest
    @MethodSource("providers")
    void namesItsArchitectures(ModelProvider provider) {
        Set<String> archs = provider.architectures();
        assertFalse(archs.isEmpty(), provider.getClass().getName() + " claims nothing");
        for (String arch : archs) {
            assertFalse(arch.isBlank(), provider.getClass().getName() + " claims a blank arch");
            assertEquals(arch, arch.strip(), provider.getClass().getName() + " claims " + arch);
        }
    }

    @ParameterizedTest
    @MethodSource("providers")
    void implementsAtLeastOneKind(ModelProvider provider) {
        assertFalse(
                implemented(provider).isEmpty(),
                provider.getClass().getName() + " overrides no load* method");
    }

    @ParameterizedTest
    @MethodSource("providers")
    void offersWellFormedCompanions(ModelProvider provider) {
        provider.companionFiles()
                .forEach(
                        (capability, file) -> {
                            assertFalse(capability.isBlank(), provider.getClass().getName());
                            assertFalse(file.isBlank(), provider.getClass().getName());
                        });
    }

    @ParameterizedTest
    @MethodSource("providers")
    void unimplementedKindsRefuseByArchitectureName(ModelProvider provider) throws Exception {
        Set<String> implemented = implemented(provider);
        for (String arch : provider.architectures()) {
            GGUF gguf = Builder.newBuilder().putString("general.architecture", arch).build();
            Path none = Path.of("none.gguf");
            List<Runnable> unimplemented = new ArrayList<>();
            if (!implemented.contains("loadLanguage"))
                unimplemented.add(
                        () ->
                                call(
                                        () ->
                                                provider.loadLanguage(
                                                        null, gguf, none, null, Map.of(), null)));
            if (!implemented.contains("loadEmbedder"))
                unimplemented.add(
                        () -> call(() -> provider.loadEmbedder(null, gguf, none, null, null)));
            if (!implemented.contains("loadReranker"))
                unimplemented.add(
                        () -> call(() -> provider.loadReranker(null, gguf, none, null, null)));
            if (!implemented.contains("loadSpeech"))
                unimplemented.add(
                        () -> call(() -> provider.loadSpeech(null, gguf, none, null, Map.of())));
            for (Runnable load : unimplemented) {
                UnsupportedOperationException refused =
                        assertThrows(UnsupportedOperationException.class, load::run);
                assertTrue(
                        refused.getMessage().matches("'" + arch + "' is not an? \\w+ architecture"),
                        provider.getClass().getName() + ": " + refused.getMessage());
            }
        }
    }

    /** The {@code load*} methods this provider's class declares itself. */
    private static Set<String> implemented(ModelProvider provider) {
        Set<String> declared = new java.util.HashSet<>();
        for (Method m : provider.getClass().getDeclaredMethods()) {
            if (KINDS.contains(m.getName())) declared.add(m.getName());
        }
        return declared;
    }

    private interface Load {
        Object run() throws Exception;
    }

    private static void call(Load load) {
        try {
            load.run();
        } catch (RuntimeException e) {
            throw e;
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }
}
