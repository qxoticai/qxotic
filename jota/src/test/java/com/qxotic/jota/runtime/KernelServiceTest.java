package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.tensor.ExecutionStream;
import com.qxotic.jota.tensor.KernelBackend;
import com.qxotic.jota.tensor.KernelCacheKey;
import com.qxotic.jota.tensor.KernelExecutable;
import com.qxotic.jota.tensor.KernelProgram;
import com.qxotic.jota.tensor.LaunchConfig;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class KernelServiceTest {

    @Test
    void registersAndLoadsNamedKernel() {
        InMemoryBackend backend = new InMemoryBackend();
        InMemoryStore sourceStore = new InMemoryStore();
        InMemoryStore binaryStore = new InMemoryStore();
        KernelService service = new KernelService(backend, sourceStore, binaryStore);

        KernelProgram program =
                new KernelProgram(
                        KernelProgram.Kind.SOURCE,
                        KernelProgram.C,
                        "void custom() {}",
                        "custom",
                        Map.of());
        KernelCacheKey key = KernelCacheKey.of("custom-kernel");

        KernelExecutable executable = service.register("custom.gelu", program, key);
        assertNotNull(executable);

        Optional<KernelProgram> byName = service.loadRegisteredKernel("custom.gelu");
        assertTrue(byName.isPresent());
        assertEquals(program, byName.get());
        assertEquals(key, service.namedKernelKeys().get("custom.gelu"));
    }

    @Test
    void loadsRegisteredExecutableByName() {
        InMemoryBackend backend = new InMemoryBackend();
        KernelService service =
                new KernelService(backend, new InMemoryStore(), new InMemoryStore());

        KernelProgram program =
                new KernelProgram(
                        KernelProgram.Kind.SOURCE,
                        KernelProgram.C,
                        "void custom() {}",
                        "custom",
                        Map.of());
        KernelCacheKey key = KernelCacheKey.of("custom-kernel");

        service.register("custom.attention", program, key);

        Optional<KernelExecutable> executable =
                service.loadRegisteredExecutable("custom.attention");
        assertTrue(executable.isPresent());
        assertSame(backend.cache.get(key), executable.get());
        assertTrue(service.loadRegisteredExecutable("does.not.exist").isEmpty());
    }

    private static final class InMemoryStore implements KernelProgramStore {
        private final Map<KernelCacheKey, KernelProgram> programs = new HashMap<>();

        @Override
        public Path root() {
            return Path.of(".");
        }

        @Override
        public void store(KernelProgram program, KernelCacheKey key) {
            programs.put(key, program);
        }

        @Override
        public Optional<KernelProgram> load(KernelCacheKey key) {
            return Optional.ofNullable(programs.get(key));
        }
    }

    private static final class InMemoryBackend implements KernelBackend {
        private final Map<KernelCacheKey, KernelExecutable> cache = new HashMap<>();

        @Override
        public KernelExecutable compile(KernelProgram program, KernelCacheKey cacheKey) {
            KernelExecutable exec = new NoopKernelExecutable();
            cache.put(cacheKey, exec);
            return exec;
        }

        @Override
        public KernelExecutable load(KernelProgram program, KernelCacheKey cacheKey) {
            return cache.get(cacheKey);
        }

        @Override
        public KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey cacheKey) {
            return cache.computeIfAbsent(cacheKey, __ -> new NoopKernelExecutable());
        }

        @Override
        public KernelExecutableCache cache() {
            return new KernelExecutableCache() {
                @Override
                public KernelExecutable get(KernelCacheKey key) {
                    return cache.get(key);
                }

                @Override
                public void put(KernelCacheKey key, KernelExecutable exec) {
                    cache.put(key, exec);
                }
            };
        }
    }

    private static final class NoopKernelExecutable implements KernelExecutable {
        @Override
        public void launch(
                LaunchConfig config,
                com.qxotic.jota.tensor.KernelArgs args,
                ExecutionStream stream) {
            // no-op
        }
    }
}
