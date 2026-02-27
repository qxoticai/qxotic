package com.qxotic.jota.testutil;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.extension.ClassTemplateInvocationContext;
import org.junit.jupiter.api.extension.ClassTemplateInvocationContextProvider;
import org.junit.jupiter.api.extension.Extension;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.InvocationInterceptor;
import org.junit.jupiter.api.extension.ReflectiveInvocationContext;
import org.opentest4j.TestAbortedException;

public final class BackendClassMatrixExtension implements ClassTemplateInvocationContextProvider {

    @Override
    public boolean supportsClassTemplate(ExtensionContext context) {
        return context.getTestClass()
                .map(c -> c.isAnnotationPresent(RunOnAllAvailableBackends.class))
                .orElse(false);
    }

    @Override
    public Stream<? extends ClassTemplateInvocationContext> provideClassTemplateInvocationContexts(
            ExtensionContext context) {
        List<Device> targets = AvailableBackends.resolveTargets();
        Assumptions.assumeTrue(!targets.isEmpty(), "No available backend targets for this test");
        return targets.stream().map(BackendClassInvocationContext::new);
    }

    private record BackendClassInvocationContext(Device device)
            implements ClassTemplateInvocationContext {

        @Override
        public String getDisplayName(int invocationIndex) {
            return "backend=" + device.name();
        }

        @Override
        public List<Extension> getAdditionalExtensions() {
            return List.of(new BackendInvocationInterceptor(device));
        }
    }

    private static final class BackendInvocationInterceptor implements InvocationInterceptor {
        private final Device target;

        private BackendInvocationInterceptor(Device target) {
            this.target = target;
        }

        @Override
        public <T> T interceptTestClassConstructor(
                Invocation<T> invocation,
                ReflectiveInvocationContext<Constructor<T>> invocationContext,
                ExtensionContext extensionContext)
                throws Throwable {
            return runInBackend(invocation);
        }

        @Override
        public void interceptBeforeAllMethod(
                Invocation<Void> invocation,
                ReflectiveInvocationContext<Method> invocationContext,
                ExtensionContext extensionContext)
                throws Throwable {
            runInBackend(invocation);
        }

        @Override
        public void interceptBeforeEachMethod(
                Invocation<Void> invocation,
                ReflectiveInvocationContext<Method> invocationContext,
                ExtensionContext extensionContext)
                throws Throwable {
            runInBackend(invocation);
        }

        @Override
        public void interceptTestMethod(
                Invocation<Void> invocation,
                ReflectiveInvocationContext<Method> invocationContext,
                ExtensionContext extensionContext)
                throws Throwable {
            runInBackend(invocation);
        }

        @Override
        public void interceptAfterEachMethod(
                Invocation<Void> invocation,
                ReflectiveInvocationContext<Method> invocationContext,
                ExtensionContext extensionContext)
                throws Throwable {
            runInBackend(invocation);
        }

        @Override
        public void interceptAfterAllMethod(
                Invocation<Void> invocation,
                ReflectiveInvocationContext<Method> invocationContext,
                ExtensionContext extensionContext)
                throws Throwable {
            runInBackend(invocation);
        }

        private <T> T runInBackend(Invocation<T> invocation) throws Throwable {
            Environment current = Environment.current();
            current.nativeRuntime();
            Assumptions.assumeTrue(
                    current.runtimes().hasRuntime(target),
                    "Backend runtime unavailable for " + target.name());

            Environment configured =
                    new Environment(target, current.defaultFloat(), current.runtimes());
            return Environment.with(
                    configured,
                    () -> {
                        TestBackendContext.set(target);
                        try {
                            return invocation.proceed();
                        } catch (TestAbortedException aborted) {
                            throw aborted;
                        } catch (Throwable throwable) {
                            throw new InvocationFailure(target, throwable);
                        } finally {
                            TestBackendContext.clear();
                        }
                    });
        }
    }

    private static final class InvocationFailure extends RuntimeException {
        private InvocationFailure(Device target, Throwable cause) {
            super(
                    "Failure while running test on backend " + target + ": " + cause.getMessage(),
                    cause);
        }
    }
}
