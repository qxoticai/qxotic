package com.qxotic.jota.testutil;

import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.extension.InvocationInterceptor;
import org.junit.jupiter.api.extension.ReflectiveInvocationContext;
import org.opentest4j.TestAbortedException;

public final class RunOnConfiguredDeviceExtension implements InvocationInterceptor {

    @Override
    public void interceptTestMethod(
            Invocation<Void> invocation,
            ReflectiveInvocationContext<java.lang.reflect.Method> invocationContext,
            ExtensionContext extensionContext)
            throws Throwable {
        runInConfiguredDevice(invocation);
    }

    @Override
    public void interceptTestTemplateMethod(
            Invocation<Void> invocation,
            ReflectiveInvocationContext<java.lang.reflect.Method> invocationContext,
            ExtensionContext extensionContext)
            throws Throwable {
        runInConfiguredDevice(invocation);
    }

    private static void runInConfiguredDevice(Invocation<Void> invocation) throws Throwable {
        Device targetDevice = ConfiguredTestDevice.resolve();
        Environment current = Environment.current();
        current.nativeRuntime();
        Assumptions.assumeTrue(
                current.runtimes().hasRuntime(targetDevice),
                "Configured test device '"
                        + targetDevice
                        + "' has no registered runtime. Set -D"
                        + ConfiguredTestDevice.TEST_DEVICE_PROPERTY
                        + "=panama|c|hip or enable corresponding Maven profile.");

        Environment configured =
                new Environment(targetDevice, current.defaultFloat(), current.runtimes());

        try {
            Environment.with(
                    configured,
                    () -> {
                        try {
                            invocation.proceed();
                            return null;
                        } catch (TestAbortedException aborted) {
                            throw aborted;
                        } catch (Throwable throwable) {
                            throw new InvocationFailure(throwable);
                        }
                    });
        } catch (InvocationFailure failure) {
            throw failure.getCause();
        }
    }

    private static final class InvocationFailure extends RuntimeException {
        private InvocationFailure(Throwable cause) {
            super(cause);
        }
    }
}
