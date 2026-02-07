package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;

public interface Kernel<I extends KernelInput, O extends KernelOutput> {

    KernelSignature signature();

    boolean supports(Device device);

    O execute(I input, ExecutionContext ctx);
}
