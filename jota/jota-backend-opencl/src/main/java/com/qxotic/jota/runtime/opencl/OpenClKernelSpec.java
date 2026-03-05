package com.qxotic.jota.runtime.opencl;

import java.nio.file.Path;

record OpenClKernelSpec(Path sourcePath, String kernelName) {}
