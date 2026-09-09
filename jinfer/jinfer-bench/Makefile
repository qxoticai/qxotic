# This module's GraalVM native image, through the root Makefile's one recipe: the executable is
# the module name with a -cli suffix dropped, and lands in the repository's bin/.
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
.DEFAULT_GOAL := help

help:
	@echo 'make native   GraalVM native image of $(notdir $(CURDIR)) for this machine -> $(ROOT)/bin/'

native:
	$(MAKE) -C $(ROOT) native NATIVE_MODULE=$(notdir $(CURDIR))

.PHONY: help native
