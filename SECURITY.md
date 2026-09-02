# Security

## Reporting a vulnerability

Email **hello@qxotic.ai**.
Do not open a public issue or pull request for anything you believe is a security problem.

Include the affected module and version, the impact, and a reproduction if you have one.

## What to expect

An acknowledgement within three working days, a verdict within two weeks, and a fix in a release within 90 days for accepted reports, sooner when the impact warrants it.
You are credited in the advisory unless you prefer otherwise.
There is no bounty program.

## Supported versions

Fixes go into the latest release only.
The `com.qxotic` artifacts are versioned together, so upgrade them together.

## Scope

Model files are trusted input.
Run only models from sources you trust; a malformed or malicious model file that crashes or misbehaves the engine is a bug to report normally, not a vulnerability.

In scope: the server's handling of client requests and its authentication, and the downloader's handling of what it fetches.

Out of scope: model files and everything derived from them, model behaviour such as prompt injection or harmful output, running the server on a non-loopback interface without an API key, resource exhaustion through legitimate use, and dependency vulnerabilities with no reachable path through this code.
