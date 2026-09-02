# Security Policy

## Reporting a vulnerability

Email **hello@qxotic.ai**.
Do not open a public issue or pull request for a suspected security problem.

Include the affected module and version, the impact as you understand it, and a reproduction if you have one.

## What to expect

Acknowledgement within three working days.
A verdict within two weeks.
Accepted reports are fixed in a release within 90 days, sooner when the impact warrants it.
The advisory credits you unless you prefer otherwise.
There is no bounty program.

## Supported versions

Fixes go into the latest release only.
The `com.qxotic` artifacts are versioned together; upgrade them together.

## Scope

Model files are trusted input.
Run only models from sources you trust.
A malformed or malicious model file that crashes or misbehaves the engine is a bug, not a vulnerability.
Report it as a normal issue.

In scope:

- The server's handling of client requests and its authentication.
- The downloader's handling of what it fetches.

Out of scope:

- Model files and everything derived from them.
- Model behaviour, including prompt injection and harmful output.
- Running the server on a non-loopback interface without an API key.
- Resource exhaustion through legitimate use.
- Dependency vulnerabilities with no reachable path through this code.
