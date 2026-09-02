# Contributing

This document says what a change needs before it can be merged.

## Before you start

Open an issue before writing anything larger than a bug fix.
Large pull requests without a prior issue are usually rejected on scope, not quality.
Small fixes, tests that pin a defect, and documentation corrections can go straight to a pull request.

Bug reports and questions go to [GitHub issues](https://github.com/qxoticai/qxotic/issues).
A bug report needs the exact steps, what happened, and what you expected.
A failing test is the best report.
Security reports do not go to issues; see [SECURITY.md](SECURITY.md).

The following are declined regardless of quality:

- A new runtime dependency for something a few hundred lines of code can do.
- A new backend, kernel, or model family without the tests that prove it.
- Options and flags nobody asked for.
- Changes to formatting or naming alone.

## Building and testing

The root README covers the build.
The default test run needs no model files and no network.
The suites that verify the project's claims are opt-in because they need models, fixtures, or hardware.
Each module's README says how to run them.
A skipped suite proves nothing.
Run the opt-in suites for every area you touch and list them in the pull request.

## Requirements for every change

- Formatted.
- Default test suite green.
- Documentation updated in the same change when behaviour changes.
- Tokenizer and kernel changes include parity evidence against the reference implementation.
- Performance claims include the numbers and the command that produced them.
- Additions to the public API of a published artifact have a use in the tree.
- Removals from the public API go through a deprecation cycle.

## Style

The formatter decides layout.
The build fails on unformatted code.
Rules the formatter cannot enforce:

- Prefer the standard library over a dependency.
- Prefer plain code over an abstraction with a single caller.
- Delete what you replace. No compatibility shims, no commented-out code.
- Comments explain why, not what.
- No em dashes. Use a plain dash.
- In long Markdown files, one sentence per line.
- Do not edit generated files by hand. Fix the generator and rerun it.

## Commits and pull requests

Commit subject: `area: what changed`.
Lowercase after the colon, no trailing period.
Add a body only when the subject cannot explain why.
One logical change per commit.
A pull request may hold several commits if the whole is reviewable in one sitting.
No co-author trailers for tools.

CI runs on pull requests once a maintainer approves the run; it covers the default suite and the release build, without models.
Pull requests are rebased onto `main`, not squashed.
Expect a first response within a week.
If none comes, one reminder is fine.
A pull request with no activity for a month after review comments is closed.
It can be reopened at any time.

## AI-assisted contributions

Using AI tools is allowed.
The rules cover disclosure, ownership, size, and communication.

**Disclosure.**
If a tool wrote a substantial part of a change, state which tool and for what in the pull request description.
This is not held against the change.

**Ownership.**
You have read every line you submit.
You can explain and defend any part of it in review without consulting the tool.
"The model wrote it" is not an answer to a review question.

**No unattended output.**
Pull requests generated and submitted without a person reading and editing them are closed without review.
This includes pull requests opened by agents or bots, unrequested bulk cleanups, and rewrites of files the submitter has not worked in.

**Size.**
Pull requests of thousands of lines are rejected on size alone.
Large diffs are cheap to generate and expensive to review, and the cost falls on the reviewer.
Split the work into steps that build and pass tests on their own, and agree on the steps in an issue first.

**Validation before submission.**
The tools that write code can also check it.
Use them to format, run the tests, and verify the requirements above before opening a pull request.
Reviewer time goes to design and correctness, not to formatting or tests you could have run.

**Communication.**
Discussion in issues and pull requests is between people.
A tool may produce supporting material such as a reproduction, a log, or a benchmark table.
Paste it in and mark it as tool output.
A tool may not argue a position, answer a reviewer, or post on its own.
Generated comments are ignored.
Accounts that post them repeatedly are blocked.

## Licensing

The project is Apache-2.0.
By submitting a contribution you license it under the same terms, per section 5 of the license.
There is no contributor license agreement.
Code you did not write must have a license that permits inclusion and must be credited in the relevant `NOTICE` file.
