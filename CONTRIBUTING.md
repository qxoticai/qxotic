# Contributing

Thank you for looking under the hood.
This document tells you what a change needs before it can land.

## Before you start

Open an issue before writing anything larger than a bug fix.
The stack has strong opinions, and a large pull request that arrives without that conversation is usually rejected on shape, not on quality.
Small fixes, test cases that pin a defect, and documentation corrections can go straight to a pull request.

Bug reports and questions go to [GitHub issues](https://github.com/qxoticai/qxotic/issues).
A good bug report has the exact steps, what happened, and what you expected; the best ones come with a failing test.
Security reports do not go to issues; see [SECURITY.md](SECURITY.md).

Some things are declined regardless of how well they are done:

- New runtime dependencies for anything a few hundred lines of Java can do.
- New backends, kernels, or model families without the tests that prove them.
- Options and flags nobody has asked for, and abstractions with one caller.
- Changes to formatting or naming alone.

## Building and testing

The root README covers the build.
The default test run covers what needs no model file and no network.
The suites that prove the project's claims are opt-in because they need models, fixtures, or hardware; each module's README says how to run its own.
A skipped suite proves nothing, so run the opt-in suites of every area you touch and say so in the pull request.

## What a change must bring

Formatted, the default suite green, and documentation updated in the same change when behaviour changes.

The bar rises with the area.
Tokenizers and kernels are held to exact parity with their references, so a change there comes with the parity evidence.
Any performance claim comes with the numbers and the command that produced them.
A change to the public API of a published artifact needs a use in the tree to add something, and a deprecation cycle to remove it.

## Style

The formatter decides layout and the build fails on drift.
The rules it cannot enforce:

- Prefer the standard library over a dependency, and plain code over an abstraction with one caller.
- Delete what you replace: no compatibility shims, no commented-out code.
- Comments explain why, not what.
- No em dashes; use a plain dash.
- In long Markdown files, write each sentence on its own line.
- Never edit a generated file by hand; fix the generator and rerun it.

## Commits and pull requests

Commit subjects name the area and state the change: `area: what changed`.
Lowercase after the colon, no trailing period, a body only when the subject cannot carry the why.
One logical change per commit; a pull request may have several as long as the whole stays reviewable in one sitting.
Do not add co-author trailers for tools.

Pull requests are rebased onto `main`, not squashed.
Expect a first response within a week; if none comes, one polite ping is welcome.
A pull request that goes quiet for a month after review comments is closed and can be reopened any time.

## AI-assisted contributions

Quixotic is built with AI pair programmers, and contributors are welcome to use them too.
The policy is about accountability, size, and who is talking.

**Disclose it.**
When a tool wrote a substantial part of a change, say so in the pull request description, in one line: which tool, and for what.
This is not a mark against the change; it tells the reviewer where to look harder.

**A human owns every line.**
You have read the whole change, you understand it, and you can explain and defend any part of it in review without asking the tool.
"The model wrote it" is not an answer to a review question.

**Unattended output is not a contribution.**
Pull requests generated and submitted without a person reading and shaping them are closed without review.
That includes PRs opened by agents or bots on their own, bulk "cleanup" changes nobody asked for, and drive-by rewrites of files the submitter has not worked in.

**Small, or rejected.**
A pull request of thousands of lines is rejected on size alone, however good the code.
Tools make large diffs cheap to produce and expensive to review, and the cost lands on the reviewer.
Split the work into steps a person can hold in their head, each one building and tested on its own, and agree the steps in an issue first.

**Arrive pristine.**
The same tools that write code can check it, so there is no excuse for a pull request that fails on the basics.
Before opening one: formatted, the default suite green, the opt-in suites of the areas you touched run and named, and the documentation updated.
A reviewer's time goes to design and correctness, never to whitespace or a test you could have run.

**Issues are conversations between people.**
Discussion in issues and pull requests is human to human.
A tool may supply supporting material such as a reproduction, a log, or a benchmark table, pasted in and marked as such.
A tool may not argue a position, answer a reviewer, or post on its own.
Comments that read as generated are ignored, and accounts that post them repeatedly are blocked.

## Licensing

The project is Apache-2.0.
By submitting a contribution you agree that it is licensed under the same terms, as section 5 of the license states; there is no contributor license agreement.
Code you did not write must carry a license that permits inclusion and must be credited in the relevant `NOTICE` file.
