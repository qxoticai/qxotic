# Pull and refresh contract

Implemented in `jinfer-hub` and `jinfer-cli`.
Linux checks are recorded below; Windows publication validation is pending.

## Ponytail rules

- Use, update, repair: three behaviors, one downloader.
- Never delete a working copy to begin a download.
- Policy in the store, transport in the downloader, presentation in the CLI.
- Reuse locks, partial files, checksums, and atomic moves.
- No new dependencies, cache database, or transaction framework.
- Every guarantee needs a regression check.

## User contract

| Operation | Behavior |
|---|---|
| Inference / `resolve` / `resolveAll` | Use installed files without network access or rehashing; download on a miss |
| `pull` / `pullAll(inputs, false)` | Check upstream and reuse matching content; download changed or missing files |
| `pull --force` / `pullAll(inputs, true)` | Bypass completed-file reuse and fetch a replacement |
| `find` / `list` | Inspect local state without network access |
| Existing local path | Return the path; force does not transfer ownership or authorize deletion |

Mutable-reference pull first selects a file from the remote repository.
When a SHA-256 is published, a candidate cached file is stream-hashed and size-checked before reuse.
That costs a sequential local read, not another payload transfer.
Without a published checksum, explicit pull downloads again; size or modification time alone cannot establish equality.
Plain URLs follow this rule, while known-host browser URLs are normalized into repository references.

Force bypasses the completed-file cache, not the existing resume protocol.
Validated partial transfers may resume.
A failed refresh exits unsuccessfully while preserving the previously usable entry.
Successful paths go to stdout in argument order; progress and errors go to stderr.
An update failure is not silently converted into cached success.

`JINFER_OFFLINE=1|true|on|yes` (case-insensitive) forbids both metadata and payload requests.
`0|false|off|no` disables the environment flag; `-Djinfer.offline=true` still forces offline mode.
Cached inference works offline; mutable pulls and forced remote downloads fail before mutation.
An ordinary pull of a cached exact filename at a full immutable commit can succeed offline.
A directory, branch, tag, abbreviated revision, or quant shorthand does not qualify for that shortcut without authoritative selection information.
The immutable-file shortcut is not an integrity scan; force requests a replacement.

Multi-file pulls are independent per-entry operations, not one transaction.
If A succeeds and B fails, A stays updated and B's previous usable copy stays available.
An already loaded model is not hot-swapped; atomic replacement affects subsequent opens.
If the OS refuses replacement of an open file, the update fails without deleting it first.

## Publication invariants

1. **No pre-eviction.** Never unlink or truncate a published file to initiate refresh.
2. **Old or complete new.** Readers see a complete published file, never partial replacement bytes or a deliberately introduced gap.
3. **Validate first.** Expected size, available checksum, and plain-URL HTML rejection are checked before publication.
   Checksum-free transfers are size-checked, not described as cryptographically verified.
4. **Preserve availability on failure.** Metadata, transfer, verification, cancellation, and pre-commit publication failures preserve the previously usable requested entry and its lookup path.
   An already corrupt or missing entry has no such availability guarantee.
5. **Publish selectors last.** HF branch/tag refs change only after the selected snapshot entry is ready.
6. **One remote revision.** When a commit can be resolved, listing, selection, and transfer use that commit.
   The original requested ref still determines flat-cache naming.
7. **No false success.** Subsequent ordinary lookup selects the refreshed content, absent another actor's mutation.
8. **Shared data remains shared.** Refresh does not garbage-collect HF snapshots or blobs.
   Same-hash repair publishes only bytes verified against that hash.
9. **One lock implementation.** Logical publication keys and payload destinations use the existing thread/process locks, released on every exit.
   Lock order is logical publication key, payload destination, then staging destination.
10. **Independent inputs.** Failure never pre-evicts another requested model or rolls back a completed sibling.
11. **Offline means no requests.** Check the offline gate before listing, size probes, or transfer.
12. **Housekeeping is not the commit.** Failure to remove stale partial metadata after publication does not report the completed replacement as a failed refresh.

These guarantees require same-filesystem atomic replacement.
An unsupported atomic move fails safely; there is no delete-then-move fallback.
This is not a power-loss durability promise or a transaction protocol shared with other HF clients.

## Implementation map

| Owner | Responsibility |
|---|---|
| `ModelStore` | Private `USE_CACHE` / `REFRESH` / `REDOWNLOAD` policy, source selection, checksummed reuse, destination precedence |
| `Fetch` | Shared locks, staged transfer, integrity checks, atomic promotion, best-effort post-commit cleanup |
| `RepositorySource` | Fetch from the selected revision and endpoint, including configured mirrors |
| Hub `Hub` | Shared blobs, snapshot links/copies, branch/tag publication |
| CLI `Hub` / `Options` | Delegate pull, validate inputs, map expected errors, print paths |

The public addition is `ModelStore.pullAll(List<String>, boolean force)`.
The existing multi-file runner preserves argument order and bounds concurrent work.
`ModelSource` remains source-compatible.
For replacement, a source receives an unpublished staging destination; even a source that writes directly cannot overwrite the working file.
`evict` remains an explicitly destructive API and is never used to start a pull.

### Selection and cache precedence

Remote selection precedes cached-content reuse for mutable pulls.
The post-publication check uses the same default-quant and sole-model rules as ordinary cache lookup.
An unrelated cached quant is harmless when the updated file still wins that lookup.
A filename change that would leave shorthand ambiguous or selecting another file is rejected with an exact-filename remedy; other cached files are not deleted to force a result.
Exact-file lookup tests the requested path itself, not a same-named child of a directory.

Non-default cache roots never write into the shared HF cache.
They may reuse matching shared content read-only.
At the default root, an owned flat entry has precedence and is refreshed in place; otherwise eligible HF downloads use the shared layout.
The original ref names a flat destination even when transfer is pinned to a newly resolved commit.

### Staging and HF publication

Replacement retains the working file while transferring into sibling staging files.
The existing size, SHA-256, ETag, retry, range, resume, and disk-space checks apply.
Staging files are excluded from model discovery and listing.
Forces bypass completed-file reuse; a staged replacement is validated again at the source boundary before promotion.

HF publication follows this order:

```text
lock logical revision
  -> resolve and pin commit
  -> select file
  -> verify and publish blob
  -> publish snapshot entry
  -> atomically publish branch/tag ref
```

Acquire the logical publication lock before selecting the revision, so an older selection cannot wait behind a newer publisher and then roll it back.
Metadata uses unique temporary names, since unrelated clients do not use Jinfer's locks.
An unchanged branch ref is not rewritten.
Without symlink support, snapshot entries are copies; a shared blob is never moved out from under another link.
A failed selector update may leave an unreferenced complete new blob or snapshot, which is preferable to breaking the old active entry.
Moving a branch does not prefetch unrequested files from its new revision.

## Regression coverage

Tests reuse JUnit, `FakeSource`, `FileServer`, and the CLI subprocess helpers.
HTTP requests go to loopback fixtures; no real model downloads are needed.
Subprocess pipes coordinate competing publishers without correctness assertions based on sleeps.

| Test class | Coverage |
|---|---|
| `FetchDownloadTest` | Transfer/integrity failures, interrupted pre-publication, failed promotion, cleanup failures, and two-JVM replacement while readers keep working |
| `ModelStorePullTest` | Use/update/repair call counts, same-size changes/corruption, custom-source failures, offline rules, pinned transfer, destination naming, default-quant coexistence, and multi-input preservation |
| `ModelStoreFindTest` | Exact-file and directory lookup, including a same-named child |
| `HubRefreshTest` | Ref-last publication, failed snapshot publication, unchanged-ref idempotence, shared-blob repair, wrong-link repair, and no-symlink copy fallback |
| `ModelStoreUrlTest` | URL refresh and HTML rejection before replacing working content |
| CLI `HubTest` | Failed force-pull preserves the flat cache; unchanged content avoids another payload fetch |
| CLI `DistributionIT` | Packaged offline preservation/exact-file rules and fake-HF updates, same-hash force, corrupt-ref repair, and flat-cache precedence |

The original flat/shared-cache deletion tests failed against the previous implementation and passed after replacement was implemented.
The default-quant, exact-file, and unchanged-ref regressions likewise reproduced their defects before refinement.

## Verification

Run from the repository root:

```sh
mvn -pl jinfer/jinfer-hub -am test
mvn -pl jinfer/jinfer-cli -am test -Dtest=HubTest,OptionsTest,WorkflowTest -Dsurefire.failIfNoSpecifiedTests=false
mvn -pl jinfer/jinfer-cli -am package -DskipTests
mvn -pl jinfer/jinfer-cli -am test -Dtest=DistributionIT -Dsurefire.failIfNoSpecifiedTests=false -Dsurefire.excludedGroups= -Dgroups=integration -Djinfer.test.jar="$(pwd)/jinfer/jinfer-cli/target/jinfer.jar"
make ci-format
make ci-test
```

The Linux refinement validation passed the full hub suite, CLI model-free tests, five freshly packaged-JAR checks, and `make ci-format`.
The last full model-free gate stopped at the pre-existing `ModelsDispatchTest.aKindThePortDoesNotImplementIsRefusedByName` assertion: it expects `UnsupportedOperationException` instead of `ModelProvider.IncompatibleModelException`.
Do not suppress that failure or report the full reactor as green.
The Kokoro/eSpeak integration failures are separate from these model-free checks.

The no-symlink fallback is tested using the JDK ZIP filesystem.
Windows open-file and atomic-replacement behavior still needs validation on Windows.
