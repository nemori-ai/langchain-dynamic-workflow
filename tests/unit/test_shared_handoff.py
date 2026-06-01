"""Unit tests for the ``/shared/`` artifact hand-off and path-traversal guard.

These pin the locked Phase 4 hand-off semantics: a producer leaf writes an
artifact under ``/shared/`` (into its own producer namespace), a consumer leaf
reads it back, per-leaf isolated paths never leak between sibling backends sharing
one shared store (the #2884 risk verified independently), and a ``..`` traversal
that tries to escape the shared route is normalized and blocked before routing.
"""

from __future__ import annotations

import threading

import pytest

from langchain_dynamic_workflow._sandbox import (
    SHARED_ROUTE_PREFIX,
    InMemorySandbox,
    SharedArtifactStore,
    build_leaf_backend,
    normalize_path,
    normalize_within_route,
)


def test_normalize_path_collapses_dot_segments() -> None:
    assert normalize_path("/shared/./a/b") == "/shared/a/b"
    assert normalize_path("/shared/a/../b") == "/shared/b"
    assert normalize_path("/a//b/") == "/a/b"


def test_normalize_path_rejects_escape_above_root() -> None:
    # A traversal that would climb above root must be rejected, not silently
    # clamped — this is the guard that stops "/shared/../secret" from escaping
    # the shared route into another backend's namespace.
    with pytest.raises(ValueError, match="escapes root"):
        normalize_path("/shared/../../etc/passwd")
    with pytest.raises(ValueError, match="escapes root"):
        normalize_path("/../escape")


def test_normalize_within_route_allows_isolated_path_named_like_route() -> None:
    # Regression: the bare-route guard must match the route EXACTLY, not by
    # endswith. An isolated path whose final segment merely happens to be "shared"
    # (/a/shared, /myproject/shared/notes.txt) is a legitimate private path the
    # composite routes to the isolated backend — NOT an escape out of /shared/.
    # The old endswith heuristic rejected these with a misleading "escapes root".
    assert normalize_within_route("/a/shared", route_prefix=SHARED_ROUTE_PREFIX) == "/a/shared"
    assert (
        normalize_within_route("/myproject/shared/notes.txt", route_prefix=SHARED_ROUTE_PREFIX)
        == "/myproject/shared/notes.txt"
    )
    # The genuine bare-route and under-route cases still pass through unchanged.
    assert normalize_within_route("/shared", route_prefix=SHARED_ROUTE_PREFIX) == "/shared"
    assert (
        normalize_within_route("/shared/r.txt", route_prefix=SHARED_ROUTE_PREFIX) == "/shared/r.txt"
    )


def test_normalize_within_route_still_blocks_escape_out_of_route() -> None:
    # The #2884 escape out of the shared route must remain blocked after the
    # exact-match fix: "/shared/../secret" canonicalizes outside /shared/.
    with pytest.raises(ValueError, match="escapes root"):
        normalize_within_route("/shared/../secret", route_prefix=SHARED_ROUTE_PREFIX)
    with pytest.raises(ValueError, match="escapes root"):
        normalize_within_route("/shared/../../etc/passwd", route_prefix=SHARED_ROUTE_PREFIX)


async def test_shared_handoff_producer_writes_consumer_reads() -> None:
    # The hand-off contract: a producer writes under /shared/, a consumer with a
    # DIFFERENT isolated backend but the SAME shared store reads it back.
    store = SharedArtifactStore()
    producer = build_leaf_backend(
        isolated=InMemorySandbox(identity="prod"), shared_store=store, producer="prod"
    )
    consumer = build_leaf_backend(
        isolated=InMemorySandbox(identity="cons"), shared_store=store, producer="cons"
    )
    await producer.awrite("/shared/report.txt", "findings")
    read = await consumer.aread("/shared/report.txt")
    assert read.file_data is not None
    assert read.file_data["content"] == "findings"


async def test_isolated_paths_do_not_leak_across_shared_store() -> None:
    # #2884 verified independently: two leaves sharing one shared store must keep
    # their NON-shared (isolated) writes private. A file written outside /shared/
    # by one leaf must be invisible to the other, even though both composite
    # backends point at the same shared store.
    store = SharedArtifactStore()
    a = build_leaf_backend(isolated=InMemorySandbox(identity="a"), shared_store=store, producer="a")
    b = build_leaf_backend(isolated=InMemorySandbox(identity="b"), shared_store=store, producer="b")
    await a.awrite("/work/private.txt", "secret-a")
    # b cannot see a's isolated file.
    read = await b.aread("/work/private.txt")
    assert read.file_data is None
    assert read.error is not None


async def test_shared_writes_are_producer_namespaced() -> None:
    # Two producers writing the same /shared/ path must not clobber each other:
    # each lands in its own producer namespace under the shared store.
    store = SharedArtifactStore()
    a = build_leaf_backend(isolated=InMemorySandbox(identity="a"), shared_store=store, producer="a")
    b = build_leaf_backend(isolated=InMemorySandbox(identity="b"), shared_store=store, producer="b")
    await a.awrite("/shared/out.txt", "from-a")
    await b.awrite("/shared/out.txt", "from-b")
    # Distinct namespaces => both artifacts coexist in the shared store.
    assert store.read_namespaced("a", "/out.txt") == "from-a"
    assert store.read_namespaced("b", "/out.txt") == "from-b"


def test_shared_store_is_concurrency_safe_under_threaded_fanout() -> None:
    # The hand-off path touches one in-memory store from multiple OS threads: under
    # ctx.parallel a producer's write_namespaced runs on one to_thread worker while
    # a consumer's read_merged / stored_paths_under reads the same dict on another.
    # This exercises that exact contention — writer threads grow the dict while
    # reader threads concurrently read/iterate it, all joined (bounded, no
    # busy-spin) — and pins two invariants: (1) no thread raises (the dict-mutation
    # race in particular), and (2) the final state reflects EVERY write with no
    # dropped or torn entry. The lock is what makes each read snapshot atomic with
    # respect to writes, the cross-thread safety the JournalStore protocol mandates
    # for fan-out, here made explicit rather than relying on the GIL.
    #
    # Note: under standard CPython the store's `sorted(dict.items())` / dict
    # comprehension snapshots are GIL-atomic, so this is a correctness + contention
    # guard rather than a deterministic reproducer of the bare-iteration race; the
    # lock additionally protects the contract on free-threaded (no-GIL) builds.
    store = SharedArtifactStore()
    writes_per_writer = 500
    num_writers = 4
    reader_iterations = 4_000
    errors: list[BaseException] = []
    errors_lock = threading.Lock()
    num_readers = 2
    # Parties: every writer + every reader + the main thread that releases them.
    start = threading.Barrier(num_writers + num_readers + 1)

    def writer(writer_index: int) -> None:
        try:
            start.wait()
            for i in range(writes_per_writer):
                # Each write adds a distinct key, so the dict grows under any reader
                # iterating it concurrently on the unlocked path.
                store.write_namespaced(
                    f"p{writer_index}", f"/w{writer_index}-{i}.txt", f"{writer_index}:{i}"
                )
        except BaseException as exc:  # catch the dict-mutation race (and anything)
            with errors_lock:
                errors.append(exc)

    def reader() -> None:
        try:
            start.wait()
            for i in range(reader_iterations):
                # read_merged sorts+iterates the whole dict; stored_paths_under
                # comprehends it — both iterate while writers mutate.
                store.read_merged(f"/w0-{i % writes_per_writer}.txt")
                if i % 50 == 0:
                    store.stored_paths_under("/")
        except BaseException as exc:  # catch the dict-mutation race (and anything)
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(num_writers)]
    threads += [threading.Thread(target=reader) for _ in range(num_readers)]
    for thread in threads:
        thread.start()
    start.wait()  # release all threads at once for maximal overlap
    for thread in threads:
        thread.join(timeout=10.0)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []  # no thread tripped the dict-mutation race (or anything else)
    # Every write landed and is readable: the lock dropped no mutation.
    assert len(store.stored_paths_under("/")) == num_writers * writes_per_writer
    assert store.read_merged("/w0-0.txt") == "0:0"


async def test_traversal_through_shared_route_is_blocked() -> None:
    # A leaf that tries to escape the shared route with ".." must be blocked at
    # the backend boundary rather than reaching another namespace.
    store = SharedArtifactStore()
    leaf = build_leaf_backend(
        isolated=InMemorySandbox(identity="x"), shared_store=store, producer="x"
    )
    res = await leaf.awrite("/shared/../escape.txt", "payload")
    assert res.error is not None
    assert "escapes root" in res.error


async def test_guarded_backend_grep_reaches_composite_not_not_implemented() -> None:
    # Regression guard: the guarded wrapper must DELEGATE grep to the wrapped
    # composite, not shadow it with the protocol's bare NotImplementedError. A
    # leaf's grep over its isolated workspace must reach the InMemorySandbox and
    # return real matches.
    store = SharedArtifactStore()
    leaf = build_leaf_backend(
        isolated=InMemorySandbox(identity="g"), shared_store=store, producer="g"
    )
    await leaf.awrite("/notes.txt", "alpha\nTODO: fix\nbeta")
    result = await leaf.agrep("TODO", "/")
    assert result.error is None
    assert result.matches is not None
    assert [(m["path"], m["line"], m["text"]) for m in result.matches] == [
        ("/notes.txt", 2, "TODO: fix")
    ]


async def test_guarded_backend_glob_reaches_composite_not_not_implemented() -> None:
    # The guarded wrapper must delegate glob to the composite too, so a leaf's
    # file-pattern search reaches the isolated backend instead of raising.
    store = SharedArtifactStore()
    leaf = build_leaf_backend(
        isolated=InMemorySandbox(identity="gl"), shared_store=store, producer="gl"
    )
    await leaf.awrite("/a.py", "x")
    await leaf.awrite("/b.txt", "y")
    result = await leaf.aglob("*.py", "/")
    assert result.error is None
    assert result.matches is not None
    assert [m["path"] for m in result.matches] == ["/a.py"]


def test_guarded_backend_execute_delegates_to_isolated() -> None:
    # execute() is not path-routable, so the guarded wrapper must delegate it
    # straight to the isolated sandbox rather than shadow it with the protocol's
    # bare NotImplementedError. This pins the execution-capability wiring for the
    # needs_execution tier the whole phase exists to serve (offline echo backend).
    store = SharedArtifactStore()
    leaf = build_leaf_backend(
        isolated=InMemorySandbox(identity="ex"), shared_store=store, producer="ex"
    )
    result = leaf.execute("echo build")
    assert result.exit_code == 0
    assert result.output == "echo build"


async def test_guarded_backend_grep_on_shared_route_is_routable() -> None:
    # The shared route the finding specifically calls out: a consumer leaf must be
    # able to grep an artifact a producer wrote under /shared/, through the guarded
    # composite — not hit NotImplementedError at the guard boundary.
    store = SharedArtifactStore()
    producer = build_leaf_backend(
        isolated=InMemorySandbox(identity="p"), shared_store=store, producer="p"
    )
    consumer = build_leaf_backend(
        isolated=InMemorySandbox(identity="c"), shared_store=store, producer="c"
    )
    await producer.awrite("/shared/report.txt", "intro\nFINDING: leak\nend")
    result = await consumer.agrep("FINDING", "/shared/report.txt")
    assert result.error is None
    assert result.matches is not None
    assert any(m["text"] == "FINDING: leak" for m in result.matches)


async def test_guarded_backend_upload_download_round_trip() -> None:
    # upload_files / download_files must delegate through the guard to the isolated
    # backend (which now implements them), round-tripping bytes — not raise.
    store = SharedArtifactStore()
    leaf = build_leaf_backend(
        isolated=InMemorySandbox(identity="u"), shared_store=store, producer="u"
    )
    uploaded = await leaf.aupload_files([("/data.bin", b"payload")])
    assert [r.error for r in uploaded] == [None]
    downloaded = await leaf.adownload_files(["/data.bin"])
    assert downloaded[0].error is None
    assert downloaded[0].content == b"payload"


async def test_guarded_backend_blocks_traversal_on_grep_glob_and_transfer() -> None:
    # The traversal guard must cover the newly-delegated surface too: a ".." escape
    # out of the shared route on grep / glob / upload / download is rejected at the
    # boundary, exactly as it is for write/read/edit/ls.
    store = SharedArtifactStore()
    leaf = build_leaf_backend(
        isolated=InMemorySandbox(identity="t"), shared_store=store, producer="t"
    )
    grep_res = await leaf.agrep("x", "/shared/../escape")
    assert grep_res.error is not None and "escapes root" in grep_res.error
    glob_res = await leaf.aglob("*.txt", "/shared/../escape")
    assert glob_res.error is not None and "escapes root" in glob_res.error
    up = await leaf.aupload_files([("/shared/../escape.bin", b"x")])
    assert up[0].error == "invalid_path"
    down = await leaf.adownload_files(["/shared/../escape.bin"])
    assert down[0].error == "invalid_path"
