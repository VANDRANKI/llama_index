"""Async utilities for running coroutines and managing concurrency.

This module provides helpers for running asyncio coroutines from both
synchronous and asynchronous contexts, batching coroutine execution,
and throttling concurrency with a semaphore-based worker pool.
"""

import asyncio
import contextvars
import concurrent.futures
from itertools import zip_longest
from typing import Any, Coroutine, Iterable, List, Optional, TypeVar

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


def get_asyncio_module(show_progress: bool = False) -> Any:
    """Return the asyncio module or its tqdm-wrapped equivalent.

    When *show_progress* is ``True`` the function imports
    ``tqdm.asyncio.tqdm_asyncio`` and returns it in place of the
    standard :mod:`asyncio` module so that callers can swap in progress
    reporting with minimal code changes.

    Args:
        show_progress: If ``True``, return ``tqdm.asyncio.tqdm_asyncio``
            instead of the standard :mod:`asyncio` module.

    Returns:
        Either ``tqdm.asyncio.tqdm_asyncio`` (when *show_progress* is
        ``True``) or the built-in :mod:`asyncio` module.

    """
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


def asyncio_module(show_progress: bool = False) -> Any:
    import warnings

    warnings.warn(
        "asyncio_module() is deprecated and will be removed in a future release. "
        "Use get_asyncio_module() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_asyncio_module(show_progress=show_progress)


def asyncio_run(coro: Coroutine) -> Any:
    """
    Gets an existing event loop to run the coroutine.

    If there is no existing event loop, creates a new one.
    If an event loop is already running, uses threading to run in a separate thread.
    """
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # Check if the loop is already running
        if loop.is_running():
            # If loop is already running, run in a separate thread
            # Snapshot the current context so we can propagate contextvars
            ctx = contextvars.copy_context()

            def run_coro_in_thread() -> Any:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return ctx.run(new_loop.run_until_complete, coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_coro_in_thread)
                return future.result()
        else:
            # If we're here, there's an existing loop but it's not running
            return loop.run_until_complete(coro)

    except RuntimeError as e:
        # If we can't get the event loop, we're likely in a different thread
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                "Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc."
            )


def run_async_tasks(
    tasks: List[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> List[Any]:
    """Run a list of async tasks."""
    tasks_to_execute: List[Any] = tasks
    if show_progress:
        try:
            import nest_asyncio
            from tqdm.asyncio import tqdm

            # jupyter notebooks already have an event loop running
            # we need to reuse it instead of creating a new one
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> List[Any]:
                return await tqdm.gather(*tasks_to_execute, desc=progress_bar_desc)

            tqdm_outputs: List[Any] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        except Exception:
            pass

    async def _gather() -> List[Any]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: List[Any] = asyncio_run(_gather())
    return outputs


def chunks(iterable: Iterable, size: int) -> Iterable:
    """Split *iterable* into consecutive chunks of length *size*.

    The last chunk may be shorter than *size* if the iterable length is
    not evenly divisible.  Internally uses :func:`itertools.zip_longest`
    with ``fillvalue=None``, so callers should filter ``None`` sentinels
    when the fill value matters.

    Args:
        iterable: Any iterable to be chunked.
        size: Maximum number of elements per chunk.

    Yields:
        Tuples of at most *size* elements from *iterable*.

    Example:
        >>> list(chunks([1, 2, 3, 4, 5], 2))
        [(1, 2), (3, 4), (5, None)]

    """
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=None)


async def batch_gather(
    tasks: List[Coroutine], batch_size: int = 10, verbose: bool = False
) -> List[Any]:
    """Gather coroutines in sequential batches to limit peak concurrency.

    Unlike :func:`asyncio.gather` which launches all coroutines at once,
    this helper divides *tasks* into batches of *batch_size* and awaits
    each batch before starting the next.  This is useful when the total
    number of coroutines is large and unbounded concurrency would exhaust
    connection pools or hit API rate limits.

    Args:
        tasks: Coroutines to execute.
        batch_size: Maximum number of coroutines running concurrently at
            any given time.  Defaults to 10.
        verbose: If ``True``, print a progress message after each batch
            completes.

    Returns:
        A flat list of results in the same order as *tasks*.

    """
    output: List[Any] = []
    for task_chunk in chunks(tasks, batch_size):
        task_chunk = (task for task in task_chunk if task is not None)
        output_chunk = await asyncio.gather(*task_chunk)
        output.extend(output_chunk)
        if verbose:
            print(f"Completed {len(output)} out of {len(tasks)} tasks")
    return output


DEFAULT_NUM_WORKERS = 4

T = TypeVar("T")


@dispatcher.span
async def run_jobs(
    jobs: List[Coroutine[Any, Any, T]],
    show_progress: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
    desc: Optional[str] = None,
) -> List[T]:
    """
    Run jobs.

    Args:
        jobs (List[Coroutine]):
            List of jobs to run.
        show_progress (bool):
            Whether to show progress bar.

    Returns:
        List[Any]:
            List of results.

    """
    semaphore = asyncio.Semaphore(workers)

    @dispatcher.span
    async def worker(job: Coroutine) -> Any:
        async with semaphore:
            return await job

    pool_jobs = [worker(job) for job in jobs]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(*pool_jobs, desc=desc)
    else:
        results = await asyncio.gather(*pool_jobs)

    return results
