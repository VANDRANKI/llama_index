"""Async utility helpers for running coroutines from both sync and async contexts.

This module provides:

- :func:`asyncio_run` -- run a coroutine from synchronous code, even when an
  event loop is already running (e.g. inside Jupyter notebooks).
- :func:`run_async_tasks` -- gather a list of coroutines with optional tqdm
  progress reporting.
- :func:`batch_gather` -- gather coroutines in fixed-size batches to limit
  concurrency without a semaphore.
- :func:`run_jobs` -- gather coroutines with a semaphore-based worker pool
  and optional tqdm progress reporting (instrumented via the dispatcher).
"""

import asyncio
import contextvars
import concurrent.futures
from itertools import zip_longest
from typing import Any, Coroutine, Iterable, List, Optional, TypeVar

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


def get_asyncio_module(show_progress: bool = False) -> Any:
    """Return the asyncio-compatible gather module to use for task collection.

    When *show_progress* is ``True``, returns ``tqdm.asyncio.tqdm_asyncio`` so
    that callers can swap it in for ``asyncio`` transparently.  Otherwise the
    standard :mod:`asyncio` module is returned.

    Args:
        show_progress (bool): If ``True``, return ``tqdm_asyncio`` for
            progress-bar-aware gathering.  Defaults to ``False``.

    Returns:
        The asyncio or tqdm_asyncio module.
    """
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


def asyncio_module(show_progress: bool = False) -> Any:
    """Deprecated alias for :func:`get_asyncio_module`.

    .. deprecated::
        Use :func:`get_asyncio_module` instead.  This alias will be removed in
        a future release.
    """
    import warnings

    warnings.warn(
        "asyncio_module() is deprecated and will be removed in a future release. "
        "Use get_asyncio_module() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_asyncio_module(show_progress=show_progress)


def asyncio_run(coro: Coroutine) -> Any:
    """Run *coro* to completion from synchronous code, handling nested loops.

    This helper attempts to reuse an already-existing event loop where
    possible.  When an event loop is currently running (as is the case inside
    Jupyter notebooks or when called from within another coroutine), the
    coroutine is submitted to a new event loop running in a background thread
    so that the caller does not block the existing loop.

    Args:
        coro (Coroutine): The coroutine to execute.

    Returns:
        Any: The return value of *coro*.

    Raises:
        RuntimeError: If a nested async environment is detected and
            ``nest_asyncio`` has not been applied.  The error message includes
            instructions for resolving the issue.
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

    except RuntimeError:
        # If we can't get the event loop, we're likely in a different thread
        try:
            return asyncio.run(coro)
        except RuntimeError:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                "Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc."
            )


def run_async_tasks(
    tasks: List[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> List[Any]:
    """Run a list of coroutines concurrently and return their results.

    All tasks are gathered via :func:`asyncio.gather`.  When *show_progress*
    is ``True``, ``tqdm.asyncio`` is used to display a progress bar; if tqdm
    is unavailable the tasks run without a progress indicator.

    Args:
        tasks (List[Coroutine]): Coroutines to execute.
        show_progress (bool): Whether to display a tqdm progress bar.
            Defaults to ``False``.
        progress_bar_desc (str): Description label shown on the tqdm bar.
            Defaults to ``"Running async tasks"``.

    Returns:
        List[Any]: Results from each coroutine in the same order as *tasks*.
    """
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
    """Split *iterable* into consecutive chunks of at most *size* items.

    The last chunk may be padded with ``None`` values to reach *size*.

    Args:
        iterable (Iterable): The source iterable to partition.
        size (int): Maximum number of items per chunk.

    Returns:
        Iterable: An iterable of tuples, each of length *size*.
    """
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=None)


async def batch_gather(
    tasks: List[Coroutine], batch_size: int = 10, verbose: bool = False
) -> List[Any]:
    """Gather *tasks* in sequential batches to bound peak concurrency.

    Unlike a semaphore-based approach, tasks in later batches do not start
    until all tasks in the current batch have completed.  This is simpler but
    less efficient when tasks have uneven durations.

    Args:
        tasks (List[Coroutine]): Coroutines to execute.
        batch_size (int): Maximum number of coroutines to run concurrently
            within a single batch.  Defaults to ``10``.
        verbose (bool): If ``True``, print a progress line after each batch.
            Defaults to ``False``.

    Returns:
        List[Any]: Results from all tasks in the same order as *tasks*.
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
    """Run *jobs* concurrently using a semaphore-based worker pool.

    At most *workers* coroutines execute at the same time.  Results are
    returned in the same order as the input list regardless of completion
    order.

    This function is instrumented with the LlamaIndex dispatcher so that span
    events are emitted for observability integrations.

    Args:
        jobs (List[Coroutine]): Coroutines to execute.
        show_progress (bool): Whether to display a tqdm progress bar.
            Defaults to ``False``.
        workers (int): Maximum number of coroutines that may run concurrently.
            Defaults to :data:`DEFAULT_NUM_WORKERS` (``4``).
        desc (str | None): Optional description label for the tqdm progress
            bar.  Ignored when *show_progress* is ``False``.

    Returns:
        List[T]: Results from all jobs in input order.
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
