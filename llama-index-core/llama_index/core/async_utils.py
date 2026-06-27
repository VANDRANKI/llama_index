"""Async utilities for running coroutines and managing concurrency."""

import asyncio
import contextvars
import concurrent.futures
from itertools import zip_longest
from typing import Any, AsyncIterator, Coroutine, Generator, Iterable, Iterator, List, Optional, TypeVar

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)

T = TypeVar("T")

DEFAULT_NUM_WORKERS = 4


def get_asyncio_module(show_progress: bool = False) -> Any:
    """Return the asyncio module or a tqdm-wrapped variant for progress display.

    Args:
        show_progress: If ``True``, return ``tqdm.asyncio.tqdm_asyncio`` so that
            gathered coroutines render a progress bar.  If ``False`` (default),
            return the standard :mod:`asyncio` module.

    Returns:
        Either :mod:`asyncio` or :class:`tqdm.asyncio.tqdm_asyncio`.
    """
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


def asyncio_module(show_progress: bool = False) -> Any:
    """Return the asyncio module, optionally with tqdm progress support.

    .. deprecated::
        Use :func:`get_asyncio_module` instead.  This function will be removed
        in a future release.

    Args:
        show_progress: Passed through to :func:`get_asyncio_module`.

    Returns:
        Either :mod:`asyncio` or :class:`tqdm.asyncio.tqdm_asyncio`.
    """
    import warnings

    warnings.warn(
        "asyncio_module() is deprecated and will be removed in a future release. "
        "Use get_asyncio_module() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_asyncio_module(show_progress=show_progress)


def asyncio_run(coro: Coroutine[Any, Any, T]) -> T:
    """Run *coro* on an event loop, handling both sync and async call sites.

    Behaviour:

    - If there is an existing event loop that is **not** running, the coroutine
      is run on that loop directly via :meth:`asyncio.AbstractEventLoop.run_until_complete`.
    - If the existing event loop **is** running (e.g. inside a Jupyter notebook
      or another async framework), the coroutine is scheduled in a brand-new
      loop on a background thread so that the caller is not blocked.
    - If no event loop exists at all, :func:`asyncio.run` is used as a fallback.

    Args:
        coro: The coroutine to execute.

    Returns:
        The value returned by *coro*.

    Raises:
        RuntimeError: If nested async cannot be handled automatically.  In that
            case, apply ``nest_asyncio.apply()`` before calling this function or
            use an async entry point (``aquery()``, ``achat()``, etc.).
    """
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # Check if the loop is already running
        if loop.is_running():
            # If loop is already running, run in a separate thread.
            # Snapshot the current context so contextvars are propagated.
            ctx = contextvars.copy_context()

            def run_coro_in_thread() -> T:
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
            # Existing loop is not running — use it directly.
            return loop.run_until_complete(coro)

    except RuntimeError:
        # No accessible event loop; fall back to asyncio.run().
        try:
            return asyncio.run(coro)
        except RuntimeError:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow "
                "nested event loops, or use async entry methods like `aquery()`, "
                "`aretriever`, `achat()`, etc."
            )


def run_async_tasks(
    tasks: List[Coroutine[Any, Any, T]],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> List[T]:
    """Run a list of async tasks concurrently and return their results.

    Args:
        tasks: List of coroutines to execute concurrently.
        show_progress: If ``True``, display a ``tqdm`` progress bar while the
            tasks are running.  Falls back to non-progress execution if tqdm or
            nest_asyncio are unavailable.
        progress_bar_desc: Label shown on the progress bar when
            *show_progress* is ``True``.

    Returns:
        List of results in the same order as *tasks*.
    """
    tasks_to_execute: List[Any] = tasks
    if show_progress:
        try:
            import nest_asyncio
            from tqdm.asyncio import tqdm

            # Jupyter notebooks already have a running event loop;
            # nest_asyncio lets us reuse it instead of creating a new one.
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> List[T]:
                return await tqdm.gather(*tasks_to_execute, desc=progress_bar_desc)

            tqdm_outputs: List[T] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        except Exception:
            # Run without tqdm if it is unsupported in the current environment.
            pass

    async def _gather() -> List[T]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: List[T] = asyncio_run(_gather())
    return outputs


def chunks(iterable: Iterable[T], size: int) -> Iterator[tuple]:
    """Split *iterable* into consecutive chunks of at most *size* items.

    The last chunk is padded with ``None`` values if the iterable length is not
    a multiple of *size*.

    Args:
        iterable: Any iterable to chunk.
        size: Maximum number of items per chunk.

    Yields:
        Tuples of length *size*.  The final tuple may contain ``None`` padding.
    """
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=None)


async def batch_gather(
    tasks: List[Coroutine[Any, Any, T]],
    batch_size: int = 10,
    verbose: bool = False,
) -> List[T]:
    """Run coroutines in batches, gathering *batch_size* at a time.

    Useful for limiting the number of in-flight requests to an external API
    without using a semaphore.

    Args:
        tasks: Coroutines to execute.
        batch_size: How many coroutines to run concurrently per batch.
        verbose: If ``True``, print progress to stdout after each batch.

    Returns:
        Flat list of results in submission order.
    """
    output: List[T] = []
    for task_chunk in chunks(tasks, batch_size):
        task_chunk = (task for task in task_chunk if task is not None)
        output_chunk: tuple = await asyncio.gather(*task_chunk)
        output.extend(output_chunk)
        if verbose:
            print(f"Completed {len(output)} out of {len(tasks)} tasks")
    return output


@dispatcher.span
async def run_jobs(
    jobs: List[Coroutine[Any, Any, T]],
    show_progress: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
    desc: Optional[str] = None,
) -> List[T]:
    """Run async jobs with a bounded concurrency semaphore.

    All *jobs* are submitted immediately, but at most *workers* coroutines are
    allowed to execute concurrently at any given time.

    Args:
        jobs: List of coroutines to run.
        show_progress: Whether to render a ``tqdm`` progress bar.
        workers: Maximum number of concurrently running coroutines.
        desc: Description label for the progress bar (used when
            *show_progress* is ``True``).

    Returns:
        List of results in the same order as *jobs*.
    """
    semaphore = asyncio.Semaphore(workers)

    @dispatcher.span
    async def worker(job: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await job

    pool_jobs = [worker(job) for job in jobs]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        results: List[T] = await tqdm_asyncio.gather(*pool_jobs, desc=desc)
    else:
        results = await asyncio.gather(*pool_jobs)

    return results
