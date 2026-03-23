import concurrent.futures
import os
from typing import Any, Callable, List
from tqdm import tqdm

def concurrent_task_executor(
    task: Callable[[Any], None],
    data_list: List[Any],
    max_workers: int | None = None,
    description: str = None,
) -> None:
    """
    Execute tasks concurrently on a list of data objects using ThreadPoolExecutor.
    Args:
        task (Callable): The function to apply to each data object.
        data_list (List): The list of data objects.
        max_workers (int): The maximum number of worker threads (default is 32).
        description (str, optional): Description for the progress bar.
    Raises:
        ValueError: If data_list is empty.
    Example:
        >>> def process_data(data):
        >>>     # Process data here
        >>>     pass
        >>> data_list = [1, 2, 3, 4, 5]
        >>> concurrent_task_executor(process_data, data_list, max_workers=8, description="Processing data")
    """

    if not data_list:
        raise ValueError("Data list is empty. No tasks to execute.")

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map streams work instead of allocating one Future per item,
        # which keeps memory use significantly lower on large datasets.
        with tqdm(total=len(data_list), desc=description) as pbar:
            for _ in executor.map(task, data_list):
                pbar.update(1)