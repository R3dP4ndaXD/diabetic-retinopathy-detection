import concurrent.futures
import os
from typing import Any, Callable, List
from tqdm import tqdm

def concurrent_task_executor(
    task: Callable[[Any], Any],  # Fixed: Task now returns Any (not None)
    data_list: List[Any],
    max_workers: int | None = None,
    description: str = None,
) -> List[Any]:                    # Fixed: Returns a List of results
    """
    Execute tasks concurrently on a list of data objects using ThreadPoolExecutor.
    """

    if not data_list:
        raise ValueError("Data list is empty. No tasks to execute.")

    # Laptop Safety: If your laptop completely freezes during preprocessing, 
    # change this to max(1, os.cpu_count() // 2) to leave cores free for your OS.
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map streams work instead of allocating one Future per item
        with tqdm(total=len(data_list), desc=description) as pbar:
            for result in executor.map(task, data_list):
                results.append(result)  # Capture the return value
                pbar.update(1)
                
    return results