# Design document

The goal of this repository is to make image analysis algorithms scalable --- and seamlessly switch between local and distributed execution on HPC or other backends.

This project is meant as an evolution of [elf](https://github.com/constantinpape/elf) and [cluster-tools](https://github.com/constantinpape/cluster_tools), providing similar functionality but in a significantly more scalable, convenient, and robust fashion.
To implement the underlying algorithms use standard scientific Python libraries (numpy, pandas, scipy, scikit-image, etc.) and [bioimage-cpp](https://github.com/computational-cell-analytics/bioimage-cpp). Prefer using the latter if it offers the required functionality.

The algorithms we implement are generally block-wise processing of (large) image data.
They should keep a fall-back for direct application of the respective algorithm, see details below.

## Runner

The core element is a flexible runner implementation that can dispatch to local execution (a thread-pool) or distributed execution (for now we will implement slurm but may extend this in the future).

This should be implemented with a `Runner` class hierachy (an abstract base class + implementations for the different runners).
The runners then dispatch the computation etc.

Below is a simple example for how the implementation for computing the max could look based on this design.
    
    
```python
# All imports that have to be copied to a script running the computation have to have the COMPUTE IMPORT comment
import numpy as np  # COMPUTE IMPORT


# This defines the per-block computation.
def _compute(input_, mask_, block_roi):
    if mask_ is None:
        return np.max(input_[block_roi])
    block_mask = mask_[block_roi].astype(bool)
    mask_sum = block_mask.sum()
    if mask_sum == 0:  # Nothing is in the mask -> we don't compute the max.
        return None
    block_input = input_[block_roi]
    if mask_sum == block_mask.size()  # Everything is in the mask, we don't need to apply it.
        return np.max(block_input)
    return np.max(block_input[block_mask])


def max(
    input: ArrayLike,  # this can be a numpy array (only local execution) or a zarr array etc. (local and distributed)
    num_workers: int,  # Number of parallel workers.
    block_shape: Optional[Tuple[int, ...]] = None,  # The block shape, if none and single worker compute the max directly, otherwise block-wise.
    job_type: str = "local",  # Either 'local' or 'slurm' (may be extended in the future)
    job_config: Optional[Dict] = None,  # Extra configuration for the runner, especially for slurm.
    mask: Optional[ArrayLike] = None,  # Optional binary mask, only values within the mask will be taken into account.
    block_ids: Optional[Sequence[int]] = None,  # Optional block ids, computation will be restricted to those, relevant for re-running failed jobs.
) -> np.float64:
    """...
    """
    # First check if we are local and directly compute the result (no blocking needed).
    if job_type == "local" and num_workers == 1 and block_shape == None:
        if (mask is None) or (block_ids is None):
            raise ValueError  # Direct computation is not supported for or masks
        return np.max(input[:])

    # Then create and initialize the runner.
    # get_runner is convenience function that will return either the local or slurm runner, depending on the job type.
    # We specify the number of inputs (= input arrays / zarr-like containers) and outputs (= output arrays / containers).
    # Here we have just a single input and no output array (the output is a single scalar).
    runner = get_runner(job_type, n_inputs=1, n_outputs=0, **job_config)
    # Then we initialize the runner by passing the actual input(s), the number of workers, whether the function has a return value,
    # the block shape (which will be derived if None), the output(s) (here we don't have any), and the mask, and block_ids.
    # This takes care of validating the outputs, extracting filepaths and wrappers for distributed jobs (see below), etc.  
    runner.initialize(
        input, function=_compute, num_workers=num_workers, has_return_val=True,
        block_shape=block_shape, mask=mask, block_ids=block_ids, name="Compute max",
    )

    # Then run the computation. The runner takes care of providing the input(s), output(s) (here we don't have any),
    # the mask (if given) and the roi for the given block. It then records block success, updates progress, etc.
    # It collects the results in a list and returns it. 
    results = runner.run()

    # We combine the result locally (here a max over all block results, excluding empty ones due to a mask) 
    result = np.max([res for res in results if res is not None])
    return result
```

The implementation of this is straight-forward locally (futures.ThreadPoolExecutor and tqdm prog bar).

For the distributed case (slurm) it's more complicated. My current idea for the design:
- The runner checks the inputs and derives filepaths / internal paths from them (e.g. from the zarr array).
  If it is a numpy array or doesn't know the input type and can't determine paths than it fails.
  It determines the wrapper logic and wires it up (see below).
- It creates a temporary folder for the job, writes a slurm job config for an array job to it and a script that 
  runs the computation per task. This script is created by parsing the file, extracting the `_compute` function
  (whatever is passed to runner.run), and the needed imports. This is put into a small harness script that gets
  the blocks for this job, runs them with a loop and after each iteration saves the block success and return value
  by writing them to files in the result folder (the block success to a json file or so, the result as a pickle file).
- After submitting the array, the runner watches the respective slurm jobs. It pools this every 10 seconds
  (by default, make param in the config), and updates the progress report based on this
  (also reports how many jobs are pending, running, have succeeded, have failed). While doing this it should keep the folder clean,
  e.g., individual tasks write each job success to a single json, the runner loads these, writes to a summary json and deletes the individualones.
- After everything is done the runner checkes if computation was overall successfull. If yes, it unpickles the return values (if any),
  cleans up the temp folder, and returns the list of return vals. Otherwise it gives an error report and throws a runtime error.
  In the latter case the temp folder is not clear to enable debugging; the error report contains the path to the temp folder.


## Wrapper / On-the-fly transformation

We want to also enable on-the-fly transformations of the input data, e.g. thresholding the data.
This is quite easy for local execution but more complex for distributed cases.
In the latter we have to parse the wrapper to rebuild it where the code is run.
This means we will need a clear design for this. I have not further designed this yet, but mention this to make clear that this will be needed.
