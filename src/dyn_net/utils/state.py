import h5py
import numpy as np
from typing import cast


def open_state_writer(
    path,
    dim,
    *,
    compression="gzip",
    compression_opts=4,
    chunk_rows=1,
    dtype=np.float64,
    swmr=True,
):
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")

    fh = h5py.File(path, "a", libver="latest")

    if "state" in fh:
        dset_state = cast(h5py.Dataset, fh["state"])
        if dset_state.shape[1] != dim:
            raise ValueError("Existing state dataset has different dimension.")
        dset_time = cast(h5py.Dataset, fh["time"])
        dset_step = cast(h5py.Dataset, fh["step"])
    else:
        chunk_rows = max(1, int(chunk_rows))
        dset_state = cast(
            h5py.Dataset,
            fh.create_dataset(
                "state",
                shape=(0, dim),
                maxshape=(None, dim),
                chunks=(chunk_rows, dim),
                dtype=np.dtype(dtype),
                compression=compression,
                compression_opts=compression_opts,
            ),
        )
        dset_time = cast(
            h5py.Dataset,
            fh.create_dataset(
                "time",
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_rows,),
                dtype=np.float64,
            ),
        )
        dset_step = cast(
            h5py.Dataset,
            fh.create_dataset(
                "step",
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_rows,),
                dtype=np.int64,
            ),
        )
        dset_state.attrs["dim"] = dim

    idx = int(dset_state.shape[0])
    if swmr:
        fh.swmr_mode = True

    return ["h5_state", fh, dset_state, dset_time, dset_step, idx]


def write_state(writer, x, t, step):
    _, fh, dset_state, dset_time, dset_step, idx = writer
    x_arr = np.asarray(x, dtype=dset_state.dtype).reshape(-1)
    if x_arr.shape[0] != dset_state.shape[1]:
        raise ValueError("State has wrong dimension.")

    new_size = idx + 1
    dset_state.resize((new_size, dset_state.shape[1]))
    dset_time.resize((new_size,))
    dset_step.resize((new_size,))

    dset_state[idx, :] = x_arr
    dset_time[idx] = float(t)
    dset_step[idx] = int(step)

    writer[5] = new_size  # update index in-place
    fh.flush()


def close_state_writer(writer):
    writer[1].close()
