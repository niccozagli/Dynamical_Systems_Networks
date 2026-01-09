import h5py
import numpy as np
from typing import cast


def open_stats_writer(
    path,
    fieldnames,
    *,
    compression="gzip",
    compression_opts=4,
    chunk_rows=1024,
    dtype=np.float64,
    swmr=True,
):
    fieldnames = tuple(fieldnames)
    if not fieldnames:
        raise ValueError("fieldnames must be non-empty.")

    fh = h5py.File(path, "a", libver="latest")
    if "stats" in fh:
        dset = cast(h5py.Dataset, fh["stats"])
        stored = dset.attrs.get("fieldnames")
        if stored is None:
            raise ValueError("Existing stats dataset missing fieldnames attribute.")
        stored = [s.decode("utf-8") for s in stored.tolist()]
        if tuple(stored) != fieldnames:
            raise ValueError("Existing stats dataset fieldnames do not match.")
    else:
        n_fields = len(fieldnames)
        chunk_rows = max(1, int(chunk_rows))
        dset = cast(h5py.Dataset, fh.create_dataset(
            "stats",
            shape=(0, n_fields),
            maxshape=(None, n_fields),
            chunks=(chunk_rows, n_fields),
            dtype=np.dtype(dtype),
            compression=compression,
            compression_opts=compression_opts,
        ))
        dset.attrs["fieldnames"] = np.asarray(fieldnames, dtype="S")

    idx = int(dset.shape[0])
    if swmr:
        fh.swmr_mode = True

    return ["h5", fh, dset, fieldnames, idx]


def write_stats(writer, row):
    _, fh, dset, fieldnames, idx = writer
    values = [row[name] for name in fieldnames]
    arr = np.asarray(values, dtype=dset.dtype)
    if arr.shape != (len(fieldnames),):
        raise ValueError("Stats row has wrong shape.")

    dset.resize((idx + 1, dset.shape[1]))
    dset[idx, :] = arr
    writer[4] = idx + 1  # update index in-place
    fh.flush()


def close_stats_writer(writer):
    writer[1].close()
