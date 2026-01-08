import csv

def open_stats_writer(path, fieldnames):
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(
        fh,
        fieldnames=fieldnames,
        extrasaction="raise",   # error on unknown keys
    )

    if fh.tell() == 0:
        writer.writeheader()
        fh.flush()

    return fh, writer

def write_stats(writer, fh, row):
    writer.writerow(row)
    fh.flush()