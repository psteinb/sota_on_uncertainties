from pathlib import Path


def main(tablefile, prefix=".", verbose=False):

    tfile = Path(tablefile)
    assert tfile.exists(), f"tablefile {tfile} does not exist"
    assert tfile.is_file()

    lines = tfile.read_text().split("\n")
    success = 0

    for line in lines:
        src_, dst_ = line.split(" ")
        src = Path(src_)
        dst = Path(dst_)

        assert src.exists() and src.is_file(), f"file {src} does not exist"
        if dst.exists():
            print(f"Warning: unable to link {src}->{dst} as latter exists")
        dstp = dst.parent
        if not dstp.exists():
            dstp.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src)
        success += 1
        if verbose:
            print(f"{src} -> {dst}")

    if success == len(lines):
        return 0
    else:
        return 1


if __name__ == "__main__":
    if "snakemake" in globals() and (
        hasattr(snakemake, "input") and hasattr(snakemake, "output")
    ):
        opath = Path(snakemake.output[0])
        value = main(snakemake.input[0], opath)
        sys.exit(value)
    else:
        inpath = sys.argv[1] if len(sys.argv) > 1 else None
        outpath = sys.argv[2] if len(sys.argv) > 2 else "."

        value = main(inpath, outpath, kfolds, seed)

    sys.exit(value)
