"""Fail early on deployment identity and output permissions, without changing them."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Iterable


def check_runtime_identity() -> None:
    """Check the runtime identity when CURASYN_UID/CURASYN_GID are configured."""
    uid, gid = os.environ.get("CURASYN_UID"), os.environ.get("CURASYN_GID")
    if uid is None and gid is None:
        return
    if not uid or not gid or not uid.isdecimal() or not gid.isdecimal():
        raise PermissionError("[PRECHECK] CURASYN_UID/CURASYN_GID must be numeric; use host id -u/id -g")
    if int(uid) == 0 or (os.geteuid(), os.getegid()) != (int(uid), int(gid)):
        raise PermissionError(
            f"[PRECHECK] runtime uid:gid={os.geteuid()}:{os.getegid()} "
            f"does not match a non-root deployment identity {uid}:{gid}; "
            "check CURASYN_UID/CURASYN_GID and the user running this process"
        )


def _existing_parent(path: Path) -> Path:
    while not path.exists():
        if path.is_symlink() or path.parent == path:
            raise FileNotFoundError(f"Unresolvable output path: {path}")
        path = path.parent
    return path


def _details(path: Path) -> str:
    try:
        existing = _existing_parent(path)
        info = existing.stat()
        return f"existing={existing} owner={info.st_uid}:{info.st_gid} mode={stat.S_IMODE(info.st_mode):04o}"
    except OSError:
        return "owner/mode unavailable (check parent directory permissions)"


def check_writable_outputs(
    *,
    files: Iterable[str | Path] = (),
    directories: Iterable[str | Path] = (),
) -> None:
    """Probe the actual output locations, never truncate files or retain probes.

    Missing directories are checked through their nearest existing parent, not
    created here. This preserves no-clobber and check-only workflow semantics.
    Permissions can still change after this point; normal write errors propagate.
    """
    check_runtime_identity()
    file_paths = list(dict.fromkeys(Path(path).absolute() for path in files))
    directory_paths = list(dict.fromkeys(
        [Path(path).absolute() for path in directories]
        + [path.parent for path in file_paths]
    ))
    checked: set[Path] = set()
    targets = [(path, True) for path in directory_paths] + [(path, False) for path in file_paths]
    for path, is_directory in targets:
        try:
            if is_directory:
                existing = _existing_parent(path)
                if not existing.is_dir():
                    raise NotADirectoryError(str(existing))
                if existing not in checked:
                    with tempfile.NamedTemporaryFile(prefix=".curasyn-write-check-", dir=existing) as probe:
                        probe.write(b"permission check\n")
                        probe.flush()
                    checked.add(existing)
            else:
                try:
                    info = path.stat()
                except FileNotFoundError:
                    if path.is_symlink():
                        raise
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise OSError("Output is not a regular file")
                # No O_CREAT or O_TRUNC: existing output bytes remain untouched.
                descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
                os.close(descriptor)
        except OSError as error:
            message = (
                f"[PRECHECK] output not writable: {path}; "
                f"runtime uid:gid={os.geteuid()}:{os.getegid()}; {_details(path)}; "
                f"reason={error}. Ask the host owner/admin to repair this exact path; "
                "no ownership or permissions were changed."
            )
            print(message, flush=True)
            raise PermissionError(message) from error
    print(
        f"[PRECHECK] output permissions OK uid:gid={os.geteuid()}:{os.getegid()} "
        f"files={len(file_paths)} directories={len(directory_paths)}",
        flush=True,
    )
