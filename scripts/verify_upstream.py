"""Verify that the vendored official INSID3 files are byte-identical to upstream.

Checks every file listed in upstream/UPSTREAM_MANIFEST.sha256 (generated from
github.com/visinf/INSID3 at the commit in upstream/UPSTREAM_COMMIT). Exit code 1 on
any mismatch. This is the evidence that the baseline was not modified.
"""
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    manifest = os.path.join(ROOT, "upstream", "UPSTREAM_MANIFEST.sha256")
    bad, n = [], 0
    with open(manifest) as fp:
        for line in fp:
            digest, rel = line.strip().split(maxsplit=1)
            path = os.path.join(ROOT, rel)
            n += 1
            if not os.path.isfile(path):
                bad.append(f"MISSING  {rel}")
                continue
            with open(path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != digest:
                    bad.append(f"MODIFIED {rel}")
    commit = open(os.path.join(ROOT, "upstream", "UPSTREAM_COMMIT")).read().strip()
    if bad:
        print("\n".join(bad))
        print(f"FAILED: {len(bad)}/{n} upstream files differ from commit {commit}")
        return 1
    print(f"OK: all {n} upstream files identical to visinf/INSID3@{commit[:7]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
