"""Resolve data.pyg.org via Google DoH wireformat (POST, RFC 8484) and
install the prebuilt torch_sparse wheel.

The local resolver fails for data.pyg.org; the JSON API returns only the
CNAME; the wireformat POST returns full answers including A records.
"""

import struct
import subprocess
import urllib.request
from pathlib import Path

WHL = ("https://data.pyg.org/whl/torch-2.11.0%2Bcu128/"
       "torch_sparse-0.6.19%2Bpt211cu128-cp311-cp311-win_amd64.whl")
HOST = "data.pyg.org"


def build_query(name: str, qtype: int = 1) -> bytes:
    q = struct.pack(">HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0)
    for part in name.split("."):
        q += bytes([len(part)]) + part.encode()
    q += b"\x00" + struct.pack(">HH", qtype, 1)
    return q


def wire_query(name: str) -> bytes:
    req = urllib.request.Request(
        "https://dns.google/dns-query",
        data=build_query(name),
        headers={"Content-Type": "application/dns-message",
                 "Accept": "application/dns-message"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read()


def parse_records(data: bytes) -> tuple[list[str], list[str]]:
    """Returns (ips, cnames) from the answer section."""
    _, _, qd, an, _, _ = struct.unpack(">HHHHHH", data[:12])
    i = 12
    while data[i] != 0:
        i += data[i] + 1
    i += 5  # null terminator + qtype + qclass
    ips, cnames = [], []
    for _ in range(an):
        if data[i] & 0xC0 == 0xC0:
            i += 2
        else:
            while data[i] != 0:
                i += data[i] + 1
            i += 1
        rtype, _rclass, _ttl, rdlen = struct.unpack(">HHIH", data[i:i + 10])
        rdata = data[i + 10:i + 10 + rdlen]
        if rtype == 1 and rdlen == 4:
            ips.append(".".join(str(b) for b in rdata))
        elif rtype == 5:
            name, j = "", 0
            while j < len(rdata) and rdata[j] != 0 and not (rdata[j] & 0xC0):
                ln = rdata[j]
                name += rdata[j + 1:j + 1 + ln].decode(errors="replace") + "."
                j += ln + 1
            cnames.append(name.rstrip("."))
        i += 10 + rdlen
    return ips, cnames


def resolve(name: str, depth: int = 0) -> str:
    if depth > 5:
        raise RuntimeError("CNAME chain too deep")
    ips, cnames = parse_records(wire_query(name))
    if ips:
        return ips[0]
    if cnames:
        print(f"  {name} CNAME -> {cnames[0]}")
        return resolve(cnames[0], depth + 1)
    raise RuntimeError(f"no records for {name}")


def main() -> int:
    ip = resolve(HOST)
    print("resolved:", HOST, "->", ip)
    out = Path("torch_sparse.whl")
    subprocess.run(
        ["curl", "-sL", "--resolve", f"{HOST}:443:{ip}", "-o", str(out),
         "--retry", "3", "--max-time", "600", WHL],
        check=True,
    )
    print("wheel bytes:", out.stat().st_size)
    return subprocess.run(
        ["uv", "pip", "install", "--python", ".venv", str(out.resolve())]
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
