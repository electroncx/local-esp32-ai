"""Run the built firmware in Espressif's ESP32 emulator and ask it questions.

    cd firmware && pio run -e esp32dev-dio
    python qemu_test.py --qemu /path/to/qemu-system-xtensa < questions.txt

Needs Espressif's QEMU fork (github.com/espressif/qemu/releases, the
qemu-xtensa-softmmu build). Prints: question <TAB> answer. Timings in the
emulator are not real ESP32 timings, so none are printed."""

import argparse
import os
import select
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def esptool():
    """esptool as PlatformIO installs it (a script up to v4, a program from v5),
    or as a pip package."""
    pkgs = os.path.expanduser("~/.platformio/packages")
    for cmd in ([sys.executable, os.path.join(pkgs, "tool-esptoolpy", "esptool.py")],
                [os.path.join(pkgs, "tool-esptool", "esptool")]):
        if os.path.exists(cmd[-1]):
            return cmd
    return [sys.executable, "-m", "esptool"]


def build_files(build):
    """Bootloader, partition table and app in a build folder: PlatformIO's
    names (firmware.bin) or the Arduino IDE's (<sketch>.ino.bin)."""
    if os.path.exists(os.path.join(build, "firmware.bin")):
        return [os.path.join(build, n) for n in ("bootloader.bin", "partitions.bin", "firmware.bin")]
    app = [n for n in os.listdir(build) if n.endswith(".ino.bin")]
    if len(app) != 1:
        sys.exit(f"{build}: no firmware.bin or <sketch>.ino.bin")
    stem = os.path.join(build, app[0][:-4])
    return [stem + ".bootloader.bin", stem + ".partitions.bin", stem + ".bin"]


def merged_flash(build, out):
    """One 4 MB flash image: bootloader, partition table and app, as the chip sees them."""
    boot, table, app = build_files(build)
    parts = ["0x1000", boot, "0x8000", table, "0x10000", app]
    tool = esptool() + ["--chip", "esp32"]
    for args in (["merge_bin", "-o", out, "--fill-flash-size", "4MB"],   # esptool 4
                 ["merge-bin", "-o", out, "--pad-to-size", "4MB"]):     # esptool 5
        r = subprocess.run(tool + args + parts, capture_output=True, text=True)
        if r.returncode == 0:
            return
    sys.exit(f"esptool could not merge the flash image:\n{r.stdout}{r.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qemu", default="qemu-system-xtensa")
    # the emulator can't switch flash to QIO, so it runs the DIO build of the same code
    ap.add_argument("--build", default=os.path.join(HERE, ".pio/build/esp32dev-dio"),
                    help="a PlatformIO build folder, or an Arduino IDE export (arduino-cli --output-dir)")
    args = ap.parse_args()
    build = args.build
    questions = [l.rstrip("\n") for l in sys.stdin if l.strip()]

    flash = os.path.join(tempfile.mkdtemp(), "flash.bin")
    merged_flash(build, flash)
    p = subprocess.Popen([args.qemu, "-nographic", "-machine", "esp32", "-m", "4M",
                          "-drive", f"file={flash},if=mtd,format=raw", "-serial", "mon:stdio"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf = b""

    def read_until(token, timeout):
        nonlocal buf
        end = time.time() + timeout
        while token not in buf and time.time() < end:
            if select.select([p.stdout], [], [], 0.2)[0]:
                buf += os.read(p.stdout.fileno(), 4096)
        if token not in buf:
            return None
        out, buf = buf.split(token, 1)
        return out.decode(errors="replace")

    boot = read_until(b"you: ", 120)
    if boot is None:
        p.kill()
        sys.exit("firmware did not reach the prompt")
    print(next((l for l in boot.splitlines() if "tinyai" in l), boot.strip()), file=sys.stderr)
    for q in questions:
        p.stdin.write(q.encode() + b"\r")
        p.stdin.flush()
        out = read_until(b"you: ", 300)
        if out is None:
            p.kill()
            sys.exit(f"no answer to {q!r}")
        ans = next((l[5:] for l in out.splitlines() if l.startswith("esp: ")), "")
        print(f"{q}\t{ans}", flush=True)
    p.kill()


if __name__ == "__main__":
    main()
