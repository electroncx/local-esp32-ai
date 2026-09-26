"""Run the built firmware in Espressif's ESP32 emulator and ask it questions.

    cd firmware && pio run -e esp32dev-dio
    python qemu_test.py --qemu /path/to/qemu-system-xtensa < questions.txt

Needs Espressif's QEMU fork (github.com/espressif/qemu/releases, the
qemu-xtensa-softmmu build). Prints: question <TAB> answer. Timings in the
emulator are not real ESP32 timings, so none are printed."""

import argparse
import glob
import os
import select
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def merged_flash(build, out):
    esptool = glob.glob(os.path.expanduser("~/.platformio/packages/tool-esptoolpy/esptool.py"))[0]
    subprocess.run([sys.executable, esptool, "--chip", "esp32", "merge_bin", "-o", out,
                    "--fill-flash-size", "4MB", "0x1000", f"{build}/bootloader.bin",
                    "0x8000", f"{build}/partitions.bin", "0x10000", f"{build}/firmware.bin"],
                   check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qemu", default="qemu-system-xtensa")
    # the emulator can't switch flash to QIO, so it runs the DIO build of the same code
    ap.add_argument("--build", default=os.path.join(HERE, ".pio/build/esp32dev-dio"))
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
