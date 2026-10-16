#!/usr/bin/env python3
"""Build or run the isolated capability example, preserving real command evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

from validation import PATTERNS, TENSOR_BYTES, ValidationError, compare_outputs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCRATCH = ROOT / ".scratch" / "adreno"
BUILD = SCRATCH / "build"
REMOTE_PREFIX = "/data/local/tmp/bend-adreno-gemv."
BINARIES = ("cpu-android", "gemv-android")
DEPENDENCIES = ("bend-ml-tensor@0.1.2.0", "bend-ml-tensor-array@0.1.7.0",
                "bend-ml-nat-lemmas@0.1.0.0")


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def atomic_json(path, value):
    """Replace a receipt, never leaving partial JSON or an earlier success current."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


class Evidence:
    def __init__(self, directory, current, kind):
        self.directory = Path(directory)
        self.current = Path(current)
        self.data = dict(schema_version=1, kind=kind, attempt_id=uuid.uuid4().hex,
                         started_at=now(), status="running", acceptance_met=False,
                         commands=[], cases=[])
        self.persist()

    def persist(self):
        atomic_json(self.directory / "receipt.json", self.data)
        atomic_json(self.current, self.data)

    def invoke(self, label, command, timeout, cwd=None, remote_deadline=False):
        item = dict(label=label, command=[str(part) for part in command],
                    cwd=str(cwd) if cwd else None, started_at=now(), status="running",
                    exit_code=None, timed_out=False, stdout="", stderr="", acceptance_met=False)
        self.data["commands"].append(item)
        self.persist()
        try:
            process = subprocess.run(item["command"], cwd=cwd, capture_output=True,
                                     text=True, errors="replace", timeout=timeout)
            item.update(exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr,
                        status="exited", timed_out=remote_deadline and process.returncode in (124, 137))
        except subprocess.TimeoutExpired as error:
            item.update(stdout=text(error.stdout), stderr=text(error.stderr),
                        status="timeout", timed_out=True)
        except OSError as error:
            item.update(status="launch-error", error=str(error))
        item["acceptance_met"] = item["exit_code"] == 0 and not item["timed_out"]
        item["finished_at"] = now()
        self.persist()
        return item

    def finish(self, accepted, error=None):
        self.data.update(status="completed" if accepted else "failed", acceptance_met=accepted,
                         finished_at=now())
        if error is not None:
            self.data["error"] = str(error)
        self.persist()


def require_success(item):
    if not item["acceptance_met"]:
        raise RuntimeError(f"{item['label']}: {item['status']}, exit={item['exit_code']}, "
                           f"timed_out={item['timed_out']}")
    return item


def executable(value):
    found = shutil.which(value)
    if found is None:
        raise RuntimeError(f"executable not found: {value}")
    return str(Path(found).resolve(strict=True))


def source_identities(bend):
    paths = [ROOT / "llama" / "main.bend", ROOT / "reference" / "test_adreno_gemv.py"]
    paths.extend(path for path in HERE.iterdir()
                 if path.is_file() and path.suffix in (".bend", ".c", ".js", ".py", ".sh", ".md"))
    paths.append(Path(bend).parent.parent / "bend2" / "base.bend")
    cache = Path.home() / ".bend" / "lib"
    for name in DEPENDENCIES:
        alias = cache / "names" / name
        content = alias.read_text().strip()
        if re.fullmatch(r"0x[0-9a-f]+", content) is None:
            raise RuntimeError(f"invalid Bend package cache identity: {alias}")
        paths.extend((alias, cache / content / "main.bend"))
    return [identity(path) for path in sorted(set(paths))]


def build():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="build-attempt-", dir=SCRATCH))
    evidence = Evidence(attempt, SCRATCH / "build-receipt.json", "build")
    try:
        bend = executable(os.environ.get("BEND", "bend"))
        if not os.environ.get("ANDROID_CC"):
            raise RuntimeError("set ANDROID_CC to the existing Android/Bionic compiler wrapper")
        cc = executable(os.environ["ANDROID_CC"])
        evidence.data.update(bend=identity(bend), android_cc=identity(cc),
                             sources=source_identities(bend), build_directory=str(BUILD))
        # Preserve the actual underlying tools for the supplied isolated wrapper as well.
        tools = [Path(cc).parent / "host" / "bin" / "ld.lld"]
        clang = os.environ.get("CLANG") or shutil.which("clang")
        if clang:
            tools.append(Path(clang))
        evidence.data["compiler_tools"] = [identity(path) for path in tools if path.is_file()]
        evidence.persist()
        version = require_success(evidence.invoke("bend-version", [bend, "version"], 30))
        if re.search(r"(?<![0-9.])2\.0\.37(?![0-9.])", version["stdout"] + version["stderr"]) is None:
            raise RuntimeError("this example requires the repository's pinned Bend 2.0.37")
        require_success(evidence.invoke("android-compiler-version", [cc, "--version"], 30))
        BUILD.mkdir(parents=True, exist_ok=True)
        commands = (
            ("emit-cpu", [bend, str(HERE / "cpu.bend"), "-o", str(BUILD / "cpu.generated.c")]),
            ("emit-gemv", [bend, str(HERE / "main.bend"), "-o", str(BUILD / "gemv.generated.c")]),
            ("emit-js", [bend, str(HERE / "main.bend"), "-o", str(BUILD / "gemv.js")]),
            ("compile-cpu", [cc, "-I" + str(HERE), "-ffp-contract=off", "-fno-fast-math",
                             str(BUILD / "cpu.generated.c"), "-o", str(BUILD / "cpu-android")]),
            ("compile-gemv", [cc, "-I" + str(HERE), "-ffp-contract=off", "-fno-fast-math",
                              str(BUILD / "gemv.generated.c"), "-o", str(BUILD / "gemv-android")]),
        )
        for label, command in commands:
            require_success(evidence.invoke(label, command, 300, cwd=ROOT))
        if source_identities(bend) != evidence.data["sources"]:
            raise RuntimeError("sources changed during the build; discard this build attempt")
        evidence.data["artifacts"] = {name: identity(BUILD / name)
                                      for name in (*BINARIES, "gemv.js", "cpu.generated.c", "gemv.generated.c")}
        evidence.finish(True)
        print(f"Build receipt: {evidence.current}; no generated program has been executed.")
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        evidence.finish(False, error)
        print(f"Build failed: {error}; evidence: {attempt}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        evidence.finish(False, "interrupted")
        raise


def validate_remote_path(remote):
    if re.fullmatch(re.escape(REMOTE_PREFIX) + r"[A-Za-z0-9]{6,32}", remote) is None:
        raise ValidationError("unexpected mktemp path; refusing device writes")
    return remote


def validate_device_uid(stdout):
    uid = stdout.strip()
    if re.fullmatch(r"[0-9]+", uid) is None or int(uid) == 0:
        raise ValidationError("an unprivileged Android shell UID is required; root is refused")
    return int(uid)


def ownership_command(remote, empty=False):
    validate_remote_path(remote)
    path = shlex.quote(remote)
    command = (f"test -d {path} && test ! -L {path} && "
               f'test "$(stat -c %u {path})" = "$(id -u)" && '
               f'test "$(stat -c %a {path})" = 700')
    if empty:
        command += f' && test -z "$(find {path} -mindepth 1 -maxdepth 1 -print)"'
    return command


def validate_remote_hashes(stdout, expected):
    observed = {}
    for line in stdout.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or match[2] in observed:
            raise ValidationError("malformed or duplicate device SHA256 output")
        observed[match[2]] = match[1]
    if observed != expected:
        raise ValidationError("device bytes differ from the recorded local inputs")


def accepted_refusal(item):
    return (item["status"] == "exited" and not item["timed_out"] and item["exit_code"] == 95
            and "Adreno.gemv: JavaScript target refused" in item["stdout"] + item["stderr"])


def run(args):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="run-", dir=SCRATCH))
    evidence = Evidence(attempt, SCRATCH / "latest.json", "device-run")
    evidence.data.update(adb_serial=args.serial, per_program_timeout_seconds=args.timeout)
    evidence.persist()
    try:
        adb = executable(args.adb)
        bun = executable(args.bun)
        evidence.data.update(adb=identity(adb), bun=identity(bun))
        manifest = json.loads((SCRATCH / "build-receipt.json").read_text())
        if manifest.get("status") != "completed" or manifest.get("acceptance_met") is not True:
            raise RuntimeError("the latest build did not complete successfully")
        if source_identities(manifest["bend"]["path"]) != manifest["sources"]:
            raise RuntimeError("sources no longer match the accepted build receipt")
        for name, expected in manifest["artifacts"].items():
            if identity(BUILD / name) != expected:
                raise RuntimeError(f"artifact changed since build: {name}")
        tensor = identity(args.tensor)
        if tensor["bytes"] != TENSOR_BYTES:
            raise ValidationError(f"tensor must be exactly {TENSOR_BYTES} bytes")
        evidence.data.update(build_receipt=manifest, tensor=tensor)
        evidence.persist()

        def adb_command(label, *arguments, remote_deadline=False):
            return evidence.invoke(label, [adb, "-s", args.serial, *arguments],
                                   args.timeout + 15, remote_deadline=remote_deadline)

        def setup(label, *arguments):
            return require_success(adb_command(label, *arguments))

        setup("adb-version", "version")
        setup("device-state", "get-state")
        setup("device-identity", "shell", "id; uname -a; getprop ro.product.manufacturer; "
              "getprop ro.product.model; getprop ro.build.fingerprint; getprop ro.serialno")
        uid = setup("unprivileged-device-uid", "shell", "id -u")
        evidence.data["device_uid"] = validate_device_uid(uid["stdout"])
        evidence.persist()
        physical = setup("reject-emulator", "shell", "getprop ro.kernel.qemu")
        if physical["stdout"].strip() == "1":
            raise RuntimeError("the capability runner requires a physical Android device")
        created = setup("create-exclusive-directory", "shell", "umask 077; mktemp -d " + REMOTE_PREFIX + "XXXXXXXX")
        remote = validate_remote_path(created["stdout"].strip())
        evidence.data["remote_directory"] = remote
        evidence.persist()
        setup("validate-empty-owned-directory", "shell", ownership_command(remote, empty=True))
        setup("create-owned-model-layout", "shell", f"umask 077; mkdir {remote}/model {remote}/model/w")
        uploads = {name: (BUILD / name, manifest["artifacts"][name]["sha256"]) for name in BINARIES}
        uploads["model/w/h0.q.bin"] = (Path(tensor["path"]), tensor["sha256"])
        expected_hashes = {}
        for name, (local, digest) in uploads.items():
            setup("validate-ownership-before-push-" + name, "shell", ownership_command(remote))
            setup("push-" + name, "push", str(local), remote + "/" + name)
            expected_hashes[remote + "/" + name] = digest
        setup("set-owned-executable-modes", "shell", f"chmod 700 {remote}/cpu-android {remote}/gemv-android")
        hash_command = "sha256sum " + " ".join(shlex.quote(path) for path in expected_hashes)
        hashed = setup("verify-uploaded-bytes", "shell", hash_command)
        validate_remote_hashes(hashed["stdout"], expected_hashes)
        evidence.data["uploaded_sha256"] = expected_hashes
        evidence.persist()
        for pattern, mode in enumerate(PATTERNS):
            case = dict(pattern=pattern, mode=mode, status="running", acceptance_met=False)
            evidence.data["cases"].append(case)
            evidence.persist()
            cpu_command = (f"cd {remote} && exec timeout -s KILL {args.timeout} ./cpu-android "
                           f"--threads 1 --gpu off {remote}/model {mode}")
            gpu_command = (f"cd {remote} && exec timeout -s KILL {args.timeout} ./gemv-android "
                           f"--threads 1 --gpu off {remote}/model/w/h0.q.bin {mode}")
            cpu = adb_command("cpu-" + mode, "shell", cpu_command, remote_deadline=True)
            gpu = adb_command("gpu-" + mode, "shell", gpu_command, remote_deadline=True)
            case["cpu_command_index"] = evidence.data["commands"].index(cpu)
            case["gpu_command_index"] = evidence.data["commands"].index(gpu)
            try:
                require_success(cpu)
                require_success(gpu)
                case.update(compare_outputs(cpu["stdout"], gpu["stdout"], pattern))
                case["status"] = "accepted" if case["acceptance_met"] else "numeric-failure"
            except (ValidationError, RuntimeError) as error:
                case.update(status="failed", error=str(error), acceptance_met=False)
            evidence.persist()
            print(f"{mode}: {case['status']}")
        final_hashes = setup("verify-final-device-bytes", "shell", hash_command)
        validate_remote_hashes(final_hashes["stdout"], expected_hashes)
        if identity(tensor["path"]) != tensor:
            raise RuntimeError("local tensor changed during the attempt")
        refusal = evidence.invoke("javascript-refusal", [bun, str(BUILD / "gemv.js"), tensor["path"], "ones"], args.timeout)
        refusal["acceptance_met"] = accepted_refusal(refusal)
        evidence.persist()
        accepted = all(case["acceptance_met"] for case in evidence.data["cases"]) and refusal["acceptance_met"]
        evidence.finish(accepted)
        print(f"Acceptance: {accepted}; full evidence: {attempt / 'receipt.json'}")
        return 0 if accepted else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        evidence.finish(False, error)
        print(f"Attempt failed: {error}; full evidence: {attempt / 'receipt.json'}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        evidence.finish(False, "interrupted")
        raise


def bounded_seconds(value):
    number = int(value)
    if not 1 <= number <= 600:
        raise argparse.ArgumentTypeError("timeout must be between 1 and 600 seconds")
    return number


def explicit_serial(value):
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is None:
        raise argparse.ArgumentTypeError("supply one nonempty explicit ADB serial")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    subcommands.add_parser("build", help="emit and cross-compile only; execute no generated program")
    device = subcommands.add_parser("run", help="run on one explicit physical ADB serial")
    device.add_argument("--serial", type=explicit_serial, required=True)
    device.add_argument("--tensor", type=Path, required=True)
    device.add_argument("--adb", default="adb")
    device.add_argument("--bun", default="bun")
    device.add_argument("--timeout", type=bounded_seconds, default=45)
    args = parser.parse_args()
    os.environ["BEND_NO_TELEMETRY"] = "1"
    return build() if args.operation == "build" else run(args)


if __name__ == "__main__":
    sys.exit(main())
