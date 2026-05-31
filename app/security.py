"""보안(오프라인) 모드 — 외부 네트워크 통신 차단 + 유출 시도 시 강제종료.

동작 범위(정직하게)
  - 이 프로세스가 파이썬 `socket`으로 여는 모든 외부 연결을 가로챈다.
  - 루프백(127.x / ::1 / localhost)과 유닉스 소켓(IPC)만 허용한다.
  - 외부 주소로 connect/sendto를 시도하면 **전송 전에** 로그를 남기고
    `os._exit()`로 즉시 강제종료한다(데이터가 나가지 못함).
  - psutil이 있으면, 네이티브 코드가 파이썬 소켓을 우회하는 경우까지
    주기적으로 점검(보조 방어선)한다.

한계: 원시 패킷 스니핑이나 OS 전역 방화벽은 앱 내부에서 보장할 수 없다.
완전 격리가 필요하면 OS 방화벽/에어갭을 병행해야 한다.
"""
from __future__ import annotations

import os
import socket
import threading
import time

_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}

MODE_QUIT = "quit"   # 외부 시도 시 즉시 강제종료
MODE_WARN = "warn"   # 외부 시도 시 차단(연결 실패)만 하고 경고 기록

_enabled = False
_mode = MODE_QUIT
_log_path: str | None = None
_orig_connect = socket.socket.connect
_orig_connect_ex = socket.socket.connect_ex
_orig_sendto = socket.socket.sendto
_patched = False
_monitor_started = False
_violations: list = []  # (시각, 종류, 주소) — 경고 모드에서 UI가 폴링


class BlockedConnection(OSError):
    """보안 모드(경고)에서 외부 연결을 막을 때 발생."""


def is_enabled() -> bool:
    return _enabled


def set_mode(mode: str):
    global _mode
    _mode = MODE_WARN if mode == MODE_WARN else MODE_QUIT


def get_mode() -> str:
    return _mode


def violation_count() -> int:
    return len(_violations)


def last_violation():
    return _violations[-1] if _violations else None


def _is_local(address) -> bool:
    # AF_UNIX(문자열) = 로컬 IPC → 허용
    if isinstance(address, (str, bytes)):
        return True
    try:
        host = address[0]
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(host, str):
        return False
    return host in _LOCAL_HOSTS or host.startswith("127.")


def _violation(kind: str, address) -> None:
    """외부 통신 시도 감지 → 기록. 모드에 따라 강제종료 또는 차단(예외)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        if _log_path:
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] BLOCKED {kind} -> {address} (mode={_mode})\n")
    except Exception:  # noqa: BLE001
        pass
    _violations.append((ts, kind, address))
    if _mode == MODE_QUIT:
        os._exit(1)  # 데이터가 나가기 전에 즉시 종료
    # 경고 모드: 연결을 막되(예외) 종료는 하지 않음 → 데이터는 나가지 않음
    raise BlockedConnection(f"외부 통신 차단됨: {address}")


def _install_patches():
    global _patched
    if _patched:
        return
    _patched = True

    def guard_connect(self, address):
        if _enabled and not _is_local(address):
            _violation("connect", address)
        return _orig_connect(self, address)

    def guard_connect_ex(self, address):
        if _enabled and not _is_local(address):
            _violation("connect_ex", address)
        return _orig_connect_ex(self, address)

    def guard_sendto(self, data, *args):
        # sendto(data, address) 또는 sendto(data, flags, address)
        address = args[-1] if args else None
        if _enabled and address is not None and not _is_local(address):
            _violation("sendto", address)
        return _orig_sendto(self, data, *args)

    socket.socket.connect = guard_connect
    socket.socket.connect_ex = guard_connect_ex
    socket.socket.sendto = guard_sendto


def _start_monitor():
    """psutil이 있으면 비루프백 연결을 주기적으로 점검(보조)."""
    global _monitor_started
    if _monitor_started:
        return
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return
    _monitor_started = True
    proc = psutil.Process()

    def loop():
        while True:
            time.sleep(1.0)
            if not _enabled:
                continue
            try:
                for c in proc.net_connections(kind="inet"):
                    raddr = getattr(c, "raddr", None)
                    if raddr and raddr.ip and not (
                        raddr.ip in _LOCAL_HOSTS or raddr.ip.startswith("127.")
                    ):
                        _violation("psutil", (raddr.ip, raddr.port))
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=loop, daemon=True).start()


def enable(log_path: str | None = None, mode: str = MODE_QUIT):
    """보안 모드 켜기. 이후 외부 연결 시도는 차단된다(모드에 따라 종료/경고)."""
    global _enabled, _log_path
    _log_path = log_path
    set_mode(mode)
    _install_patches()
    _start_monitor()
    _enabled = True


def disable():
    global _enabled
    _enabled = False
