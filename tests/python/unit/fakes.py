"""Test doubles: synthetic tar streams, an in-process fake NIM HTTP server, fake clients and sources."""
from __future__ import annotations

import email
import email.policy
import socket
import tarfile
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from earth2_dku.arrays import npy_bytes
from earth2_dku.clients.base import ForecastStep, HealthStatus, NimClient
from earth2_dku.errors import NimStreamBrokenError
from earth2_dku.extract.grid import ERA5_0P25
from earth2_dku.models.registry import ModelSpec
from earth2_dku.sources.base import InitialConditionSource
from earth2_dku.times import floor_to_cycle, parse_time

# --------------------------------------------------------------------------- models


def make_model(n_channels=4, model_id="fake-model", supports="unverified", min_gpu=40, variables_from="fake_e2s:VARIABLES"):
    return ModelSpec(
        model_id=model_id, label="Fake", nim_image=f"nvcr.io/nim/nvidia/fourcastnet:{model_id}",
        input_variables_from=variables_from, n_channels=n_channels, step_hours=6,
        min_gpu_mem_gb=min_gpu, supports_variables_param=supports, grid="era5_0p25",
    )


# ------------------------------------------------------------------ synthetic data


def encoded_fields(variables, lead_hours=0, grid=ERA5_0P25):
    """Fields whose values encode their own position so extraction is checkable.

    lat     -> latitude
    lon     -> longitude in 0..360 (discontinuous at 0 deg)
    lonc    -> (lon + 180) % 360, continuous across the 0 deg meridian
    lead    -> lead_hours
    other   -> lat + 1000 * index
    """
    lat2d, lon2d = np.meshgrid(grid.lats, grid.lons, indexing="ij")
    out = []
    for k, v in enumerate(variables):
        if v == "lat":
            f = lat2d
        elif v == "lon":
            f = lon2d
        elif v == "lonc":
            f = np.mod(lon2d + 180.0, 360.0)
        elif v == "lead":
            f = np.full_like(lat2d, float(lead_hours))
        else:
            f = lat2d + 1000.0 * k + lead_hours
        out.append(f)
    return np.stack(out).astype("float32")[None, None]  # (1, 1, n, 721, 1440)


def step_name(lead, batch=0):
    return f"{lead:03d}_{batch:03d}.npy"


SMALL_STEP_SHAPE = (1, 1, 2, 150, 300)  # ~350 KB per step: larger than the client's read pieces


def small_steps(n_steps=3, shape=SMALL_STEP_SHAPE):
    return [(step_name(6 * i), npy_bytes(np.full(shape, i, dtype="float32"))) for i in range(n_steps)]


def tar_member_bytes(name: str, data: bytes) -> bytes:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    return info.tobuf(format=tarfile.USTAR_FORMAT) + data + b"\0" * ((512 - len(data) % 512) % 512)


# ------------------------------------------------------------- fake NIM over HTTP


class FakeNimServer:
    """Serves /v1/health/ready and /v1/infer (chunked tar stream) on localhost.

    break_after=k: send k complete members, then half of the next one, then drop the connection.
    reject_variables: answer 400 mentioning `variables` when the field is sent.
    """

    def __init__(self, files, break_after=None, reject_variables=False, infer_status=200, health_status=200,
                 chunk_size=16 * 1024):
        self.files = list(files)
        self.break_after = break_after
        self.reject_variables = reject_variables
        self.infer_status = infer_status
        self.health_status = health_status
        self.chunk_size = chunk_size
        self.requests: list[dict] = []
        self._server = None

    def __enter__(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                body = b"ok" if fake.health_status == 200 else b"not ready"
                self.send_response(fake.health_status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                msg = email.message_from_bytes(
                    b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw,
                    policy=email.policy.HTTP,
                )
                fields = {}
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    payload = part.get_payload(decode=True)
                    fields[name] = payload if name == "input_array" else payload.decode()
                fields["accept"] = self.headers.get("accept")
                fake.requests.append(fields)
                if fake.reject_variables and "variables" in fields:
                    return self._plain(400, b'{"detail": "unexpected field: variables"}')
                if fake.infer_status != 200:
                    return self._plain(fake.infer_status, b'{"detail": "boom"}')
                self.send_response(200)
                self.send_header("Content-Type", "application/x-tar")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                members = [tar_member_bytes(n, d) for n, d in fake.files]
                if fake.break_after is None:
                    payload, broken = b"".join(members) + b"\0" * 1024, False
                else:
                    k = fake.break_after
                    payload = b"".join(members[:k]) + (members[k][: len(members[k]) // 2] if k < len(members) else b"")
                    broken = True
                for i in range(0, len(payload), fake.chunk_size):
                    chunk = payload[i:i + fake.chunk_size]
                    self.wfile.write(b"%X\r\n" % len(chunk) + chunk + b"\r\n")
                if broken:
                    self.wfile.write(b"%X\r\n" % 5000 + b"partial")  # chunk promised but never delivered
                    self.wfile.flush()
                    self.close_connection = True
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    return
                self.wfile.write(b"0\r\n\r\n")

            def _plain(self, status, body):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


# ---------------------------------------------------------------- in-process fakes


class FakeNimClient(NimClient):
    """Yields given steps; optionally breaks after ``break_after`` steps."""

    def __init__(self, files, break_after=None, ready=True, variables_dropped=False):
        self.files = list(files)
        self.break_after = break_after
        self.ready = ready
        self.variables_dropped = variables_dropped
        self.calls = []
        self.last_variables_sent = None

    def health(self):
        return HealthStatus(self.ready, "fake ready" if self.ready else "fake endpoint down")

    def infer_stream(self, input_array_bytes, init_time, simulation_length, variables):
        self.calls.append(dict(init_time=init_time, simulation_length=simulation_length, variables=variables))
        self.last_variables_sent = None if self.variables_dropped else (list(variables) if variables else None)
        from earth2_dku.clients.base import parse_step_filename

        for i, (name, data) in enumerate(self.files):
            if self.break_after is not None and i >= self.break_after:
                raise NimStreamBrokenError(steps_received=i, detail="fake break")
            lead, batch = parse_step_filename(name)
            yield ForecastStep(lead, batch, name, data)


class FakeSource(InitialConditionSource):
    name = "fake"

    def __init__(self, now="2024-06-10T09:30", shape_override=None):
        self.now = parse_time(now)
        self.calls = []
        self.shape_override = shape_override

    def latest_available_time(self):
        return floor_to_cycle(self.now - timedelta(hours=6))

    def get_state(self, time, variables):
        self.calls.append((time, list(variables)))
        shape = self.shape_override or (1, 1, len(variables), 721, 1440)
        return np.zeros(shape, dtype="float32")


def write_synthetic_run(storage, init_time="2023-01-01T00", variables=("lat", "lon", "lonc"), leads=(0, 6),
                        status="complete", model=None, source="era5_arco"):
    """Write a run whose step fields encode their own coordinates (see encoded_fields)."""
    from earth2_dku.forecast.manifest import Manifest, StepRecord, run_dir_for, write_manifest

    model = model or make_model()
    run_dir = run_dir_for(init_time)
    m = Manifest.new(model, source, init_time, max(len(leads) - 1, 0), list(variables), ERA5_0P25.to_dict())
    for lead in leads:
        rel = f"steps/{step_name(lead)}"
        storage.write_bytes(f"{run_dir}/{rel}", npy_bytes(encoded_fields(variables, lead)))
        m.steps.append(StepRecord(rel, lead, 0))
    m.status, m.steps_received = status, len(leads)
    write_manifest(storage, m, run_dir)
    return m
