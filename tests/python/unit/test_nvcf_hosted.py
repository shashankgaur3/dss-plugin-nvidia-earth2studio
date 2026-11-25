import io
import tarfile

import pytest

from earth2_dku.clients.nvcf_hosted import (
    ASSETS_URL, INFER_URL, NvcfHostedClient, is_large_asset_body, sniff_archive, unpack_archive,
)
from earth2_dku.errors import ConfigError, HostedLargeAssetError, NimRequestError
from earth2_dku.times import parse_time
from fakes import FakeResponse, FakeSession, build_tar, small_steps, zip_bytes

STATUS = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
UPLOAD = "https://s3.example/upload"
FILES = small_steps(3, shape=(1, 1, 2, 3, 4))


def _session(*infer_responses, status=None, extra=None):
    routes = {
        ("POST", ASSETS_URL): [FakeResponse(200, json_body={"uploadUrl": UPLOAD, "assetId": "asset-1"})],
        ("PUT", UPLOAD): [FakeResponse(200)],
        ("POST", INFER_URL): list(infer_responses),
    }
    if status:
        routes[("GET", STATUS)] = status
    routes.update(extra or {})
    return FakeSession(routes)


def _run(session):
    client = NvcfHostedClient("key", session=session, sleep=lambda s: None)
    return client, list(client.infer_stream(b"in", parse_time("2023-01-01"), 2, ["u10m", "v10m"]))


def test_requires_api_key():
    with pytest.raises(ConfigError):
        NvcfHostedClient("")


def test_sniffing():
    assert sniff_archive(zip_bytes(FILES)) == "zip"
    assert sniff_archive(build_tar(FILES)) == "tar"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo("000_000.npy")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"abc"))
    assert sniff_archive(buf.getvalue()) == "gzip"
    assert [s.filename for s in unpack_archive(buf.getvalue())] == ["000_000.npy"]
    with pytest.raises(NimRequestError):
        unpack_archive(b"hello world")


def test_direct_200_zip():
    session = _session(FakeResponse(200, zip_bytes(list(reversed(FILES)))))
    client, steps = _run(session)
    assert [s.lead_hours for s in steps] == [0, 6, 12]
    method, url, kw = session.calls[2]
    assert kw["json"]["input_id"] == "asset-1" and kw["json"]["input_time"] == "2023-01-01T00:00:00Z"
    assert kw["json"]["variables"] == "u10m,v10m" and kw["json"]["simulation_length"] == 2
    assert kw["headers"]["Authorization"] == "Bearer key" and kw["headers"]["NVCF-POLL-SECONDS"] == "300"
    assert client.last_variables_sent == ["u10m", "v10m"]


def test_202_poll_then_302_without_auth():
    session = _session(
        FakeResponse(202, headers={"NVCF-REQID": "r1"}),
        status=[FakeResponse(202, headers={"NVCF-REQID": "r1"}),
                FakeResponse(302, headers={"Location": "https://s3.example/result"})],
        extra={("GET", "https://s3.example/result"): [FakeResponse(200, build_tar(FILES))]},
    )
    _, steps = _run(session)
    assert len(steps) == 3
    polls = [c for c in session.calls if c[1].startswith(STATUS)]
    assert len(polls) == 2 and all(c[2]["allow_redirects"] is False for c in polls)
    assert polls[0][1] == STATUS + "r1" and "Authorization" in polls[0][2]["headers"]
    download = [c for c in session.calls if c[1] == "https://s3.example/result"][0]
    assert "headers" not in download[2] or "Authorization" not in (download[2].get("headers") or {})


def test_large_asset_written_body():
    assert is_large_asset_body(b'{"message":"Large asset written"}')
    assert not is_large_asset_body(b'{"message":"ok"}')
    with pytest.raises(HostedLargeAssetError, match="simulation_length"):
        _run(_session(FakeResponse(200, b'{"message":"Large asset written"}')))


def test_status_410_is_large_asset():
    with pytest.raises(HostedLargeAssetError, match="410"):
        _run(_session(FakeResponse(202, headers={"NVCF-REQID": "r1"}), status=[FakeResponse(410)]))


def test_http_error():
    with pytest.raises(NimRequestError, match="HTTP 401"):
        _run(_session(FakeResponse(401, b"unauthorized")))
