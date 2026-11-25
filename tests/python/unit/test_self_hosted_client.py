import numpy as np
import pytest

from earth2_dku.arrays import load_npy
from earth2_dku.clients.base import parse_step_filename
from earth2_dku.clients.self_hosted import SelfHostedNimClient
from earth2_dku.errors import ConfigError, NimRequestError, NimStreamBrokenError
from earth2_dku.times import parse_time
from fakes import SMALL_STEP_SHAPE, FakeNimServer, small_steps

INIT = parse_time("2023-01-01T00:00:00.5Z")


def _run(client, n=2, variables=("u10m", "v10m")):
    return list(client.infer_stream(b"INPUT", INIT, n, list(variables) if variables else None))


def test_parse_step_filename():
    assert parse_step_filename("006_000.npy") == (6, 0)
    assert parse_step_filename("out/120_002.npy") == (120, 2)
    with pytest.raises(ValueError):
        parse_step_filename("manifest.json")


def test_empty_base_url():
    with pytest.raises(ConfigError):
        SelfHostedNimClient("")


def test_health_ready_and_not_ready():
    with FakeNimServer([]) as srv:
        assert SelfHostedNimClient(srv.base_url).health().ready
    with FakeNimServer([], health_status=503) as srv:
        status = SelfHostedNimClient(srv.base_url).health()
        assert not status.ready and "503" in status.detail


def test_health_unreachable():
    status = SelfHostedNimClient("http://127.0.0.1:9", timeout_seconds=2).health()
    assert not status.ready and "Cannot reach" in status.detail


def test_streams_all_steps_and_sends_expected_request():
    files = small_steps(3)
    with FakeNimServer(files) as srv:
        client = SelfHostedNimClient(srv.base_url)
        steps = _run(client)
        req = srv.requests[0]
    assert [(s.lead_hours, s.batch_idx) for s in steps] == [(0, 0), (6, 0), (12, 0)]
    assert [s.raw_bytes for s in steps] == [d for _, d in files]
    np.testing.assert_array_equal(load_npy(steps[2].raw_bytes), np.full(SMALL_STEP_SHAPE, 2, dtype="float32"))
    assert req["input_time"] == "2023-01-01T00:00:00Z"  # no fractional seconds
    assert req["simulation_length"] == "2"
    assert req["variables"] == "u10m,v10m"
    assert req["input_array"] == b"INPUT"
    assert req["accept"] == "application/x-tar"
    assert client.last_variables_sent == ["u10m", "v10m"]


def test_mid_stream_break_raises_with_count():
    with FakeNimServer(small_steps(4), break_after=2) as srv:
        client = SelfHostedNimClient(srv.base_url)
        received = []
        with pytest.raises(NimStreamBrokenError) as exc:
            for s in client.infer_stream(b"x", INIT, 3, ["u10m"]):
                received.append(s)
    assert exc.value.steps_received == 2 and len(received) == 2
    assert "pod logs" in str(exc.value)


def test_clean_but_short_stream_is_broken():
    with FakeNimServer(small_steps(2)) as srv:
        with pytest.raises(NimStreamBrokenError) as exc:
            _run(SelfHostedNimClient(srv.base_url), n=4)
    assert exc.value.steps_received == 2


def test_unverified_variables_param_retries_without_it(caplog):
    with FakeNimServer(small_steps(3), reject_variables=True) as srv:
        client = SelfHostedNimClient(srv.base_url, supports_variables_param="unverified")
        steps = _run(client)
        reqs = srv.requests
    assert len(steps) == 3 and len(reqs) == 2
    assert "variables" in reqs[0] and "variables" not in reqs[1]
    assert client.last_variables_sent is None
    assert "Retrying once without it" in caplog.text


def test_verified_variables_param_does_not_retry():
    with FakeNimServer(small_steps(3), reject_variables=True) as srv:
        with pytest.raises(NimRequestError, match="HTTP 400"):
            _run(SelfHostedNimClient(srv.base_url, supports_variables_param="true"))
        assert len(srv.requests) == 1


def test_unsupported_variables_param_never_sent():
    with FakeNimServer(small_steps(3)) as srv:
        client = SelfHostedNimClient(srv.base_url, supports_variables_param="false")
        _run(client)
        assert "variables" not in srv.requests[0]
    assert client.last_variables_sent is None


def test_server_error_message():
    with FakeNimServer([], infer_status=500) as srv:
        with pytest.raises(NimRequestError, match="RFC 3339"):
            _run(SelfHostedNimClient(srv.base_url))
