from __future__ import annotations

from typing import Any, cast

import docker.errors  # type: ignore[import-untyped]
import pytest

from agents import Agent
from agents.run_context import RunContextWrapper
from agents.run_state import CURRENT_SCHEMA_VERSION, RunState
from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE
from agents.sandbox.manifest import Manifest
from agents.sandbox.sandboxes.docker import (
    DockerSandboxClient,
    DockerSandboxClientOptions,
    DockerSandboxSession,
    DockerSandboxSessionState,
)
from agents.sandbox.session import BaseSandboxClientOptions
from agents.sandbox.snapshot import NoopSnapshot


class _Images:
    def get(self, image: str) -> object:
        _ = image
        return object()

    def pull(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected image pull: {args!r} {kwargs!r}")


class _Container:
    id = "replacement-container"
    status = "created"
    attrs: dict[str, object] = {"Mounts": [], "Config": {"Labels": {}}}

    def reload(self) -> None:
        return None

    def start(self) -> None:
        self.status = "running"


class _ExistingContainer(_Container):
    def __init__(self, labels: dict[str, str]) -> None:
        self.id = "existing-container"
        self.status = "running"
        self.attrs = {"Mounts": [], "Config": {"Labels": labels}}


class _Containers:
    def __init__(self, existing: _Container | None = None) -> None:
        self.created = _Container()
        self.existing = existing
        self.create_calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _Container:
        self.create_calls.append(dict(kwargs))
        return self.created

    def get(self, container_id: str) -> _Container:
        _ = container_id
        if self.existing is not None:
            return self.existing
        raise docker.errors.NotFound("container not found")


class _DockerClient:
    def __init__(self, existing: _Container | None = None) -> None:
        self.images = _Images()
        self.containers = _Containers(existing)


class _NoDockerProviderAccess:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected Docker provider access: {name}")


def _client(
    existing: _Container | None = None,
) -> tuple[DockerSandboxClient, _DockerClient]:
    docker_client = _DockerClient(existing)
    client = DockerSandboxClient(docker_client=cast(object, docker_client))
    return client, docker_client


def _state(*, labels: dict[str, str] | None = None) -> DockerSandboxSessionState:
    payload: dict[str, object] = {
        "manifest": Manifest(),
        "snapshot": NoopSnapshot(id="snapshot"),
        "image": DEFAULT_PYTHON_SANDBOX_IMAGE,
        "container_id": "missing-container",
    }
    if labels is not None:
        payload["labels"] = labels
    return DockerSandboxSessionState.model_validate(payload)


def test_docker_options_accept_labels() -> None:
    labels = {
        "com.example.owner": "worker-123",
        "com.example.job": "job-456",
    }

    options = DockerSandboxClientOptions(
        image=DEFAULT_PYTHON_SANDBOX_IMAGE,
        labels=labels,
    )

    assert options.labels == labels


def test_docker_options_labels_round_trip() -> None:
    labels = {"com.example.job": "job-456"}
    options = DockerSandboxClientOptions(
        image=DEFAULT_PYTHON_SANDBOX_IMAGE,
        labels=labels,
    )

    restored = BaseSandboxClientOptions.parse(options.model_dump(mode="json"))

    assert isinstance(restored, DockerSandboxClientOptions)
    assert restored.labels == labels


def test_docker_options_omitted_labels_preserve_default_behavior() -> None:
    options = DockerSandboxClientOptions(image=DEFAULT_PYTHON_SANDBOX_IMAGE)

    assert options.labels == {}


@pytest.mark.parametrize(
    "labels",
    [
        {1: "value"},
        {"key": 1},
    ],
    ids=["non-string-key", "non-string-value"],
)
def test_docker_options_reject_invalid_labels(labels: object) -> None:
    with pytest.raises(ValueError):
        DockerSandboxClientOptions(
            image=DEFAULT_PYTHON_SANDBOX_IMAGE,
            labels=cast(Any, labels),
        )


@pytest.mark.asyncio
async def test_docker_client_create_applies_and_persists_labels() -> None:
    client, docker_client = _client()
    labels = {"com.example.job": "job-456"}

    session = await client.create(
        options=DockerSandboxClientOptions(
            image=DEFAULT_PYTHON_SANDBOX_IMAGE,
            labels=labels,
        )
    )

    assert docker_client.containers.create_calls[0]["labels"] == labels
    assert isinstance(session._inner, DockerSandboxSession)
    assert session._inner.state.labels == labels


@pytest.mark.asyncio
async def test_docker_create_container_omits_labels_by_default() -> None:
    client, docker_client = _client()

    await client._create_container(DEFAULT_PYTHON_SANDBOX_IMAGE)

    assert "labels" not in docker_client.containers.create_calls[0]


def test_docker_session_state_labels_round_trip() -> None:
    client, _ = _client()
    labels = {"com.example.job": "job-456"}
    state = _state(labels=labels)

    restored = client.deserialize_session_state(state.model_dump(mode="json"))

    assert isinstance(restored, DockerSandboxSessionState)
    assert restored.labels == labels


def test_docker_session_state_rejects_invalid_labels_before_provider_access() -> None:
    client = DockerSandboxClient(docker_client=cast(object, _NoDockerProviderAccess()))
    payload = _state().model_dump(mode="json")
    payload["labels"] = {"job": 123}

    with pytest.raises(ValueError):
        client.deserialize_session_state(payload)


def test_docker_session_state_without_labels_preserves_old_payloads() -> None:
    client, _ = _client()
    payload = _state().model_dump(mode="json")
    payload.pop("labels", None)

    restored = client.deserialize_session_state(payload)

    assert isinstance(restored, DockerSandboxSessionState)
    assert restored.labels == {}


@pytest.mark.asyncio
async def test_docker_resume_reapplies_labels_to_replacement_container() -> None:
    client, docker_client = _client()
    labels = {"com.example.job": "job-456"}
    state = _state(labels=labels)

    await client.resume(state)

    assert docker_client.containers.create_calls[0]["labels"] == labels
    assert state.container_id == "replacement-container"


@pytest.mark.asyncio
async def test_docker_resume_reuses_container_with_requested_labels() -> None:
    labels = {"com.example.job": "job-456"}
    existing = _ExistingContainer({**labels, "com.example.extra": "preserved"})
    client, docker_client = _client(existing)
    state = _state(labels=labels)
    state.container_id = existing.id

    await client.resume(state)

    assert docker_client.containers.create_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actual_labels",
    [{}, {"com.example.job": "different"}],
    ids=["missing", "different-value"],
)
async def test_docker_resume_rejects_container_missing_requested_labels(
    actual_labels: dict[str, str],
) -> None:
    expected_labels = {"com.example.job": "job-456"}
    existing = _ExistingContainer(actual_labels)
    client, docker_client = _client(existing)
    state = _state(labels=expected_labels)
    state.container_id = existing.id

    with pytest.raises(ValueError, match="labels"):
        await client.resume(state)

    assert docker_client.containers.create_calls == []


@pytest.mark.asyncio
async def test_run_state_round_trip_preserves_docker_labels() -> None:
    agent = Agent(name="sandbox")
    labels = {"com.example.job": "job-456"}
    run_state = RunState(
        context=RunContextWrapper(context={}),
        original_input="resume sandbox",
        starting_agent=agent,
    )
    run_state._sandbox = {
        "backend_id": "docker",
        "current_agent_name": agent.name,
        "session_state": _state(labels=labels).model_dump(mode="json"),
    }

    serialized = run_state.to_json()
    restored = await RunState.from_json(agent, serialized)

    assert serialized["$schemaVersion"] == "1.17"
    assert CURRENT_SCHEMA_VERSION == "1.17"
    assert restored._sandbox is not None
    restored_session_state = restored._sandbox["session_state"]
    assert isinstance(restored_session_state, dict)
    assert restored_session_state["labels"] == labels
