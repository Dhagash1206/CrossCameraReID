import numpy as np
import pytest

from PersonReid.IdentityManager import GlobalIdentityManager
from PersonReid.OsnetEmbedder import l2_normalize
from PersonReid.PipelineConfig import PipelineConfig

EMBEDDING_SIZE = 512


@pytest.fixture
def random_generator():
    return np.random.default_rng(seed=7)


def make_person(random_generator) -> np.ndarray:
    return l2_normalize(random_generator.normal(size=EMBEDDING_SIZE))


def noisy_view(person: np.ndarray, random_generator) -> np.ndarray:
    return l2_normalize(person + random_generator.normal(scale=0.02, size=EMBEDDING_SIZE))


def feed(manager, random_generator, camera_id, frames, people_by_local_id, positions=None):
    """Feed several frames of one camera; returns the last {local_id: global_id} mapping."""
    mapping = {}
    local_ids = np.array(list(people_by_local_id), dtype=np.int64)
    for frame_index in frames:
        embeddings = [noisy_view(person, random_generator) for person in people_by_local_id.values()]
        mapping = manager.update(camera_id, frame_index, local_ids, embeddings, positions)
    return mapping


def test_same_person_in_two_cameras_shares_global_id(random_generator):
    manager = GlobalIdentityManager(PipelineConfig())
    person = make_person(random_generator)
    first = feed(manager, random_generator, camera_id=0, frames=range(10), people_by_local_id={1: person})
    second = feed(manager, random_generator, camera_id=1, frames=range(10), people_by_local_id={4: person})
    assert first[1] == second[4] == 1


def test_different_people_get_different_ids(random_generator):
    manager = GlobalIdentityManager(PipelineConfig())
    first = feed(manager, random_generator, 0, range(10), {1: make_person(random_generator)})
    second = feed(manager, random_generator, 1, range(10), {1: make_person(random_generator)})
    assert first[1] != second[1]


def test_simultaneous_tracks_in_one_camera_never_share_id(random_generator):
    manager = GlobalIdentityManager(PipelineConfig())
    person = make_person(random_generator)
    mapping = feed(manager, random_generator, 0, range(10), {1: person, 2: person})
    assert mapping[1] != mapping[2]


def test_spatial_gate_blocks_lookalikes_far_apart(random_generator):
    manager = GlobalIdentityManager(PipelineConfig(spatial_gate_meters=1.5))
    person = make_person(random_generator)
    first = feed(manager, random_generator, 0, range(10), {1: person}, positions=np.array([[0.0, 0.0]]))
    second = feed(manager, random_generator, 1, range(10), {1: person}, positions=np.array([[8.0, 8.0]]))
    assert first[1] != second[1]


def test_spatial_gate_allows_nearby_match(random_generator):
    manager = GlobalIdentityManager(PipelineConfig(spatial_gate_meters=1.5))
    person = make_person(random_generator)
    first = feed(manager, random_generator, 0, range(10), {1: person}, positions=np.array([[2.0, 2.0]]))
    second = feed(manager, random_generator, 1, range(10), {1: person}, positions=np.array([[2.3, 2.1]]))
    assert first[1] == second[1]


def test_consolidation_merges_fragmented_identity(random_generator):
    # An impossible match threshold forces fragmentation that consolidation must repair
    manager = GlobalIdentityManager(PipelineConfig(match_similarity_threshold=2.0))
    person = make_person(random_generator)
    feed(manager, random_generator, 0, range(10), {1: person})
    feed(manager, random_generator, 1, range(10), {1: person})
    assert len(manager.identities) == 2

    assert manager.consolidate() == [(2, 1)]
    assert manager.resolve(2) == 1
    assert manager.local_tracks[(1, 1)].global_id == 1


def test_cannot_link_prevents_merging_coexisting_people(random_generator):
    manager = GlobalIdentityManager(PipelineConfig(match_similarity_threshold=2.0))
    person = make_person(random_generator)
    feed(manager, random_generator, 0, range(10), {1: person, 2: person})
    assert manager.consolidate() == []
    assert len(manager.identities) == 2
