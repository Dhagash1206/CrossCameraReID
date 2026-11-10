from collections import deque
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from PersonReid.OsnetEmbedder import l2_normalize
from PersonReid.PipelineConfig import PipelineConfig

# Finite stand-in for "impossible"; linear_sum_assignment rejects rows that are all infinite
FORBIDDEN_SIMILARITY = -1e6


@dataclass
class LocalTrack:
    camera_id: int
    local_id: int
    embeddings: deque
    last_seen_frame: int
    hits: int = 0
    global_id: int | None = None
    world_position: np.ndarray | None = None

    def query_embedding(self) -> np.ndarray:
        return l2_normalize(np.mean(np.stack(self.embeddings), axis=0))


@dataclass
class GlobalIdentity:
    global_id: int
    gallery: deque
    active_local_tracks: set = field(default_factory=set)
    # camera_id -> (frame_index, world_xy) of the latest floor-plan sighting
    last_world_sighting: dict = field(default_factory=dict)

    def similarity_to(self, query: np.ndarray, top_k: int) -> float:
        similarities = np.stack(self.gallery) @ query
        k = min(top_k, len(similarities))
        return float(np.mean(np.partition(similarities, -k)[-k:]))

    def mean_embedding(self) -> np.ndarray:
        return l2_normalize(np.mean(np.stack(self.gallery), axis=0))


class GlobalIdentityManager:
    """Links per-camera local tracks to global person IDs.

    Matching combines appearance similarity (OSNet gallery), a same-camera exclusivity
    constraint, optional floor-plan spatial gating, and periodic merging of fragmented IDs.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.local_tracks: dict[tuple[int, int], LocalTrack] = {}
        self.identities: dict[int, GlobalIdentity] = {}
        self.next_global_id = 1
        # Pairs of global IDs seen together in one camera frame: they can never be one person
        self.cannot_link: set[tuple[int, int]] = set()
        self.merged_into: dict[int, int] = {}

    def resolve(self, global_id: int | None) -> int | None:
        while global_id in self.merged_into:
            global_id = self.merged_into[global_id]
        return global_id

    def update(
        self,
        camera_id: int,
        frame_index: int,
        local_track_ids: np.ndarray,
        embeddings: list[np.ndarray | None],
        world_positions: np.ndarray | None = None,
    ) -> dict[int, int | None]:
        """Ingest one camera frame; return {local_track_id: global_id, or None while pending}."""
        for detection_index, (local_id, embedding) in enumerate(zip(local_track_ids.tolist(), embeddings)):
            key = (camera_id, local_id)
            track = self.local_tracks.get(key)
            if track is None:
                track = LocalTrack(
                    camera_id, local_id,
                    deque(maxlen=self.config.local_track_embedding_buffer),
                    frame_index,
                )
                self.local_tracks[key] = track
            track.last_seen_frame = frame_index
            track.hits += 1
            track.world_position = None if world_positions is None else world_positions[detection_index]
            if embedding is not None:
                track.embeddings.append(embedding)
            if track.global_id is not None:
                self._refresh_identity(track, embedding, frame_index)

        self._assign_pending_tracks(camera_id, frame_index, local_track_ids)
        self._record_cannot_link(camera_id, local_track_ids)
        self._expire_stale_tracks(camera_id, frame_index)
        return {
            local_id: self.local_tracks[(camera_id, local_id)].global_id
            for local_id in local_track_ids.tolist()
        }

    def _refresh_identity(self, track: LocalTrack, embedding: np.ndarray | None, frame_index: int) -> None:
        identity = self.identities[track.global_id]
        if embedding is not None and track.hits % self.config.gallery_update_interval == 0:
            identity.gallery.append(embedding)
        if track.world_position is not None:
            identity.last_world_sighting[track.camera_id] = (frame_index, track.world_position)

    def _spatial_adjustment(self, track: LocalTrack, identity: GlobalIdentity, frame_index: int) -> float:
        if track.world_position is None:
            return 0.0
        recent_distances = [
            float(np.linalg.norm(track.world_position - position))
            for camera_id, (seen_frame, position) in identity.last_world_sighting.items()
            if camera_id != track.camera_id
            and frame_index - seen_frame <= self.config.spatial_time_window_frames
        ]
        if not recent_distances:
            return 0.0
        nearest = min(recent_distances)
        if nearest > self.config.spatial_gate_meters:
            return FORBIDDEN_SIMILARITY
        return self.config.spatial_bonus * (1.0 - nearest / self.config.spatial_gate_meters)

    def _assign_pending_tracks(self, camera_id: int, frame_index: int, visible_local_ids: np.ndarray) -> None:
        visible_tracks = [self.local_tracks[(camera_id, local_id)] for local_id in visible_local_ids.tolist()]
        pending_tracks = [
            track for track in visible_tracks
            if track.global_id is None and len(track.embeddings) >= self.config.min_track_hits_before_match
        ]
        if not pending_tracks:
            return

        # A person cannot appear as two simultaneous tracks in the same camera
        identities_visible_here = {track.global_id for track in visible_tracks if track.global_id is not None}
        eligible_identities = [
            identity for identity in self.identities.values()
            if identity.global_id not in identities_visible_here
        ]

        matched_pairs: list[tuple[LocalTrack, GlobalIdentity]] = []
        if eligible_identities:
            similarity_matrix = np.array([
                [
                    identity.similarity_to(track.query_embedding(), self.config.top_k_similarities)
                    + self._spatial_adjustment(track, identity, frame_index)
                    for identity in eligible_identities
                ]
                for track in pending_tracks
            ])
            track_indices, identity_indices = linear_sum_assignment(-similarity_matrix)
            for track_index, identity_index in zip(track_indices, identity_indices):
                if similarity_matrix[track_index, identity_index] >= self.config.match_similarity_threshold:
                    matched_pairs.append((pending_tracks[track_index], eligible_identities[identity_index]))

        matched_tracks = {id(track) for track, _ in matched_pairs}
        for track, identity in matched_pairs:
            self._bind(track, identity, frame_index)
        for track in pending_tracks:
            if id(track) not in matched_tracks:
                self._bind(track, self._create_identity(), frame_index)

    def _create_identity(self) -> GlobalIdentity:
        identity = GlobalIdentity(self.next_global_id, deque(maxlen=self.config.global_gallery_size))
        self.identities[identity.global_id] = identity
        self.next_global_id += 1
        return identity

    @staticmethod
    def _bind(track: LocalTrack, identity: GlobalIdentity, frame_index: int) -> None:
        track.global_id = identity.global_id
        identity.gallery.extend(track.embeddings)
        identity.active_local_tracks.add((track.camera_id, track.local_id))
        if track.world_position is not None:
            identity.last_world_sighting[track.camera_id] = (frame_index, track.world_position)

    def _record_cannot_link(self, camera_id: int, visible_local_ids: np.ndarray) -> None:
        visible_global_ids = sorted({
            self.local_tracks[(camera_id, local_id)].global_id
            for local_id in visible_local_ids.tolist()
            if self.local_tracks[(camera_id, local_id)].global_id is not None
        })
        self.cannot_link.update(combinations(visible_global_ids, 2))

    def _expire_stale_tracks(self, camera_id: int, frame_index: int) -> None:
        stale_keys = [
            key for key, track in self.local_tracks.items()
            if track.camera_id == camera_id
            and frame_index - track.last_seen_frame > self.config.local_track_ttl_frames
        ]
        for key in stale_keys:
            track = self.local_tracks.pop(key)
            if track.global_id is not None:
                self.identities[track.global_id].active_local_tracks.discard(key)

    def consolidate(self) -> list[tuple[int, int]]:
        """Merge identities whose galleries look like the same person and never co-occurred.

        Returns (merged_id, kept_id) pairs.
        """
        mean_embeddings = {
            global_id: identity.mean_embedding()
            for global_id, identity in self.identities.items() if identity.gallery
        }
        merges = []
        for first_id, second_id in combinations(sorted(mean_embeddings), 2):
            if first_id not in self.identities or second_id not in self.identities:
                continue
            if (first_id, second_id) in self.cannot_link:
                continue
            similarity = float(mean_embeddings[first_id] @ mean_embeddings[second_id])
            if similarity >= self.config.merge_similarity_threshold:
                self._merge(source_id=second_id, target_id=first_id)
                mean_embeddings[first_id] = self.identities[first_id].mean_embedding()
                merges.append((second_id, first_id))
        return merges

    def _merge(self, source_id: int, target_id: int) -> None:
        source = self.identities.pop(source_id)
        target = self.identities[target_id]
        target.gallery.extend(source.gallery)
        target.active_local_tracks |= source.active_local_tracks
        for key in source.active_local_tracks:
            if key in self.local_tracks:
                self.local_tracks[key].global_id = target_id
        for camera_id, sighting in source.last_world_sighting.items():
            if camera_id not in target.last_world_sighting or sighting[0] > target.last_world_sighting[camera_id][0]:
                target.last_world_sighting[camera_id] = sighting
        self.merged_into[source_id] = target_id
        self.cannot_link = {
            tuple(sorted(target_id if member == source_id else member for member in pair))
            for pair in self.cannot_link
        }

    def identity_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Global IDs and their mean embeddings, for offline person search."""
        populated = [identity for identity in self.identities.values() if identity.gallery]
        if not populated:
            return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        return (
            np.array([identity.global_id for identity in populated], dtype=np.int64),
            np.stack([identity.mean_embedding() for identity in populated]).astype(np.float32),
        )
