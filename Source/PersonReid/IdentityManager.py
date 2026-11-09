from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from PersonReid.PipelineConfig import PipelineConfig


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


@dataclass
class LocalTrack:
    camera_id: int
    local_id: int
    embeddings: deque
    last_seen_frame: int
    hits: int = 0
    global_id: int | None = None

    def query_embedding(self) -> np.ndarray:
        return l2_normalize(np.mean(np.stack(self.embeddings), axis=0))


@dataclass
class GlobalIdentity:
    global_id: int
    gallery: deque
    # Cameras where this identity is currently held by a live local track
    active_local_tracks: set = field(default_factory=set)

    def similarity_to(self, query: np.ndarray, top_k: int) -> float:
        similarities = np.stack(self.gallery) @ query
        k = min(top_k, len(similarities))
        return float(np.mean(np.partition(similarities, -k)[-k:]))


class GlobalIdentityManager:
    """Links per-camera local tracks to global person IDs using appearance similarity."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.local_tracks: dict[tuple[int, int], LocalTrack] = {}
        self.identities: dict[int, GlobalIdentity] = {}
        self.next_global_id = 1

    def update(
        self,
        camera_id: int,
        frame_index: int,
        local_track_ids: np.ndarray,
        embeddings: list[np.ndarray | None],
    ) -> dict[int, int | None]:
        """Ingest one camera's frame; return {local_track_id: global_id or None if pending}."""
        for local_id, embedding in zip(local_track_ids.tolist(), embeddings):
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
            if embedding is not None:
                track.embeddings.append(embedding)
                if track.global_id is not None and track.hits % self.config.gallery_update_interval == 0:
                    self.identities[track.global_id].gallery.append(embedding)

        self._assign_pending_tracks(camera_id, local_track_ids)
        self._expire_stale_tracks(camera_id, frame_index)
        return {
            local_id: self.local_tracks[(camera_id, local_id)].global_id
            for local_id in local_track_ids.tolist()
        }

    def _assign_pending_tracks(self, camera_id: int, visible_local_ids: np.ndarray) -> None:
        pending_tracks = [
            self.local_tracks[(camera_id, local_id)]
            for local_id in visible_local_ids.tolist()
            if self.local_tracks[(camera_id, local_id)].global_id is None
            and len(self.local_tracks[(camera_id, local_id)].embeddings) >= self.config.min_track_hits_before_match
        ]
        if not pending_tracks:
            return

        # A person cannot appear as two simultaneous tracks in the same camera
        eligible_identities = [
            identity for identity in self.identities.values()
            if camera_id not in {cam for cam, _ in identity.active_local_tracks}
        ]

        matched_pairs: list[tuple[LocalTrack, GlobalIdentity]] = []
        if eligible_identities:
            queries = [track.query_embedding() for track in pending_tracks]
            similarity_matrix = np.array([
                [identity.similarity_to(query, self.config.top_k_similarities) for identity in eligible_identities]
                for query in queries
            ])
            track_indices, identity_indices = linear_sum_assignment(-similarity_matrix)
            for track_index, identity_index in zip(track_indices, identity_indices):
                if similarity_matrix[track_index, identity_index] >= self.config.match_similarity_threshold:
                    matched_pairs.append((pending_tracks[track_index], eligible_identities[identity_index]))

        matched_track_keys = {(track.camera_id, track.local_id) for track, _ in matched_pairs}
        for track, identity in matched_pairs:
            self._bind(track, identity)
        for track in pending_tracks:
            if (track.camera_id, track.local_id) not in matched_track_keys:
                self._bind(track, self._create_identity())

    def _create_identity(self) -> GlobalIdentity:
        identity = GlobalIdentity(self.next_global_id, deque(maxlen=self.config.global_gallery_size))
        self.identities[identity.global_id] = identity
        self.next_global_id += 1
        return identity

    @staticmethod
    def _bind(track: LocalTrack, identity: GlobalIdentity) -> None:
        track.global_id = identity.global_id
        identity.gallery.extend(track.embeddings)
        identity.active_local_tracks.add((track.camera_id, track.local_id))

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
