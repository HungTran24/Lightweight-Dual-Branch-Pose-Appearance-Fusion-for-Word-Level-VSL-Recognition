"""Deterministic RTMW extraction manifest builder, validator, and sharding logic."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

HF_DATASET_REPO = "hungchill24/vsl400-front-view"
HF_DATASET_REVISION = "df5509963c62b2adaa18525d2990c8b30a100829"
KNOWN_DECODE_DEFECT = "019442"
EXPECTED_TOTAL_VIDEOS = 24_753
EXPECTED_GLOSS_COUNT = 400
DEFAULT_FULL_SHARD_SIZE = 500
DEFAULT_SMOKE_SHARD_SIZE = 5
DEFAULT_CANONICAL_MANIFEST_PATH = Path("manifests/vsl400_front_400gloss/all.csv")
VALID_RUN_MODES = ("smoke", "full")
SPLIT_ORDER = ("train", "val", "test")

CANONICAL_COLUMNS = (
    "video_id",
    "video_path",
    "gloss",
    "class_id",
    "signer_id",
    "fps",
    "resolution",
    "num_frames",
    "length_seconds",
    "part_id",
    "split",
)


class RTMWManifestError(ValueError):
    """Raised when an RTMW extraction manifest violates contract rules."""


@dataclass(frozen=True)
class RTMWVideoRecord:
    """Canonical metadata for a single video in an extraction manifest."""

    video_id: str
    video_path: str
    gloss: str
    class_id: int
    signer_id: str
    fps: float
    resolution: str
    num_frames: int
    length_seconds: float
    part_id: int
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RTMWVideoRecord:
        video_id = str(data.get("video_id", "")).strip()
        if video_id.isdigit():
            video_id = video_id.zfill(6)
        signer_id = str(data.get("signer_id", "")).strip()
        if signer_id.isdigit():
            signer_id = signer_id.zfill(3)

        return cls(
            video_id=video_id,
            video_path=str(data.get("video_path", "")).strip(),
            gloss=str(data.get("gloss", "")).strip(),
            class_id=int(data.get("class_id", 0)),
            signer_id=signer_id,
            fps=float(data.get("fps", 0.0)),
            resolution=str(data.get("resolution", "")).strip(),
            num_frames=int(data.get("num_frames", 0)),
            length_seconds=float(data.get("length_seconds", 0.0)),
            part_id=int(data.get("part_id", 0)),
            split=str(data.get("split", "")).strip(),
        )


@dataclass(frozen=True)
class RTMWShardSpec:
    """Deterministic shard specification holding an ordered partition of videos."""

    shard_id: str
    shard_index: int
    video_count: int
    video_ids: tuple[str, ...]
    videos: tuple[RTMWVideoRecord, ...]
    split_counts: dict[str, int]

    def to_dict(self, *, include_videos: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "shard_id": self.shard_id,
            "shard_index": self.shard_index,
            "video_count": self.video_count,
            "video_ids": list(self.video_ids),
            "split_counts": dict(self.split_counts),
        }
        if include_videos:
            data["videos"] = [v.to_dict() for v in self.videos]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RTMWShardSpec:
        videos = tuple(RTMWVideoRecord.from_dict(v) for v in data.get("videos", []))
        raw_video_ids = data.get("video_ids", [])
        video_ids = tuple(
            str(vid).zfill(6) if str(vid).isdigit() else str(vid)
            for vid in raw_video_ids
        )
        return cls(
            shard_id=str(data["shard_id"]),
            shard_index=int(data["shard_index"]),
            video_count=int(data["video_count"]),
            video_ids=video_ids,
            videos=videos,
            split_counts=dict(data.get("split_counts", {})),
        )


@dataclass(frozen=True)
class RTMWExtractionManifest:
    """Immutable extraction manifest contract for RTMW extraction runs."""

    mode: str
    dataset_repo: str
    dataset_revision: str
    shard_size: int
    total_videos: int
    shard_count: int
    inventory_sha256: str
    shards: tuple[RTMWShardSpec, ...]

    def get_shard(self, shard_ref: int | str) -> RTMWShardSpec:
        """Retrieve a shard by its integer index or string ID."""
        if isinstance(shard_ref, int):
            if not (0 <= shard_ref < len(self.shards)):
                raise RTMWManifestError(
                    f"Shard index {shard_ref} out of range [0, {len(self.shards) - 1}]"
                )
            return self.shards[shard_ref]
        for shard in self.shards:
            if shard.shard_id == shard_ref:
                return shard
        raise RTMWManifestError(f"Shard {shard_ref!r} not found in manifest")

    @property
    def all_video_ids(self) -> tuple[str, ...]:
        """Flattened tuple of all video IDs in canonical shard order."""
        return tuple(vid for shard in self.shards for vid in shard.video_ids)

    def to_dict(self, *, include_videos: bool = True) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "dataset_repo": self.dataset_repo,
            "dataset_revision": self.dataset_revision,
            "shard_size": self.shard_size,
            "total_videos": self.total_videos,
            "shard_count": self.shard_count,
            "inventory_sha256": self.inventory_sha256,
            "shards": [s.to_dict(include_videos=include_videos) for s in self.shards],
        }

    def to_json(self, *, indent: int | None = 2, include_videos: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_videos=include_videos),
            ensure_ascii=False,
            indent=indent,
        )

    def save_json(self, path: Path | str, *, include_videos: bool = True) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            self.to_json(indent=2, include_videos=include_videos) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RTMWExtractionManifest:
        shards = tuple(RTMWShardSpec.from_dict(s) for s in data.get("shards", []))
        return cls(
            mode=str(data["mode"]),
            dataset_repo=str(data["dataset_repo"]),
            dataset_revision=str(data["dataset_revision"]),
            shard_size=int(data["shard_size"]),
            total_videos=int(data["total_videos"]),
            shard_count=int(data["shard_count"]),
            inventory_sha256=str(data["inventory_sha256"]),
            shards=shards,
        )

    @classmethod
    def load_json(cls, path: Path | str) -> RTMWExtractionManifest:
        p = Path(path)
        if not p.is_file():
            raise RTMWManifestError(f"Manifest file not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RTMWManifestError(f"Invalid JSON in manifest file {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise RTMWManifestError(f"Manifest root must be a JSON object: {p}")
        manifest = cls.from_dict(data)
        manifest.validate()
        return manifest

    def validate(self) -> None:
        validate_rtmw_manifest(self)


def compute_inventory_sha256(
    mode: str,
    dataset_repo: str,
    dataset_revision: str,
    shards: Sequence[RTMWShardSpec],
) -> str:
    """Compute deterministic SHA256 hex digest over the shard inventory structure."""
    inventory_dict = {
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "mode": mode,
        "shard_count": len(shards),
        "shards": [
            {
                "shard_id": s.shard_id,
                "shard_index": s.shard_index,
                "split_counts": dict(sorted(s.split_counts.items())),
                "video_count": s.video_count,
                "video_ids": list(s.video_ids),
            }
            for s in shards
        ],
        "total_videos": sum(s.video_count for s in shards),
    }
    canonical_json = json.dumps(
        inventory_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def read_canonical_csv(path: Path | str) -> list[RTMWVideoRecord]:
    """Read and validate canonical all.csv into a list of RTMWVideoRecord."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise RTMWManifestError(f"Canonical CSV manifest not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = tuple(reader.fieldnames or ())
        if headers != CANONICAL_COLUMNS:
            raise RTMWManifestError(
                f"CSV headers {headers} do not match canonical columns {CANONICAL_COLUMNS}."
            )
        records = [RTMWVideoRecord.from_dict(row) for row in reader]

    if not records:
        raise RTMWManifestError(f"Canonical CSV is empty: {csv_path}")
    return records


def sort_records_canonical(records: Sequence[RTMWVideoRecord]) -> list[RTMWVideoRecord]:
    """Sort records deterministically using the canonical split-first ranking."""
    split_rank = {name: index for index, name in enumerate(SPLIT_ORDER)}
    return sorted(
        records,
        key=lambda r: (
            split_rank.get(r.split, 99),
            r.class_id,
            r.gloss,
            r.signer_id,
            r.video_id,
        ),
    )


def build_rtmw_manifest(
    canonical_source: Path | str | Sequence[RTMWVideoRecord | dict[str, Any]] = DEFAULT_CANONICAL_MANIFEST_PATH,
    *,
    mode: str = "full",
    shard_size: int | None = None,
    dataset_repo: str = HF_DATASET_REPO,
    dataset_revision: str = HF_DATASET_REVISION,
    smoke_video_ids: Sequence[str] | None = None,
) -> RTMWExtractionManifest:
    """Build an immutable RTMWExtractionManifest for 'smoke' or 'full' mode."""
    if mode not in VALID_RUN_MODES:
        raise RTMWManifestError(
            f"Invalid mode {mode!r}; must be one of {VALID_RUN_MODES}"
        )

    if isinstance(canonical_source, (str, Path)):
        raw_records = read_canonical_csv(canonical_source)
    else:
        raw_records = [
            r if isinstance(r, RTMWVideoRecord) else RTMWVideoRecord.from_dict(r)
            for r in canonical_source
        ]

    records = sort_records_canonical(raw_records)
    records_by_id = {r.video_id: r for r in records}

    if mode == "full":
        target_shard_size = (
            shard_size if shard_size is not None else DEFAULT_FULL_SHARD_SIZE
        )
        if target_shard_size <= 0:
            raise RTMWManifestError(
                f"shard_size must be positive, got {target_shard_size}"
            )
        if len(records) != EXPECTED_TOTAL_VIDEOS:
            raise RTMWManifestError(
                f"Expected {EXPECTED_TOTAL_VIDEOS} records for full manifest, got {len(records)}"
            )
        if len(records_by_id) != len(records):
            duplicates = [
                vid for vid, count in Counter(r.video_id for r in records).items() if count > 1
            ]
            raise RTMWManifestError(f"Duplicate video IDs found: {duplicates[:10]}")
        if KNOWN_DECODE_DEFECT not in records_by_id:
            raise RTMWManifestError(
                f"Known decode defect {KNOWN_DECODE_DEFECT!r} must be present in full manifest"
            )

        shard_specs: list[RTMWShardSpec] = []
        num_shards = math.ceil(len(records) / target_shard_size)
        for i in range(num_shards):
            chunk = records[i * target_shard_size : (i + 1) * target_shard_size]
            shard_id = f"shard_{i:04d}"
            split_counts = dict(Counter(r.split for r in chunk))
            shard_specs.append(
                RTMWShardSpec(
                    shard_id=shard_id,
                    shard_index=i,
                    video_count=len(chunk),
                    video_ids=tuple(r.video_id for r in chunk),
                    videos=tuple(chunk),
                    split_counts=split_counts,
                )
            )

    else:  # mode == "smoke"
        target_shard_size = (
            shard_size if shard_size is not None else DEFAULT_SMOKE_SHARD_SIZE
        )
        if target_shard_size <= 0:
            raise RTMWManifestError(
                f"shard_size must be positive, got {target_shard_size}"
            )

        if smoke_video_ids is not None:
            if KNOWN_DECODE_DEFECT not in smoke_video_ids:
                raise RTMWManifestError(
                    f"Smoke video IDs must include known decode defect {KNOWN_DECODE_DEFECT!r}"
                )
            smoke_records = []
            for vid in smoke_video_ids:
                vid_str = str(vid).zfill(6) if str(vid).isdigit() else str(vid)
                if vid_str not in records_by_id:
                    raise RTMWManifestError(
                        f"Smoke video ID {vid_str!r} not found in canonical records"
                    )
                smoke_records.append(records_by_id[vid_str])

            num_shards = math.ceil(len(smoke_records) / target_shard_size)
            if num_shards < 2:
                raise RTMWManifestError(
                    f"Smoke manifest must contain at least 2 shards; got {num_shards} with shard_size={target_shard_size}"
                )
            shard_specs = []
            for i in range(num_shards):
                chunk = smoke_records[i * target_shard_size : (i + 1) * target_shard_size]
                shard_id = f"shard_{i:04d}"
                split_counts = dict(Counter(r.split for r in chunk))
                shard_specs.append(
                    RTMWShardSpec(
                        shard_id=shard_id,
                        shard_index=i,
                        video_count=len(chunk),
                        video_ids=tuple(r.video_id for r in chunk),
                        videos=tuple(chunk),
                        split_counts=split_counts,
                    )
                )
        else:
            # Deterministic default smoke selection:
            # Shard 0: (target_shard_size - 1) train records + KNOWN_DECODE_DEFECT (019442)
            # Shard 1: target_shard_size val records (or val + test)
            if KNOWN_DECODE_DEFECT not in records_by_id:
                raise RTMWManifestError(
                    f"Known decode defect {KNOWN_DECODE_DEFECT!r} not in canonical records"
                )
            defect_record = records_by_id[KNOWN_DECODE_DEFECT]

            train_records = [
                r for r in records if r.split == "train" and r.video_id != KNOWN_DECODE_DEFECT
            ]
            val_records = [r for r in records if r.split == "val"]
            test_records = [r for r in records if r.split == "test"]

            if not train_records or not val_records:
                raise RTMWManifestError("Canonical records must contain train and val splits")

            # Shard 0: train videos including 019442
            s0_train_count = max(1, target_shard_size - 1)
            s0_records = sort_records_canonical(train_records[:s0_train_count] + [defect_record])

            # Shard 1: val videos (plus test if needed)
            s1_val_count = min(len(val_records), target_shard_size)
            s1_records = list(val_records[:s1_val_count])
            if len(s1_records) < target_shard_size and test_records:
                needed = target_shard_size - len(s1_records)
                s1_records.extend(test_records[:needed])
            s1_records = sort_records_canonical(s1_records)

            shard_specs = [
                RTMWShardSpec(
                    shard_id="shard_0000",
                    shard_index=0,
                    video_count=len(s0_records),
                    video_ids=tuple(r.video_id for r in s0_records),
                    videos=tuple(s0_records),
                    split_counts=dict(Counter(r.split for r in s0_records)),
                ),
                RTMWShardSpec(
                    shard_id="shard_0001",
                    shard_index=1,
                    video_count=len(s1_records),
                    video_ids=tuple(r.video_id for r in s1_records),
                    videos=tuple(s1_records),
                    split_counts=dict(Counter(r.split for r in s1_records)),
                ),
            ]

    inventory_sha256 = compute_inventory_sha256(
        mode=mode,
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        shards=shard_specs,
    )

    manifest = RTMWExtractionManifest(
        mode=mode,
        dataset_repo=dataset_repo,
        dataset_revision=dataset_revision,
        shard_size=target_shard_size,
        total_videos=sum(s.video_count for s in shard_specs),
        shard_count=len(shard_specs),
        inventory_sha256=inventory_sha256,
        shards=tuple(shard_specs),
    )
    manifest.validate()
    return manifest


def validate_rtmw_manifest(manifest: RTMWExtractionManifest | dict[str, Any]) -> None:
    """Validate an RTMWExtractionManifest against all integrity and contract invariants."""
    if isinstance(manifest, dict):
        manifest = RTMWExtractionManifest.from_dict(manifest)

    if manifest.mode not in VALID_RUN_MODES:
        raise RTMWManifestError(
            f"Invalid manifest mode: {manifest.mode!r}; expected one of {VALID_RUN_MODES}"
        )

    if not manifest.dataset_repo.strip():
        raise RTMWManifestError("dataset_repo must be non-empty")
    if not manifest.dataset_revision.strip():
        raise RTMWManifestError("dataset_revision must be non-empty")

    if manifest.shard_size <= 0:
        raise RTMWManifestError(f"shard_size must be positive, got {manifest.shard_size}")

    if manifest.shard_count != len(manifest.shards):
        raise RTMWManifestError(
            f"shard_count {manifest.shard_count} does not match len(shards) {len(manifest.shards)}"
        )

    if manifest.mode == "smoke" and manifest.shard_count < 2:
        raise RTMWManifestError(
            f"smoke manifest must have at least 2 shards, got {manifest.shard_count}"
        )

    all_vids = manifest.all_video_ids
    if len(all_vids) != len(set(all_vids)):
        duplicates = [vid for vid, count in Counter(all_vids).items() if count > 1]
        raise RTMWManifestError(f"Duplicate video IDs found across shards: {duplicates[:10]}")

    if manifest.total_videos != len(all_vids):
        raise RTMWManifestError(
            f"total_videos {manifest.total_videos} does not match total video count in shards {len(all_vids)}"
        )

    if manifest.mode == "full" and manifest.total_videos != EXPECTED_TOTAL_VIDEOS:
        raise RTMWManifestError(
            f"full manifest must have exactly {EXPECTED_TOTAL_VIDEOS} videos, got {manifest.total_videos}"
        )

    if KNOWN_DECODE_DEFECT not in all_vids:
        raise RTMWManifestError(
            f"Known decode defect {KNOWN_DECODE_DEFECT!r} must be present in manifest video inventory"
        )

    for expected_idx, shard in enumerate(manifest.shards):
        if shard.shard_index != expected_idx:
            raise RTMWManifestError(
                f"Shard index {shard.shard_index} at position {expected_idx} is not contiguous"
            )
        if shard.video_count != len(shard.video_ids):
            raise RTMWManifestError(
                f"Shard {shard.shard_id} video_count {shard.video_count} does not match len(video_ids) {len(shard.video_ids)}"
            )
        for vid in shard.video_ids:
            if len(vid) != 6 or not vid.isdigit():
                raise RTMWManifestError(f"video_id must be a 6-digit numeric string, got {vid!r}")

        if shard.videos:
            if len(shard.videos) != shard.video_count:
                raise RTMWManifestError(
                    f"Shard {shard.shard_id} video list length {len(shard.videos)} does not match video_count {shard.video_count}"
                )
            if tuple(v.video_id for v in shard.videos) != shard.video_ids:
                raise RTMWManifestError(
                    f"Shard {shard.shard_id} video record IDs do not match video_ids"
                )
            actual_split_counts = dict(Counter(v.split for v in shard.videos))
            if shard.split_counts != actual_split_counts:
                raise RTMWManifestError(
                    f"Shard {shard.shard_id} split_counts {shard.split_counts} do not match actual counts {actual_split_counts}"
                )
            for v in shard.videos:
                if v.split not in SPLIT_ORDER:
                    raise RTMWManifestError(f"Invalid split {v.split!r} in video {v.video_id}")
                if v.num_frames <= 0 or v.fps <= 0 or v.length_seconds < 0:
                    raise RTMWManifestError(
                        f"Invalid numeric stats in video record {v.video_id}"
                    )

    expected_sha256 = compute_inventory_sha256(
        mode=manifest.mode,
        dataset_repo=manifest.dataset_repo,
        dataset_revision=manifest.dataset_revision,
        shards=manifest.shards,
    )
    if manifest.inventory_sha256 != expected_sha256:
        raise RTMWManifestError(
            f"inventory_sha256 mismatch: manifest has {manifest.inventory_sha256}, expected {expected_sha256}"
        )


def build_manifest_report(manifests: dict[str, RTMWExtractionManifest]) -> str:
    """Generate a Markdown report summarizing materialized manifests."""
    lines = [
        "---",
        "title: RTMW extraction manifest report",
        "description: Deterministic RTMW extraction manifest inventory and sharding summary for smoke and full runs",
        "status: ready",
        f"update_at: {date.today():%d-%m-%y}",
        "---",
        "",
        "# RTMW extraction manifest report",
        "",
        "## Contract",
        "",
        f"- Hugging Face dataset: `{HF_DATASET_REPO}`",
        f"- Hugging Face revision: `{HF_DATASET_REVISION}`",
        f"- Known decode defect: `{KNOWN_DECODE_DEFECT}`",
        "",
        "## Manifest summaries",
        "",
        "| Mode | Total videos | Shards | Shard target | Inventory SHA256 | Known defect present |",
        "|---|---:|---:|---:|---|---|",
    ]
    for mode, manifest in manifests.items():
        defect_present = "YES" if KNOWN_DECODE_DEFECT in manifest.all_video_ids else "NO"
        lines.append(
            f"| `{mode}` | {manifest.total_videos:,} | {manifest.shard_count:,} | "
            f"{manifest.shard_size} | `{manifest.inventory_sha256[:16]}...` | {defect_present} |"
        )
    lines.append("")
    return "\n".join(lines)


def materialize_rtmw_manifests(
    canonical_csv_path: Path | str = DEFAULT_CANONICAL_MANIFEST_PATH,
    output_dir: Path | str = Path("manifests/rtmw_full400"),
    *,
    modes: Sequence[str] = ("smoke", "full"),
    full_shard_size: int = DEFAULT_FULL_SHARD_SIZE,
    smoke_shard_size: int = DEFAULT_SMOKE_SHARD_SIZE,
    overwrite: bool = False,
) -> dict[str, RTMWExtractionManifest]:
    """Materialize RTMW manifest JSON files and markdown summary report."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    records = read_canonical_csv(canonical_csv_path)

    results: dict[str, RTMWExtractionManifest] = {}
    for mode in modes:
        if mode not in VALID_RUN_MODES:
            raise RTMWManifestError(f"Invalid mode {mode!r}; must be one of {VALID_RUN_MODES}")
        shard_size = full_shard_size if mode == "full" else smoke_shard_size
        manifest = build_rtmw_manifest(
            records,
            mode=mode,
            shard_size=shard_size,
        )
        manifest_file = destination / f"rtmw_manifest_{mode}.json"
        if manifest_file.exists() and not overwrite:
            raise RTMWManifestError(f"Refusing to overwrite existing manifest: {manifest_file}")
        manifest.save_json(manifest_file)
        results[mode] = manifest

    report_file = destination / "manifest_report.md"
    report_file.write_text(build_manifest_report(results), encoding="utf-8")
    return results
