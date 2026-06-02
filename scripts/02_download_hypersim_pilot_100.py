from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


REPO = "ritianyu/Hypersim"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE_ROOT = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def fetch_json(url: str) -> list[dict]:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def list_dir(path: str, recursive: bool = False) -> list[dict]:
    encoded_path = quote(path.strip("/"))
    url = f"{API_ROOT}/{encoded_path}?recursive={'true' if recursive else 'false'}&expand=true"
    return fetch_json(url)


def download_file(
    remote_path: str,
    local_path: Path,
    overwrite: bool = False,
    retries: int = 3,
) -> bool:
    if local_path.exists() and not overwrite:
        return False

    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RESOLVE_ROOT}/{quote(remote_path)}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=120) as response:
                local_path.write_bytes(response.read())
            return True
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if local_path.exists():
                local_path.unlink()
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"failed to download {remote_path}") from last_error


def find_scene_names(limit: int) -> list[str]:
    rows = fetch_json(f"{API_ROOT}?recursive=false&expand=true")
    scenes = sorted(
        row["path"]
        for row in rows
        if row.get("type") == "directory" and row.get("path", "").startswith("ai_")
    )
    if len(scenes) < limit:
        raise RuntimeError(f"only found {len(scenes)} scenes, need {limit}")
    return scenes[:limit]


def find_frame_ids(scene: str, frames_per_scene: int) -> list[str]:
    rows = list_dir(f"{scene}/images/scene_cam_00_geometry_hdf5", recursive=True)
    frame_ids: list[str] = []
    for row in rows:
        path = row.get("path", "")
        suffix = ".depth_meters.hdf5"
        if path.endswith(suffix):
            frame_ids.append(Path(path).name.removesuffix(suffix).removeprefix("frame."))
    frame_ids = sorted(set(frame_ids))
    if len(frame_ids) < frames_per_scene:
        raise RuntimeError(f"{scene} has only {len(frame_ids)} frames, need {frames_per_scene}")
    return frame_ids[:frames_per_scene]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small Hypersim RGB-D pilot subset.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/hypersim_pilot_100"))
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--frames-per-scene", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    scene_names = find_scene_names(args.scenes)
    manifest: list[dict] = []
    downloaded = 0
    skipped = 0

    for scene in scene_names:
        frame_ids = find_frame_ids(scene, args.frames_per_scene)
        metadata_paths = [
            f"{scene}/_detail/metadata_scene.csv",
            f"{scene}/_detail/metadata_cameras.csv",
            f"{scene}/_detail/cam_00/metadata_camera.csv",
            f"{scene}/_detail/cam_00/camera_keyframe_positions.hdf5",
            f"{scene}/_detail/cam_00/camera_keyframe_orientations.hdf5",
            f"{scene}/_detail/cam_00/camera_keyframe_frame_indices.hdf5",
        ]
        for remote_path in metadata_paths:
            try:
                changed = download_file(
                    remote_path,
                    args.out / remote_path,
                    overwrite=args.overwrite,
                    retries=args.retries,
                )
            except (HTTPError, URLError) as exc:
                print(f"warn: failed metadata {remote_path}: {exc}")
                continue
            downloaded += int(changed)
            skipped += int(not changed)

        for frame_id in frame_ids:
            remote_rgb = f"{scene}/images/scene_cam_00_final_preview/frame.{frame_id}.tonemap.jpg"
            remote_depth = f"{scene}/images/scene_cam_00_geometry_hdf5/frame.{frame_id}.depth_meters.hdf5"
            for remote_path in [remote_rgb, remote_depth]:
                changed = download_file(
                    remote_path,
                    args.out / remote_path,
                    overwrite=args.overwrite,
                    retries=args.retries,
                )
                downloaded += int(changed)
                skipped += int(not changed)
            manifest.append(
                {
                    "scene": scene,
                    "camera": "scene_cam_00",
                    "frame_id": frame_id,
                    "rgb": remote_rgb,
                    "depth_meters": remote_depth,
                }
            )

    manifest_path = args.out / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"scenes={len(scene_names)} frames={len(manifest)}")
    print(f"downloaded_files={downloaded} skipped_existing={skipped}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
