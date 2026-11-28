#!/usr/bin/env python3
"""仅整理 Umbra 官方公开前缀的 STAC 元数据，不读取影像载荷。"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def item_row(path: Path, item: dict) -> dict:
    p = item.get("properties", {})
    bbox = item.get("bbox")
    pols = p.get("sar:polarizations") or []
    return {
        "retrieval_file": path.name,
        "metadata_sha256": sha256(path),
        "item_id": item.get("id"),
        "collection": item.get("collection"),
        "datetime": p.get("datetime"),
        "date": str(p.get("datetime", ""))[:10],
        "platform": p.get("platform"),
        "polarizations": "+".join(sorted(pols)),
        "looks_azimuth": p.get("sar:looks_azimuth"),
        "looks_range": p.get("sar:looks_range"),
        "incidence_angle": p.get("view:incidence_angle"),
        "resolution_azimuth": p.get("sar:resolution_azimuth"),
        "resolution_range": p.get("sar:resolution_range"),
        "task_id": p.get("umbra:task_id"),
        "bbox": bbox,
        "metadata_asset_hrefs": [
            asset.get("href")
            for asset in item.get("assets", {}).values()
            if "metadata" in (asset.get("roles") or [])
        ],
        "payload_asset_count": sum(
            "data" in (asset.get("roles") or [])
            for asset in item.get("assets", {}).values()
        ),
    }


def overlap_fraction_of_smaller(a: list[float], b: list[float]) -> float:
    # 仅用于粗筛；经纬度平面面积不等于投影/地面面积。
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    aw, ah = max(0.0, ax1 - ax0), max(0.0, ay1 - ay0)
    bw, bh = max(0.0, bx1 - bx0), max(0.0, by1 - by0)
    aa, ba = aw * ah, bw * bh
    if aa <= 0 or ba <= 0:
        return 0.0
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / min(aa, ba)


def make_components(items: list[dict], threshold: float = 0.5) -> list[list[dict]]:
    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            if overlap_fraction_of_smaller(items[i]["bbox"], items[j]["bbox"]) >= threshold:
                union(i, j)
    groups: dict[int, list[dict]] = defaultdict(list)
    for i, row in enumerate(items):
        groups[find(i)].append(row)
    return sorted(groups.values(), key=lambda g: min(x["item_id"] for x in g))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Umbra 官方公开 SAR 元数据筛查记录（不下载影像载荷）")
    parser.add_argument("--metadata-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--listing-xml", type=Path, default=Path("/tmp/umbra_s3_open_data_listing_20261128.xml"))
    parser.add_argument("--catalog-json", type=Path, default=Path("/tmp/umbra_stac_catalog_20261128.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.metadata_dir.glob("umbra_open_data_item_*_20261128.json"))
    if len(files) != 134:
        raise SystemExit(f"错误：预期 134 份 STAC 元数据，实际找到 {len(files)} 份。")
    if not args.listing_xml.exists() or not args.catalog_json.exists():
        raise SystemExit("错误：缺少公开对象清单 XML 或 STAC 根目录 JSON。")

    raw_items = []
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        raw_items.append((path, item))
    rows = [item_row(path, item) for path, item in raw_items]

    root = ET.parse(args.listing_xml).getroot()
    object_rows = []
    for entry in root.findall("s:Contents", NS):
        object_rows.append({
            "key": entry.findtext("s:Key", default="", namespaces=NS),
            "last_modified": entry.findtext("s:LastModified", default="", namespaces=NS),
            "size_bytes": int(entry.findtext("s:Size", default="0", namespaces=NS)),
            "etag": entry.findtext("s:ETag", default="", namespaces=NS).strip('"'),
        })
    meta_objects = [r for r in object_rows if r["key"].endswith(".stac.v2.json")]
    if len(object_rows) != 537 or len(meta_objects) != 134:
        raise SystemExit(
            f"错误：对象清单计数与本次核对不符（总数 {len(object_rows)}，STAC {len(meta_objects)}）。"
        )
    metadata_by_task = {r["key"].split("/")[1]: r for r in meta_objects}
    if len(metadata_by_task) != len(meta_objects):
        raise SystemExit("错误：公开清单中的 STAC 对象 task ID 不唯一。")
    for row in rows:
        source_obj = metadata_by_task.get(str(row["task_id"]))
        if source_obj is None:
            raise SystemExit(f"错误：STAC task ID 无法对应公开清单：{row['task_id']}")
        row["open_data_sidecar_key"] = source_obj["key"]
        row["open_data_sidecar_last_modified"] = source_obj["last_modified"]
        row["open_data_sidecar_size_bytes"] = source_obj["size_bytes"]
        row["open_data_sidecar_etag"] = source_obj["etag"]

    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_id[str(row["item_id"])].append(row)
    unique_rows = [sorted(v, key=lambda r: r["retrieval_file"])[0] for v in by_id.values()]
    unique_rows.sort(key=lambda r: (r["datetime"] or "", r["item_id"] or ""))
    if any(not row["bbox"] for row in unique_rows):
        raise SystemExit("错误：存在缺少 bbox 的唯一条目，停止空间粗分组。")

    components = make_components(unique_rows)
    repeated_groups = []
    all_pairs = []
    for gi, group in enumerate(components, 1):
        signatures: dict[tuple, list[dict]] = defaultdict(list)
        for row in group:
            sig = (row["platform"], row["polarizations"], row["looks_azimuth"], row["looks_range"])
            signatures[sig].append(row)
        for sig, matches in signatures.items():
            matches.sort(key=lambda r: (r["datetime"] or "", r["item_id"] or ""))
            if len(matches) >= 2:
                rg = {
                    "screen_group": f"bbox_group_{gi:02d}",
                    "platform_polarization_looks": list(sig),
                    "item_ids": [r["item_id"] for r in matches],
                    "dates": [r["datetime"] for r in matches],
                }
                repeated_groups.append(rg)
                for i in range(len(matches)):
                    for j in range(i + 1, len(matches)):
                        d1, d2 = parse_dt(matches[i]["datetime"]), parse_dt(matches[j]["datetime"])
                        days = abs((d2 - d1).total_seconds()) / 86400 if d1 and d2 else None
                        a1, a2 = matches[i]["incidence_angle"], matches[j]["incidence_angle"]
                        all_pairs.append({
                            "screen_group": rg["screen_group"],
                            "item_id_a": matches[i]["item_id"],
                            "item_id_b": matches[j]["item_id"],
                            "interval_days": days,
                            "incidence_angle_difference_degrees": abs(a2 - a1) if a1 is not None and a2 is not None else None,
                        })

    dates = sorted({r["date"] for r in unique_rows if r["date"]})
    platforms = Counter(r["platform"] for r in unique_rows)
    polarizations = Counter(r["polarizations"] for r in unique_rows)
    looks = Counter(str(r["looks_azimuth"]) for r in unique_rows)
    component_sizes = Counter(len(g) for g in components)
    item_fingerprints = defaultdict(set)
    for path, item in raw_items:
        # 相同 item ID 的原始载荷指纹单独保留，不根据 ID 假定内容相同。
        item_fingerprints[str(item.get("id"))].add(sha256(path))
    id_conflicts = [k for k, v in item_fingerprints.items() if len(v) > 1]

    catalog = json.loads(args.catalog_json.read_text(encoding="utf-8"))
    catalog_info = {
        "file": args.catalog_json.name,
        "sha256": sha256(args.catalog_json),
        "stac_version": catalog.get("stac_version"),
        "catalog_type": catalog.get("type"),
        "links": catalog.get("links", []),
    }
    xml_info = {
        "file": args.listing_xml.name,
        "sha256": sha256(args.listing_xml),
        "prefix": root.findtext("s:Prefix", default="", namespaces=NS),
        "is_truncated": root.findtext("s:IsTruncated", default="", namespaces=NS).lower() == "true",
        "object_count": len(object_rows),
        "stac_sidecar_count": len(meta_objects),
        "listed_object_total_bytes": sum(r["size_bytes"] for r in object_rows),
        "max_last_modified": max((r["last_modified"] for r in object_rows), default=None),
        "stac_sidecar_last_modified_min": min((r["last_modified"] for r in meta_objects), default=None),
        "stac_sidecar_last_modified_max": max((r["last_modified"] for r in meta_objects), default=None),
    }
    metadata_archive = args.output_dir / "umbra_stac_items_20261128.jsonl.gz"
    with metadata_archive.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as f:
                for path, item in raw_items:
                    f.write(json.dumps({"retrieval_file": path.name, "sha256": sha256(path), "item": item}, ensure_ascii=False, sort_keys=True) + "\n")

    with (args.output_dir / "公开前缀对象清单.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "last_modified", "size_bytes", "etag"])
        w.writeheader()
        w.writerows(object_rows)
    with (args.output_dir / "STAC条目清单.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["retrieval_file", "metadata_sha256", "open_data_sidecar_key", "open_data_sidecar_last_modified", "open_data_sidecar_size_bytes", "open_data_sidecar_etag", "item_id", "collection", "datetime", "platform", "polarizations", "looks_azimuth", "looks_range", "incidence_angle", "resolution_azimuth", "resolution_range", "task_id", "payload_asset_count", "bbox"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(row[k], ensure_ascii=False) if k == "bbox" else row[k] for k in fields})

    result = {
        "audit_id": "umbra_open_data_metadata_screen_20261128_v1",
        "retrieved_date_utc": "2026-11-28",
        "access_scope": "仅读取公开 S3 对象清单与 STAC JSON 元数据；未下载或解码任何影像、NITF、GeoTIFF、CPHD 等数据资产。",
        "source_urls": {
            "official_open_data_page": "https://umbra.space/open-data/",
            "aws_open_data_registry": "https://registry.opendata.aws/umbra-open-data/",
            "archive_stac_documentation": "https://docs.canopy.umbra.space/docs/archive-catalog-searching-via-stac-api",
            "public_s3_prefix": "s3://umbra-open-data-catalog/open-data/",
        },
        "catalog_snapshot": catalog_info,
        "public_prefix_listing": xml_info,
        "metadata_file_count": len(files),
        "metadata_files_total_bytes": sum(p.stat().st_size for p in files),
        "metadata_archive": {"file": metadata_archive.name, "sha256": sha256(metadata_archive), "size_bytes": metadata_archive.stat().st_size},
        "items": {
            "retrieved_sidecars": len(rows),
            "unique_item_ids": len(unique_rows),
            "duplicate_id_occurrences": len(rows) - len(unique_rows),
            "duplicate_item_ids": {k: len(v) for k, v in sorted(by_id.items()) if len(v) > 1},
            "same_id_with_different_raw_json_hashes": id_conflicts,
            "acquisition_date_min": dates[0] if dates else None,
            "acquisition_date_max": dates[-1] if dates else None,
            "unique_acquisition_days": len(dates),
            "platform_counts": dict(sorted(platforms.items())),
            "polarization_counts": dict(sorted(polarizations.items())),
            "azimuth_look_counts": dict(sorted(looks.items())),
            "spatial_bbox_components": {
                "method": "两个 STAC bbox 的经纬度平面相交面积 / 较小 bbox 经纬度平面面积 >= 0.5 即连边，取连通分量；仅为粗筛，不是米制地理分组或正式 split。",
                "threshold": 0.5,
                "component_count": len(components),
                "component_size_distribution": {str(k): v for k, v in sorted(component_sizes.items())},
                "components": [
                    {"screen_group": f"bbox_group_{i:02d}", "item_ids": [r["item_id"] for r in sorted(g, key=lambda x: (x["datetime"] or "", x["item_id"] or ""))]}
                    for i, g in enumerate(components, 1)
                ],
                "same_platform_polarization_look_repeated_groups": repeated_groups,
                "repeated_acquisition_pair_count": len(all_pairs),
                "repeated_acquisition_pairs": all_pairs,
                "pair_interval_days_min": min((x["interval_days"] for x in all_pairs if x["interval_days"] is not None), default=None),
                "pair_interval_days_max": max((x["interval_days"] for x in all_pairs if x["interval_days"] is not None), default=None),
                "max_incidence_angle_difference_degrees": max((x["incidence_angle_difference_degrees"] for x in all_pairs if x["incidence_angle_difference_degrees"] is not None), default=None),
            },
        },
        "not_verified": [
            "产品的绝对定标和数据域与目标任务的对应关系",
            "同一组件内影像的像素级配准、共用栅格与相干性",
            "多时相目标变化、散斑独立性及其可否构造伪参考",
            "bbox 连通分量是否对应真实场景或独立来源组",
            "所列开放前缀在本次查询时点之后的持续更新情况",
        ],
        "object_listing": object_rows,
        "stac_sidecar_manifest": rows,
    }
    manifest_path = args.output_dir / "元数据筛查清单.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已归档 STAC 元数据：{len(rows)} 份，{len(unique_rows)} 个唯一条目 ID。")
    print(f"公开前缀对象：{len(object_rows)} 个；列出字节总量：{xml_info['listed_object_total_bytes']}（未下载资产）。")
    print(f"粗略包围框连通组：{len(components)} 组；同平台/极化/视数重复候选组：{len(repeated_groups)} 组、{len(all_pairs)} 对。")
    print(f"元数据清单 SHA-256：{sha256(manifest_path)}")
    print(f"输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()
