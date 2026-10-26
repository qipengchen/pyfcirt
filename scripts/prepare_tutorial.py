"""Convert an explicitly supplied MGPPM tutorial folder into PyFCIRT data.

Input response codes are zero-based score-row IDs, and block_position and
selection_orders are zero-based positions in the shared tutorial. No response
data is bundled, downloaded or licensed for redistribution by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pyfcirt import FCData, FCDesign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["rank", "graded"], required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    target = args.output.resolve()
    targets = [
        target / name
        for name in (
            "design.csv",
            "responses.csv",
            "administration.csv",
            "design_options.json",
            "conversion.json",
        )
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Conversion output already exists in {target}")
    source_files = [source / "statement_design.csv", source / "responses.csv"]
    design_frame = pd.read_csv(source_files[0]).rename(
        columns={"block_position": "position", "loading_sign": "keying"}
    )
    design_frame["position"] += 1
    if (source / "trait_labels.csv").exists():
        labels = pd.read_csv(source / "trait_labels.csv")["trait_name"].tolist()
        traits = dict(enumerate(labels))
        if set(design_frame["trait"]) - set(traits):
            raise ValueError("Trait indices do not match trait_labels.csv")
        design_frame["trait"] = design_frame["trait"].map(traits)
        source_files.append(source / "trait_labels.csv")
    responses = pd.read_csv(source_files[1])
    options = {"response_format": args.task}
    if args.task == "rank":
        orders_file = source / "selection_orders.csv"
        orders = pd.read_csv(orders_file).to_numpy(dtype=int) + 1
        codes = responses["response"]
        if not codes.between(0, len(orders) - 1).all() or not (codes % 1 == 0).all():
            raise ValueError("Response codes do not index the supplied selection orders")
        responses["response"] = [orders[int(code)].tolist() for code in codes]
        source_files.append(orders_file)
    else:
        options["graded_scale"] = {"categories": [0, 1, 2, 3, 4], "higher_prefers": "position_1"}
    design = FCDesign.from_frame(design_frame, **options)
    persons = responses["person_id"].drop_duplicates().tolist()
    if (source / "person_table.csv").exists():
        persons = pd.read_csv(source / "person_table.csv")["person_id"].tolist()
        source_files.append(source / "person_table.csv")
    known = set(zip(responses["person_id"], responses["block_id"], strict=True))
    administration = pd.DataFrame(
        [
            (person, block, "answered" if (person, block) in known else "missing")
            for person in persons
            for block in design.block_sizes
        ],
        columns=["person_id", "block_id", "status"],
    )
    data = FCData.from_frames(responses, design, administration=administration)
    exported = data.to_frame()
    if args.task == "rank":
        exported["response"] = exported["response"].map(json.dumps)
    manifest = {
        "task": args.task,
        "data_kind": "empirical_complete_rankings" if args.task == "rank" else "simulated_example",
        "summary": data.summary(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "position_conversion": "zero-based tutorial position to one-based PyFCIRT position",
        "missing_status": (
            "no complete event in the tutorial response file; not inferred unadministered"
        ),
        "redistribution": "not established by conversion; preserve source attribution and license",
    }
    if (source / "source_manifest.json").exists():
        manifest["upstream_manifest"] = json.loads((source / "source_manifest.json").read_text())
        expected = manifest["upstream_manifest"].get("complete_transitive_rankings")
        if expected is not None and expected != len(responses):
            raise ValueError("Converted event count differs from the tutorial manifest")
    target.mkdir(parents=True, exist_ok=True)
    design.to_frame().to_csv(target / "design.csv", index=False)
    exported.to_csv(target / "responses.csv", index=False)
    administration.to_csv(target / "administration.csv", index=False)
    (target / "design_options.json").write_text(json.dumps(options, indent=2), encoding="utf8")
    (target / "conversion.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
