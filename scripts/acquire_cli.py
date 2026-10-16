from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable when run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset_acquisition import acquire_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset Acquisition & Optimization Agent — CLI"
    )
    parser.add_argument("--request", help="Natural-language description of the ML task")
    parser.add_argument("--target", help="Explicit target column name")
    parser.add_argument("--file", help="Path to a user-provided dataset file")
    parser.add_argument("--query", help="Search query for external sources")
    parser.add_argument(
        "--format", default="csv",
        choices=["csv", "xlsx", "excel", "parquet", "json", "sql"],
        help="Output format",
    )
    parser.add_argument("--out", default="./acquired_datasets", help="Output directory")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--features", type=int, default=None, help="Requested input features")
    parser.add_argument("--no-save", action="store_true", help="Do not write output file")
    args = parser.parse_args()

    if not args.request and not args.file and not args.query:
        parser.error("Provide --request, --file, or --query")

    report = acquire_dataset(
        request=args.request,
        local_file_path=args.file,
        query=args.query,
        output_format=args.format,
        output_dir=args.out,
        save_csv=not args.no_save,
        max_candidates=args.max_candidates,
        requested_input_features=args.features,
    )

    print("\n=== ACQUISITION REPORT ===")
    print(f"Status            : {report.get('status')}")
    if report.get("status") == "success":
        prob = report.get("problem", {})
        dim = report.get("dimensions", {})
        dq = report.get("dataset_quality", {})
        print(f"Source            : {report['dataset_selected'].get('source')} / {report['dataset_selected'].get('name')}")
        print(f"Target            : {prob.get('target')} ({prob.get('target_type')})")
        print(f"Rows              : {dim.get('final_rows')}")
        print(f"Input features    : {dim.get('final_input_features')}")
        print(f"Overall quality   : {dq.get('overall_score')}")
        print(f"Removed features  : {[r['feature'] for r in report.get('removed_features', [])]}")
        print(f"Output file       : {report.get('output_path')}")
        if report.get("warnings"):
            print(f"Warnings          : {report['warnings']}")
    else:
        print(report.get("message", ""))

    print("\nFull report written to acquire_report.json")
    with open(Path(args.out) / "acquire_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
