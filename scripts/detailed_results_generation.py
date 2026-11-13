
import sys
import json
import ast
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from settings import INPUT_DIR, OUTPUT_DIR

WORKLOADS = [
    "product_catalogue_extraction",
    "customer_issue_classification",
    "product_review_summarization",
    "customer_response_drafting",
]


def parse_scores(value):
    """Parse the list of criterion evaluations stored in scores."""
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    if isinstance(parsed, dict):
        parsed = parsed.get("scores", [])

    return parsed if isinstance(parsed, list) else []


def expand_criteria(df):
    """Create a score and reason column for each criterion."""
    parsed_scores = df["scores"].apply(parse_scores)

    criterion_rows = []

    for evaluations in parsed_scores:
        row = {}

        for evaluation in evaluations:
            if not isinstance(evaluation, dict):
                continue

            criterion = evaluation.get("criterion")
            if not criterion:
                continue

            row[f"{criterion}_score"] = evaluation.get("score")
            row[f"{criterion}_reason"] = evaluation.get("reason")

        criterion_rows.append(row)

    criteria_df = pd.DataFrame(
        criterion_rows,
        index=df.index,
    )

    return pd.concat([df, criteria_df], axis=1)


def get_workload_distributions(workload):
    evaluation_file = INPUT_DIR / f"{workload}_evaluation.csv"
    generation_file = INPUT_DIR / f"{workload}_results.csv"

    if not evaluation_file.exists():
        print(f"Skipping {workload}: evaluation file missing.")
        return None

    evaluation_df = pd.read_csv(evaluation_file)

    # Keep all evaluations and expand criterion-specific details.
    if "scores" in evaluation_df.columns:
        evaluation_df = expand_criteria(evaluation_df)

    # Generation metrics are optional.
    if generation_file.exists():
        generation_df = pd.read_csv(generation_file)

        candidate_keys = [
            "id", "sample_id", "record_id",
            "row_id", "item_id", "index",
        ]

        shared_id = next(
            (
                col for col in candidate_keys
                if col in evaluation_df.columns
                and col in generation_df.columns
                and not evaluation_df[col].isna().any()
                and not generation_df[col].isna().any()
                and not evaluation_df.duplicated(
                    ["model_name", col]
                ).any()
                and not generation_df.duplicated(
                    ["model_name", col]
                ).any()
            ),
            None,
        )

        if shared_id:
            join_keys = ["model_name", shared_id]
        else:
            evaluation_df["_row_seq"] = (
                evaluation_df.groupby(
                    "model_name", dropna=False
                ).cumcount()
            )
            generation_df["_row_seq"] = (
                generation_df.groupby(
                    "model_name", dropna=False
                ).cumcount()
            )
            join_keys = ["model_name", "_row_seq"]

        generation_columns = [
            "latency_seconds",
            "input_tokens",
            "output_tokens",
        ]

        available_columns = [
            col for col in generation_columns
            if col in generation_df.columns
            and col not in evaluation_df.columns
        ]

        evaluation_df = pd.merge(
            evaluation_df,
            generation_df[join_keys + available_columns],
            on=join_keys,
            how="left",
            validate="one_to_one",
        )

        evaluation_df.drop(
            columns=["_row_seq"],
            inplace=True,
            errors="ignore",
        )

    evaluation_df["workload"] = workload

    # Put metadata first, then criterion score/reason columns.
    metadata_columns = [
        "workload",
        "model_name",
        "input_data",
        "generated_output",
        "overall_score",
        "overall_comment",
        "judge_model",
        "judge_latency_seconds",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "status",
    ]

    metadata_columns = [
        col for col in metadata_columns
        if col in evaluation_df.columns
    ]

    criterion_columns = [
        col for col in evaluation_df.columns
        if col.endswith(("_score", "_reason"))
        and col not in metadata_columns
    ]

    remaining_columns = [
        col for col in evaluation_df.columns
        if col not in metadata_columns + criterion_columns
    ]

    evaluation_df = evaluation_df[
        metadata_columns
        + criterion_columns
        + remaining_columns
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{workload}_distributions.csv"
    evaluation_df.to_csv(output_file, index=False)

    print(f"Saved: {output_file} ({len(evaluation_df)} rows)")

    return evaluation_df


def main():
    all_workloads = []

    for workload in WORKLOADS:
        print(f"\nProcessing: {workload}")

        df = get_workload_distributions(workload)

        if df is not None:
            all_workloads.append(df)

    if all_workloads:
        combined_df = pd.concat(
            all_workloads,
            ignore_index=True,
        )

        combined_file = (
            OUTPUT_DIR / "all_workloads_distributions.csv"
        )

        combined_df.to_csv(combined_file, index=False)

        print(
            f"\nSaved: {combined_file} "
            f"({len(combined_df)} evaluations)"
        )

        return combined_df

    return pd.DataFrame()


if __name__ == "__main__":
    df = main()
