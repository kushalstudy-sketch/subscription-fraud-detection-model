
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "num_logins",
    "num_unique_ips",
    "num_unique_devices",
    "num_unique_countries",
    "num_unique_cities",
    "primary_ip_share",
    "max_implied_speed_kmh",
    "max_concurrent_streams",
    "logins_per_day",
]

LOG_FEATURE_COLS = [
    "num_unique_ips",
    "num_unique_countries",
    "num_unique_cities",
    "max_implied_speed_kmh",
]


def load_data(input_path: Path) -> pd.DataFrame:
    """Load account features from a CSV file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    missing_cols = [col for col in FEATURE_COLS if col not in df.columns]

    if missing_cols:
        raise ValueError(
            "Input file is missing required feature columns: "
            + ", ".join(missing_cols)
        )

    return df


def preprocess_features(df: pd.DataFrame) -> np.ndarray:
    """
    Apply transformations and standardization to model features.
    """
    X = df[FEATURE_COLS].copy()

    # Log-transform selected heavy-tailed features.
    for col in LOG_FEATURE_COLS:
        X[col] = np.log1p(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def add_isolation_forest_scores(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    """
    Compute unsupervised anomaly scores using Isolation Forest.

    sklearn's score_samples() returns higher values for more normal
    observations, so we negate the score so that higher values mean
    greater anomaly.
    """
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    df["anomaly_score"] = -model.score_samples(X_scaled)

    return df


def add_kmeans_tiers(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
) -> pd.DataFrame:
    """
    Cluster accounts using KMeans and assign descriptive tiers
    based on average anomaly score within each cluster.
    """
    kmeans = KMeans(
        n_clusters=3,
        n_init=10,
        random_state=42,
    )

    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Order clusters according to their average anomaly score.
    cluster_rank = (
        df.groupby("cluster")["anomaly_score"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    cluster_tier_map = {
        cluster_rank[0]: "Safe",
        cluster_rank[1]: "Potential Abuse",
        cluster_rank[2]: "Abusive",
    }

    df["cluster_tier"] = df["cluster"].map(cluster_tier_map)

    return df


def add_percentile_tiers(
    df: pd.DataFrame,
    safe_pct: float,
    potential_pct: float,
) -> pd.DataFrame:
    """
    Convert anomaly scores into business-oriented tiers.

    Example:
        safe_pct = 0.70
        potential_pct = 0.22

    Results in approximately:
        70% -> Safe
        22% -> Potential Abuse
        8%  -> Abusive
    """
    q_safe = df["anomaly_score"].quantile(safe_pct)
    q_potential = df["anomaly_score"].quantile(
        safe_pct + potential_pct
    )

    def tier(score: float) -> str:
        if score <= q_safe:
            return "Safe"
        if score <= q_potential:
            return "Potential Abuse"
        return "Abusive"

    df["anomaly_tier"] = df["anomaly_score"].apply(tier)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print model output summaries."""
    print("\nPercentile-based anomaly tier distribution:")
    print(df["anomaly_tier"].value_counts())

    print("\nKMeans-based cluster tier distribution:")
    print(df["cluster_tier"].value_counts())

    # Ground truth is never used during model fitting.
    # This comparison is only for post-hoc evaluation.
    if "ground_truth_label" in df.columns:
        print("\nSanity check: predictions vs ground-truth labels")
        print("Ground truth vs anomaly tier:")
        print(
            pd.crosstab(
                df["ground_truth_label"],
                df["anomaly_tier"],
            )
        )

        print("\nGround truth vs cluster tier:")
        print(
            pd.crosstab(
                df["ground_truth_label"],
                df["cluster_tier"],
            )
        )


def validate_percentages(
    safe_pct: float,
    potential_pct: float,
) -> None:
    """Validate percentile configuration."""
    if not 0 < safe_pct < 1:
        raise ValueError("--safe_pct must be between 0 and 1.")

    if not 0 < potential_pct < 1:
        raise ValueError("--potential_pct must be between 0 and 1.")

    if safe_pct + potential_pct >= 1:
        raise ValueError(
            "--safe_pct + --potential_pct must be less than 1."
        )


def score_accounts(
    input_path: Path,
    output_path: Path,
    safe_pct: float,
    potential_pct: float,
) -> None:
    """Run the complete account anomaly-scoring pipeline."""
    validate_percentages(safe_pct, potential_pct)

    print(f"Loading data from: {input_path}")

    df = load_data(input_path)
    X_scaled = preprocess_features(df)

    # Isolation Forest anomaly detection.
    df = add_isolation_forest_scores(df, X_scaled)

    # KMeans provides an independent unsupervised grouping.
    df = add_kmeans_tiers(df, X_scaled)

    # Convert anomaly scores into business-oriented tiers.
    df = add_percentile_tiers(
        df,
        safe_pct=safe_pct,
        potential_pct=potential_pct,
    )

    # Create output directory if necessary.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(
        f"\nWrote {len(df):,} scored accounts -> {output_path}"
    )

    print_summary(df)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Unsupervised account abuse/anomaly scoring."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/account_features.csv"),
        help="Path to the input CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/account_features_scored.csv"),
        help="Path to the output CSV file.",
    )

    parser.add_argument(
        "--safe_pct",
        type=float,
        default=0.70,
        help="Fraction of accounts labeled Safe (default: 0.70).",
    )

    parser.add_argument(
        "--potential_pct",
        type=float,
        default=0.22,
        help=(
            "Fraction of accounts labeled Potential Abuse "
            "(default: 0.22)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    score_accounts(
        input_path=args.input,
        output_path=args.output,
        safe_pct=args.safe_pct,
        potential_pct=args.potential_pct,
    )


if __name__ == "__main__":
    main()