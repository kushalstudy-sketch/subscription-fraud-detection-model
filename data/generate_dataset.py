"""
Synthetic Subscription Account-Sharing Dataset Generator
==========================================================
Simulates a streaming service's raw login/session logs for a population of
accounts belonging to three behavior profiles, then engineers account-level
features and labels of the kind a real abuse-detection model would use.

Outputs two CSVs:
  1. login_events.csv     -> one row per login/streaming session (raw log)
  2. account_features.csv -> one row per account, aggregated features + labels

Usage:
  python generate_dataset.py --num_accounts 2000 --days 60 --seed 42
"""

import argparse
import random
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Reference data: cities with coordinates and country, used to simulate
#    where logins geographically originate from.
# ---------------------------------------------------------------------------
CITIES = [
    ("New York", "US", 40.7128, -74.0060),
    ("Los Angeles", "US", 34.0522, -118.2437),
    ("Chicago", "US", 41.8781, -87.6298),
    ("Toronto", "CA", 43.6532, -79.3832),
    ("Mexico City", "MX", 19.4326, -99.1332),
    ("London", "UK", 51.5074, -0.1278),
    ("Paris", "FR", 48.8566, 2.3522),
    ("Berlin", "DE", 52.5200, 13.4050),
    ("Madrid", "ES", 40.4168, -3.7038),
    ("Rome", "IT", 41.9028, 12.4964),
    ("Mumbai", "IN", 19.0760, 72.8777),
    ("Delhi", "IN", 28.7041, 77.1025),
    ("Bangalore", "IN", 12.9716, 77.5946),
    ("Singapore", "SG", 1.3521, 103.8198),
    ("Tokyo", "JP", 35.6762, 139.6503),
    ("Sydney", "AU", -33.8688, 151.2093),
    ("Sao Paulo", "BR", -23.5505, -46.6333),
    ("Lagos", "NG", 6.5244, 3.3792),
    ("Dubai", "AE", 25.2048, 55.2708),
    ("Johannesburg", "ZA", -26.2041, 28.0473),
]

DEVICE_TYPES = ["Smart TV", "Phone", "Laptop", "Tablet", "Game Console", "Browser"]

# ---------------------------------------------------------------------------
# 2. Behavior profiles: control how "spread out" an account's logins are.
#    These are the GROUND-TRUTH generative labels (used to validate the
#    feature engineering + rule-based labeling further down).
# ---------------------------------------------------------------------------
PROFILES = {
    "safe": {
        "label": "Safe",
        "weight": 0.70,
        "num_home_cities": (1, 1),      # usually just one household
        "num_devices": (1, 3),
        "sessions_per_week": (3, 10),
        "far_login_prob": 0.01,         # chance a session is from a random far city
        "concurrent_prob": 0.03,        # chance of a simultaneous second stream
        "max_concurrent_extra": (0, 1),
    },
    "potential_abuse": {
        "label": "Potential Abuse",
        "weight": 0.22,
        "num_home_cities": (2, 2),      # e.g. account holder + one relative elsewhere
        "num_devices": (3, 6),
        "sessions_per_week": (5, 14),
        "far_login_prob": 0.05,
        "concurrent_prob": 0.15,
        "max_concurrent_extra": (0, 2),
    },
    "abusive": {
        "label": "Abusive",
        "weight": 0.08,
        "num_home_cities": (4, 8),
        "num_devices": (6, 15),
        "sessions_per_week": (10, 25),
        "far_login_prob": 0.15,
        "concurrent_prob": 0.35,
        "max_concurrent_extra": (1, 5),
    },
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def pick_profile():
    names = list(PROFILES.keys())
    weights = [PROFILES[n]["weight"] for n in names]
    return random.choices(names, weights=weights, k=1)[0]


def make_account_setup(account_id):
    """Fix an account's home cities, devices, and per-city IP pool up front,
    so repeated logins from the same place plausibly reuse the same IP
    (like a household behind one router), while abusive accounts still end
    up with far more distinct IPs overall."""
    profile_name = pick_profile()
    profile = PROFILES[profile_name]

    n_cities = random.randint(*profile["num_home_cities"])
    home_cities = random.sample(CITIES, n_cities)

    n_devices = random.randint(*profile["num_devices"])
    devices = [
        (f"dev_{account_id}_{i}", random.choice(DEVICE_TYPES)) for i in range(n_devices)
    ]

    # 1-2 "household" IPs per home city
    city_ip_pool = {}
    for city in home_cities:
        prefix = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
        n_ips = random.choice([1, 1, 2])
        city_ip_pool[city[0]] = [f"{prefix}.{random.randint(1, 254)}" for _ in range(n_ips)]

    return {
        "account_id": account_id,
        "profile_name": profile_name,
        "ground_truth_label": profile["label"],
        "home_cities": home_cities,
        "devices": devices,
        "city_ip_pool": city_ip_pool,
        "sessions_per_week": random.randint(*profile["sessions_per_week"]),
        "far_login_prob": profile["far_login_prob"],
        "concurrent_prob": profile["concurrent_prob"],
        "max_concurrent_extra": profile["max_concurrent_extra"],
    }


def simulate_account_events(setup, days, start_date):
    events = []
    n_weeks = max(1, days // 7)
    total_sessions = setup["sessions_per_week"] * n_weeks

    for _ in range(total_sessions):
        # Choose location: usually a home city, occasionally a random far one
        if random.random() < setup["far_login_prob"]:
            city = random.choice(CITIES)
            ip = f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        else:
            city = random.choice(setup["home_cities"])
            ip = random.choice(setup["city_ip_pool"][city[0]])

        device_id, device_type = random.choice(setup["devices"])
        start_time = start_date + timedelta(
            seconds=random.randint(0, days * 24 * 3600)
        )
        duration_min = random.randint(15, 180)

        events.append(
            {
                "account_id": setup["account_id"],
                "timestamp": start_time,
                "duration_min": duration_min,
                "ip_address": ip,
                "city": city[0],
                "country": city[1],
                "lat": city[2],
                "lon": city[3],
                "device_id": device_id,
                "device_type": device_type,
            }
        )

        # Occasionally add 1+ near-simultaneous sessions on OTHER devices to
        # simulate concurrent streaming (a hallmark of shared accounts)
        if random.random() < setup["concurrent_prob"]:
            n_extra = random.randint(*setup["max_concurrent_extra"])
            other_devices = [d for d in setup["devices"] if d[0] != device_id]
            for _ in range(min(n_extra, len(other_devices))):
                extra_device_id, extra_device_type = random.choice(other_devices)
                # abusive accounts' concurrent streams are more likely to be
                # from a totally different city (i.e., a different person)
                if random.random() < setup["far_login_prob"] * 3:
                    extra_city = random.choice(CITIES)
                    extra_ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
                else:
                    extra_city = city
                    extra_ip = ip
                events.append(
                    {
                        "account_id": setup["account_id"],
                        "timestamp": start_time + timedelta(minutes=random.randint(-10, 10)),
                        "duration_min": random.randint(15, 180),
                        "ip_address": extra_ip,
                        "city": extra_city[0],
                        "country": extra_city[1],
                        "lat": extra_city[2],
                        "lon": extra_city[3],
                        "device_id": extra_device_id,
                        "device_type": extra_device_type,
                    }
                )
    return events


# ---------------------------------------------------------------------------
# 3. Feature engineering: turn raw login events into account-level features
#    that a real abuse-detection model would use.
# ---------------------------------------------------------------------------
def compute_account_features(df_account):
    df_account = df_account.sort_values("timestamp").reset_index(drop=True)

    num_logins = len(df_account)
    num_unique_ips = df_account["ip_address"].nunique()
    num_unique_devices = df_account["device_id"].nunique()
    num_unique_countries = df_account["country"].nunique()
    num_unique_cities = df_account["city"].nunique()
    primary_ip_share = df_account["ip_address"].value_counts(normalize=True).iloc[0]

    # Max implied travel speed (km/h) between any two consecutive logins
    # from different locations -> flags "impossible travel"
    max_speed_kmh = 0.0
    for i in range(1, len(df_account)):
        prev, curr = df_account.iloc[i - 1], df_account.iloc[i]
        dist_km = haversine_km(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
        hours = max((curr["timestamp"] - prev["timestamp"]).total_seconds() / 3600, 1 / 60)
        speed = dist_km / hours
        max_speed_kmh = max(max_speed_kmh, speed)

    # Max concurrent sessions: sweep-line over session intervals
    intervals = sorted(
        (row["timestamp"], row["timestamp"] + timedelta(minutes=row["duration_min"]))
        for _, row in df_account.iterrows()
    )
    events_pm = [(s, 1) for s, e in intervals] + [(e, -1) for s, e in intervals]
    events_pm.sort()
    running, max_concurrent = 0, 0
    for _, delta in events_pm:
        running += delta
        max_concurrent = max(max_concurrent, running)

    span_days = max((df_account["timestamp"].max() - df_account["timestamp"].min()).total_seconds() / 86400, 1)
    logins_per_day = num_logins / span_days

    return pd.Series(
        {
            "num_logins": num_logins,
            "num_unique_ips": num_unique_ips,
            "num_unique_devices": num_unique_devices,
            "num_unique_countries": num_unique_countries,
            "num_unique_cities": num_unique_cities,
            "primary_ip_share": round(primary_ip_share, 3),
            "max_implied_speed_kmh": round(max_speed_kmh, 1),
            "max_concurrent_streams": max_concurrent,
            "logins_per_day": round(logins_per_day, 2),
        }
    )


# ---------------------------------------------------------------------------
# 4. Rule-based labeling: a simple, transparent heuristic you can later try
#    to replace/beat with a trained classifier. This does NOT look at the
#    ground-truth profile label -- it only uses the engineered features.
#    (~900 km/h is faster than a commercial flight, i.e. physically implausible)
# ---------------------------------------------------------------------------
def rule_based_label(row):
    # Thresholds below were picked by inspecting this simulator's own feature
    # distributions per ground-truth group (run the script, then compare
    # account_features.csv grouped by ground_truth_label). In a real system
    # you'd calibrate these against your platform's actual traffic instead.
    if (
        row["num_unique_ips"] >= 20
        or row["num_unique_countries"] >= 8
        or row["max_concurrent_streams"] >= 6
    ):
        return "Abusive"
    if (
        row["num_unique_ips"] >= 3
        or row["num_unique_countries"] >= 2
        or row["max_concurrent_streams"] >= 3
    ):
        return "Potential Abuse"
    return "Safe"


def main(num_accounts, days, seed, out_dir):
    random.seed(seed)
    np.random.seed(seed)

    start_date = datetime(2026, 1, 1)
    all_events = []
    ground_truth = {}  # account_id -> ground-truth label, from the generative profile
    for i in range(num_accounts):
        account_id = f"acct_{i:06d}"
        setup = make_account_setup(account_id)
        ground_truth[account_id] = setup["ground_truth_label"]
        all_events.extend(simulate_account_events(setup, days, start_date))

    login_events = pd.DataFrame(all_events)
    login_events.to_csv(f"{out_dir}/login_events.csv", index=False)

    features = login_events.groupby("account_id").apply(compute_account_features).reset_index()
    features["ground_truth_label"] = features["account_id"].map(ground_truth)
    features["rule_based_label"] = features.apply(rule_based_label, axis=1)

    features.to_csv(f"{out_dir}/account_features.csv", index=False)

    print(f"Wrote {len(login_events):,} login events -> {out_dir}/login_events.csv")
    print(f"Wrote {len(features):,} account feature rows -> {out_dir}/account_features.csv")
    print("\nRule-based label distribution:")
    print(features["rule_based_label"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_accounts", type=int, default=2000)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()
    main(args.num_accounts, args.days, args.seed, args.out_dir)
