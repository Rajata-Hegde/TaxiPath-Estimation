import pandas as pd
import glob
import os
from math import radians, sin, cos, sqrt, atan2

# ================== HAVERSINE ==================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


# ================== LOAD AIRPORT GRAPH ==================
xlsx = pd.ExcelFile("airport_data.xlsx")
taxiways_df = xlsx.parse(0)    # Ident, Vertex_Indices
vertices_df = xlsx.parse(1)    # Vertex_Index, Latitude, Longitude

# vertex_id → (lat, lon)
vertex_lookup = dict(zip(
    vertices_df["Vertex_Index"],
    zip(vertices_df["Latitude"], vertices_df["Longitude"])
))

# vertex_id → taxiway_id
vertex_to_taxiway = {}
for _, row in taxiways_df.iterrows():
    taxiway_id = row["Ident"]
    vertex_ids = list(map(int, row["Vertex_Indices"].split(",")))
    for v in vertex_ids:
        vertex_to_taxiway[v] = taxiway_id


# ================== NEAREST VERTEX ==================
def nearest_vertex(lat, lon):
    return min(
        vertex_lookup,
        key=lambda v: haversine(lat, lon, *vertex_lookup[v])
    )


# ================== EXTRACT TAXI PATH ==================
def extract_taxi_path(taxi_df):
    path = []

    for _, r in taxi_df.iterrows():
        v = nearest_vertex(r["Latitude"], r["Longitude"])
        if v in vertex_to_taxiway:
            t = vertex_to_taxiway[v]
            if not path or path[-1] != t:
                path.append(t)

    return path


# ================== LOAD TAXIWAY LENGTHS ==================
taxiway_length = pd.read_csv(
    "taxiway_lengths.csv",
    index_col=0,
    header=None
)[1].to_dict()


# ================== PROCESS FLIGHT CSVs ==================
rows = []
log_rows = []

csv_files = glob.glob("Landing/*.csv") + glob.glob("Takeoff/*.csv")

for file in csv_files:
    df = pd.read_csv(file)

    # ---- FIX 1: SPLIT POSITION INTO LAT / LON ----
    if "Position" not in df.columns:
        raise ValueError(f"❌ Position column missing in {file}")

    df[["Latitude", "Longitude"]] = (
        df["Position"]
        .astype(str)
        .str.split(",", expand=True)
        .astype(float)
    )

    df["Timestamp"] = df["Timestamp"].astype(float)

    source_file = os.path.basename(file)
    operation_type = "Landing" if "Landing" in file else "Takeoff"

    # ---- FILTER TRUE TAXI ----
    taxi_df = df[
        (df["Altitude"] == 0) &
        (df["Speed"] > 1) &
        (df["Speed"] < 30)
    ]

    if len(taxi_df) < 2:
        log_rows.append({
            "source_file": source_file,
            "status": "SKIPPED",
            "reason": "Not enough taxi points"
        })
        continue

    taxi_time = taxi_df["Timestamp"].max() - taxi_df["Timestamp"].min()

    if taxi_time > 1500:
        log_rows.append({
            "source_file": source_file,
            "status": "SKIPPED",
            "reason": f"Unrealistic taxi_time={taxi_time:.1f}"
        })
        continue

    # ---- DYNAMIC TAXI PATH ----
    path = extract_taxi_path(taxi_df)

    if len(path) < 1:
        log_rows.append({
            "source_file": source_file,
            "status": "SKIPPED",
            "reason": "Taxi path extraction failed"
        })
        continue

    lengths = [taxiway_length[p] for p in path if p in taxiway_length]

    if len(lengths) == 0:
        log_rows.append({
            "source_file": source_file,
            "status": "SKIPPED",
            "reason": "No valid taxiway lengths"
        })
        continue

    # ---- DATASET ROW ----
    rows.append({
        "source_file": source_file,
        "operation_type": operation_type,
        "taxi_path": "-".join(path),

        "total_distance": sum(lengths),
        "num_segments": len(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "max_length": max(lengths),
        "turn_count": len(lengths) - 1,

        "taxi_time": taxi_time
    })

    log_rows.append({
        "source_file": source_file,
        "operation_type": operation_type,
        "status": "USED",
        "taxi_path": "-".join(path),
        "taxi_time": round(taxi_time, 2)
    })


# ================== SAVE OUTPUTS ==================
dataset = pd.DataFrame(rows)
dataset.to_csv("dataset.csv", index=False)

log_df = pd.DataFrame(log_rows)
log_df.to_csv("dataset_build_log.csv", index=False)

print(f"✅ dataset.csv created with {len(dataset)} samples")
print("📝 dataset_build_log.csv created")
