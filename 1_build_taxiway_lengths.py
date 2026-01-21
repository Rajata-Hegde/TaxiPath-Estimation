import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

xlsx = pd.ExcelFile("airport_data.xlsx")
taxiways_df = xlsx.parse(0)
vertices_df = xlsx.parse(1)

vertex_lookup = dict(
    zip(vertices_df["Vertex_Index"],
        zip(vertices_df["Latitude"], vertices_df["Longitude"]))
)

taxiway_length = {}

for _, row in taxiways_df.iterrows():
    tid = row["Ident"]
    vertex_ids = list(map(int, row["Vertex_Indices"].split(",")))

    length = 0.0
    for i in range(len(vertex_ids) - 1):
        lat1, lon1 = vertex_lookup[vertex_ids[i]]
        lat2, lon2 = vertex_lookup[vertex_ids[i+1]]
        length += haversine(lat1, lon1, lat2, lon2)

    taxiway_length[tid] = taxiway_length.get(tid, 0) + length

pd.Series(taxiway_length).to_csv("taxiway_lengths.csv")
print("✅ taxiway_lengths.csv created")
