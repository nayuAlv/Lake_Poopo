import ee 
import pandas as pd
import time
ee.Initialize(project="lake-poopo-climate-2026")

# ID of the dataset containing maps of the location and temporal distribution of surface water from 1984 to 2021 in Google Earth Engine
jrc_monthly = "JRC/GSW1_4/MonthlyHistory"

# Coordinates for the bounds of a rectagle in which the Lake is contained.
# Google earth engine uses the format: (min_lon, min_lat, max_lon, max_lat) i.e. (West, South, East, North)
poopo_area = [-67.50, -19.20, -66.60,-18.30]
# Defines a rectangular geographic area within Earth engine
poopo_geom = ee.Geometry.Rectangle(list(poopo_area))



def query_chunk(chunk_start, chunk_end):
    #Define a collection of monthly global surface water images filtered by time and location
    coll = (ee.ImageCollection(jrc_monthly)
                .filterDate(chunk_start, chunk_end)
                .filterBounds(poopo_geom))
    # The algorithm to map over the collection of images
    def per_image(img):
        water = img.select("water")
        # Generate an image in which the value of each pixel is the area of that pixel in square meters.
        pixel_area = ee.Image.pixelArea() 
        # Combine both metrics into a single reduceRegion call
        stats = (pixel_area.updateMask(water.eq(2)).rename("water_area")
                .addBands(pixel_area.updateMask(water.gt(0)).rename("valid_area"))
                .reduceRegion(reducer=ee.Reducer.sum(), geometry=poopo_geom,
                scale=30, maxPixels=1e10))
        return ee.Feature(None, {
              "date": img.date().format("YYYY-MM-dd"),
                "water_area_km2": ee.Number(stats.get("water_area")).divide(1e6),
                "valid_area_km2": ee.Number(stats.get("valid_area")).divide(1e6),
        })

    return coll.map(per_image).getInfo()["features"]

def monthly_water_area_km2(start: str, end: str,
                           chunk_years: int = 2,
                           max_retries: int = 3) -> pd.DataFrame:
    """
    Function to extract the monthly surface water extent in m^2

    Splits the time range into smaller chunks and retries with backoff
    on transient failures (concurrency, timeouts).

    """

    # Build the chunk boundaries
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
    chunks, cursor = [], start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + pd.DateOffset(years=chunk_years), end_dt)
        chunks.append((cursor.strftime("%Y-%m-%d"),
                       chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end

    all_features = []
    for cs, ce in chunks:
        for attempt in range(max_retries):
            try:
                print(f"  Querying {cs} → {ce} (attempt {attempt + 1})...")
                features = query_chunk(cs, ce)
                all_features.extend(features)
                time.sleep(1)
                break
            # ee.EEException is the exception class that the Earth Engine library raises when GEE itself rejected a request
            except ee.EEException as exc:
                if "concurrent" in str(exc).lower() and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 2)
                    print(f"  Concurrency limit — waiting {wait}s and retrying...")
                    time.sleep(wait)
                else:
                    print(f"  Chunk {cs}-{ce} failed permanently: {exc}")
                    break

    rows = [f["properties"] for f in all_features]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df