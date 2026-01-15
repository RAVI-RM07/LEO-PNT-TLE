import numpy as np
import requests
from math import pi
from skyfield.api import load, wgs84
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

# ================= CONFIG =================
MU = 398600.4418
EARTH_RADIUS_KM = 6378.137

# ================= FLASK APP =================
app = Flask(__name__)
CORS(app)

# ================= SATELLITE TRACKER =================
class SatelliteTracker:
    def __init__(self):
        self.ts = load.timescale()
        self.lat = 18.4939
        self.lon = 74.0193
        self.sats = []
        self.leo_sats = []
        self.freq_map = {}
        self.initialized = False

    def initialize(self):
        print("Initializing Satellite Tracker...")

        # Load TLEs
        tle_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        self.sats = load.tle_file(tle_url)
        self.leo_sats = [s for s in self.sats if self._is_leo(s)]

        print(f"Loaded {len(self.leo_sats)} LEO satellites")

        self.initialized = True

    def set_location(self, lat, lon):
        self.lat = float(lat)
        self.lon = float(lon)

    def _is_leo(self, sat):
        try:
            n_rad = sat.model.no_kozai
            rev_day = (n_rad / (2 * pi)) * 1440
            rad_sec = rev_day * (2 * pi) / 86400
            a_km = (MU / (rad_sec ** 2)) ** (1 / 3)
            alt = a_km - EARTH_RADIUS_KM
            return 160 < alt < 2000
        except Exception:
            return False

    def get_visible_satellites(self, min_elev=10):
        if not self.initialized:
            return []

        t = self.ts.now()
        observer = wgs84.latlon(self.lat, self.lon)
        visible = []

        for sat in self.leo_sats:
            try:
                topocentric = (sat - observer).at(t)
                alt, az, _ = topocentric.altaz()

                if alt.degrees >= min_elev:
                    geo = sat.at(t).subpoint()
                    visible.append({
                        "unique_id": sat.model.satnum,
                        "name": sat.name,
                        "sub_latitude": round(geo.latitude.degrees, 6),
                        "sub_longitude": round(geo.longitude.degrees, 6),
                        "elevation_deg": round(alt.degrees, 1),
                        "azimuth_deg": round(az.degrees, 1),
                        "frequency_mhz": None
                    })
            except Exception:
                continue

            if len(visible) >= 20:
                break

        return sorted(visible, key=lambda x: x["elevation_deg"], reverse=True)

# ================= INITIALIZATION =================
tracker = SatelliteTracker()

def init_tracker():
    tracker.initialize()

threading.Thread(target=init_tracker, daemon=True).start()

# ================= API ROUTE =================
@app.route("/api/visible", methods=["POST"])
def api_visible():
    if not tracker.initialized:
        return jsonify({"status": "initializing", "satellites": []})

    data = request.get_json() or {}
    tracker.set_location(data.get("latitude", tracker.lat),
                         data.get("longitude", tracker.lon))

    sats = tracker.get_visible_satellites()

    return jsonify({
        "latitude": tracker.lat,
        "longitude": tracker.lon,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "count": len(sats),
        "satellites": sats
    })

# ================= RUN =================
if __name__ == "__main__":
    print("Backend running at http://127.0.0.1:5500")
    app.run(debug=True, port=5500)
