import numpy as np
import requests
from math import pi, cos, radians, atan2
from skyfield.api import load, wgs84
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time

# --- CONFIG ---
MU = 398600.4418
EARTH_RADIUS_KM = 6378.137
C = 299792458.0
R_E = 6378137.0

class SatelliteTracker:
    def __init__(self): 
        self.ts = load.timescale()
        self.lat = 18.4939
        self.lon = 74.0193
        self.sats = []
        self.leo_sats = []
        self.freq_map = {}
        self.name_map = {}
        self.initialized = False

    def initialize(self):
        print("Initializing Satellite Tracker...")
        # 1. Location
        try:
            res = requests.get("https://ipinfo.io/json", timeout=5).json()
            if "loc" in res:
                lat, lon = res["loc"].split(",")
                self.lat, self.lon = float(lat), float(lon)
        except Exception as e:
            print(f"Location API failed, using fallback: {e}")

        # 2. TLE
        print("Loading TLEs (remote preferred, fallback to local)...")
        tle_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
        local_tle_path = "backend/gp.php"
        self.sats = []
        try:
            self.sats = load.tle_file(tle_url)
            print("Loaded TLEs from remote URL.")
        except Exception as e:
            print(f"Remote TLE load failed: {e}")
            try:
                self.sats = load.tle_file(local_tle_path)
                print(f"Loaded TLEs from local file: {local_tle_path}")
            except Exception as e2:
                print(f"Local TLE load failed: {e2}")
                raise RuntimeError("No valid TLE data available from remote or local file.")
        self.leo_sats = [s for s in self.sats if self._is_LEO(s)]
        print(f"Loaded {len(self.leo_sats)} LEO satellites.")

        # 3. SatNOGS DB
        print("Fetching SatNOGS DB...")
        tx_url = "https://db.satnogs.org/api/transmitters/?format=json"
        try:
            tx_data = requests.get(tx_url, timeout=15).json()
            for tx in tx_data:
                norad = tx.get("norad_cat_id")
                sat_name = (tx.get("sat_id") or "").lower().replace(" ", "")
                down = tx.get("downlink_low")
                mode = tx.get("mode")
                if down:
                    try:
                        freq = down / 1e6
                    except:
                        freq = None
                    if norad and freq:
                        self.freq_map.setdefault(norad, []).append((freq, mode))
                    if sat_name and freq:
                        self.name_map.setdefault(sat_name, []).append((freq, mode))
        except Exception as e:
            print(f"SatNOGS fetch failed: {e}")
        
        self.initialized = True
        return True

    def set_location(self, lat, lon):
        try:
            self.lat = float(lat)
            self.lon = float(lon)
            print(f"Location updated to: {self.lat}, {self.lon}")
            return True
        except ValueError:
            return False

    def _is_LEO(self, sat):
        try:
            n_rad = sat.model.no_kozai
            rev_day = (n_rad/(2*pi))*1440
            rad_sec = rev_day*(2*pi)/86400
            a_km = (MU/(rad_sec**2))**(1/3)
            alt = a_km - EARTH_RADIUS_KM
            T_min = (2*pi/rad_sec)/60
            return 160 < alt < 2000 and 85 < T_min < 130
        except:
            return False

    def get_visible_satellites(self, min_elev=10):
        if not self.initialized:
            return []
        t = self.ts.now()
        observer = wgs84.latlon(self.lat, self.lon)
        visible = []
        for sat in self.leo_sats:
            try:
                difference = sat - observer
                topocentric = difference.at(t)
                alt, az, distance = topocentric.altaz()
                if alt.degrees >= min_elev:
                    # Frequency from SatNOGS if available
                    f0_mhz = None
                    norad = sat.model.satnum
                    if norad in self.freq_map:
                        f0_mhz = self.freq_map[norad][0][0]
                    # Doppler only if freq available
                    doppler_shift = None
                    if f0_mhz:
                        _, _, range_rate_km_s = topocentric.frame_latlon_and_rates(observer)
                        doppler_shift = -(range_rate_km_s * 1000 / C) * (f0_mhz * 1e6)
                    # Subpoint (lat/lon under satellite)
                    geo = sat.at(t).subpoint()
                    sub_lat = round(geo.latitude.degrees, 6)
                    sub_lon = round(geo.longitude.degrees, 6)
                    visible.append({
                        "name": sat.name,
                        "id": norad,
                        "sub_lat": sub_lat,
                        "sub_lon": sub_lon,
                        "elev": round(alt.degrees, 1),
                        "az": round(az.degrees, 1),
                        "dist_km": round(distance.km, 1),
                        "doppler": round(doppler_shift, 1) if doppler_shift is not None else None,
                        "freq": f0_mhz
                    })
            except Exception:
                continue
            if len(visible) > 20:
                break
        visible.sort(key=lambda x: x['elev'], reverse=True)
        return visible

    def get_position_estimate(self):
        return {
            "lat": self.lat,
            "lon": self.lon,
            "accuracy": random_accuracy(),
            "doppler_error": random_doppler_error()
        }

def random_accuracy():
    return round(np.random.normal(5, 1.5), 2)

def random_doppler_error():
    return round(np.random.normal(0, 50), 1)

app = Flask(__name__)
CORS(app)

tracker = SatelliteTracker()

# Background thread to initialize data without blocking server start
def init_tracker():
    tracker.initialize()
    print("Tracker Initialized in Background")

@app.route('/api/visible', methods=['POST'])
def get_visible():
    if not tracker.initialized:
        return jsonify({"status": "initializing", "satellites": [], "latitude": None, "longitude": None, "timestamp": None, "count": 0})

    data = request.get_json(force=True)
    lat = data.get('latitude')
    lon = data.get('longitude')
    try:
        tracker.set_location(lat, lon)
    except Exception:
        return jsonify({"error": "Invalid coordinates"}), 400

    visible_sats = tracker.get_visible_satellites(min_elev=10)
    satellites = []
    for sat in visible_sats:
        elev = sat.get('elev', 0)
        error = round(2.0 - min(elev, 90) / 90.0, 2)
        satellites.append({
            "unique_id": sat.get("id"),
            "name": sat.get("name"),
            "sub_latitude": sat.get("sub_lat"),
            "sub_longitude": sat.get("sub_lon"),
            "elevation_deg": sat.get("elev"),
            "azimuth_deg": sat.get("az"),
            "frequency_mhz": sat.get("freq"),
            "error": error
        })

    resp = {
        "latitude": float(lat),
        "longitude": float(lon),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(satellites),
        "satellites": satellites
    }
    return jsonify(resp)

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>Satellite Tracker</title>
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <h1>Visible Satellites</h1>
            <table>
                <thead>
                    <tr>
                        <th>Satellite Name</th>
                        <th>NORAD</th>
                        <th>Latitude (°)</th>
                        <th>Longitude (°)</th>
                        <th>Elev (°)</th>
                        <th>Az (°)</th>
                        <th>Freq (MHz)</th>
                    </tr>
                </thead>
                <tbody id="satellite-data">
                </tbody>
            </table>
            <script>
                async function fetchData() {
                    const response = await fetch('/api/visible', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            latitude: 18.4939,
                            longitude: 74.0193
                        })
                    });
                    const data = await response.json();
                    const tableBody = document.getElementById('satellite-data');
                    tableBody.innerHTML = '';
                    data.satellites.forEach(sat => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${sat.name}</td>
                            <td>${sat.unique_id}</td>
                            <td>${sat.sub_latitude}</td>
                            <td>${sat.sub_longitude}</td>
                            <td>${sat.elevation_deg}</td>
                            <td>${sat.azimuth_deg}</td>
                            <td>${sat.frequency_mhz}</td>
                        `;
                        tableBody.appendChild(row);
                    });
                }
                fetchData();
                setInterval(fetchData, 5000);
            </script>
        </body>
    </html>
    '''

if __name__ == '__main__':
    # Start init thread
    threading.Thread(target=init_tracker, daemon=True).start()
    app.run(debug=True, port=5000)
