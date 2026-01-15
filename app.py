
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
from datetime import datetime
from leo_engine import SatelliteTracker

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

if __name__ == '__main__':
    # Start init thread
    threading.Thread(target=init_tracker, daemon=True).start()
    app.run(debug=True, port=5000)



