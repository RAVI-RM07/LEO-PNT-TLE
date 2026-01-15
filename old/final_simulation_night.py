# Auto-generated merged script: leo_pnt_full.py

# --- Begin part1.py ---
# === part1.py ===
# Foundations: imports, config, TLE + SatNOGS load, helper functions

import tkinter as tk
from tkinter import ttk
from math import pi, sin, cos, radians, atan2
from skyfield.api import load, wgs84
import time
import requests
import numpy as np
import random
import sys
from tkinter import messagebox


# --------------------------
# BASIC CONFIG
# --------------------------
MU = 398600.4418
EARTH_RADIUS_KM = 6378.137

import requests

def get_initial_location():
    """
    Fetch live location using API.
    If API fails, use fixed fallback coordinates:
    lat = 18.4943645
    lon = 74.0197876
    """
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5).json()

        if "loc" in res:
            lat, lon = res["loc"].split(",")
            print("LIVE LOCATION:", lat, lon)
            return float(lat), float(lon)

        else:
            print("API returned no location, using fallback.")
    except Exception as e:
        print("Location API failed:", e)

    # FALLBACK LOCATION
    fallback_lat = 18.493945904684335
    fallback_lon = 74.01936650276186
    print("Using fallback fixed location:", fallback_lat, fallback_lon)
    return fallback_lat, fallback_lon


LAT, LON = get_initial_location()
LAT = 18.493945904684335
LON = 74.01936650276186
print("Latitude:", LAT, "Longitude:", LON)


# Elevation/elevation thresholds for the three display windows
ELEV_LIMIT = 10          # main window shows >= 10°
THRESH_30 = 30           # window2 threshold
THRESH_60 = 60           # window3 threshold

# UI sizes (kept same as you provided)
GRAPH_SIZE = 750
GRAPH_MAX_R = 320
POLL_MS = 1000           # update interval ms (1 second tick by default)

CAN2_SIZE = 600
CAN2_MAX_R = 250

CAN3_SIZE = 500
CAN3_MAX_R = 210

# color palette for satellite markers
PALETTE = [
    "#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#46f0f0",
    "#f032e6","#bcf60c","#fabebe","#008080","#e6beff","#9A6324","#fffac8"
]

# --------------------------
# SKYFIELD / TLE / SatNOGS
# --------------------------
print("Initializing time scale and location...")
ts = load.timescale()
loc = wgs84.latlon(LAT, LON)


print("Downloading TLEs from CelesTrak...")
tle_url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
try:
    sats = load.tle_file(tle_url)
    print("TLEs loaded:", len(sats))
except Exception as e:
    print("Error loading TLEs:", e)
    sats = []

print("Downloading SatNOGS transmitter DB...")
tx_url = "https://db.satnogs.org/api/transmitters/?format=json"
try:
    tx_data = requests.get(tx_url, timeout=15).json()
    print("SatNOGS entries:", len(tx_data))
except Exception as e:
    print("SatNOGS fetch failed:", e)
    tx_data = []

# Build frequency / name maps from SatNOGS for Doppler/frequency lookup
freq_map = {}
name_map = {}
for tx in tx_data:
    norad = tx.get("norad_cat_id")
    sat_name = (tx.get("sat_id") or "").lower().replace(" ", "")
    down = tx.get("downlink_low")
    mode = tx.get("mode")
    if down:
        try:
            freq = down / 1e6
        except Exception:
            freq = None
        if norad and freq:
            freq_map.setdefault(norad, []).append((freq, mode))
        if sat_name and freq:
            name_map.setdefault(sat_name, []).append((freq, mode))

def is_LEO(sat):
    try:
        n_rad = sat.model.no_kozai
        rev_day = (n_rad/(2*pi))*1440
        rad_sec = rev_day*(2*pi)/86400
        a_km = (MU/(rad_sec**2))**(1/3)
        alt = a_km - EARTH_RADIUS_KM
        T_min = (2*pi/rad_sec)/60
        return 160 < alt < 2000 and 85 < T_min < 130
    except Exception:
        return False

LEO_sats = [s for s in sats if is_LEO(s)]
print("Filtered LEO sats:", len(LEO_sats))

# --------------------------
# HELPER GEOMETRY & CONVERSIONS
# --------------------------
C = 299792458.0
R_E = 6378137.0  # meters (WGS84 approx)

def latlon_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = (R_E + alt_m) * np.cos(lat) * np.cos(lon)
    y = (R_E + alt_m) * np.cos(lat) * np.sin(lon)
    z = (R_E + alt_m) * np.sin(lat)
    return np.array([x, y, z])

def ecef_to_latlon(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return np.rad2deg(lat), np.rad2deg(lon)

def en_to_ecef_offset(east_m, north_m, lat0_deg, lon0_deg):
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    de = east_m
    dn = north_m
    dx = -np.sin(lon0)*de - np.sin(lat0)*np.cos(lon0)*dn
    dy =  np.cos(lon0)*de - np.sin(lat0)*np.sin(lon0)*dn
    dz =  np.cos(lat0)*dn
    return np.array([dx, dy, dz])

def ecef_to_en_delta(xyz, lat0_deg, lon0_deg):
    lat0 = np.deg2rad(lat0_deg); lon0 = np.deg2rad(lon0_deg)
    ref = latlon_to_ecef(lat0_deg, lon0_deg)
    d = xyz - ref
    east = -np.sin(lon0)*d[0] + np.cos(lon0)*d[1]
    north = -np.sin(lat0)*np.cos(lon0)*d[0] - np.sin(lat0)*np.sin(lon0)*d[1] + np.cos(lat0)*d[2]
    return east, north

def en_to_latlon(east_m, north_m, lat0_deg, lon0_deg):
    dlat = (north_m / R_E) * (180.0 / pi)
    dlon = (east_m / (R_E * cos(radians(lat0_deg)))) * (180.0 / pi)
    return lat0_deg + dlat, lon0_deg + dlon

# --------------------------
# SATELLITE STATE & DOPPLER
# --------------------------
def sat_state_ecef_m(sat, t):
    g = sat.at(t)
    pos_km = g.position.km
    vel_km_s = g.velocity.km_per_s
    pos_m = np.array(pos_km) * 1000.0
    vel_m = np.array(vel_km_s) * 1000.0
    return pos_m, vel_m

def predicted_doppler_for_state(sat_pos_m, sat_vel_m, rec_pos_m, rec_vel_m, f0_hz):
    los = rec_pos_m - sat_pos_m
    r = np.linalg.norm(los)
    if r == 0:
        return 0.0
    u = los / r
    v_rel = sat_vel_m - rec_vel_m
    range_rate = np.dot(v_rel, u)
    df = - (range_rate / C) * f0_hz
    return df

def pick_best_six(visible_list):
    if not visible_list:
        return []
    v_sorted = sorted(visible_list, key=lambda x: -x["elev"])
    chosen = []
    for cand in v_sorted:
        if len(chosen) == 0:
            chosen.append(cand)
        else:
            az_c = cand["az"]
            sep = min(abs((az_c - ch["az"] + 180) % 360 - 180) for ch in chosen)
            if sep >= 15 or len(chosen) < 2:
                chosen.append(cand)
        if len(chosen) >= 6:
            break
    if len(chosen) < 6:
        for cand in v_sorted:
            if cand not in chosen:
                chosen.append(cand)
            if len(chosen) >= 6:
                break
    return chosen[:6]

def solve_receiver_from_dopplers_lm(sat_list, meas_df, t_obs, lat0_deg, lon0_deg, alt0_m=560.0):
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    try:
        x[2] = sim_state.get("ve", 0.0)
        x[3] = sim_state.get("vn", 0.0)
    except Exception:
        pass

    sat_pos = []
    sat_vel = []
    f0s = []
    for s in sat_list:
        p_m, v_m = sat_state_ecef_m(s["sat"], t_obs)
        sat_pos.append(p_m)
        sat_vel.append(v_m)
        sid = s["id"]
        f0_mhz = None
        if sid in freq_map:
            f0_mhz = freq_map[sid][0][0]
        else:
            key = s["name"].lower().replace(" ", "")
            if key in name_map:
                f0_mhz = name_map[key][0][0]
        if not f0_mhz:
            f0_mhz = 435.0
        f0s.append(f0_mhz * 1e6)
    sat_pos = np.array(sat_pos)
    sat_vel = np.array(sat_vel)
    f0s = np.array(f0s)

    lam = 1e-2
    for it in range(20):
        e, n, ve, vn = x
        rec_pos_ref = latlon_to_ecef(lat0_deg, lon0_deg, alt0_m)
        delta_ecef = en_to_ecef_offset(e, n, lat0_deg, lon0_deg)
        rec_pos = rec_pos_ref + delta_ecef
        lat0 = radians(lat0_deg); lon0 = radians(lon0_deg)
        rec_vel = np.array([
            -np.sin(lon0)*ve - np.sin(lat0)*np.cos(lon0)*vn,
             np.cos(lon0)*ve - np.sin(lat0)*np.sin(lon0)*vn,
             np.cos(lat0)*vn
        ])
        N = len(sat_list)
        res = np.zeros(N)
        for i in range(N):
            df_pred = predicted_doppler_for_state(sat_pos[i], sat_vel[i], rec_pos, rec_vel, f0s[i])
            res[i] = meas_df[i] - df_pred

        eps = 1e-3
        J = np.zeros((N, 4))
        for k in range(4):
            dx = np.zeros(4); dx[k] = eps
            xp = x + dx
            e2, n2, ve2, vn2 = xp
            delta_ecef_p = en_to_ecef_offset(e2, n2, lat0_deg, lon0_deg)
            rec_pos_p = rec_pos_ref + delta_ecef_p
            rec_vel_p = np.array([
                -np.sin(lon0)*ve2 - np.sin(lat0)*np.cos(lon0)*vn2,
                 np.cos(lon0)*ve2 - np.sin(lat0)*np.sin(lon0)*vn2,
                 np.cos(lat0)*vn2
            ])
            rp = np.zeros(N)
            for i in range(N):
                df_pred_p = predicted_doppler_for_state(sat_pos[i], sat_vel[i], rec_pos_p, rec_vel_p, f0s[i])
                rp[i] = meas_df[i] - df_pred_p
            J[:, k] = (rp - res) / eps

        JTJ = J.T @ J
        A = JTJ + lam * np.diag(np.diag(JTJ) + 1e-6)
        g = J.T @ res
        try:
            dx = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        x_new = x + dx
        e_n, n_n, ve_n, vn_n = x_new
        delta_ecef_n = en_to_ecef_offset(e_n, n_n, lat0_deg, lon0_deg)
        rec_pos_n = rec_pos_ref + delta_ecef_n
        rec_vel_n = np.array([
            -np.sin(lon0)*ve_n - np.sin(lat0)*np.cos(lon0)*vn_n,
             np.cos(lon0)*ve_n - np.sin(lat0)*np.sin(lon0)*vn_n,
             np.cos(lat0)*vn_n
        ])
        res_n = np.zeros(N)
        for i in range(N):
            df_pred_n = predicted_doppler_for_state(sat_pos[i], sat_vel[i], rec_pos_n, rec_vel_n, f0s[i])
            res_n[i] = meas_df[i] - df_pred_n
        if np.linalg.norm(res_n) < np.linalg.norm(res):
            x = x_new
            lam *= 0.7
        else:
            lam *= 2.5
        if np.linalg.norm(dx) < 1e-4:
            break

    e, n, ve, vn = x
    est_lat = lat0_deg + (n / R_E) * (180.0 / pi)
    est_lon = lon0_deg + (e / (R_E * cos(radians(lat0_deg)))) * (180.0 / pi)
    return est_lat, est_lon, x

def color_for(sid, palette=PALETTE):
    try:
        sid_int = int(sid) % len(palette)
    except Exception:
        sid_int = abs(hash(str(sid))) % len(palette)
    return palette[sid_int]

def direction_from_vector(e, n):
    ang = (atan2(e, n) * 180/pi) % 360
    if ang < 22.5: return "N"
    if ang < 67.5: return "NE"
    if ang < 112.5: return "E"
    if ang < 157.5: return "SE"
    if ang < 202.5: return "S"
    if ang < 247.5: return "SW"
    if ang < 292.5: return "W"
    if ang < 337.5: return "NW"
    return "N"





# --- End part1.py ---

# --- Begin part2.py ---
# === part2.py ===
# Window 1: Main window (>=10°) with scrollbars for canvas and table

# Create root window and main layout (root will be created/used in final merge)
root = tk.Tk()
root.title(f"Real-Time LEO Satellites Over Pune (Lat {LAT}, Lon {LON})")
root.geometry("1400x820")

title = tk.Label(root, text=f"Real-Time LEO Satellites Over Pune (Lat {LAT}, Lon {LON})",
                 font=("Arial", 16, "bold"))
title.pack(pady=6)

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

left_main = tk.Frame(main_frame)
left_main.pack(side=tk.LEFT, padx=8, pady=6)

right_main = tk.Frame(main_frame)
right_main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

# main header label for window 1 (missing earlier — this fixes NameError)
main_header = tk.Label(right_main,
                       text="Visible satellites: 0 | Updated: --:--:--",
                       font=("Arial", 11))
main_header.pack(pady=5)


# Canvas frame with scrollbars (always enabled)
main_canvas_frame = tk.Frame(left_main)
main_canvas_frame.pack()

scroll_y_main = tk.Scrollbar(main_canvas_frame, orient="vertical")
scroll_x_main = tk.Scrollbar(main_canvas_frame, orient="horizontal")

canvas_main = tk.Canvas(
    main_canvas_frame, width=GRAPH_SIZE, height=GRAPH_SIZE,
    bg="white", bd=2, relief=tk.SUNKEN,
    yscrollcommand=scroll_y_main.set, xscrollcommand=scroll_x_main.set
)

scroll_y_main.config(command=canvas_main.yview)
scroll_x_main.config(command=canvas_main.xview)

scroll_y_main.pack(side=tk.RIGHT, fill=tk.Y)
scroll_x_main.pack(side=tk.BOTTOM, fill=tk.X)
canvas_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

canvas_main.bind("<Configure>", lambda e: canvas_main.configure(scrollregion=canvas_main.bbox("all")))

mx = GRAPH_SIZE//2
my = GRAPH_SIZE//2

# draw static rings on main canvas
radius60_main = GRAPH_MAX_R * (90 - THRESH_60) / (90 - ELEV_LIMIT)
canvas_main.create_rectangle(2,2,GRAPH_SIZE-2,GRAPH_SIZE-2, outline="#444")
canvas_main.create_oval(mx-GRAPH_MAX_R, my-GRAPH_MAX_R, mx+GRAPH_MAX_R, my+GRAPH_MAX_R, outline="#bbb", dash=(4,6))
canvas_main.create_oval(mx-radius60_main, my-radius60_main, mx+radius60_main, my+radius60_main, outline="#777", dash=(3,5))
canvas_main.create_oval(mx-6,my-6,mx+6,my+6, fill="blue")
canvas_main.create_text(mx, my+14, text="You", font=("Arial", 10))

# Table frame with scrollbars
main_table_frame = tk.Frame(right_main)
main_table_frame.pack(fill=tk.BOTH, expand=True)

scroll_y = ttk.Scrollbar(main_table_frame, orient=tk.VERTICAL)
scroll_x = ttk.Scrollbar(main_table_frame, orient=tk.HORIZONTAL)

cols = ("name","id","lat","lon","elev","az","freq","mode")
titles = ["Satellite Name","NORAD","Latitude (°)","Longitude (°)","Elev (°)","Az (°)","Freq (MHz)","Mode"]

tree_main = ttk.Treeview(
    main_table_frame, columns=cols, show="headings",
    yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set
)

scroll_y.config(command=tree_main.yview)
scroll_x.config(command=tree_main.xview)

scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
tree_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

for c,t in zip(cols,titles):
    tree_main.heading(c, text=t)
tree_main.column("name", width=220)
tree_main.column("id", width=70)
tree_main.column("lat", width=115)
tree_main.column("lon", width=115)
tree_main.column("elev", width=70)
tree_main.column("az", width=70)
tree_main.column("freq", width=100)
tree_main.column("mode", width=80)

items_main = {}














# --- End part2.py ---

# --- Begin part3.py ---
# === part3.py ===
# Window 2: ≥30° with scrollbars (canvas + table)

win2 = tk.Toplevel(root)
win2.title(f"Real-Time LEO Satellites Over Pune (Lat {LAT}, Lon {LON})")
win2.geometry("1200x700")

win2_header = tk.Label(win2, text="Visible satellites: 0 | Updated: --:--:--", font=("Arial", 14, "bold"))
win2_header.pack(pady=6)

frame2 = tk.Frame(win2)
frame2.pack(fill=tk.BOTH, expand=True)

left2 = tk.Frame(frame2)
left2.pack(side=tk.LEFT, padx=8, pady=6)

right2 = tk.Frame(frame2)
right2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

# Canvas 2 with scrollbars
canvas2_frame = tk.Frame(left2)
canvas2_frame.pack()

scroll_y2 = tk.Scrollbar(canvas2_frame, orient="vertical")
scroll_x2 = tk.Scrollbar(canvas2_frame, orient="horizontal")

canvas2 = tk.Canvas(
    canvas2_frame, width=CAN2_SIZE, height=CAN2_SIZE, bg="white", bd=2, relief=tk.SUNKEN,
    yscrollcommand=scroll_y2.set, xscrollcommand=scroll_x2.set
)
scroll_y2.config(command=canvas2.yview)
scroll_x2.config(command=canvas2.xview)
scroll_y2.pack(side=tk.RIGHT, fill=tk.Y)
scroll_x2.pack(side=tk.BOTTOM, fill=tk.X)
canvas2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
canvas2.bind("<Configure>", lambda e: canvas2.configure(scrollregion=canvas2.bbox("all")))

c2x = CAN2_SIZE//2
c2y = CAN2_SIZE//2

# rings for window2
r2_outer = CAN2_MAX_R
r2_30 = CAN2_MAX_R * (90 - THRESH_30) / (90 - ELEV_LIMIT)
r2_60 = CAN2_MAX_R * (90 - THRESH_60) / (90 - ELEV_LIMIT)

canvas2.create_rectangle(2,2,CAN2_SIZE-2,CAN2_SIZE-2, outline="#444")
canvas2.create_oval(c2x-r2_outer, c2y-r2_outer, c2x+r2_outer, c2y+r2_outer, outline="#bbb", dash=(4,6))
canvas2.create_oval(c2x-r2_30, c2y-r2_30, c2x+r2_30, c2y+r2_30, outline="#999", dash=(4,6))
canvas2.create_oval(c2x-r2_60, c2y-r2_60, c2x+r2_60, c2y+r2_60, outline="#666", dash=(3,5))
canvas2.create_oval(c2x-5,c2y-5,c2x+5,c2y+5, fill="blue")
canvas2.create_text(c2x, c2y+14, text="You (Pune)", font=("Arial",9))

# Table 2 with scrollbars
tree2_frame = tk.Frame(right2)
tree2_frame.pack(fill=tk.BOTH, expand=True)

scroll_y2_t = ttk.Scrollbar(tree2_frame, orient=tk.VERTICAL)
scroll_x2_t = ttk.Scrollbar(tree2_frame, orient=tk.HORIZONTAL)

tree2 = ttk.Treeview(
    tree2_frame, columns=cols, show="headings",
    yscrollcommand=scroll_y2_t.set, xscrollcommand=scroll_x2_t.set
)

scroll_y2_t.config(command=tree2.yview)
scroll_x2_t.config(command=tree2.xview)
scroll_y2_t.pack(side=tk.RIGHT, fill=tk.Y)
scroll_x2_t.pack(side=tk.BOTTOM, fill=tk.X)
tree2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

for c,t in zip(cols,titles):
    tree2.heading(c, text=t)
tree2.column("name", width=220); tree2.column("id", width=70)
tree2.column("lat", width=115); tree2.column("lon", width=115)
tree2.column("elev", width=70); tree2.column("az", width=70)
tree2.column("freq", width=100); tree2.column("mode", width=80)

items_win2 = {}














# --- End part3.py ---

# --- Begin part4.py ---
# === part4.py ===
# Window 3: ≥60° with scrollbars (canvas + table)

win3 = tk.Toplevel(root)
win3.title(f"Real-Time LEO Satellites Over Pune (Lat {LAT}, Lon {LON})")
win3.geometry("1000x600")

win3_header = tk.Label(win3, text="Visible satellites: 0 | Updated: --:--:--", font=("Arial", 14, "bold"))
win3_header.pack(pady=6)

frame3 = tk.Frame(win3)
frame3.pack(fill=tk.BOTH, expand=True)

left3 = tk.Frame(frame3)
left3.pack(side=tk.LEFT, padx=8, pady=6)

right3 = tk.Frame(frame3)
right3.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

# Canvas 3 with scrollbars
canvas3_frame = tk.Frame(left3)
canvas3_frame.pack()

scroll_y3 = tk.Scrollbar(canvas3_frame, orient="vertical")
scroll_x3 = tk.Scrollbar(canvas3_frame, orient="horizontal")

canvas3 = tk.Canvas(
    canvas3_frame, width=CAN3_SIZE, height=CAN3_SIZE, bg="white", bd=2, relief=tk.SUNKEN,
    yscrollcommand=scroll_y3.set, xscrollcommand=scroll_x3.set
)
scroll_y3.config(command=canvas3.yview)
scroll_x3.config(command=canvas3.xview)
scroll_y3.pack(side=tk.RIGHT, fill=tk.Y)
scroll_x3.pack(side=tk.BOTTOM, fill=tk.X)
canvas3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
canvas3.bind("<Configure>", lambda e: canvas3.configure(scrollregion=canvas3.bbox("all")))

c3x = CAN3_SIZE//2
c3y = CAN3_SIZE//2

# rings for window3 (we draw outer, 30°, 60°)
r3_outer = CAN3_MAX_R
r3_30 = CAN3_MAX_R * (90 - THRESH_30) / (90 - ELEV_LIMIT)
r3_60 = CAN3_MAX_R * (90 - THRESH_60) / (90 - ELEV_LIMIT)

canvas3.create_rectangle(2,2,CAN3_SIZE-2,CAN3_SIZE-2, outline="#444")
canvas3.create_oval(c3x-r3_outer, c3y-r3_outer, c3x+r3_outer, c3y+r3_outer, outline="#bbb", dash=(4,6))
canvas3.create_oval(c3x-r3_30, c3y-r3_30, c3x+r3_30, c3y+r3_30, outline="#999", dash=(4,6))
canvas3.create_oval(c3x-r3_60, c3y-r3_60, c3x+r3_60, c3y+r3_60, outline="#666", dash=(3,5))
canvas3.create_oval(c3x-5,c3y-5,c3x+5,c3y+5, fill="blue")
canvas3.create_text(c3x, c3y+14, text="You (Pune)", font=("Arial",9))

# Table 3 with scrollbars
tree3_frame = tk.Frame(right3)
tree3_frame.pack(fill=tk.BOTH, expand=True)

scroll_y3_t = ttk.Scrollbar(tree3_frame, orient=tk.VERTICAL)
scroll_x3_t = ttk.Scrollbar(tree3_frame, orient=tk.HORIZONTAL)

tree3 = ttk.Treeview(
    tree3_frame, columns=cols, show="headings",
    yscrollcommand=scroll_y3_t.set, xscrollcommand=scroll_x3_t.set
)
scroll_y3_t.config(command=tree3.yview)
scroll_x3_t.config(command=tree3.xview)
scroll_y3_t.pack(side=tk.RIGHT, fill=tk.Y)
scroll_x3_t.pack(side=tk.BOTTOM, fill=tk.X)
tree3.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

for c,t in zip(cols,titles):
    tree3.heading(c, text=t)
tree3.column("name", width=220); tree3.column("id", width=70)
tree3.column("lat", width=115); tree3.column("lon", width=115)
tree3.column("elev", width=70); tree3.column("az", width=70)
tree3.column("freq", width=100); tree3.column("mode", width=80)

items_win3 = {}
















# --- End part4.py ---

# --- Begin part5.py ---
# === part5.py (FINAL UPDATED) ===
# Window 4: IMU fusion + doppler prediction, track visualization,
# scrollable canvas + scrollable table + Zoom + LOS + Advanced Waterfall (Pillow)
# + Selected satellites shown on canvas + Predicted Lat/Lon boxes (bottom-right)

from PIL import Image, ImageDraw, ImageTk, ImageFilter  # requires pillow

# Fusion gain
K_FUSION = 0.40

win4 = tk.Toplevel(root)
win4.title("LEO-PNT Simulation & Estimator (IMU Fusion K=0.40) — Zoom+LOS+Waterfall")
win4.geometry("1400x760")  # widen to accommodate waterfall

w4_header = tk.Label(
    win4,
    text="LEO-PNT Navigation (IMU fusion) | Updated: --:--:--",
    font=("Arial", 13, "bold")
)
w4_header.pack(pady=6)


# --- ADDED: IMU Status Label & Startup Mode ---
imu_status_label = tk.Label(
    win4,
    text="Waiting for IMU...",
    font=("Arial", 11),
    fg="darkblue"
)
imu_status_label.pack(pady=3)

STARTUP_DONE = False     # ensures update_window4 will not run immediately
REQUEST_SIMULATION = False


# ----- CASE SWITCH CONTROLS -----
case_frame = tk.Frame(win4)
case_frame.pack(pady=4)

btn_case1 = tk.Button(case_frame, text="Case 1: Simulation", width=18,
                      command=lambda: set_navigation_mode("SIM"))
btn_case1.pack(side=tk.LEFT, padx=6)

btn_case2 = tk.Button(case_frame, text="Case 2: IMU Sensor", width=18,
                      command=lambda: set_navigation_mode("IMU"))
btn_case2.pack(side=tk.LEFT, padx=6)

# Display which case is active
active_case_label = tk.Label(case_frame, text="Active Mode: Simulation", font=("Arial", 10, "bold"))
active_case_label.pack(side=tk.LEFT, padx=12)


frame4 = tk.Frame(win4)
frame4.pack(fill=tk.BOTH, expand=True)

left4 = tk.Frame(frame4)
left4.pack(side=tk.LEFT, padx=8, pady=6, fill=tk.BOTH, expand=False)

right4 = tk.Frame(frame4)
right4.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=6)

# =======================
# LEFT: Canvas 4 + scrollbars + zoom controls
# =======================
canvas4_frame = tk.Frame(left4)
canvas4_frame.pack()

scroll_y4c = tk.Scrollbar(canvas4_frame, orient="vertical")
scroll_x4c = tk.Scrollbar(canvas4_frame, orient="horizontal")

CAN4_W = 700
CAN4_H = 700

canvas4 = tk.Canvas(
    canvas4_frame,
    width=CAN4_W,
    height=CAN4_H,
    bg="white",
    bd=2,
    relief=tk.SUNKEN,
    yscrollcommand=scroll_y4c.set,
    xscrollcommand=scroll_x4c.set
)

scroll_y4c.config(command=canvas4.yview)
scroll_x4c.config(command=canvas4.xview)

scroll_y4c.pack(side=tk.RIGHT, fill=tk.Y)
scroll_x4c.pack(side=tk.BOTTOM, fill=tk.X)
canvas4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Update scrollregion on resize
canvas4.bind("<Configure>", lambda e: canvas4.configure(scrollregion=canvas4.bbox("all")))

# Zoom controls panel
zoom_frame = tk.Frame(left4)
zoom_frame.pack(fill=tk.X, pady=6)

zoom_label = tk.Label(zoom_frame, text="Zoom:")
zoom_label.pack(side=tk.LEFT, padx=(4,8))

btn_zoom_in = tk.Button(zoom_frame, text="+", width=3)
btn_zoom_in.pack(side=tk.LEFT)
btn_zoom_out = tk.Button(zoom_frame, text="-", width=3)
btn_zoom_out.pack(side=tk.LEFT, padx=(6,4))
btn_reset_zoom = tk.Button(zoom_frame, text="Reset", width=6)
btn_reset_zoom.pack(side=tk.LEFT, padx=(6,4))

# canvas top-left info text
canvas4.create_text(
    10, 10, anchor="nw",
    text="Blue = True Track\nRed = Predicted (IMU+LEO)\nGreen Dot = Last Live GPS\nDashed lines = LOS to selected sats",
    font=("Arial", 9)
)

c4cx = CAN4_W // 2
c4cy = CAN4_H // 2

# Dynamic zoom scale (smaller => larger visible area)
zoom_scale = 0.00010  # initial; used as global mapping from meters -> degrees->canvas

def set_zoom(scale):
    global zoom_scale
    zoom_scale = max(zoom_scale, 1e-7)
    zoom_scale = min(zoom_scale, 0.01)

def zoom_in(evt=None):
    global zoom_scale
    zoom_scale *= 0.85
    canvas4.configure(scrollregion=canvas4.bbox("all"))

def zoom_out(evt=None):
    global zoom_scale
    zoom_scale /= 0.85
    canvas4.configure(scrollregion=canvas4.bbox("all"))

def reset_zoom(evt=None):
    global zoom_scale
    zoom_scale = 0.00010
    canvas4.configure(scrollregion=canvas4.bbox("all"))

btn_zoom_in.config(command=zoom_in)
btn_zoom_out.config(command=zoom_out)
btn_reset_zoom.config(command=reset_zoom)

# Mouse-wheel zoom bindings (Windows/Mac)
def on_mousewheel_zoom(event):
    delta = getattr(event, 'delta', 0)
    if event.state & 0x4:  # Ctrl pressed on Windows
        factor = 0.85 if delta > 0 else 1/0.85
    else:
        factor = 0.92 if delta > 0 else 1/0.92
    global zoom_scale
    zoom_scale *= factor
    canvas4.configure(scrollregion=canvas4.bbox("all"))

canvas4.bind_all("<MouseWheel>", on_mousewheel_zoom)     # Windows & Mac
canvas4.bind_all("<Button-4>", lambda e: zoom_in(e))     # Linux scroll up
canvas4.bind_all("<Button-5>", lambda e: zoom_out(e))    # Linux scroll down

# =======================
# RIGHT: Waterfall + Table  (waterfall above table)
# =======================
wf_frame = tk.Frame(right4)
wf_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

WF_WIDTH = 400
WF_HEIGHT = 300
WF_FREQ_SPAN_HZ = 12000.0  # ±6 kHz shown
WF_BINS = 256

waterfall_canvas = tk.Canvas(wf_frame, width=WF_WIDTH, height=WF_HEIGHT, bg="black", bd=2, relief=tk.SUNKEN)
waterfall_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

# Prepare Pillow image buffer and PhotoImage
wf_image = Image.new("RGB", (WF_WIDTH, WF_HEIGHT), "black")
wf_draw = ImageDraw.Draw(wf_image)
wf_photo = ImageTk.PhotoImage(wf_image)
wf_img_id = waterfall_canvas.create_image(0, 0, anchor="nw", image=wf_photo)

# Waterfall intensity buffer (list of rows); newest appended at top then shifted
waterfall_buf = []  # each row: list of WF_BINS float intensities 0..1

def gen_waterfall_row_from_dopplers(doppler_list_hz):
    row = np.zeros(WF_BINS, dtype=float)
    for df in doppler_list_hz:
        if df < -WF_FREQ_SPAN_HZ/2 or df > WF_FREQ_SPAN_HZ/2:
            continue
        bin_pos = (df + WF_FREQ_SPAN_HZ/2) / WF_FREQ_SPAN_HZ * (WF_BINS-1)
        widths = [1.5 + abs(df)/3000.0]
        for w in widths:
            xs = np.arange(WF_BINS)
            gauss = np.exp(-0.5 * ((xs - bin_pos) / (3.0*w))**2)
            row += gauss
    row += np.random.rand(WF_BINS) * 0.06
    row = row / (row.max() + 1e-9)
    row = np.sqrt(row)
    return row.clip(0.0, 1.0)

def draw_waterfall():
    global wf_image, wf_draw, wf_photo, wf_img_id
    h = WF_HEIGHT
    w = WF_WIDTH
    rows = len(waterfall_buf)
    wf_draw.rectangle((0,0,w,h), fill="black")
    if rows > 0:
        max_rows = min(rows, h)
        start = max(0, rows - max_rows)
        for i in range(max_rows):
            row = waterfall_buf[start + i]
            for px in range(w):
                binf = px * (WF_BINS - 1) / (w - 1)
                b0 = int(binf)
                b1 = min(b0 + 1, WF_BINS-1)
                frac = binf - b0
                val = (1-frac)*row[b0] + frac*row[b1]
                if val <= 0:
                    c = (0,0,0)
                elif val < 0.25:
                    t = val/0.25
                    c = (0, int(80*t), int(200*t)+30)
                elif val < 0.5:
                    t = (val-0.25)/0.25
                    c = (int(0 + 255*t), int(80 + 100*t), 255)
                elif val < 0.75:
                    t = (val-0.5)/0.25
                    c = (255, int(180 + 50*t), int(255 - 200*t))
                else:
                    t = (val-0.75)/0.25
                    c = (255, int(230 + 25*t), int(55 + 200*t))
                y = h - 1 - i
                wf_image.putpixel((px, y), c)
    wf_display = wf_image.filter(ImageFilter.GaussianBlur(radius=0.3))
    wf_photo = ImageTk.PhotoImage(wf_display)
    waterfall_canvas.itemconfig(wf_img_id, image=wf_photo)
    waterfall_canvas.image = wf_photo

# =======================
# TABLE 4 WITH SCROLLBARS
# =======================
cols4 = ("time","true_lat","true_lon","pred_lat","pred_lon","err_m","dist_m","dir","sats")
titles4 = ["Time","True Lat","True Lon","Pred Lat","Pred Lon","Error (m)","Dist (m)","Dir","Sats Used"]

tree4_frame = tk.Frame(right4)
tree4_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

scroll_y4t = ttk.Scrollbar(tree4_frame, orient=tk.VERTICAL)
scroll_x4t = ttk.Scrollbar(tree4_frame, orient=tk.HORIZONTAL)

tree4 = ttk.Treeview(
    tree4_frame, columns=cols4, show="headings",
    yscrollcommand=scroll_y4t.set, xscrollcommand=scroll_x4t.set, height=14
)
scroll_y4t.config(command=tree4.yview)
scroll_x4t.config(command=tree4.xview)
scroll_y4t.pack(side=tk.RIGHT, fill=tk.Y)
scroll_x4t.pack(side=tk.BOTTOM, fill=tk.X)
tree4.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

for c, t in zip(cols4, titles4):
    tree4.heading(c, text=t)
    tree4.column(c, width=120, minwidth=100)

# =======================
# SIMULATION STATE
# =======================
sim_state = {
    "t": ts.now(),
    "east": 0.0,
    "north": 0.0,
    "ve": 18.0,
    "vn": 20.0,
    "alt": 560.0
}

sim_heading = atan2(sim_state["vn"], sim_state["ve"])

# ------------------------------
# MPU6050 IMU (CASE-2) integration
# ------------------------------
# Place this block right AFTER sim_state definition and BEFORE existing step_motion() in your file.

USE_IMU = False   # set True to use real MPU6050 (CASE-2). Keep False to use current simulated CASE-1.

# I2C / MPU6050 configuration - change bus/address if required
try:
    from smbus2 import SMBus
    I2C_AVAILABLE = True
except Exception:
    I2C_AVAILABLE = False

MPU_I2C_ADDR = 0x68
MPU_BUS_NUM = 1  # Raspberry Pi default I2C bus. Change to 0 on older Pi or if required.

# MPU6050 register map (minimal)
MPU_PWR_MGMT_1 = 0x6B
MPU_ACCEL_XOUT_H = 0x3B
MPU_GYRO_XOUT_H  = 0x43
MPU_SMPLRT_DIV   = 0x19
MPU_CONFIG       = 0x1A
MPU_GYRO_CONFIG  = 0x1B
MPU_ACCEL_CONFIG = 0x1C

# scale factors (default full scale: accel±2g, gyro±250 °/s)
ACCEL_SCALE = 16384.0   # LSB/g
GYRO_SCALE  = 131.0     # LSB/(°/s)

# complementary filter gains
CF_ALPHA = 0.98   # gyro contribution (high), accel contribution = 1 - CF_ALPHA

# internal state for IMU fusion
_imu_state = {
    "inited": False,
    "bus": None,
    "bias_accel": np.zeros(3),
    "bias_gyro": np.zeros(3),
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "last_t": time.time(),
    "calibrated": False
}

def init_mpu(calibrate=True, cal_samples=200):
    """Initialize MPU6050 and optionally calibrate biases."""
    if not I2C_AVAILABLE:
        print("smbus2 not available: MPU disabled.")
        return False
    try:
        bus = SMBus(MPU_BUS_NUM)
        # Wake up MPU
        bus.write_byte_data(MPU_I2C_ADDR, MPU_PWR_MGMT_1, 0x00)
        # Set sample rate divider (optional)
        bus.write_byte_data(MPU_I2C_ADDR, MPU_SMPLRT_DIV, 0x07)
        # Configure DLPF (set moderate bandwidth)
        bus.write_byte_data(MPU_I2C_ADDR, MPU_CONFIG, 0x03)
        # Set gyro full-scale ±250 deg/s
        bus.write_byte_data(MPU_I2C_ADDR, MPU_GYRO_CONFIG, 0x00)
        # Set accel full-scale ±2g
        bus.write_byte_data(MPU_I2C_ADDR, MPU_ACCEL_CONFIG, 0x00)
        _imu_state["bus"] = bus
        _imu_state["inited"] = True
        print("MPU6050 initialized on I2C bus", MPU_BUS_NUM)
        if calibrate:
            calibrate_mpu(cal_samples)
        return True
    except Exception as e:
        print("MPU init failed:", e)
        _imu_state["inited"] = False
        return False

def read_mpu_raw():
    """Read raw accel and gyro (signed 16-bit) from MPU registers."""
    if not _imu_state["inited"]:
        raise RuntimeError("MPU not initialized")
    bus = _imu_state["bus"]
    def read_word(reg):
        high = bus.read_byte_data(MPU_I2C_ADDR, reg)
        low  = bus.read_byte_data(MPU_I2C_ADDR, reg+1)
        val = (high << 8) + low
        if val >= 0x8000:
            val = -((65535) - val + 1)
        return val
    ax = read_word(MPU_ACCEL_XOUT_H)
    ay = read_word(MPU_ACCEL_XOUT_H + 2)
    az = read_word(MPU_ACCEL_XOUT_H + 4)
    gx = read_word(MPU_GYRO_XOUT_H)
    gy = read_word(MPU_GYRO_XOUT_H + 2)
    gz = read_word(MPU_GYRO_XOUT_H + 4)
    return np.array([ax, ay, az], dtype=float), np.array([gx, gy, gz], dtype=float)

def calibrate_mpu(n=200):
    """Compute average biases (stationary) for accel and gyro."""
    if not _imu_state["inited"]:
        return
    print("Calibrating MPU (keep device stationary)...")
    a_sum = np.zeros(3)
    g_sum = np.zeros(3)
    valid = 0
    for i in range(n):
        try:
            a_raw, g_raw = read_mpu_raw()
        except Exception:
            break
        a_sum += a_raw
        g_sum += g_raw
        valid += 1
        time.sleep(0.01)
    if valid == 0:
        return
    a_avg = a_sum / valid
    g_avg = g_sum / valid
    # Convert to physical units: g and deg/s
    _imu_state["bias_accel"] = a_avg / ACCEL_SCALE - np.array([0.0, 0.0, 1.0])  # remove gravity on Z
    _imu_state["bias_gyro"] = g_avg / GYRO_SCALE
    _imu_state["calibrated"] = True
    print("MPU calibration done. accel_bias:", _imu_state["bias_accel"], "gyro_bias(deg/s):", _imu_state["bias_gyro"])

def process_mpu(dt_s):
    """Read MPU, apply calibration, complementary filter -> returns body accel (m/s^2) in ENU and velocities update."""
    if not _imu_state["inited"]:
        raise RuntimeError("MPU not initialized")
    a_raw, g_raw = read_mpu_raw()
    # convert to physical units
    a = a_raw / ACCEL_SCALE   # g
    a = a * 9.80665           # m/s^2
    g = g_raw / GYRO_SCALE    # deg/s
    g = np.deg2rad(g)         # rad/s

    # remove biases (gyro in deg/s bias converted earlier)
    if _imu_state["calibrated"]:
        a -= _imu_state["bias_accel"] * 9.80665
        g -= np.deg2rad(_imu_state["bias_gyro"])

    # simple complementary filter for roll/pitch using accel + gyro
    # compute accel-based roll/pitch
    ax, ay, az = a
    # protect division
    try:
        roll_acc = np.arctan2(ay, az)
    except Exception:
        roll_acc = _imu_state["roll"]
    try:
        pitch_acc = np.arctan2(-ax, np.sqrt(ay*ay + az*az))
    except Exception:
        pitch_acc = _imu_state["pitch"]

    # integrate gyro to update angles
    gx, gy, gz = g  # rad/s in body axes
    # Note: body-to-earth axis sign conventions may need tuning for your mounting orientation
    roll_gyro = _imu_state["roll"] + gx * dt_s
    pitch_gyro = _imu_state["pitch"] + gy * dt_s
    yaw_gyro = _imu_state["yaw"] + gz * dt_s

    # complementary blend
    roll = CF_ALPHA * roll_gyro + (1.0 - CF_ALPHA) * roll_acc
    pitch = CF_ALPHA * pitch_gyro + (1.0 - CF_ALPHA) * pitch_acc
    yaw = yaw_gyro  # yaw from gyro only (drift) - recommend magnetometer for long runs

    _imu_state["roll"] = roll
    _imu_state["pitch"] = pitch
    _imu_state["yaw"] = yaw

    # rotation matrix from body -> NED then to ENU mapping
    # build DCM from roll,pitch,yaw (ZYX or appropriate) - we use standard aerospace sequence
    cr = np.cos(roll); sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw); sy = np.sin(yaw)

    # body to earth (NED) rotation matrix (one common convention)
    R_b2n = np.array([
        [cp*cy, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [cp*sy, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,             cp*cr]
    ])
    # convert to ENU from NED: swap axes
    # NED -> ENU mapping: [E, N, U] = [  N,  E, -D ] depending on conventions.
    # To keep things simple, we will compute acceleration in NED then map to ENU
    acc_ned = R_b2n.dot(np.array([ax, ay, az]))
    # remove gravity in NED frame (gravity ~ +9.80665 m/s^2 down in NED => add +9.80665 on D)
    # In our acc_ned the third component is down (D), so subtract gravity:
    acc_ned[2] += 9.80665  # remove gravity effect (since accel measured includes gravity)
    # Map NED -> ENU: east = north_ned, north = east_ned
    acc_enu = np.array([acc_ned[1], acc_ned[0], -acc_ned[2]])

    return acc_enu, np.rad2deg(yaw)  # return ENU accel (m/s^2) and yaw degrees

def step_motion_mpu(dt_s=1.0):
    """
    Replace simulated motion with IMU-driven motion.
    This function mirrors the original step_motion signature and updates sim_state dict.
    """
    global sim_state, sim_heading
    # Initialize MPU if not yet
    if USE_IMU and not _imu_state["inited"]:
        ok = init_mpu(calibrate=True)
        if not ok:
            # fallback to simulation if MPU init fails
            print("Using simulated motion (MPU init failed).")
            return step_motion(dt_s)

    if not USE_IMU:
        # If flag off, keep original behaviour
        return step_motion(dt_s)

    # Read and process IMU
    try:
        acc_enu, yaw_deg = process_mpu(dt_s)
    except Exception as e:
        # on any error, fallback gracefully to simulation
        print("MPU read error:", e)
        return step_motion(dt_s)

    # integrate acceleration -> velocities -> position (east/north)
    # assume sim_state stores velocities and positions in meters/meters-per-sec

    # -------------------------------
    # OPTION B — use IMU CSV PDR data
    # -------------------------------
    dE, dN, ok = read_latest_imu_motion()

    if ok:
        # IMU PDR motion detected → use it
        sim_state["east"]  += dE
        sim_state["north"] += dN
    else:
        # IMU is not producing step updates → fallback to simulation
        return step_motion(dt_s)


    # sim_state["ve"] += acc_enu[0] * dt_s   # east velocity
    # sim_state["vn"] += acc_enu[1] * dt_s   # north velocity
    # sim_state["east"] += sim_state["ve"] * dt_s
    # sim_state["north"] += sim_state["vn"] * dt_s

    # update sim heading from yaw (optional)
    sim_heading = np.deg2rad(yaw_deg)

    # update timestamp
    sim_state["t"] = ts.now()
    return sim_state


def step_motion(dt_s=1.0):
    global sim_heading
    yaw_change = random.uniform(-0.15, 0.15)
    sim_heading += yaw_change * dt_s
    base_speed = np.hypot(sim_state["ve"], sim_state["vn"])
    speed = max(18, base_speed + random.uniform(-4, 4))
    sim_state["ve"] = speed * np.cos(sim_heading)
    sim_state["vn"] = speed * np.sin(sim_heading)
    sim_state["east"] += sim_state["ve"] * dt_s
    sim_state["north"] += sim_state["vn"] * dt_s
    sim_state["t"] = ts.now()
    return sim_state

# IMU predicted state (dead reckoning)
imu_pred = {
    "east": 0.0,
    "north": 0.0,
    "ve": sim_state["ve"],
    "vn": sim_state["vn"]
}

true_track = []
pred_track = []
last_visible = []
last_gps_lat = LAT
last_gps_lon = LON
start_time = time.time()

# Helper: projection for LOS markers on canvas edge
def azimuth_to_canvas_edge(az_deg, radius):
    a = radians(az_deg)
    x = c4cx + radius * sin(a)
    y = c4cy - radius * cos(a)
    return x, y


# Navigation mode: "SIM" or "IMU"
NAV_MODE = "SIM"

def set_navigation_mode(mode):
    global NAV_MODE, USE_IMU
    if mode == "SIM":
        NAV_MODE = "SIM"
        USE_IMU = False
        STARTUP_DONE = True   # <-- START SIMULATION NOW
        imu_status_label.config(text="Simulation Mode Activated")
        active_case_label.config(text="Active Mode: Simulation")
        print("Switched to CASE 1: Simulation Mode")

    elif mode == "IMU":
        NAV_MODE = "IMU"
        USE_IMU = True
        active_case_label.config(text="Active Mode: IMU Sensor")
        print("Switched to CASE 2: IMU Motion Mode")


# --- ADDED: IMU Detection + Popup + Startup Logic ---
# --- ADDED GLOBALS ---
IMU_WAIT_START = time.time()
POPUP_SHOWN = False

def try_initialize_imu():
    global USE_IMU, NAV_MODE, STARTUP_DONE, POPUP_SHOWN, IMU_WAIT_START

    # If IMU already initialized, start immediately
    if _imu_state.get("inited", False):
        USE_IMU = True
        NAV_MODE = "IMU"
        imu_status_label.config(text="IMU detected — Starting IMU Mode")
        active_case_label.config(text="Active Mode: IMU")
        STARTUP_DONE = True
        root.after(POLL_MS, update_window4)
        return

    # Show waiting text (NO POPUPS YET)
    imu_status_label.config(text="Waiting for IMU...")

    # Check elapsed time
    elapsed = time.time() - IMU_WAIT_START

    # Try initializing IMU *only once*
    if elapsed < 2:  # first 2 seconds: one attempt only
        ok = False
        try:
            ok = init_mpu(calibrate=True)
        except:
            ok = False

        if ok:
            USE_IMU = True
            NAV_MODE = "IMU"
            imu_status_label.config(text="IMU detected — Starting IMU Mode")
            active_case_label.config(text="Active Mode: IMU")
            STARTUP_DONE = True
            root.after(POLL_MS, update_window4)
            return

    # If less than 60 seconds → continue waiting (NO POPUP)
    if elapsed < 20:
        root.after(1000, try_initialize_imu)  # check again after 1 second
        return

    # After 60 seconds → show popup ONLY ONCE
    if not POPUP_SHOWN:
        POPUP_SHOWN = True
        ask_simulation_permission()  # You already have this function


def ask_simulation_permission():
    global REQUEST_SIMULATION, STARTUP_DONE
    res = messagebox.askokcancel(
        "IMU Not Found",
        "IMU was not detected.\nShall I start Simulation Mode?"
    )
    if res:
        REQUEST_SIMULATION = True
        enable_simulation()
    else:
        imu_status_label.config(text="Idle — Waiting for user input")
        STARTUP_DONE = False  # do nothing until user interacts

def enable_simulation():
    global NAV_MODE, USE_IMU, STARTUP_DONE
    NAV_MODE = "SIM"
    USE_IMU = False
    imu_status_label.config(text="Simulation Mode Enabled")
    active_case_label.config(text="Active Mode: Simulation")
    STARTUP_DONE = True
    root.after(POLL_MS, update_window4)


def safe_canvas4_delete(tag):
    try:
        canvas4.delete(tag)
    except Exception:
        return


# =======================
# MAIN UPDATE FUNCTION
# =======================
def update_window4():

        # BLOCK navigation until startup explicitly allowed
    if not STARTUP_DONE:
        # keep screen alive but do not move the blue dot
        w4_header.config(text=f"Waiting for start… ({time.strftime('%H:%M:%S')})")
        canvas4.configure(scrollregion=canvas4.bbox("all"))
        root.after(POLL_MS, update_window4)
        return


    global imu_pred, true_track, pred_track, last_visible, waterfall_buf, wf_image, wf_draw, wf_photo

    dt_s = POLL_MS / 1000.0
    if USE_IMU:
        st = step_motion_mpu(dt_s)
    else:
        # Choose CASE 1 (Simulation) or CASE 2 (IMU)
        if NAV_MODE == "IMU":
            st = step_motion_mpu(dt_s)
        else:
            st = step_motion(dt_s)


    true_lat = LAT + (st["north"] / R_E) * (180.0 / pi)
    true_lon = LON + (st["east"]  / (R_E * cos(radians(LAT)))) * (180.0 / pi)

    # choose best sats (up to 6)
    chosen = pick_best_six(last_visible)

    meas_dfs = []
    used_sats = []
    doppler_list_hz = []

    for ch in chosen:
        sat = ch["sat"]
        p_m, v_m = sat_state_ecef_m(sat, st["t"])
        rec_pos = latlon_to_ecef(true_lat, true_lon, st["alt"])

        lat0r = radians(true_lat)
        lon0r = radians(true_lon)
        rec_vel = np.array([
            -np.sin(lon0r)*st["ve"] - np.sin(lat0r)*np.cos(lon0r)*st["vn"],
             np.cos(lon0r)*st["ve"] - np.sin(lat0r)*np.sin(lon0r)*st["vn"],
             np.cos(lat0r)*st["vn"]
        ])

        sid = ch["id"]
        if sid in freq_map:
            f0 = freq_map[sid][0][0] * 1e6
        else:
            key = ch["name"].lower().replace(" ", "")
            f0 = name_map.get(key, [(435.0, "")])[0][0] * 1e6

        df = predicted_doppler_for_state(p_m, v_m, rec_pos, rec_vel, f0)
        doppler_list_hz.append(df)

        noise = random.gauss(0, 0.7)
        meas_dfs.append(df + noise)
        used_sats.append(ch)

    # IMU propagation
    imu_pred["east"]  += imu_pred["ve"] * dt_s
    imu_pred["north"] += imu_pred["vn"] * dt_s
    imu_pred["ve"] += random.gauss(0, 0.05)
    imu_pred["vn"] += random.gauss(0, 0.05)

    # Doppler solution
    if len(used_sats) >= 4:
        est_lat, est_lon, est_state = solve_receiver_from_dopplers_lm(
            used_sats, np.array(meas_dfs), st["t"], LAT, LON, st["alt"]
        )
        e_solver, n_solver, ve_solver, vn_solver = est_state
    else:
        e_solver = imu_pred["east"]
        n_solver = imu_pred["north"]
        ve_solver = imu_pred["ve"]
        vn_solver = imu_pred["vn"]
        est_lat = LAT + (n_solver / R_E) * (180.0 / pi)
        est_lon = LON + (e_solver / (R_E * cos(radians(LAT)))) * (180.0 / pi)

    # FUSION IMU + Doppler
    fused_e  = imu_pred["east"]  + K_FUSION*(e_solver  - imu_pred["east"])
    fused_n  = imu_pred["north"] + K_FUSION*(n_solver  - imu_pred["north"])
    fused_ve = imu_pred["ve"]    + K_FUSION*(ve_solver - imu_pred["ve"])
    fused_vn = imu_pred["vn"]    + K_FUSION*(vn_solver - imu_pred["vn"])

    imu_pred["east"]  = fused_e
    imu_pred["north"] = fused_n
    imu_pred["ve"]    = fused_ve
    imu_pred["vn"]    = fused_vn

    pred_lat = LAT + (fused_n / R_E) * (180.0 / pi)
    pred_lon = LON + (fused_e / (R_E * cos(radians(LAT)))) * (180.0 / pi)

    err = np.linalg.norm(latlon_to_ecef(pred_lat, pred_lon) - latlon_to_ecef(true_lat, true_lon))
    dist_from_last = np.hypot(imu_pred["east"], imu_pred["north"])
    travel_dir = direction_from_vector(imu_pred["east"], imu_pred["north"])

    # store tracks
    true_track.append((true_lat, true_lon))
    pred_track.append((pred_lat, pred_lon))
    if len(true_track) > 800:
        true_track.pop(0)
        pred_track.pop(0)

    # DRAW canvas: clear group 'trk'
    safe_canvas4_delete("trk")

    # -------------------------
    # DRAW SELECTED SAT LIST (Top-right corner)
    # -------------------------
    sat_list_text = "Selected Satellites:\n"
    for i, ch in enumerate(used_sats):
        # ensure name is available
        name_display = ch.get("name", "N/A")
        sat_list_text += f"{i+1}) {name_display} ({ch['id']})\n"

    # position top-right inside canvas (offset from right edge)
    txt_x = c4cx + (CAN4_W // 2) - 260
    txt_y = 12
    canvas4.create_rectangle(txt_x-6, txt_y-6, txt_x+250, txt_y+20 + 16*len(used_sats),
                             fill="#fffefe", outline="#ccc", tags="trk")
    canvas4.create_text(txt_x, txt_y, text=sat_list_text, anchor="nw", font=("Arial", 10, "bold"), fill="black", tags="trk")

    # draw last GPS marker
    gx = c4cx + (last_gps_lon - LON)/zoom_scale
    gy = c4cy - (last_gps_lat - LAT)/zoom_scale
    canvas4.create_oval(gx-6, gy-6, gx+6, gy+6, fill="green", tags="trk")
    canvas4.create_text(gx+10, gy, text="Last Live GPS", anchor="w", tags="trk")

    # true track (blue)
    for i in range(len(true_track)-1):
        x1 = c4cx + (true_track[i][1]-LON)/zoom_scale
        y1 = c4cy - (true_track[i][0]-LAT)/zoom_scale
        x2 = c4cx + (true_track[i+1][1]-LON)/zoom_scale
        y2 = c4cy - (true_track[i+1][0]-LAT)/zoom_scale
        canvas4.create_line(x1,y1,x2,y2,fill="blue",width=3,tags="trk")

    # predicted track (red)
    for i in range(len(pred_track)-1):
        x1 = c4cx + (pred_track[i][1]-LON)/zoom_scale
        y1 = c4cy - (pred_track[i][0]-LAT)/zoom_scale
        x2 = c4cx + (pred_track[i+1][1]-LON)/zoom_scale
        y2 = c4cy - (pred_track[i+1][0]-LAT)/zoom_scale
        canvas4.create_line(x1,y1,x2,y2,fill="red",width=2,tags="trk")

    # markers
    tx = c4cx + (true_lon - LON)/zoom_scale
    ty = c4cy - (true_lat - LAT)/zoom_scale
    canvas4.create_oval(tx-6, ty-6, tx+6, ty+6, fill="blue", tags="trk")
    canvas4.create_text(tx+8, ty, text="True", anchor="w", tags="trk", font=("Arial",8))

    px = c4cx + (pred_lon - LON)/zoom_scale
    py = c4cy - (pred_lat - LAT)/zoom_scale
    canvas4.create_oval(px-6, py-6, px+6, py+6, fill="red", tags="trk")
    canvas4.create_text(px+8, py, text="Predicted", anchor="w", tags="trk", font=("Arial",8))

    # Draw LOS lines for used_sats (dashed lines) and sat markers on edge
    radius_edge = min(CAN4_W, CAN4_H)//2 - 30
    for idx, ch in enumerate(used_sats):
        try:
            az = ch["az"]
        except Exception:
            try:
                diff = ch["sat"] - wgs84.latlon(true_lat, true_lon)
                _, az, _ = diff.at(st["t"]).altaz()
                az = az.degrees
            except Exception:
                az = 0.0
        color = color_for(ch["id"])
        sx, sy = azimuth_to_canvas_edge(az, radius_edge)
        canvas4.create_line(px, py, sx, sy, fill=color, dash=(4,6), width=1, tags="trk")
        canvas4.create_oval(sx-6, sy-6, sx+6, sy+6, outline=color, tags="trk")
        canvas4.create_text(sx+8, sy, text=str(ch["id"]), anchor="w", tags="trk", font=("Arial",8))

    # -------------------------
    # DRAW PREDICTED LAT/LON BOXES (Bottom-right)
    # -------------------------
    box_x1 = c4cx + (CAN4_W // 2) - 270
    box_y1 = c4cy + (CAN4_H // 2) - 160
    box_x2 = box_x1 + 260
    box_y2 = box_y1 + 70

    box2_y1 = box_y2 + 12
    box2_y2 = box2_y1 + 70

    # Rectangle 1 (Latitude)
    canvas4.create_rectangle(box_x1, box_y1, box_x2, box_y2,
                             outline="black", width=2, fill="#e6f2ff", tags="trk")
    canvas4.create_text(box_x1 + 12, box_y1 + 10,
                        text=f"Predicted Latitude:", anchor="nw",
                        font=("Arial", 11, "bold"), tags="trk")
    canvas4.create_text(box_x1 + 12, box_y1 + 36,
                        text=f"{pred_lat:.6f}°", anchor="nw",
                        font=("Arial", 12), fill="blue", tags="trk")

    # Rectangle 2 (Longitude)
    canvas4.create_rectangle(box_x1, box2_y1, box_x2, box2_y2,
                             outline="black", width=2, fill="#fff0e6", tags="trk")
    canvas4.create_text(box_x1 + 12, box2_y1 + 10,
                        text=f"Predicted Longitude:", anchor="nw",
                        font=("Arial", 11, "bold"), tags="trk")
    canvas4.create_text(box_x1 + 12, box2_y1 + 36,
                        text=f"{pred_lon:.6f}°", anchor="nw",
                        font=("Arial", 12), fill="red", tags="trk")

    # Update table
    used_ids_str = ", ".join(str(s["id"]) for s in used_sats)
    tree4.insert("", "end", values=(
        time.strftime("%H:%M:%S"),
        f"{true_lat:.6f}",
        f"{true_lon:.6f}",
        f"{pred_lat:.6f}",
        f"{pred_lon:.6f}",
        f"{err:.1f}",
        f"{dist_from_last:.1f}",
        travel_dir,
        used_ids_str
    ))
    if len(tree4.get_children()) > 200:
        tree4.delete(tree4.get_children()[0])

    # Update header
    w4_header.config(
        text=f"LEO-PNT IMU-Fusion | {time.strftime('%H:%M:%S')} | "
             f"Error: {err:.1f} m | Dist from GPS: {dist_from_last:.1f} m "
             f"| Dir: {travel_dir}"
    )

    # WATERFALL: generate row, push to buffer, render
    wf_row = gen_waterfall_row_from_dopplers(doppler_list_hz)
    waterfall_buf.append(wf_row)
    if len(waterfall_buf) > WF_HEIGHT:
        waterfall_buf = waterfall_buf[-WF_HEIGHT:]
    draw_waterfall()

    # ensure scrollregion updates
    canvas4.configure(scrollregion=canvas4.bbox("all"))

    # --- ADDED: IMU-first Startup ---
    #win4.after(1200, try_initialize_imu)


    root.after(POLL_MS, update_window4)

# End of part5.py (FINAL)
















# --- End part5.py ---

# --- Begin part6.py ---
# === part6.py ===
# Main update loop (windows 1-3) and start loops

import csv
import os
import time

last_imu_lat = None
last_imu_lon = None
last_imu_timestamp = 0

def read_latest_imu_motion():
    """
    Reads last line of pdr_steps.csv and returns (d_east_m, d_north_m, is_valid)
    If no IMU movement available → return (0,0,False)
    """
    global last_imu_lat, last_imu_lon, last_imu_timestamp

    csv_path = "/home/pi/pdr_steps.csv"   # <-- update if needed

    if not os.path.exists(csv_path):
        return 0, 0, False

    try:
        with open(csv_path, "r") as f:
            rows = list(csv.reader(f))
            if len(rows) < 2:
                return 0, 0, False
            last = rows[-1]

        imu_lat = float(last[0])
        imu_lon = float(last[1])

    except Exception as e:
        print("IMU CSV read error:", e)
        return 0, 0, False

    now = time.time()

    # FIRST CALL → initialize but no movement yet
    if last_imu_lat is None:
        last_imu_lat = imu_lat
        last_imu_lon = imu_lon
        last_imu_timestamp = now
        return 0, 0, False

    # If file not updated recently → IMU stalled
    if now - last_imu_timestamp > 2.0:
        return 0, 0, False

    # Compute motion in meters from lat/lon differences
    d_north = (imu_lat - last_imu_lat) * (R_E * pi / 180)
    d_east  = (imu_lon - last_imu_lon) * (R_E * cos(radians(imu_lat)) * pi / 180)

    # Update stored values
    last_imu_lat = imu_lat
    last_imu_lon = imu_lon
    last_imu_timestamp = now

    return d_east, d_north, True


items_main = {}
items_win2 = {}
items_win3 = {}
sat_colors = {}

def azel_to_main(az, elev):
    if elev > 90: elev = 90
    if elev < ELEV_LIMIT: elev = ELEV_LIMIT
    r = GRAPH_MAX_R * (90 - elev) / (90 - ELEV_LIMIT)
    a = radians(az)
    x = mx + r * sin(a)
    y = my - r * cos(a)
    return x, y

last_visible = []

def update_all():
    global last_visible
    t = ts.now()
    visible = []

    for sat in LEO_sats:
        diff = sat - loc
        alt, az, _ = diff.at(t).altaz()
        if alt.degrees > ELEV_LIMIT:
            sp = wgs84.subpoint(sat.at(t))
            lat = sp.latitude.degrees
            lon = sp.longitude.degrees
            sid = sat.model.satnum

            freq = "N/A"; mode = "N/A"
            if sid in freq_map:
                freq, mode = freq_map[sid][0]
            else:
                key = sat.name.lower().replace(" ", "")
                if key in name_map:
                    freq, mode = name_map[key][0]

            visible.append({
                "sat": sat,
                "name": sat.name,
                "id": sid,
                "lat": lat,
                "lon": lon,
                "elev": alt.degrees,
                "az": az.degrees,
                "freq": freq if freq == "N/A" else f"{freq:.4f}",
                "mode": mode
            })

    visible.sort(key=lambda x: -x["elev"])
    last_visible = visible

    main_header.config(text=f"Visible satellites: {len(visible)} | Updated: {time.strftime('%H:%M:%S')}")
    tree_main.delete(*tree_main.get_children())
    for r in visible:
        tree_main.insert("", "end", values=(
            r["name"], r["id"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
            f"{r['elev']:.1f}", f"{r['az']:.1f}", r["freq"], r["mode"]
        ))

    current_ids = {r["id"] for r in visible}
    for sid in list(items_main.keys()):
        if sid not in current_ids:
            for obj in items_main[sid]:
                canvas_main.delete(obj)
            del items_main[sid]

    for r in visible:
        sid = r["id"]
        x, y = azel_to_main(r["az"], r["elev"])
        color = color_for(sid)
        size = 8
        if sid in items_main:
            circ, txt = items_main[sid]
            canvas_main.coords(circ, x-size, y-size, x+size, y+size)
            canvas_main.coords(txt, x+size+4, y)
            canvas_main.itemconfigure(txt, text=f"{r['name']}\n{r['elev']:.0f}°")
        else:
            circ = canvas_main.create_oval(x-size, y-size, x+size, y+size, fill=color, outline="#222")
            txt = canvas_main.create_text(x+size+4, y, text=f"{r['name']}\n{r['elev']:.0f}°", anchor="w", font=("Arial",9))
            items_main[sid] = (circ, txt)

    canvas_main.delete("legend")
    Lx = 10; Ly = GRAPH_SIZE - 100
    canvas_main.create_rectangle(Lx-6, Ly-6, Lx+350, Ly+72, fill="white", outline="#ccc", tags="legend")
    canvas_main.create_text(Lx, Ly, anchor="nw", text="Legend:", font=("Arial",9,"bold"), tags="legend")
    canvas_main.create_text(Lx, Ly+18, anchor="nw", text=f"Inner dotted ring = Elev ≥ {THRESH_60}°", font=("Arial",8), tags="legend")
    canvas_main.create_text(Lx, Ly+34, anchor="nw", text=f"Outer ring = Elev ≥ {ELEV_LIMIT}° (approx horizon)", font=("Arial",8), tags="legend")
    canvas_main.create_text(Lx, Ly+50, anchor="nw", text="Marker shows name + elevation", font=("Arial",8), tags="legend")

    # WINDOW 2 updates
    high30 = [r for r in visible if r["elev"] >= THRESH_30]
    win2_header.config(text=f"Visible satellites: {len(high30)} | Updated: {time.strftime('%H:%M:%S')}")
    tree2.delete(*tree2.get_children())
    for r in high30:
        tree2.insert("", "end", values=(
            r["name"], r["id"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
            f"{r['elev']:.1f}", f"{r['az']:.1f}", r["freq"], r["mode"]
        ))

    ids2 = {r["id"] for r in high30}
    for sid in list(items_win2.keys()):
        if sid not in ids2:
            for obj in items_win2[sid]:
                canvas2.delete(obj)
            del items_win2[sid]

    for r in high30:
        sid = r["id"]
        elev = r["elev"]; az = r["az"]
        if elev > 90: elev = 90
        if elev < ELEV_LIMIT: elev = ELEV_LIMIT
        r_pix = r2_outer * (90 - elev) / (90 - ELEV_LIMIT)
        a = radians(az)
        x = c2x + r_pix * sin(a)
        y = c2y - r_pix * cos(a)
        col = color_for(sid); size = 8
        if sid in items_win2:
            circ, txt = items_win2[sid]
            canvas2.coords(circ, x-size, y-size, x+size, y+size)
            canvas2.coords(txt, x+size+5, y)
            canvas2.itemconfigure(txt, text=f"{r['name']}\n{r['elev']:.0f}°")
        else:
            circ = canvas2.create_oval(x-size, y-size, x+size, y+size, fill=col, outline="#222")
            txt = canvas2.create_text(x+size+5, y, text=f"{r['name']}\n{r['elev']:.0f}°", anchor="w", font=("Arial",9))
            items_win2[sid] = (circ, txt)

    canvas2.delete("legend")
    Lx2 = 12; Ly2 = CAN2_SIZE - 130
    canvas2.create_rectangle(Lx2-6, Ly2-6, Lx2+430, Ly2+98, fill="white", outline="#ccc", tags="legend")
    canvas2.create_text(Lx2, Ly2, anchor="nw", text="Legend:", font=("Arial",9,"bold"), tags="legend")
    canvas2.create_text(Lx2, Ly2+18, anchor="nw", text=f"Inner dotted ring = Elev ≥ {THRESH_60}° (High zone)", font=("Arial",8), tags="legend")
    canvas2.create_text(Lx2, Ly2+36, anchor="nw", text=f"Middle dotted ring = Elev ≥ {THRESH_30}° (Mid zone)", font=("Arial",8), tags="legend")
    canvas2.create_text(Lx2, Ly2+54, anchor="nw", text=f"Outer ring = Elev ≥ {ELEV_LIMIT}° (approx horizon)", font=("Arial",8), tags="legend")
    canvas2.create_text(Lx2, Ly2+72, anchor="nw", text="Marker = Satellite name + elevation", font=("Arial",8), tags="legend")

    # WINDOW 3 updates
    high60 = [r for r in visible if r["elev"] >= THRESH_60]
    win3_header.config(text=f"Visible satellites: {len(high60)} | Updated: {time.strftime('%H:%M:%S')}")
    tree3.delete(*tree3.get_children())
    for r in high60:
        tree3.insert("", "end", values=(
            r["name"], r["id"], f"{r['lat']:.6f}", f"{r['lon']:.6f}",
            f"{r['elev']:.1f}", f"{r['az']:.1f}", r["freq"], r["mode"]
        ))

    ids3 = {r["id"] for r in high60}
    for sid in list(items_win3.keys()):
        if sid not in ids3:
            for obj in items_win3[sid]:
                canvas3.delete(obj)
            del items_win3[sid]

    for r in high60:
        sid = r["id"]
        elev = r["elev"]; az = r["az"]
        if elev > 90: elev = 90
        if elev < ELEV_LIMIT: elev = ELEV_LIMIT
        r_pix = r3_outer * (90 - elev) / (90 - ELEV_LIMIT)
        a = radians(az)
        x = c3x + r_pix * sin(a)
        y = c3y - r_pix * cos(a)
        col = color_for(sid); size = 8
        if sid in items_win3:
            circ, txt = items_win3[sid]
            canvas3.coords(circ, x-size, y-size, x+size, y+size)
            canvas3.coords(txt, x+size+5, y)
            canvas3.itemconfigure(txt, text=f"{r['name']}\n{r['elev']:.0f}°")
        else:
            circ = canvas3.create_oval(x-size, y-size, x+size, y+size, fill=col, outline="#222")
            txt = canvas3.create_text(x+size+5, y, text=f"{r['name']}\n{r['elev']:.0f}°", anchor="w", font=("Arial",9))
            items_win3[sid] = (circ, txt)

    canvas3.delete("legend")
    Lx3 = 10; Ly3 = CAN3_SIZE - 120
    canvas3.create_rectangle(Lx3-6, Ly3-6, Lx3+380, Ly3+92, fill="white", outline="#ccc", tags="legend")
    canvas3.create_text(Lx3, Ly3, anchor="nw", text="Legend:", font=("Arial",9,"bold"), tags="legend")
    canvas3.create_text(Lx3, Ly3+18, anchor="nw", text=f"Inner dotted ring = Elev ≥ {THRESH_60}° (High zone)", font=("Arial",8), tags="legend")
    canvas3.create_text(Lx3, Ly3+36, anchor="nw", text=f"Middle dotted ring = Elev ≥ {THRESH_30}° (Mid zone)", font=("Arial",8), tags="legend")
    canvas3.create_text(Lx3, Ly3+54, anchor="nw", text=f"Outer ring = Elev ≥ {ELEV_LIMIT}° (approx horizon)", font=("Arial",8), tags="legend")
    canvas3.create_text(Lx3, Ly3+72, anchor="nw", text="Marker = Satellite name + elevation", font=("Arial",8), tags="legend")

    root.after(POLL_MS, update_all)

# start loops
root.after(100, update_all)
win4.after(1000, try_initialize_imu)
root.after(500, update_window4)
root.mainloop()





















# --- End part6.py ---
