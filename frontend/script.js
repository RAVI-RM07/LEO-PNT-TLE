// ======================================
// GLOBAL STATE
// ======================================
const API_URL = "http://localhost:5000/api/visible";

let manualLat = null;
let manualLon = null;
let elevationFilter = 30;

let satQueue = [];
let isAnimating = false;
let dummyCounter = 1;

// ======================================
// DOM READY
// ======================================
document.addEventListener("DOMContentLoaded", () => {
    setupClock();
    setupDrawer();
    setupManualInput();
    setupElevationFilter();

    fetchSatelliteData();
    setInterval(fetchSatelliteData, 3000);

    updateClock();
    setInterval(updateClock, 1000);
});

// ======================================
// CLOCK (12H UTC)
// ======================================
function updateClock() {
    const badge = document.querySelector(".time-badge");
    if (!badge) return;

    const now = new Date();
    let h = now.getUTCHours();
    const m = String(now.getUTCMinutes()).padStart(2, "0");
    const s = String(now.getUTCSeconds()).padStart(2, "0");

    const period = h >= 12 ? "PM" : "AM";
    h = h % 12 || 12;

    badge.textContent = `${h}:${m}:${s} ${period} UTC`;
}

function setupClock() {}

// ======================================
// DRAWER
// ======================================
function setupDrawer() {
    const drawer = document.getElementById("sideDrawer");
    document.getElementById("menuToggle")?.addEventListener("click", () => drawer.classList.add("open"));
    document.getElementById("closeDrawerBtn")?.addEventListener("click", () => drawer.classList.remove("open"));
}

// ======================================
// MANUAL LAT / LON
// ======================================
function setupManualInput() {
    const btn = document.querySelector(".btn-primary");
    if (!btn) return;

    btn.addEventListener("click", async () => {
        const inputs = document.querySelectorAll(".styled-input");
        const lat = parseFloat(inputs[0].value);
        const lon = parseFloat(inputs[1].value);

        if (isNaN(lat) || isNaN(lon)) {
            alert("Invalid coordinates");
            return;
        }

        manualLat = lat;
        manualLon = lon;

        btn.textContent = "Updating...";
        await fetchSatelliteData();
        btn.textContent = "Update Coordinates";
    });
}

// ======================================
// ELEVATION FILTER
// ======================================
function setupElevationFilter() {
    document.querySelectorAll('input[name="elevation"]').forEach(radio => {
        radio.addEventListener("change", () => {
            elevationFilter = parseInt(radio.value);
            fetchSatelliteData();
        });
    });
}

// ======================================
// FETCH DATA
// ======================================
async function fetchSatelliteData() {
    try {
        const lat = manualLat ?? 13.045612;
        const lon = manualLon ?? 77.568921;

        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ latitude: lat, longitude: lon })
        });

        if (!res.ok) throw new Error("API error");

        const data = await res.json();

        updateStatus(true);
        updateSystemOutput(data);
        processSatellites(data.satellites);
    } catch {
        updateStatus(false);
    }
}

// ======================================
// STATUS
// ======================================
function updateStatus(ok) {
    const label = document.querySelector(".status-label-lg");
    const dot = document.querySelector(".pulse-base");

    if (!label || !dot) return;

    label.textContent = ok ? "ONLINE" : "OFFLINE";
    label.style.color = ok ? "#10B981" : "#EF4444";
    dot.style.background = ok ? "#10B981" : "#EF4444";
}

// ======================================
// SYSTEM OUTPUT
// ======================================
function updateSystemOutput(data) {
    const vals = document.querySelectorAll(".stat-val");
    if (vals.length < 2) return;

    vals[0].innerHTML = `${data.latitude.toFixed(4)}°`;
    vals[1].innerHTML = `${data.longitude.toFixed(4)}°`;
}

// ======================================
// SATELLITE FILTER PIPELINE (FINAL)
// ======================================
function processSatellites(sats) {
    if (!Array.isArray(sats)) return;

    let filtered = sats.filter(s => s.elevation_deg >= elevationFilter);

    // 🔁 60° fallback → 50°
    if (elevationFilter === 60 && filtered.length < 4) {
        filtered = sats.filter(s => s.elevation_deg >= 50);
    }

    // 🔁 still < 4 → inject dummy
    if (filtered.length < 4) {
        const needed = 4 - filtered.length;
        filtered = filtered.concat(generateDummySatellites(needed));
    }

    updateRadar(filtered);
    prepareTableQueue(filtered);
}

// ======================================
// RADAR (ICON + NAME)
// ======================================
function updateRadar(sats) {
    const radar = document.querySelector(".radar-ui");
    if (!radar) return;

    radar.querySelectorAll(".sat-point").forEach(p => p.remove());
    const maxR = radar.clientWidth / 2;

    sats.forEach(s => {
        const angle = (s.azimuth_deg - 90) * Math.PI / 180;
        const r = (90 - s.elevation_deg) / 90 * maxR;

        const x = Math.cos(angle) * r;
        const y = Math.sin(angle) * r;

        const p = document.createElement("div");
        p.className = "sat-point";
        p.style.transform = `translate(${x}px, ${y}px)`;
        p.innerHTML = `<i class="fa-solid fa-satellite"></i><span class="sat-label">${s.name}</span>`;
        radar.appendChild(p);
    });
}

// ======================================
// TABLE FLOW (SMOOTH UPWARD)
// ======================================
function prepareTableQueue(sats) {
    satQueue = [...sats];
    const tbody = document.querySelector("tbody");
    if (!tbody) return;

    tbody.innerHTML = "";
    isAnimating = true;
    animateTable();
}

function animateTable() {
    if (!isAnimating) return;

    const tbody = document.querySelector("tbody");
    if (!tbody) return;

    if (tbody.children.length >= 10) {
        tbody.children[0].remove();
    }

    const sat = satQueue.shift();
    if (sat) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${sat.name}</td>
            <td>${sat.unique_id}</td>
            <td>${sat.sub_latitude.toFixed(4)}</td>
            <td>${sat.sub_longitude.toFixed(4)}</td>
            <td>${sat.elevation_deg.toFixed(1)}°</td>
            <td>${sat.azimuth_deg.toFixed(1)}°</td>
            <td>${sat.frequency_mhz ? sat.frequency_mhz.toFixed(2) : "N/A"}</td>
        `;
        tbody.appendChild(row);
    }

    setTimeout(animateTable, 700);
}

// ======================================
// DUMMY SATELLITES (NON-REPEATING)
// ======================================
function generateDummySatellites(count) {
    const list = [];
    for (let i = 0; i < count; i++) {
        list.push({
            name: `SIM-SAT-${dummyCounter}`,
            unique_id: `SIM${dummyCounter}`,
            sub_latitude: (manualLat ?? 13.0) + Math.random(),
            sub_longitude: (manualLon ?? 77.0) + Math.random(),
            elevation_deg: 50 + Math.random() * 10,
            azimuth_deg: Math.random() * 360,
            frequency_mhz: null
        });
        dummyCounter++;
    }
    return list;
}
