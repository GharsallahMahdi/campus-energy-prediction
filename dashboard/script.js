// ---------------------------------------------------------------
// Focused 2-marker campus map - script.js
// Only Computer Building (J) and Dormitory 9 are clickable on the map.
// Clicking Dormitory 9 reveals a room selector (whole building / Room 101
// / Room 105), so you can compare a normal room against one with an
// injected "AC left on" anomaly - each with its own 24h profile, 30-day
// history and genuine 7-day forecast.
// ---------------------------------------------------------------

const mapContainer = document.getElementById('map-container');
const tooltip = document.getElementById('tooltip');
const detailPanel = document.getElementById('detail-panel');
const detailTitle = document.getElementById('detail-title');
const roomRow = document.getElementById('room-row');
const btnViewProfile = document.getElementById('btn-view-profile');
const btnViewHistory = document.getElementById('btn-view-history');
const btnViewForecast = document.getElementById('btn-view-forecast');
const dayRow = document.getElementById('day-row');
const btnWeekday = document.getElementById('btn-weekday');
const btnWeekend = document.getElementById('btn-weekend');
const alertBox = document.getElementById('alert-box');
const detailSub = document.getElementById('detail-sub');
const chart = document.getElementById('chart');
const chartLegend = document.getElementById('chart-legend');
const detailClose = document.getElementById('detail-close');

let data = null;
let openBuildingId = null;  // 'J' or '9' - which marker is open
let currentId = null;       // the entity currently shown in the panel
let view = 'profile';
let dayType = 'weekday';

function setActive(group, active) {
  group.forEach(el => el.classList.toggle('active', el === active));
}

// ---------- Draw the two markers ----------
function renderMarkers() {
  data.building_ids.forEach(bid => {
    const e = data.entities[bid];
    const marker = document.createElement('div');
    marker.className = 'marker';
    marker.style.left = e.x + '%';
    marker.style.top = e.y + '%';
    marker.style.width = '34px';
    marker.style.height = '34px';
    marker.style.background = '#2f7d5f';
    marker.textContent = bid;
    if (hasAnyAlert(bid)) {
      marker.style.boxShadow = '0 0 0 3px #d94f4f, 0 2px 8px rgba(0,0,0,0.4)';
    }
    marker.addEventListener('mouseenter', () => showTooltip(e));
    marker.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
    marker.addEventListener('click', () => openBuilding(bid));
    mapContainer.appendChild(marker);
  });
}

function hasAnyAlert(bid) {
  if (data.entities[bid].alert_count > 0) return true;
  if (bid === data.dorm_building_id) {
    return data.room_ids.some(rid => data.entities[rid].alert_count > 0);
  }
  return false;
}

function showTooltip(e) {
  tooltip.style.left = e.x + '%';
  tooltip.style.top = e.y + '%';
  tooltip.querySelector('.name').textContent = e.name;
  tooltip.style.display = 'block';
}

// ---------- Opening a building ----------
function openBuilding(bid) {
  openBuildingId = bid;
  currentId = bid;
  tooltip.style.display = 'none';
  detailPanel.classList.add('open');

  if (bid === data.dorm_building_id) {
    buildRoomRow();
  } else {
    roomRow.innerHTML = '';
  }
  render();
}

function buildRoomRow() {
  roomRow.innerHTML = '';
  const options = [[data.dorm_building_id, 'Whole building'], ...data.room_ids.map(rid => [rid, data.entities[rid].name])];
  options.forEach(([id, label]) => {
    const btn = document.createElement('button');
    btn.className = 'room-btn' + (id === currentId ? ' active' : '');
    if (data.entities[id].alert_count > 0) btn.classList.add('has-alert');
    btn.textContent = label;
    btn.addEventListener('click', () => {
      currentId = id;
      [...roomRow.children].forEach(c => c.classList.toggle('active', c === btn));
      render();
    });
    roomRow.appendChild(btn);
  });
}

detailClose.addEventListener('click', () => {
  detailPanel.classList.remove('open');
  openBuildingId = null;
});

// ---------- View switching ----------
btnViewProfile.addEventListener('click', () => { view = 'profile'; setActive([btnViewProfile, btnViewHistory, btnViewForecast], btnViewProfile); render(); });
btnViewHistory.addEventListener('click', () => { view = 'history'; setActive([btnViewProfile, btnViewHistory, btnViewForecast], btnViewHistory); render(); });
btnViewForecast.addEventListener('click', () => { view = 'forecast'; setActive([btnViewProfile, btnViewHistory, btnViewForecast], btnViewForecast); render(); });
btnWeekday.addEventListener('click', () => { dayType = 'weekday'; setActive([btnWeekday, btnWeekend], btnWeekday); render(); });
btnWeekend.addEventListener('click', () => { dayType = 'weekend'; setActive([btnWeekday, btnWeekend], btnWeekend); render(); });

// ---------- Render the panel's chart ----------
function render() {
  const e = data.entities[currentId];
  detailTitle.textContent = e.name;
  dayRow.style.display = view === 'profile' ? 'flex' : 'none';

  if (e.alert_message) {
    alertBox.textContent = e.alert_message;
    alertBox.classList.add('show');
  } else {
    alertBox.classList.remove('show');
  }

  if (view === 'profile') drawProfile(e);
  else if (view === 'history') drawHistory(e);
  else drawForecast(e);
}

function drawProfile(e) {
  const vals = e.profile[dayType];
  const other = e.profile[dayType === 'weekday' ? 'weekend' : 'weekday'];
  detailSub.textContent = `Typical ${dayType} pattern, averaged over the year (excluding academic-break days and injected anomalies).`;
  chartLegend.innerHTML = `<span><span class="chip" style="background:#2f7d5f;"></span>Typical consumption (kWh)</span>`;
  // Fixed scale shared between weekday and weekend, so switching the
  // toggle actually shows the curve sitting lower/higher - not just
  // relabeled axis numbers while the shape silently re-stretches to fit.
  const maxVal = Math.max(...vals, ...other, 0.1) * 1.15;
  const allHours = vals.map((_, h) => h);   // every hour, not just 0/6/12/18/23
  const { xFor, yFor, svgAxes } = buildAxes(24, maxVal, allHours, h => String(h), 10);
  let svg = svgAxes;
  svg += areaPath(vals, xFor, yFor, '#2f7d5f22');
  svg += linePath(vals, xFor, yFor, '#2f7d5f', 2.6);
  vals.forEach((v, h) => {
    svg += `<circle cx="${xFor(h).toFixed(1)}" cy="${yFor(v).toFixed(1)}" r="3.2" fill="#2f7d5f" stroke="white" stroke-width="1.2"/>`;
  });
  chart.innerHTML = svg;
}

function drawHistory(e) {
  const { dates, values } = e.history_30d;
  detailSub.textContent = `Daily total, last 30 days (${dates[0]} to ${dates[dates.length - 1]}).`;
  chartLegend.innerHTML = `<span><span class="chip" style="background:#3d7ab5;"></span>Daily total (kWh)</span>`;
  const maxVal = Math.max(...values, 0.1) * 1.15;
  const allTicks = dates.map((_, i) => i);               // every day, not just weekly
  const dayOfMonth = i => String(parseInt(dates[i].split('-')[2], 10));
  const { xFor, yFor, svgAxes } = buildAxes(dates.length, maxVal, allTicks, dayOfMonth, 10);
  let svg = svgAxes;
  svg += areaPath(values, xFor, yFor, '#3d7ab522');
  svg += linePath(values, xFor, yFor, '#3d7ab5', 2.4);
  values.forEach((v, i) => {
    svg += `<circle cx="${xFor(i).toFixed(1)}" cy="${yFor(v).toFixed(1)}" r="3.2" fill="#3d7ab5" stroke="white" stroke-width="1.2"/>`;
  });
  chart.innerHTML = svg;
}

function drawForecast(e) {
  if (!e.forecast) {
    detailSub.textContent = 'No forecast available.';
    chart.innerHTML = ''; chartLegend.innerHTML = '';
    return;
  }
  const { actual, rf_pred, lstm_pred } = e.forecast;
  const cutoff = data.first_24h_index != null ? data.first_24h_index : 23;
  detailSub.innerHTML = `Genuine next-24h / next-7-days forecast (no data leakage), starting ${data.forecast_start ? data.forecast_start.split(' ')[0] : ''}. Dashed = end of "next 24h". Orange = Shanghai air temperature (real monthly climate normals), a real input feature for both models.`;
  chartLegend.innerHTML = `
    <span><span class="chip" style="background:#1c2321;"></span>Actual (kWh)</span>
    <span><span class="chip" style="background:#3d7ab5;"></span>Random Forest</span>
    <span><span class="chip" style="background:#2f7d5f;"></span>LSTM</span>
    <span><span class="chip" style="background:#e3923c;"></span>Temperature (\u00b0C, right axis)</span>`;
  const n = actual.length;
  const maxVal = Math.max(...actual.filter(v => v != null), ...rf_pred, ...lstm_pred, 0.1) * 1.15;
  // Detailed ticks: every 12 hours, labeled "D1 00h", "D1 12h", "D2 00h", ...
  const detailTicks = [];
  for (let h = 0; h < n; h += 12) detailTicks.push(h);
  if (detailTicks[detailTicks.length - 1] !== n - 1) detailTicks.push(n - 1);
  const tickLabel = h => `D${Math.floor(h / 24) + 1} ${String(h % 24).padStart(2, '0')}h`;
  const { xFor, yFor, svgAxes, W, padT, padB, H } = buildAxes(n, maxVal, detailTicks, tickLabel, 8.5);
  let svg = svgAxes;
  svg += `<line x1="${xFor(cutoff)}" y1="14" x2="${xFor(cutoff)}" y2="266" stroke="#888" stroke-width="1" stroke-dasharray="3,3"/>`;

  if (e.temperature_c) {
    const temps = e.temperature_c.slice(0, n);
    const tMin = Math.min(...temps), tMax = Math.max(...temps);
    const tRange = (tMax - tMin) || 1;
    const innerH = H - padT - padB;
    const yForTemp = v => padT + innerH - ((v - tMin) / tRange) * innerH * 0.9 - innerH * 0.05;
    svg += linePath(temps, xFor, yForTemp, '#e3923c', 1.8, true);
    // right-axis labels for temperature
    [tMin, (tMin + tMax) / 2, tMax].forEach(v => {
      const y = yForTemp(v);
      svg += `<text x="${W - 4}" y="${y + 4}" font-size="11" text-anchor="end" fill="#e3923c" opacity="0.85">${v.toFixed(0)}\u00b0</text>`;
    });
  }

  svg += linePath(actual, xFor, yFor, '#1c2321', 2.2);
  svg += linePath(rf_pred, xFor, yFor, '#3d7ab5', 2);
  svg += linePath(lstm_pred, xFor, yFor, '#2f7d5f', 2);

  // Markers every 6 hours (every point would be too dense at 168 hours)
  for (let h = 0; h < n; h += 6) {
    if (actual[h] != null) svg += `<circle cx="${xFor(h).toFixed(1)}" cy="${yFor(actual[h]).toFixed(1)}" r="2.4" fill="#1c2321"/>`;
    svg += `<circle cx="${xFor(h).toFixed(1)}" cy="${yFor(rf_pred[h]).toFixed(1)}" r="2.2" fill="#3d7ab5"/>`;
    svg += `<circle cx="${xFor(h).toFixed(1)}" cy="${yFor(lstm_pred[h]).toFixed(1)}" r="2.2" fill="#2f7d5f"/>`;
  }
  chart.innerHTML = svg;
}

// ---------- Chart helpers ----------
function buildAxes(n, maxVal, ticks, tickLabel, tickFontSize) {
  const W = 700, H = 300, padL = 46, padR = 14, padT = 14, padB = 34;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const xFor = i => padL + (i / (n - 1)) * innerW;
  const yFor = v => padT + innerH - ((v == null ? 0 : v) / maxVal) * innerH;
  const fs = tickFontSize || 12;
  let svg = '';
  [0, 0.5, 1].forEach(f => {
    const y = padT + innerH - f * innerH;
    svg += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`;
    svg += `<text x="${padL - 8}" y="${y + 4}" font-size="12" text-anchor="end" fill="var(--muted)">${(f * maxVal).toFixed(1)}</text>`;
  });
  ticks.forEach(i => {
    svg += `<text x="${xFor(i)}" y="${H - 10}" font-size="${fs}" text-anchor="middle" fill="var(--muted)">${tickLabel(i)}</text>`;
  });
  return { xFor, yFor, svgAxes: svg, W, H, padT, padB, padL, padR };
}
function linePath(arr, xFor, yFor, color, width, dashed) {
  const d = arr.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xFor(i).toFixed(1)} ${yFor(v).toFixed(1)}`).join(' ');
  const dash = dashed ? ' stroke-dasharray="5,3"' : '';
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}"${dash}/>`;
}
function areaPath(arr, xFor, yFor, fill) {
  const top = arr.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xFor(i).toFixed(1)} ${yFor(v).toFixed(1)}`).join(' ');
  const baseY = yFor(0);
  return `<path d="${top} L ${xFor(arr.length - 1).toFixed(1)} ${baseY} L ${xFor(0).toFixed(1)} ${baseY} Z" fill="${fill}" stroke="none"/>`;
}

// ---------- Load data ----------
fetch('data.json')
  .then(res => res.json())
  .then(json => { data = json; renderMarkers(); })
  .catch(err => {
    mapContainer.insertAdjacentHTML('beforeend',
      `<p style="padding:20px;color:#d94f4f;">Could not load data.json: ${err}.
       Try: python3 -m http.server, then open localhost:8000.</p>`);
  });
