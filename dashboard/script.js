// ---------------------------------------------------------------
// Campus Energy Dashboard - script.js
// Loads building data from data.json and draws colored markers on
// campus_map.jpg. Clicking a marker opens a detail panel showing that
// building's recent actual usage vs. its own typical pattern, plus any
// anomaly alerts, as a 24-hour curve.
// ---------------------------------------------------------------

const mapContainer = document.getElementById('map-container');
const tooltip = document.getElementById('tooltip');
const hourSlider = document.getElementById('hour-slider');
const hourLabel = document.getElementById('hour-label');
const btnWeekday = document.getElementById('btn-weekday');
const btnWeekend = document.getElementById('btn-weekend');
const btnActual = document.getElementById('btn-actual');
const btnTypical = document.getElementById('btn-typical');

const detailPanel = document.getElementById('detail-panel');
const detailTitle = document.getElementById('detail-title');
const detailSub = document.getElementById('detail-sub');
const detailChart = document.getElementById('detail-chart');
const detailClose = document.getElementById('detail-close');

let buildings = [];
let dayType = 'weekday';   // 'weekday' | 'weekend'
let source = 'actual';     // 'actual'  | 'typical'
let globalMax = 1;
let openBuildingId = null;

function seriesFor(b) {
  return b[dayType + '_' + source];
}

function colorFor(ratio) {
  // green -> yellow -> red
  if (ratio < 0.5) {
    return lerpColor([63, 174, 92], [227, 178, 60], ratio / 0.5);
  }
  return lerpColor([227, 178, 60], [217, 79, 79], (ratio - 0.5) / 0.5);
}

function lerpColor(a, b, t) {
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---------- Draw the markers on the map ----------
function render() {
  mapContainer.querySelectorAll('.marker').forEach(m => m.remove());
  const hour = parseInt(hourSlider.value, 10);
  hourLabel.textContent = String(hour).padStart(2, '0') + ':00';

  if (openBuildingId) {
    const openB = buildings.find(b => b.id === openBuildingId);
    if (openB) drawDetailChart(openB);
  }

  buildings.forEach(b => {
    const series = seriesFor(b);
    const value = series[hour];
    const ratio = Math.min(value / b.max, 1);
    const sizeRatio = Math.min(value / globalMax, 1);
    const size = 14 + sizeRatio * 26;

    const marker = document.createElement('div');
    marker.className = 'marker';
    marker.style.left = b.x + '%';
    marker.style.top = b.y + '%';
    marker.style.width = size + 'px';
    marker.style.height = size + 'px';
    marker.style.background = colorFor(ratio);
    if (b.alert_count > 0) {
      marker.style.boxShadow = '0 0 0 3px #d94f4f, 0 1px 4px rgba(0,0,0,0.35)';
    }

    marker.addEventListener('mouseenter', e => showTooltip(e, b, value));
    marker.addEventListener('mousemove', e => showTooltip(e, b, value));
    marker.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
    marker.addEventListener('click', () => openDetail(b));

    mapContainer.appendChild(marker);
  });
}

function showTooltip(e, b, value) {
  const x = parseFloat(e.target.style.left);
  const y = parseFloat(e.target.style.top);
  tooltip.style.left = x + '%';
  tooltip.style.top = y + '%';
  tooltip.querySelector('.name').textContent = `${b.name} (${b.id})`;
  tooltip.querySelector('.val').textContent = value.toFixed(1) + ' kWh';
  tooltip.style.display = 'block';
}

// ---------- Detail panel: click a building to see its 24h curve ----------
function openDetail(b) {
  openBuildingId = b.id;
  tooltip.style.display = 'none';
  detailPanel.classList.add('open');
  detailTitle.textContent = `${b.name} (${b.id})`;
  drawDetailChart(b);
}

detailClose.addEventListener('click', () => {
  detailPanel.classList.remove('open');
  openBuildingId = null;
});

function drawDetailChart(b) {
  const actual = dayType === 'weekday' ? b.weekday_actual : b.weekend_actual;
  const typical = dayType === 'weekday' ? b.weekday_typical : b.weekend_typical;

  const alertLine = b.alert_count > 0
    ? `\u26a0 ${b.alert_count} anomaly alert${b.alert_count > 1 ? 's' : ''} in the last 14 days`
    : 'No anomalies in the last 14 days';
  const alertColor = b.alert_count > 0 ? '#d94f4f' : 'var(--muted)';
  const alertWeight = b.alert_count > 0 ? '700' : '400';
  detailSub.innerHTML = (dayType === 'weekday' ? 'Weekday' : 'Weekend') +
                        ' profile, 24 hours' +
                        `<br><span style="color:${alertColor}; font-weight:${alertWeight};">${alertLine}</span>`;

  const W = 560, H = 220, padL = 34, padR = 10, padT = 10, padB = 24;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const maxVal = Math.max(...actual, ...typical, 1) * 1.1;
  const xFor = h => padL + (h / 23) * innerW;
  const yFor = v => padT + innerH - (v / maxVal) * innerH;
  const pathFor = arr => arr.map((v, h) => `${h === 0 ? 'M' : 'L'} ${xFor(h).toFixed(1)} ${yFor(v).toFixed(1)}`).join(' ');

  const hour = parseInt(hourSlider.value, 10);
  let svg = '';

  // gridlines + y-axis labels (0, half, max)
  [0, 0.5, 1].forEach(f => {
    const y = padT + innerH - f * innerH;
    svg += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="var(--border)" stroke-width="1"/>`;
    svg += `<text x="${padL - 6}" y="${y + 3}" font-size="9" text-anchor="end" fill="var(--muted)">${(f * maxVal).toFixed(0)}</text>`;
  });
  // x-axis labels every 6 hours
  [0, 6, 12, 18, 23].forEach(h => {
    svg += `<text x="${xFor(h)}" y="${H - 6}" font-size="9" text-anchor="middle" fill="var(--muted)">${String(h).padStart(2, '0')}h</text>`;
  });
  // current-hour marker (follows the slider)
  svg += `<line x1="${xFor(hour)}" y1="${padT}" x2="${xFor(hour)}" y2="${padT + innerH}" stroke="var(--accent)" stroke-width="1" stroke-dasharray="3,3"/>`;
  // the two curves
  svg += `<path d="${pathFor(actual)}" fill="none" stroke="#9aa6a0" stroke-width="2"/>`;
  svg += `<path d="${pathFor(typical)}" fill="none" stroke="#2f7d5f" stroke-width="2.4"/>`;
  // dots on the current hour
  svg += `<circle cx="${xFor(hour)}" cy="${yFor(actual[hour])}" r="3.5" fill="#9aa6a0"/>`;
  svg += `<circle cx="${xFor(hour)}" cy="${yFor(typical[hour])}" r="3.5" fill="#2f7d5f"/>`;

  detailChart.innerHTML = svg;
}

// ---------- Controls ----------
function setActiveButton(group, active) {
  group.forEach(btn => btn.classList.toggle('active', btn === active));
}

hourSlider.addEventListener('input', render);

btnWeekday.addEventListener('click', () => {
  dayType = 'weekday';
  setActiveButton([btnWeekday, btnWeekend], btnWeekday);
  render();
});
btnWeekend.addEventListener('click', () => {
  dayType = 'weekend';
  setActiveButton([btnWeekday, btnWeekend], btnWeekend);
  render();
});
btnActual.addEventListener('click', () => {
  source = 'actual';
  setActiveButton([btnActual, btnTypical], btnActual);
  render();
});
btnTypical.addEventListener('click', () => {
  source = 'typical';
  setActiveButton([btnActual, btnTypical], btnTypical);
  render();
});

// ---------- Load the data, then draw the first frame ----------
fetch('data.json')
  .then(res => res.json())
  .then(data => {
    buildings = data;
    buildings.forEach(b => {
      b.max = Math.max(...b.weekday_actual, ...b.weekend_actual,
                       ...b.weekday_typical, ...b.weekend_typical, 1);
    });
    globalMax = Math.max(...buildings.map(b => b.max));
    render();
  })
  .catch(err => {
    mapContainer.insertAdjacentHTML(
      'beforeend',
      `<p style="padding:20px;color:#d94f4f;">Could not load data.json: ${err}.
       If you opened this file directly as index.html, some browsers block
       local file loading - try the "Live Server" extension in VS Code,
       or run: python3 -m http.server, then open localhost:8000.</p>`
    );
  });
