// ---------------------------------------------------------------
// Campus Energy Dashboard - script.js
// Loads building data from data.json and draws colored markers
// on top of campus_map.jpg, positioned by each building's x/y %.
// ---------------------------------------------------------------

const mapContainer = document.getElementById('map-container');
const tooltip = document.getElementById('tooltip');
const hourSlider = document.getElementById('hour-slider');
const hourLabel = document.getElementById('hour-label');
const btnWeekday = document.getElementById('btn-weekday');
const btnWeekend = document.getElementById('btn-weekend');
const btnActual = document.getElementById('btn-actual');
const btnPred = document.getElementById('btn-pred');

let buildings = [];
let dayType = 'weekday';   // 'weekday' | 'weekend'
let source = 'actual';     // 'actual'  | 'pred'
let globalMax = 1;

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

function render() {
  mapContainer.querySelectorAll('.marker').forEach(m => m.remove());
  const hour = parseInt(hourSlider.value, 10);
  hourLabel.textContent = String(hour).padStart(2, '0') + ':00';

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

    marker.addEventListener('mouseenter', e => showTooltip(e, b, value));
    marker.addEventListener('mousemove', e => showTooltip(e, b, value));
    marker.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
    marker.addEventListener('click', e => showTooltip(e, b, value));

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
  setActiveButton([btnActual, btnPred], btnActual);
  render();
});
btnPred.addEventListener('click', () => {
  source = 'pred';
  setActiveButton([btnActual, btnPred], btnPred);
  render();
});

// ---------------------------------------------------------------
// Load the data, then draw the first frame
// ---------------------------------------------------------------
fetch('data.json')
  .then(res => res.json())
  .then(data => {
    buildings = data;
    buildings.forEach(b => {
      b.max = Math.max(...b.weekday_actual, ...b.weekend_actual,
                       ...b.weekday_pred, ...b.weekend_pred, 1);
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
       or run: python3 -m http.server, then open localhost:8000/dashboard/.</p>`
    );
  });
