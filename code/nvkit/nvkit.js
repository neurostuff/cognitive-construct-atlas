/* nvkit:start */
// NVK: a NiiVue viewer for maps on the atlases' 4 mm grid (28,549 MNI152 brain voxels), over a 2 mm MNI152 template.
// One source for both atlas pages: nvkit/nvkit.js, spliced into each page by nvkit/inline.py.
// NVK.viewer(el, {grid, onMove}) -> {show(map), layout(name), moveTo(mm), threshold(f)}; map = {vals (Float32Array in grid
// voxel order), signed, cal_min, cal_max, colormap, colormapNegative, label, unit}; grid = {ijk (Uint8Array, 3 per
// voxel), shape [nx, ny, nz], origin [x, y, z] mm of index 0, mm}. NaN in vals means no value: drawn empty, read out by fmt.
var NVK = (function () {
  'use strict';
  var LIB = 'https://cdn.jsdelivr.net/npm/@niivue/niivue@0.68.1/dist/niivue.umd.js', TEMPLATE = 'mni152_2mm.b64.txt';
  var libP = null, tmplP = null;
  function lib() {
    if (!libP) libP = new Promise(function (res, rej) {
      if (window.niivue) return res(window.niivue);
      var s = document.createElement('script'); s.src = LIB; s.async = true;
      s.onload = function () { window.niivue ? res(window.niivue) : rej(new Error('NiiVue did not load')); };
      s.onerror = function () { libP = null; rej(new Error('NiiVue could not be fetched')); };
      document.head.appendChild(s);
    });
    return libP;
  }
  function template() {
    if (!tmplP) tmplP = fetch(TEMPLATE).then(function (r) { if (!r.ok) throw new Error('template ' + r.status); return r.text(); })
      .then(function (s) { var b = atob(s.trim()), u = new Uint8Array(b.length); for (var i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; })
      .catch(function (e) { tmplP = null; throw e; });
    return tmplP;
  }
  function nifti(data, g) {             // float32 NIfTI-1 with an MNI sform, as one ArrayBuffer
    var h = new ArrayBuffer(352 + 4 * data.length), v = new DataView(h), mm = g.mm || 4, o = g.origin, i;
    v.setInt32(0, 348, true); [3, g.shape[0], g.shape[1], g.shape[2], 1, 1, 1, 1].forEach(function (x, i) { v.setInt16(40 + 2 * i, x, true); });
    v.setInt16(70, 16, true); v.setInt16(72, 32, true); [1, mm, mm, mm, 1, 1, 1, 1].forEach(function (x, i) { v.setFloat32(76 + 4 * i, x, true); });
    v.setFloat32(108, 352, true); v.setFloat32(112, 1, true); v.setUint8(123, 2); v.setInt16(254, 4, true);
    [[mm, 0, 0, o[0]], [0, mm, 0, o[1]], [0, 0, mm, o[2]]].forEach(function (r, k) { r.forEach(function (x, i) { v.setFloat32(280 + 16 * k + 4 * i, x, true); }); });
    for (i = 0; i < 4; i++) v.setUint8(344 + i, [0x6e, 0x2b, 0x31, 0][i]);
    new Float32Array(h, 352).set(data); return h;
  }
  function scatter(vals, g) {           // grid voxel order -> full volume (0 outside the brain)
    var n = g.shape[0] * g.shape[1] * g.shape[2], out = new Float32Array(n), V = vals.length;
    for (var v = 0; v < V; v++) if (vals[v] === vals[v]) out[g.ijk[3 * v] + g.shape[0] * (g.ijk[3 * v + 1] + g.shape[1] * g.ijk[3 * v + 2])] = vals[v];
    return out;
  }
  function voxelAt(g, mm) {             // grid voxel nearest a point, or -1
    if (!g.lut) { g.lut = new Int32Array(g.shape[0] * g.shape[1] * g.shape[2]).fill(-1); for (var v = 0; v < g.ijk.length / 3; v++) g.lut[g.ijk[3 * v] + g.shape[0] * (g.ijk[3 * v + 1] + g.shape[1] * g.ijk[3 * v + 2])] = v; }
    var q = [0, 1, 2].map(function (a) { return Math.round((mm[a] - g.origin[a]) / (g.mm || 4)); });
    if (q.some(function (x, a) { return x < 0 || x >= g.shape[a]; })) return -1;
    return g.lut[q[0] + g.shape[0] * (q[1] + g.shape[1] * q[2])];
  }
  function mmOf(g, v) { return [0, 1, 2].map(function (a) { return g.origin[a] + (g.mm || 4) * g.ijk[3 * v + a]; }); }
  function css(name, fb) { var x = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); return x || fb; }
  function rgba(c) {                    // CSS colour -> [r, g, b, 1] in 0..1
    var cv = document.createElement('canvas').getContext('2d'); cv.fillStyle = '#000'; cv.fillStyle = c; var s = cv.fillStyle, m;
    if (s[0] === '#') { var n = parseInt(s.slice(1), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255, 1]; }
    m = s.match(/[\d.]+/g); return m ? [m[0] / 255, m[1] / 255, m[2] / 255, 1] : [0, 0, 0, 1];
  }
  var LAYOUTS = {slices: 'sliceTypeMultiplanar', axial: 'sliceTypeAxial', coronal: 'sliceTypeCoronal', sagittal: 'sliceTypeSagittal', '3d': 'sliceTypeRender'};

  function viewer(el, cfg) {
    var g = cfg.grid, nv = null, cur = null, frac = 0.35, ready, layoutName = 'slices';
    el.classList.add('nvk');
    el.innerHTML = '<div class="nvk-box"><canvas></canvas><p class="nvk-wait">Loading the 3D viewer\u2026</p></div><p class="nvk-read" aria-live="polite"></p>';
    var box = el.querySelector('.nvk-box'), cv = el.querySelector('canvas'), wait = el.querySelector('.nvk-wait'), read = el.querySelector('.nvk-read');
    cv.setAttribute('aria-label', '3D brain viewer: click or drag to move the crosshair; scroll to step through slices');
    function back() { return rgba(css('--nvk-bg', css('--panel', '#ffffff'))); }
    ready = Promise.all([lib(), template()]).then(function (r) {
      var N = r[0];
      nv = new N.Niivue({backColor: back(), crosshairColor: [0.95, 0.35, 0.15, 1], crosshairWidth: 1, show3Dcrosshair: true, isColorbar: true,
        multiplanarShowRender: N.SHOW_RENDER ? N.SHOW_RENDER.ALWAYS : true, dragAndDropEnabled: false, isResizeCanvas: true, textHeight: 0.04, loadingText: ''});
      return nv.attachToCanvas(cv).then(function () {
        // bytes handed over directly (no blob: or network fetch, which a page's content policy may refuse)
        return nv.addVolumeFromUrl({url: 'mni152.nii.gz', name: 'mni152.nii.gz', buffer: r[1].slice().buffer, colormap: 'gray'});
      }).then(function () {
        return nv.addVolumeFromUrl({url: 'map.nii', name: 'map.nii', buffer: nifti(new Float32Array(g.shape[0] * g.shape[1] * g.shape[2]), g), colormap: 'warm', opacity: 0.9});
      }).then(function () {
        wait.hidden = true; nv.volumes[0].colorbarVisible = false; setLayout(layoutName);
        nv.onLocationChange = function (d) { report(d && d.mm ? [d.mm[0], d.mm[1], d.mm[2]] : null); };
        if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function () { nv.opts.backColor = back(); nv.drawScene(); });
        new MutationObserver(function () { nv.opts.backColor = back(); nv.drawScene(); }).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
        return nv;
      });
    }).catch(function (e) { wait.hidden = false; wait.textContent = 'The 3D viewer could not start: ' + e.message + '. Reload the page to try again.'; throw e; });
    function setLayout(name) {
      layoutName = name; box.setAttribute('data-layout', name);
      if (nv) { nv.setSliceType(nv[LAYOUTS[name]]); if (nv.resizeListener) nv.resizeListener(); nv.drawScene(); }
    }
    setLayout(layoutName);
    function report(mm) {
      if (!mm || !cur) { read.textContent = ''; return; }
      mm = [mm[0], mm[1], mm[2]];
      var v = voxelAt(g, mm), val = v >= 0 ? cur.vals[v] : null;
      read.textContent = 'MNI (' + mm.map(function (x) { return Math.round(x); }).join(', ') + ')' + (cfg.region && v >= 0 ? ' \u00b7 ' + cfg.region(v) : '') +
        (val == null ? ' \u00b7 outside the brain mask' : ' \u00b7 ' + cur.label + ' ' + (cur.fmt ? cur.fmt(val) : val.toFixed(2)) + (cur.unit ? ' ' + cur.unit : ''));
    }
    function move(mm) { nv.scene.crosshairPos = nv.mm2frac(mm); nv.drawScene(); report(mm); }
    function apply() {
      if (!nv || !cur) return;
      var ov = nv.volumes[1], full = scatter(cur.vals, g);
      ov.img.set(full); ov.colormap = cur.colormap || 'warm'; ov.colormapNegative = cur.signed ? (cur.colormapNegative || 'winter') : '';
      ov.cal_max = cur.cal_max; ov.cal_min = cur.cal_max * frac; ov.cal_minNeg = NaN; ov.cal_maxNeg = NaN; ov.opacity = 0.9;
      nv.updateGLVolume(); report(nv.frac2mm(nv.scene.crosshairPos));
    }
    return {
      ready: ready,
      // show a map; the crosshair jumps to its strongest point unless map.jump is false
      show: function (map) {
        cur = map; frac = map.frac != null ? map.frac : (map.cal_min != null ? map.cal_min / map.cal_max : frac);
        return ready.then(function () { apply(); if (map.jump !== false) { var b = -1, bv = -Infinity; for (var v = 0; v < map.vals.length; v++) if (map.vals[v] > bv) { bv = map.vals[v]; b = v; } if (b >= 0) move(mmOf(g, b)); } });
      },
      threshold: function (f) { frac = f; apply(); },
      getThreshold: function () { return frac; },
      layout: setLayout,
      refresh: function () { if (nv) { if (nv.resizeListener) nv.resizeListener(); nv.drawScene(); } },   // after the viewer moves in the page
      moveTo: function (mm) { ready.then(function () { move(mm); }); },
      peakOf: function (test) { var b = -1, bv = -Infinity; if (!cur) return null; for (var v = 0; v < cur.vals.length; v++) if (test(v) && cur.vals[v] > bv) { bv = cur.vals[v]; b = v; } return b >= 0 ? mmOf(g, b) : null; },
      grid: g
    };
  }
  // the controls row both pages use: layout buttons and a threshold slider
  function controls(el, view, opts) {
    var id = 'nvk' + Math.random().toString(36).slice(2, 8);
    el.innerHTML = '<div class="nvk-seg" role="group" aria-label="View">' + [['slices', 'Slices'], ['axial', 'Axial'], ['coronal', 'Coronal'], ['sagittal', 'Sagittal'], ['3d', '3D']].map(function (x, i) {
        return '<button type="button" data-l="' + x[0] + '" aria-pressed="' + (i === 0) + '">' + x[1] + '</button>'; }).join('') + '</div>' +
      '<label class="nvk-th" for="' + id + '">Show above <input id="' + id + '" type="range" min="0" max="95" step="5" value="35"> <output>35%</output> of full colour</label>';
    var r = el.querySelector('input'), o = el.querySelector('output');
    Array.prototype.forEach.call(el.querySelectorAll('[data-l]'), function (b) { b.addEventListener('click', function () {
      Array.prototype.forEach.call(el.querySelectorAll('[data-l]'), function (x) { x.setAttribute('aria-pressed', String(x === b)); }); view.layout(b.getAttribute('data-l')); }); });
    r.addEventListener('input', function () { o.textContent = r.value + '%'; view.threshold(r.value / 100); });
    return {sync: function () { var f = Math.round(view.getThreshold() * 100); r.value = String(Math.min(95, Math.max(0, f))); o.textContent = r.value + '%'; }};
  }
  return {viewer: viewer, controls: controls, voxelAt: voxelAt, mmOf: mmOf, lib: lib};
})();
/* nvkit:end */
