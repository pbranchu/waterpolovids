/* Polygon mask editor — adapted from label_ball.py MASK_HTML_TEMPLATE. */

(function() {
  var cfg = window.MASK_CONFIG;
  var ZOOM_SIZE = 200;
  var ZOOM_FACTOR = 8;

  var points = cfg.existingPoly ? cfg.existingPoly.slice() : [];
  var showCones = false;
  var coneData = null; // loaded async

  var canvas = document.getElementById('mask-canvas');
  var ctx = canvas.getContext('2d');
  var zoomInset = document.getElementById('zoom-inset');
  var zoomCanvas = document.getElementById('zoom-canvas');
  var zoomCtx = zoomCanvas.getContext('2d');

  var img = new Image();

  img.onload = function() {
    var area = document.getElementById('canvas-area');
    var maxW = area.clientWidth - 8;
    var maxH = area.clientHeight - 8;
    var scale = Math.min(maxW / img.width, maxH / img.height, 1);
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    canvas._scale = scale;
    canvas._imgW = img.width;
    canvas._imgH = img.height;
    draw();
  };
  img.src = cfg.frameUrl;

  // Load cone candidates
  fetch(cfg.conesUrl)
    .then(function(r) { return r.json(); })
    .then(function(data) {
      coneData = data.markers || [];
      // If no existing polygon, use auto-detected one
      if (points.length === 0 && data.polygon && data.polygon.length >= 3) {
        points = data.polygon.slice();
        draw();
      }
    })
    .catch(function() {});

  function draw() {
    if (!canvas._scale) return;
    var s = canvas._scale;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Draw cone markers
    if (showCones && coneData) {
      coneData.forEach(function(c) {
        var cx = c[0] * s, cy = c[1] * s;
        var color = c[2] === 'yellow' ? 'rgba(255,255,0,0.8)' : 'rgba(255,50,50,0.8)';
        ctx.beginPath();
        ctx.arc(cx, cy, 8, 0, Math.PI * 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(cx, cy, 12, 0, Math.PI * 2);
        ctx.strokeStyle = color.replace('0.8', '0.3');
        ctx.lineWidth = 2;
        ctx.stroke();
      });
    }

    if (points.length === 0) { updateInfo(); return; }

    // Fill
    if (points.length >= 3) {
      ctx.beginPath();
      ctx.moveTo(points[0][0] * s, points[0][1] * s);
      for (var i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0] * s, points[i][1] * s);
      }
      ctx.closePath();
      ctx.fillStyle = 'rgba(42, 157, 143, 0.2)';
      ctx.fill();
    }

    // Lines
    ctx.beginPath();
    ctx.moveTo(points[0][0] * s, points[0][1] * s);
    for (var i = 1; i < points.length; i++) {
      ctx.lineTo(points[i][0] * s, points[i][1] * s);
    }
    if (points.length >= 3) ctx.closePath();
    ctx.strokeStyle = '#2a9d8f';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Vertices
    points.forEach(function(p, i) {
      ctx.beginPath();
      ctx.arc(p[0] * s, p[1] * s, 6, 0, Math.PI * 2);
      ctx.fillStyle = i === points.length - 1 ? '#e94560' : '#2a9d8f';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = 'bold 13px system-ui';
      ctx.fillText('' + (i + 1), p[0] * s + 9, p[1] * s - 9);
    });

    updateInfo();
  }

  function updateInfo() {
    var el = document.getElementById('mask-info');
    if (el) {
      el.textContent = points.length + ' points. Click to mark pool boundary corners. Need >= 3 to save.';
    }
  }

  // Click to add point
  canvas.addEventListener('click', function(e) {
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;
    var s = canvas._scale;
    var ix = Math.round(mx / s);
    var iy = Math.round(my / s);

    // Snap to cone
    if (coneData) {
      var bestDist = 30;
      var snapX = ix, snapY = iy;
      coneData.forEach(function(c) {
        var dx = c[0] * s - mx;
        var dy = c[1] * s - my;
        var d = Math.sqrt(dx * dx + dy * dy);
        if (d < bestDist) {
          bestDist = d;
          snapX = c[0];
          snapY = c[1];
        }
      });
      ix = snapX;
      iy = snapY;
    }

    points.push([ix, iy]);
    draw();
  });

  // Zoom inset
  canvas.addEventListener('mousemove', function(e) {
    var rect = canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;
    var s = canvas._scale;

    zoomInset.style.display = 'block';
    var zx = mx + 25, zy = my - 100;
    if (zx + ZOOM_SIZE + 10 > canvas.width) zx = mx - ZOOM_SIZE - 25;
    if (zy < 0) zy = my + 25;
    zoomInset.style.left = zx + 'px';
    zoomInset.style.top = zy + 'px';

    var srcSize = ZOOM_SIZE / ZOOM_FACTOR;
    var srcX = mx / s - srcSize / 2;
    var srcY = my / s - srcSize / 2;
    zoomCtx.clearRect(0, 0, ZOOM_SIZE, ZOOM_SIZE);
    zoomCtx.drawImage(img, srcX, srcY, srcSize, srcSize, 0, 0, ZOOM_SIZE, ZOOM_SIZE);

    // Cone markers in zoom
    if (showCones && coneData) {
      coneData.forEach(function(c) {
        var rx = (c[0] - srcX) * ZOOM_FACTOR;
        var ry = (c[1] - srcY) * ZOOM_FACTOR;
        if (rx > -20 && rx < ZOOM_SIZE + 20 && ry > -20 && ry < ZOOM_SIZE + 20) {
          var color = c[2] === 'yellow' ? 'rgba(255,255,0,0.9)' : 'rgba(255,50,50,0.9)';
          zoomCtx.beginPath();
          zoomCtx.arc(rx, ry, 6, 0, Math.PI * 2);
          zoomCtx.strokeStyle = color;
          zoomCtx.lineWidth = 2;
          zoomCtx.stroke();
        }
      });
    }

    // Polygon in zoom
    if (points.length >= 2) {
      zoomCtx.beginPath();
      var p0 = points[0];
      zoomCtx.moveTo((p0[0] - srcX) * ZOOM_FACTOR, (p0[1] - srcY) * ZOOM_FACTOR);
      for (var i = 1; i < points.length; i++) {
        var p = points[i];
        zoomCtx.lineTo((p[0] - srcX) * ZOOM_FACTOR, (p[1] - srcY) * ZOOM_FACTOR);
      }
      if (points.length >= 3) zoomCtx.closePath();
      zoomCtx.strokeStyle = '#2a9d8f';
      zoomCtx.lineWidth = 2;
      zoomCtx.stroke();
    }

    // Crosshair
    var ch = ZOOM_SIZE / 2;
    zoomCtx.strokeStyle = 'rgba(255,255,255,0.7)';
    zoomCtx.lineWidth = 1;
    zoomCtx.beginPath(); zoomCtx.moveTo(ch, ch - 12); zoomCtx.lineTo(ch, ch + 12); zoomCtx.stroke();
    zoomCtx.beginPath(); zoomCtx.moveTo(ch - 12, ch); zoomCtx.lineTo(ch + 12, ch); zoomCtx.stroke();
  });

  canvas.addEventListener('mouseleave', function() {
    zoomInset.style.display = 'none';
  });

  // Global functions for buttons
  window.undoPoint = function() {
    if (points.length > 0) {
      points.pop();
      draw();
      showToast('Removed last point');
    }
  };

  window.clearPoly = function() {
    points = [];
    draw();
    showToast('Cleared all points');
  };

  window.toggleCones = function() {
    showCones = !showCones;
    var btn = document.getElementById('cone-btn');
    if (btn) btn.classList.toggle('active', showCones);
    draw();
  };

  window.saveMask = function() {
    if (points.length < 3) {
      showToast('Need at least 3 points!');
      return;
    }
    var applyAll = document.getElementById('apply-all').checked;
    fetch(cfg.saveUrl, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        clip_index: cfg.clipIdx,
        polygon: points,
        apply_all: applyAll
      })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.ok) {
        showToast('Saved!');
        setTimeout(function() {
          if (data.next_clip !== null && data.next_clip !== undefined) {
            window.location.href = '/game/' + cfg.gameId + '/masks/' + data.next_clip;
          } else {
            window.location.href = '/game/' + cfg.gameId + '/review';
          }
        }, 400);
      } else {
        showToast(data.error || 'Error saving');
      }
    });
  };

  window.navClip = function(delta) {
    var next = cfg.clipIdx + delta;
    if (next >= 0 && next < cfg.totalClips) {
      window.location.href = '/game/' + cfg.gameId + '/masks/' + next;
    }
  };

  // Hide nav buttons at boundaries
  if (cfg.clipIdx <= 0) {
    var prev = document.getElementById('prev-clip');
    if (prev) prev.style.display = 'none';
  }
  if (cfg.clipIdx >= cfg.totalClips - 1) {
    var next = document.getElementById('next-clip');
    if (next) next.style.display = 'none';
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    // Don't capture if typing in an input
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { navClip(-1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { navClip(1); e.preventDefault(); }
    else if (e.key === 'z' || e.key === 'Z') { undoPoint(); e.preventDefault(); }
    else if (e.key === 'c' || e.key === 'C') { clearPoly(); e.preventDefault(); }
    else if (e.key === 'h' || e.key === 'H') { toggleCones(); e.preventDefault(); }
    else if (e.key === 'Enter') { saveMask(); e.preventDefault(); }
  });

  // Resize handler
  window.addEventListener('resize', function() {
    if (img.complete && img.naturalWidth > 0) {
      img.onload();
    }
  });
})();
