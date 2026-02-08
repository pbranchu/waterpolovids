/* Chunked file upload + drag-drop for the upload page. */

var CHUNK_SIZE = (typeof window.CHUNK_SIZE !== 'undefined') ? window.CHUNK_SIZE : 10 * 1024 * 1024;

function handleFileSelect(input, clipIndex) {
  if (input.files.length > 0) {
    uploadFile(input.files[0], clipIndex);
  }
}

function uploadFile(file, clipIndex) {
  var totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  var progressEl = document.getElementById('progress-' + clipIndex);
  var fillEl = document.getElementById('fill-' + clipIndex);
  var textEl = document.getElementById('ptext-' + clipIndex);
  var dropEl = document.getElementById('drop-' + clipIndex);

  if (dropEl) dropEl.style.display = 'none';
  if (progressEl) progressEl.style.display = 'block';

  var uploaded = 0;

  function sendChunk(idx) {
    if (idx >= totalChunks) {
      if (textEl) textEl.textContent = 'Complete!';
      // Reload card to show checkmark
      setTimeout(function() { location.reload(); }, 800);
      return;
    }

    var start = idx * CHUNK_SIZE;
    var end = Math.min(start + CHUNK_SIZE, file.size);
    var chunk = file.slice(start, end);

    var fd = new FormData();
    fd.append('game_id', GAME_ID);
    fd.append('clip_index', clipIndex);
    fd.append('chunk_idx', idx);
    fd.append('total_chunks', totalChunks);
    fd.append('filename', file.name);
    fd.append('chunk', chunk);

    fetch('/api/upload-chunk', { method: 'POST', body: fd })
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (!data.ok) {
          if (textEl) textEl.textContent = 'Error: ' + (data.error || 'upload failed');
          return;
        }
        uploaded++;
        var pct = Math.round(uploaded / totalChunks * 100);
        if (fillEl) fillEl.style.width = pct + '%';
        if (textEl) textEl.textContent = pct + '%';
        sendChunk(idx + 1);
      })
      .catch(function(err) {
        if (textEl) textEl.textContent = 'Error: ' + err.message;
      });
  }

  sendChunk(0);
}

function setServerPath(gameId, clipIndex) {
  var input = document.getElementById('path-' + clipIndex);
  var path = input ? input.value.trim() : '';
  if (!path) return;

  fetch('/api/set-server-path', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ game_id: gameId, clip_index: clipIndex, path: path })
  })
  .then(function(r) { return r.json(); })
  .then(function(data) {
    if (data.ok) {
      location.reload();
    } else {
      showToast(data.error || 'Error setting path');
    }
  })
  .catch(function(err) { showToast('Error: ' + err.message); });
}

// Drag and drop setup
document.addEventListener('DOMContentLoaded', function() {
  var drops = document.querySelectorAll('.drop-zone');
  drops.forEach(function(zone) {
    var card = zone.closest('.upload-card');
    var clipIndex = parseInt(card.getAttribute('data-clip-index'));

    zone.addEventListener('dragover', function(e) {
      e.preventDefault();
      zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', function() {
      zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', function(e) {
      e.preventDefault();
      zone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        uploadFile(e.dataTransfer.files[0], clipIndex);
      }
    });
  });
});
