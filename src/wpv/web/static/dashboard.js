/* Poll progress for games that are processing or queued. */
(function() {
  function pollActive() {
    var rows = document.querySelectorAll('tr[data-status="processing"], tr[data-status="queued"]');
    if (rows.length === 0) return;

    rows.forEach(function(row) {
      var gid = row.getAttribute('data-game-id');
      fetch('/api/game/' + gid + '/progress')
        .then(function(r) { return r.json(); })
        .then(function(data) {
          // Update badge
          var badge = row.querySelector('.badge');
          if (badge && data.status) {
            badge.textContent = data.status;
            badge.className = 'badge badge-' + data.status;
          }
          // Update progress bar
          var fill = row.querySelector('.progress-fill-sm');
          if (fill) {
            fill.style.width = (data.progress_pct || 0) + '%';
          }
          var pct = row.querySelector('.progress-pct');
          if (pct) {
            pct.textContent = Math.round(data.progress_pct || 0) + '%';
          }
          // Update progress message
          var msgEl = row.querySelector('.progress-msg');
          if (msgEl && data.message) {
            msgEl.textContent = data.message;
          }
          // Update row status attribute
          if (data.status) {
            row.setAttribute('data-status', data.status);
          }
          // If completed, failed, or cancelled, refresh to update action buttons
          if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
            setTimeout(function() { location.reload(); }, 1500);
          }
        })
        .catch(function() {});
    });
  }

  setInterval(pollActive, 3000);
})();
