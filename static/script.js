function uploadVideo() {
  const input = document.getElementById('videoInput');
  const file = input.files[0];
  if (!file) {
    alert("Please select a video.");
    return;
  }

  // Show loading spinner
  document.getElementById('spinner').style.display = 'block';
  document.getElementById('videoContainer').style.display = 'none';
  const moveListDiv = document.getElementById('moveList');
  moveListDiv.innerHTML = '';

  const formData = new FormData();
  formData.append('video', file);

  fetch('/upload', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      // Hide spinner
      document.getElementById('spinner').style.display = 'none';

      // Convert base64 to blob
      const byteCharacters = atob(data.video);
      const byteNumbers = Array.from(byteCharacters, c => c.charCodeAt(0));
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'video/mp4' });
      const videoURL = URL.createObjectURL(blob);

      // Update video player
      const videoSource = document.getElementById('videoSource');
      const videoPlayer = document.getElementById('videoPlayer');
      const downloadLink = document.getElementById('downloadLink');

      videoSource.src = videoURL;
      videoPlayer.load();
      document.getElementById('videoContainer').style.display = 'block';
      downloadLink.href = videoURL;
      downloadLink.download = data.filename || 'annotated_output.mp4';

      // Show detected moves
      if (data.moves && data.moves.length > 0) {
        const moveItems = data.moves.map((m, idx) => `<li><strong>${idx + 1}.</strong> ${m}</li>`).join('');
        moveListDiv.innerHTML = `
          <h2>📋 Detected Moves</h2>
          <ul>${moveItems}</ul>
        `;
      } else {
        moveListDiv.innerHTML = `<p>No moves were detected.</p>`;
      }
    })
    .catch(error => {
      console.error("Upload failed:", error);
      alert("Something went wrong. Please try again.");
      document.getElementById('spinner').style.display = 'none';
    });
}
