const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultDiv = document.getElementById("result");

// Get access to webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    video.play();
  })
  .catch(err => {
    console.error("Error accessing webcam: ", err);
  });

// Capture frame and send to backend every second
setInterval(() => {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL("image/jpeg");

  fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ image: imageData })
  })
    .then(res => res.json())
    .then(data => {
      resultDiv.textContent = "Prediction: " + data.prediction;
    })
    .catch(err => {
      console.error("Prediction error:", err);
    });
}, 1000);
