const video = document.getElementById('video');
const output = document.getElementById('output');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');

let stream;
let intervalId;

async function startDetection() {
  stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  intervalId = setInterval(captureAndSendFrame, 500);
}

function stopDetection() {
  if (intervalId) clearInterval(intervalId);
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

async function captureAndSendFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
  const formData = new FormData();
  formData.append('frame', blob, 'frame.jpg');

  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    output.innerText = data.prediction;
  } catch (err) {
    console.error('Prediction error:', err);
  }
}

startBtn.addEventListener('click', startDetection);
const modeSelect = document.getElementById('modeSelect');

modeSelect.addEventListener('change', () => {
  const selectedMode = modeSelect.value; // "0" for Alphabets, "1" for Numbers
  console.log("Selected mode:", selectedMode); // You can send this to your backend or model
});
let sentence = "";

const nextBtn = document.getElementById("nextBtn");
const resetBtn = document.getElementById("resetBtn");
const sentenceBox = document.getElementById("sentence");

nextBtn.addEventListener("click", () => {
  const currentLetter = output.textContent.trim();
  if (currentLetter.length) {
    sentence += currentLetter;
    sentenceBox.textContent = sentence;
  }
});

resetBtn.addEventListener("click", () => {
  sentence = "";
  sentenceBox.textContent = "[Empty]";
});
const speakBtn = document.getElementById('speakBtn');
const sentenceDisplay = document.getElementById('sentence'); // assuming you have a div/span with id="sentence"

speakBtn.addEventListener('click', () => {
  const sentenceText = sentenceDisplay.textContent.trim();
  if (sentenceText.length > 0) {
    const utterance = new SpeechSynthesisUtterance(sentenceText);
    utterance.lang = 'en-US'; // or 'en-IN', or change based on your TTS preference
    speechSynthesis.speak(utterance);
  }
});

stopBtn.addEventListener('click', stopDetection);
