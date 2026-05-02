// MuseFlow Web UI — App Logic

const API_BASE = '';

// --- DOM Elements ---
const statusBadge = document.getElementById('statusBadge');
const statusText = statusBadge.querySelector('.status-text');
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileRemove = document.getElementById('fileRemove');
const randomBtn = document.getElementById('randomBtn');
const genreCards = document.querySelectorAll('.genre-card');
const loopsSlider = document.getElementById('loopsSlider');
const loopsValue = document.getElementById('loopsValue');
const tokenCount = document.getElementById('tokenCount');
const tempSlider = document.getElementById('tempSlider');
const tempValue = document.getElementById('tempValue');
const generateBtn = document.getElementById('generateBtn');
const controlsCard = document.getElementById('controlsCard');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const progressGenre = document.getElementById('progressGenre');
const progressLoops = document.getElementById('progressLoops');
const resultSection = document.getElementById('resultSection');
const resultInfo = document.getElementById('resultInfo');
const tracksList = document.getElementById('tracksList');
const downloadBtn = document.getElementById('downloadBtn');
const newBtn = document.getElementById('newBtn');

// --- State ---
let selectedGenre = 'classical';
let selectedFile = null;
let useRandom = false;
let lastResult = null;

// --- MIDI Program Names ---
const PROGRAM_NAMES = {
  0: 'Piano', 1: 'Bright Piano', 2: 'Electric Grand', 3: 'Honky-Tonk',
  4: 'E. Piano 1', 5: 'E. Piano 2', 6: 'Harpsichord', 7: 'Clavinet',
  24: 'Nylon Guitar', 25: 'Steel Guitar', 26: 'Jazz Guitar', 27: 'Clean Guitar',
  28: 'Muted Guitar', 29: 'Overdrive Guitar', 30: 'Distortion Guitar', 31: 'Guitar Harmonics',
  32: 'Acoustic Bass', 33: 'Finger Bass', 34: 'Pick Bass', 35: 'Fretless Bass',
  36: 'Slap Bass 1', 37: 'Slap Bass 2', 38: 'Synth Bass 1', 39: 'Synth Bass 2',
  40: 'Violin', 41: 'Viola', 42: 'Cello', 43: 'Contrabass',
  44: 'Tremolo Strings', 45: 'Pizzicato', 46: 'Harp', 47: 'Timpani',
  48: 'String Ensemble 1', 49: 'String Ensemble 2', 50: 'Synth Strings 1', 51: 'Synth Strings 2',
  56: 'Trumpet', 57: 'Trombone', 58: 'Tuba', 59: 'Muted Trumpet',
  60: 'French Horn', 61: 'Brass Section', 64: 'Soprano Sax', 65: 'Alto Sax',
  66: 'Tenor Sax', 67: 'Baritone Sax', 68: 'Oboe', 69: 'English Horn',
  70: 'Bassoon', 71: 'Clarinet', 73: 'Flute', 74: 'Recorder',
  80: 'Square Lead', 81: 'Saw Lead', 88: 'New Age Pad', 89: 'Warm Pad',
  90: 'Polysynth Pad', 91: 'Choir Pad', 92: 'Bowed Pad', 93: 'Metallic Pad',
  128: 'Drums'
};

function getProgramName(program) {
  return PROGRAM_NAMES[program] || `Program ${program}`;
}

// --- Check API Status ---
async function checkStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    const data = await res.json();
    if (data.status === 'ready') {
      statusBadge.classList.add('ready');
      statusText.textContent = `Ready · ${data.device.toUpperCase()}`;
    }
  } catch (e) {
    statusBadge.classList.add('error');
    statusText.textContent = 'Server Offline';
  }
}

// --- File Upload ---
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  selectedFile = file;
  useRandom = false;
  fileName.textContent = file.name;
  fileInfo.style.display = 'block';
  uploadZone.style.display = 'none';
  randomBtn.style.display = 'none';
}

fileRemove.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  fileInfo.style.display = 'none';
  uploadZone.style.display = '';
  randomBtn.style.display = '';
});

randomBtn.addEventListener('click', () => {
  useRandom = true;
  selectedFile = null;
  fileInput.value = '';
  randomBtn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
      <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>
    Random Melody Selected
  `;
  randomBtn.style.color = 'var(--accent-green)';
  randomBtn.style.borderColor = 'rgba(61, 255, 192, 0.3)';
});

// --- Genre Selection ---
genreCards.forEach(card => {
  card.addEventListener('click', () => {
    genreCards.forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    selectedGenre = card.dataset.genre;
  });
});

// --- Sliders ---
loopsSlider.addEventListener('input', () => {
  const val = loopsSlider.value;
  loopsValue.textContent = val;
  tokenCount.textContent = val * 150;
});

tempSlider.addEventListener('input', () => {
  const val = (tempSlider.value / 100).toFixed(1);
  tempValue.textContent = val;
});

// --- Generate ---
generateBtn.addEventListener('click', async () => {
  // Disable button
  const btnContent = generateBtn.querySelector('.btn-content');
  const btnLoading = generateBtn.querySelector('.btn-loading');
  generateBtn.disabled = true;
  btnContent.style.display = 'none';
  btnLoading.style.display = 'flex';

  // Show progress
  resultSection.style.display = 'none';
  progressSection.style.display = 'block';
  progressGenre.textContent = selectedGenre;
  progressLoops.textContent = `${loopsSlider.value} loops`;
  progressText.textContent = 'Sending melody to the AI band...';
  progressBar.style.width = '10%';

  // Simulate progress
  const progressInterval = setInterval(() => {
    const current = parseFloat(progressBar.style.width);
    if (current < 85) {
      progressBar.style.width = `${current + Math.random() * 3}%`;
    }
  }, 800);

  const phases = [
    { at: 2000, text: 'Loading model and tokenizer...' },
    { at: 5000, text: 'Generating band arrangement tokens...' },
    { at: 10000, text: 'AI band is improvising...' },
    { at: 18000, text: 'Decoding tokens to MIDI...' },
    { at: 25000, text: 'Humanizing note durations...' },
  ];

  phases.forEach(p => {
    setTimeout(() => {
      if (progressSection.style.display !== 'none') {
        progressText.textContent = p.text;
      }
    }, p.at);
  });

  try {
    const formData = new FormData();
    formData.append('genre', selectedGenre);
    formData.append('loops', loopsSlider.value);
    formData.append('temperature', (tempSlider.value / 100).toFixed(2));
    
    if (selectedFile) {
      formData.append('melody', selectedFile);
    }

    const res = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      body: formData
    });

    clearInterval(progressInterval);
    const data = await res.json();

    if (data.success) {
      lastResult = data;
      progressBar.style.width = '100%';
      progressText.textContent = 'Done!';

      setTimeout(() => {
        progressSection.style.display = 'none';
        showResult(data);
      }, 600);
    } else {
      throw new Error(data.error || 'Generation failed');
    }
  } catch (err) {
    clearInterval(progressInterval);
    progressSection.style.display = 'none';
    showError(err.message);
  } finally {
    generateBtn.disabled = false;
    btnContent.style.display = 'flex';
    btnLoading.style.display = 'none';
  }
});

// --- Show Result ---
function showResult(data) {
  resultSection.style.display = 'block';
  resultInfo.textContent = `Your AI band generated a ${data.genre.toUpperCase()} arrangement with ${data.tokens_generated} tokens`;

  // Build tracks
  tracksList.innerHTML = '';
  data.tracks.forEach(track => {
    const item = document.createElement('div');
    item.className = 'track-item';
    item.innerHTML = `
      <div class="track-left">
        <div class="track-dot"></div>
        <span class="track-name">${getProgramName(track.program)}</span>
      </div>
      <span class="track-notes">${track.notes} notes</span>
    `;
    tracksList.appendChild(item);
  });

  // Scroll to result
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// --- Download ---
downloadBtn.addEventListener('click', () => {
  if (lastResult) {
    const a = document.createElement('a');
    a.href = `${API_BASE}/api/download/${lastResult.filename}`;
    a.download = lastResult.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
});

// --- New Generation ---
newBtn.addEventListener('click', () => {
  resultSection.style.display = 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

// --- Error Toast ---
function showError(msg) {
  let toast = document.querySelector('.error-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.className = 'error-toast';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  setTimeout(() => toast.classList.add('show'), 50);
  setTimeout(() => {
    toast.classList.remove('show');
  }, 5000);
}

// --- Init ---
checkStatus();
