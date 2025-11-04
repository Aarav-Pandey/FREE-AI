// Elements
const askBtn = document.getElementById('askBtn');
const clearBtn = document.getElementById('clearBtn');
const contextEl = document.getElementById('context');
const questionEl = document.getElementById('question');
const resultEl = document.getElementById('result');
const answerEl = document.getElementById('answer');
const scoreEl = document.getElementById('score');
const statusEl = document.getElementById('status');

const API = '/ask'; // same origin

// Ask button
askBtn.addEventListener('click', async () => {
  const context = contextEl.value.trim();
  const question = questionEl.value.trim();
  if (!context) {
    statusEl.textContent = "Please provide a context paragraph.";
    return;
  }
  if (!question) {
    statusEl.textContent = "Please type a question.";
    return;
  }
  statusEl.textContent = "Asking model... (this may take a second)";
  resultEl.classList.add('hidden');

  try {
    const resp = await fetch(API, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({context, question})
    });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = "Error: " + (data.error || resp.statusText);
      return;
    }
    answerEl.textContent = data.answer || "(no answer)";
    scoreEl.textContent = "Confidence: " + (data.score !== undefined ? (data.score.toFixed(4)) : "n/a");
    resultEl.classList.remove('hidden');
    statusEl.textContent = "Done";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Request failed: " + err.message;
  }
});

// Clear button
clearBtn.addEventListener('click', () => {
  contextEl.value = '';
  questionEl.value = '';
  resultEl.classList.add('hidden');
  statusEl.textContent = '';
});
