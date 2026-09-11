/* ── State ── */
let currentDocId = null;
let chatHistory = [];

/* ── Tab Navigation ── */
function switchTab(tabName) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.getElementById('tab-' + tabName).classList.add('active');
  document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
}

document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', (e) => {
    e.preventDefault();
    switchTab(link.dataset.tab);
  });
});

/* ── Upload Mode Toggle ── */
function setUploadMode(mode) {
  document.getElementById('pdfMode').style.display = mode === 'pdf' ? 'block' : 'none';
  document.getElementById('textMode').style.display = mode === 'text' ? 'block' : 'none';
  document.getElementById('modePdf').classList.toggle('active', mode === 'pdf');
  document.getElementById('modeText').classList.toggle('active', mode === 'text');
}

/* ── Drop Zone ── */
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.pdf')) {
    fileInput.files = e.dataTransfer.files;
    dropZone.querySelector('p').innerHTML = `<strong>📄 ${file.name}</strong> ready to upload`;
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) {
    dropZone.querySelector('p').innerHTML = `<strong>📄 ${fileInput.files[0].name}</strong> ready to upload`;
  }
});

/* ── Upload Document ── */
async function uploadDocument() {
  const btn = document.getElementById('uploadBtn');
  const btnText = document.getElementById('uploadBtnText');
  const resultEl = document.getElementById('uploadResult');
  const isPdf = document.getElementById('pdfMode').style.display !== 'none';

  btnText.innerHTML = '<span class="spinner"></span> Processing...';
  btn.disabled = true;
  resultEl.style.display = 'none';

  try {
    let response;

    if (isPdf) {
      if (!fileInput.files[0]) {
        showUploadResult('error', 'Please select a PDF file first.');
        return;
      }
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      response = await fetch('/upload/', { method: 'POST', body: formData });
    } else {
      const text = document.getElementById('pasteArea').value.trim();
      if (!text) {
        showUploadResult('error', 'Please paste some text first.');
        return;
      }
      response = await fetch('/upload/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
    }

    const data = await response.json();

    if (!response.ok || data.error) {
      showUploadResult('error', data.error || 'Upload failed.');
      return;
    }

    currentDocId = data.doc_id;
    updateDocStatus(true, data.doc_id);
    showUploadResult('success',
      `✅ Document ingested! <strong>${data.chunk_count} chunks</strong> indexed.<br/>
       <span style="font-size:12px;opacity:0.8;">Preview: "${data.preview.slice(0, 120)}..."</span>`
    );

    // Auto-suggest going to chat
    setTimeout(() => {
      showUploadResult('success',
        `✅ Ready! ${data.chunk_count} chunks indexed. <a href="#" onclick="switchTab('chat');return false;" style="color:inherit;text-decoration:underline;">Go to Chat →</a>`
      );
    }, 1500);

  } catch (err) {
    showUploadResult('error', 'Network error: ' + err.message);
  } finally {
    btnText.textContent = '⚡ Ingest Document';
    btn.disabled = false;
  }
}

function showUploadResult(type, html) {
  const el = document.getElementById('uploadResult');
  el.className = 'upload-result ' + type;
  el.innerHTML = html;
  el.style.display = 'block';
}

/* ── Doc Status ── */
function updateDocStatus(active, docId) {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  if (active) {
    dot.className = 'status-dot active';
    text.textContent = 'Paper loaded';
  } else {
    dot.className = 'status-dot inactive';
    text.textContent = 'No paper loaded';
  }
}

/* ── Chat ── */
function chatKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function sendSuggestion(el) {
  document.getElementById('chatInput').value = el.textContent;
  sendMessage();
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const question = input.value.trim();
  if (!question) return;

  appendChatBubble('user', question);
  chatHistory.push({ role: 'user', content: question });
  input.value = '';

  const loadingId = appendLoadingBubble();

  try {
    const response = await fetch('/chat/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        doc_id: currentDocId,
        history: chatHistory.slice(-6),
      }),
    });

    const data = await response.json();
    removeLoadingBubble(loadingId);

    if (data.error) {
      appendChatBubble('assistant', `⚠️ ${data.error}`);
    } else {
      appendChatBubble('assistant', data.answer);
      chatHistory.push({ role: 'assistant', content: data.answer });
    }
  } catch (err) {
    removeLoadingBubble(loadingId);
    appendChatBubble('assistant', '⚠️ Network error. Please try again.');
  }
}

function appendChatBubble(role, content) {
  const messages = document.getElementById('chatMessages');
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  if (role === 'assistant') {
    bubble.innerHTML = marked.parse(content);
    // Highlight code blocks
    bubble.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
  } else {
    bubble.textContent = content;
  }
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function appendLoadingBubble() {
  const messages = document.getElementById('chatMessages');
  const bubble = document.createElement('div');
  const id = 'loading-' + Date.now();
  bubble.id = id;
  bubble.className = 'chat-bubble loading';
  bubble.innerHTML = '<span class="spinner" style="border-color:rgba(139,146,184,0.3);border-top-color:#8b92b8;"></span> Thinking...';
  messages.appendChild(bubble);
  messages.scrollTop = messages.scrollHeight;
  return id;
}

function removeLoadingBubble(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

/* ── Summary ── */
async function generateSummary() {
  const btn = document.querySelector('#tab-summary .btn-primary');
  const btnText = document.getElementById('summaryBtnText');
  const output = document.getElementById('summaryOutput');

  btnText.innerHTML = '<span class="spinner"></span> Generating...';
  btn.disabled = true;
  output.style.display = 'none';

  try {
    const response = await fetch('/summary/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id: currentDocId }),
    });

    const data = await response.json();

    if (data.error) {
      output.innerHTML = `<p style="color:var(--danger)">⚠️ ${data.error}</p>`;
    } else {
      output.innerHTML = marked.parse(data.summary);
      output.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
      document.getElementById('copyBtn').style.display = 'inline-flex';
    }
    output.style.display = 'block';
  } catch (err) {
    output.innerHTML = `<p style="color:var(--danger)">⚠️ Network error: ${err.message}</p>`;
    output.style.display = 'block';
  } finally {
    btnText.textContent = '✨ Generate Summary';
    btn.disabled = false;
  }
}

function copySummary() {
  const output = document.getElementById('summaryOutput');
  navigator.clipboard.writeText(output.innerText).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✅ Copied!';
    setTimeout(() => btn.textContent = '📋 Copy', 2000);
  });
}

/* ── Concept Explainer ── */
function quickExplain(el) {
  document.getElementById('conceptInput').value = el.textContent;
  explainConcept();
}

async function explainConcept() {
  const input = document.getElementById('conceptInput');
  const concept = input.value.trim();
  if (!concept) return;

  const btn = document.querySelector('#tab-explain .btn-primary');
  const btnText = document.getElementById('explainBtnText');
  const output = document.getElementById('explainOutput');

  btnText.innerHTML = '<span class="spinner"></span> Explaining...';
  btn.disabled = true;
  output.style.display = 'none';

  try {
    const response = await fetch('/explain/concept', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ concept, doc_id: currentDocId }),
    });

    const data = await response.json();

    if (data.error) {
      output.innerHTML = `<p style="color:var(--danger)">⚠️ ${data.error}</p>`;
    } else {
      output.innerHTML = marked.parse(data.explanation);
      output.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
    }
    output.style.display = 'block';
  } catch (err) {
    output.innerHTML = `<p style="color:var(--danger)">⚠️ Network error: ${err.message}</p>`;
    output.style.display = 'block';
  } finally {
    btnText.textContent = '💡 Explain';
    btn.disabled = false;
  }
}

/* ── Init ── */
marked.setOptions({
  gfm: true,
  breaks: true,
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  }
});
