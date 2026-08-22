(function () {
  'use strict';

  var duration = 90;
  var stages = [
    { start: 8, end: 16, number: '01', type: 'OFFLINE / INGEST', title: 'Start with the<br /><em>knowledge source.</em>', description: 'The indexer reads MSMARCO-XI in batches, keeps usable passage and query pairs, and prepares the corpus before any user request arrives.', tags: ['MSMARCO-XI', 'BATCHES', 'DATASET'], operation: 'Dataset ingest', latency: 'BATCHED', status: 'Corpus ready for indexing', caption: 'Offline build · raw knowledge enters', node: 0 },
    { start: 16, end: 27, number: '02', type: 'OFFLINE / CHUNK', title: 'Split with<br /><em>context intact.</em>', description: 'The multi-tier chunking engine supports atomic passages, sentence-aware sliding windows with overlap, metadata augmentation, and query anchors.', tags: ['ATOMIC', '256 / 64', 'METADATA', 'ANCHORS'], operation: 'Multi-tier chunking', latency: 'OFFLINE', status: 'Retrieval units created', caption: 'Offline build · chunks preserve meaning', node: 1 },
    { start: 27, end: 39, number: '03', type: 'OFFLINE / EMBED', title: 'Turn chunks into<br /><em>searchable signals.</em>', description: 'BGE-M3 encodes every prepared chunk into normalized vectors. Query-anchor vectors are created too, so intent has its own retrieval track.', tags: ['BGE-M3', '1024-D', 'NORMALIZED', 'QUERY ANCHOR'], operation: 'Dense embedding generation', latency: 'OFFLINE', status: 'Vectors computed and aligned', caption: 'Offline build · meaning becomes a vector', node: 2 },
    { start: 39, end: 50, number: '04', type: 'OFFLINE / STORE', title: 'Build once.<br /><em>Serve many times.</em>', description: 'FAISS HNSW stores dense vectors, BM25s stores lexical terms, and metadata keeps every result traceable back to its source chunk.', tags: ['FAISS HNSW', 'BM25S', 'METADATA'], operation: 'Index construction & persistence', latency: 'CHECKPOINT', status: 'Artifacts saved to disk', caption: 'Offline build · indices checkpointed', node: 3 },
    { start: 50, end: 58, number: '05', type: 'ONLINE / READY', title: 'The server wakes<br /><em>with memory.</em>', description: 'At startup, the orchestrator loads the saved FAISS, BM25s, query-anchor indices, metadata, and the embedding model into the serving process.', tags: ['LOAD', 'WARMUP', 'READY'], operation: 'Index loading & warmup', latency: 'STARTUP', status: 'Online pipeline ready', caption: 'Online path · persisted knowledge is live', node: 4 },
    { start: 58, end: 69, number: '06', type: 'ONLINE / GUARD', title: 'A request arrives.<br /><em>Trust comes first.</em>', description: 'Voice is transcribed by Sarvam STT. The input guardrail checks empty or low-confidence audio, prompt injection, unsafe content, and off-topic patterns.', tags: ['VOICE', 'SARVAM STT', 'SAFETY'], operation: 'STT + input guardrail', latency: '<85 MS', status: 'Request cleared to search', caption: 'Online path · safe query acquired', node: 1 },
    { start: 69, end: 80, number: '07', type: 'ONLINE / RETRIEVE', title: 'Search in parallel.<br /><em>Fuse with intent.</em>', description: 'One query embedding fans out to passage FAISS, query-anchor FAISS, and BM25s concurrently. Tri-track RRF ranks and deduplicates the best context.', tags: ['PARALLEL', '3 TRACKS', 'RRF K=60'], operation: 'Hybrid retrieval + fusion', latency: '<30 MS', status: 'Evidence shortlist ranked', caption: 'Online path · three signals become one', node: 2 },
    { start: 80, end: 90, number: '08', type: 'ONLINE / ANSWER', title: 'Generate.<br /><em>Then prove it.</em>', description: 'The harness calls the configured LLM with structured context. The output guardrail checks grounding, detects hallucination, and abstains when evidence is weak.', tags: ['LLM', 'GROUNDING', 'ABSTAIN'], operation: 'Generation + output guardrail', latency: '<200 MS', status: 'Grounded answer delivered', caption: 'Online path · answer verified before delivery', node: 4 }
  ];
  var film = document.querySelector('.process-film');
  var stageMap = document.querySelector('.stage-map');
  var scenes = document.querySelectorAll('.scene');
  var title = document.getElementById('stage-title');
  var description = document.getElementById('stage-description');
  var tags = document.getElementById('stage-tags');
  var number = document.querySelector('.stage-number');
  var type = document.querySelector('.stage-type');
  var operation = document.getElementById('detail-operation');
  var latency = document.getElementById('stage-latency');
  var status = document.getElementById('detail-status-text');
  var caption = document.getElementById('node-caption-text');
  var progress = document.getElementById('timeline-progress');
  var scrubber = document.getElementById('timeline-scrubber');
  var current = document.getElementById('time-current');
  var total = document.getElementById('time-total');
  var play = document.getElementById('play-toggle');
  var nodes = document.querySelectorAll('.stage-node');
  var elapsed = 0;
  var playing = true;
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var last = performance.now();
  var storyboardDuration = 90;

  function formatTime(value) { var seconds = Math.max(0, Math.floor(value)); var minutes = Math.floor(seconds / 60); var remainder = seconds % 60; return String(minutes).padStart(2, '0') + ':' + String(remainder).padStart(2, '0'); }
  function activateScene(time) { Array.prototype.forEach.call(scenes, function (scene) { scene.classList.toggle('active', time >= Number(scene.dataset.start) && time < Number(scene.dataset.end)); }); }
  function updateStage(time) {
    var item = stages.find(function (stage) { return time >= stage.start && time < stage.end; });
    if (!item) return;
    number.textContent = item.number; type.textContent = item.type; title.innerHTML = item.title; description.textContent = item.description; operation.textContent = item.operation; latency.textContent = item.latency; status.textContent = item.status; caption.textContent = item.caption;
    tags.innerHTML = item.tags.map(function (tag) { return '<span>' + tag + '</span>'; }).join('');
    Array.prototype.forEach.call(nodes, function (node) { node.classList.toggle('active', Number(node.dataset.node) === item.node); });
    var stageIndex = stages.indexOf(item);
    stageMap.classList.toggle('right-animation-layout', stageIndex >= 1);
  }
  function render(time) { elapsed = Math.max(0, Math.min(duration, time)); var storyboardTime = elapsed / duration * storyboardDuration; activateScene(storyboardTime); updateStage(storyboardTime); progress.style.width = (elapsed / duration * 100) + '%'; scrubber.style.left = (elapsed / duration * 100) + '%'; current.textContent = formatTime(elapsed); total.textContent = formatTime(duration); film.setAttribute('data-reduced-motion', reduced ? 'true' : 'false'); }
  function tick(now) { var delta = (now - last) / 1000; last = now; if (playing && !reduced) { render(elapsed + delta); if (elapsed >= duration) { playing = false; render(duration); play.textContent = '▶'; } } requestAnimationFrame(tick); }
  function seek(event) { var rect = event.currentTarget.getBoundingClientRect(); render((event.clientX - rect.left) / rect.width * duration); }

  play.addEventListener('click', function () { playing = !playing; play.textContent = playing ? 'Ⅱ' : '▶'; last = performance.now(); });
  document.getElementById('replay').addEventListener('click', function () { render(0); playing = true; play.textContent = 'Ⅱ'; last = performance.now(); });
  document.getElementById('motion-toggle').addEventListener('click', function (event) { reduced = !reduced; event.currentTarget.classList.toggle('active', reduced); if (reduced) render(elapsed); });
  document.querySelector('.timeline').addEventListener('click', seek);
  document.querySelector('.timeline').addEventListener('keydown', function (event) { if (event.key === 'ArrowRight') render(elapsed + 5); if (event.key === 'ArrowLeft') render(elapsed - 5); });
  render(0); requestAnimationFrame(tick);
})();