/* Page chrome and the two example players (interactive generation, world models). */
(() => {
  'use strict';
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = t => { t = Math.max(0, t || 0); return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`; };

  /* ---------------- chrome ---------------- */
  const nav = $('#nav');
  const darkSections = $$('.dark');
  const onScroll = () => {
    nav.classList.toggle('is-scrolled', window.scrollY > 8);
    const y = nav.offsetHeight / 2;
    nav.classList.toggle('is-dark', darkSections.some(s => { const r = s.getBoundingClientRect(); return r.top <= y && r.bottom > y; }));
  };
  window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  const toggle = $('.nav-toggle'), links = $('#nav-links');
  toggle.addEventListener('click', () => {
    const open = toggle.getAttribute('aria-expanded') !== 'true';
    toggle.setAttribute('aria-expanded', String(open)); links.classList.toggle('is-open', open);
  });
  links.addEventListener('click', e => { if (e.target.closest('a')) { toggle.setAttribute('aria-expanded', 'false'); links.classList.remove('is-open'); } });

  // each link lists the section ids it covers (Demo covers both galleries)
  const navLinks = $$('a', links);
  const covers = a => (a.dataset.sections || a.hash.slice(1)).split(/\s+/);
  const sectionSpy = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) navLinks.forEach(a => a.classList.toggle('is-active', covers(a).includes(entry.target.id)));
  }), { rootMargin: '-45% 0px -50% 0px' });
  [...new Set(navLinks.flatMap(covers))].map(id => document.getElementById(id)).filter(Boolean).forEach(s => sectionSpy.observe(s));

  const reveal = new IntersectionObserver(entries => entries.forEach(entry => {
    if (entry.isIntersecting) { entry.target.classList.add('is-in'); reveal.unobserve(entry.target); }
  }), { rootMargin: '0px 0px -8% 0px', threshold: .08 });
  $$('.reveal').forEach(el => reveal.observe(el));

  /* ---------------- How it works: step overview (video + copy, timeline of steps below) ---------------- */
  const overview = $('#how-steps');
  if (overview) {
    const panels = $$('.method-step', overview), tabs = $$('.method-tab', overview), timeline = $('.method-timeline', overview);
    const clips = panels.map(p => $('video', p)), stepsBox = $('.method-steps', overview);
    let index = 0, inView = false, started = false, raf = 0;
    let manualPaused = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const sync = v => {
      const box = v.closest('.method-visual'), btn = $('.method-toggle', box);
      box.classList.toggle('is-paused', v.paused);
      btn.setAttribute('aria-pressed', String(v.paused));
      btn.setAttribute('aria-label', `${v.paused ? 'Play' : 'Pause'} ${v.getAttribute('aria-label')}`);
    };
    const warm = v => { if (v.preload !== 'auto') { v.preload = 'auto'; v.load(); } };
    const progress = () => {
      const v = clips[index];
      if (v.duration) tabs[index].style.setProperty('--progress', Math.min(1, v.currentTime / v.duration));
      if (!v.paused) raf = requestAnimationFrame(progress);
    };
    // play only when at least 45% of the clip is on screen (below the 64 px bar)
    const enoughOnScreen = () => {
      const r = clips[index].getBoundingClientRect();
      const shown = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 64));
      return document.fullscreenElement || shown / Math.max(1, Math.min(r.height, innerHeight - 64)) >= .45;
    };
    const stopAll = () => clips.forEach(v => v.pause());
    const play = () => {
      const v = clips[index];
      if (manualPaused || document.hidden || !inView || !enoughOnScreen()) { if (!v.paused) v.pause(); return; }
      warm(v);
      if (v.paused) v.play().then(() => { cancelAnimationFrame(raf); progress(); }).catch(() => {});
    };
    const reveal = i => {
      const r = tabs[i].getBoundingClientRect(), box = timeline.getBoundingClientRect();
      if (r.left < box.left) timeline.scrollLeft += r.left - box.left; else if (r.right > box.right) timeline.scrollLeft += r.right - box.right;
    };
    function select(i, { focus = false } = {}) {
      stopAll();
      index = (i + panels.length) % panels.length;
      panels.forEach((p, k) => { p.hidden = k !== index; p.inert = k !== index; });
      tabs.forEach((t, k) => { t.setAttribute('aria-selected', String(k === index)); t.tabIndex = k === index ? 0 : -1; t.style.setProperty('--progress', k < index ? 1 : 0); });
      const v = clips[index];
      warm(v);
      if (v.readyState) v.currentTime = 0;
      warm(clips[(index + 1) % clips.length]);  // keep the next step ready
      if (focus) tabs[index].focus({ preventScroll: true });
      reveal(index); clips.forEach(sync); play();
    }
    clips.forEach((v, k) => {
      v.addEventListener('play', () => sync(v));
      v.addEventListener('pause', () => sync(v));
      // after the last step the overview starts again from the first one
      v.addEventListener('ended', () => { if (k === index && !manualPaused && !document.fullscreenElement) select(index + 1); else sync(v); });
      const box = v.closest('.method-visual');
      $('.method-toggle', box).addEventListener('click', () => {
        if (v.paused) { manualPaused = false; if (v.ended) v.currentTime = 0; play(); } else { manualPaused = true; stopAll(); }
      });
      $('.method-fullscreen', box).addEventListener('click', async () => {
        try {
          if (document.fullscreenElement) await document.exitFullscreen();
          else if (box.requestFullscreen) await box.requestFullscreen();
          else if (v.webkitEnterFullscreen) v.webkitEnterFullscreen();
        } catch { v.controls = true; }
      });
    });
    tabs.forEach((t, k) => {
      t.addEventListener('click', () => { manualPaused = false; select(k); });
      t.addEventListener('keydown', e => {
        const next = { ArrowRight: k + 1, ArrowLeft: k - 1, Home: 0, End: tabs.length - 1 }[e.key];
        if (next === undefined) return;
        e.preventDefault(); manualPaused = false; select(next, { focus: true });
      });
    });
    new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting;
      if (!inView) { stopAll(); return; }
      if (!started) { started = true; select(0); } else play();
    }, { threshold: [0, .25, .5, .75, 1] }).observe(stepsBox);
    addEventListener('scroll', () => { if (inView) play(); }, { passive: true });
    addEventListener('resize', () => reveal(index));
    document.addEventListener('visibilitychange', () => (document.hidden ? stopAll() : play()));
    document.addEventListener('fullscreenchange', play);
  }

  $$('.copy').forEach(btn => btn.addEventListener('click', async () => {
    const text = $(btn.dataset.copy).textContent;
    try { await navigator.clipboard.writeText(text); btn.textContent = 'Copied'; }
    catch { const r = document.createRange(); r.selectNodeContents($(btn.dataset.copy)); const s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = 'Selected'; }
    setTimeout(() => { btn.textContent = 'Copy'; }, 1600);
  }));

  /* ---------------- players ---------------- */
  const EXAMPLES = window.EXAMPLES || [], CAMERA = window.CAMERA_MOTION || {};
  const STEP_COLORS = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)'];
  const ICON_PLAY = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M3 1.5v11l9.5-5.5z"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 14 14" aria-hidden="true"><rect x="2.5" y="1.5" width="3.2" height="11" rx="1"/><rect x="8.3" y="1.5" width="3.2" height="11" rx="1"/></svg>';
  const cameraSVG = () => `<svg viewBox="0 0 240 88" aria-hidden="true">${[['W', 65, 12], ['A', 35, 43], ['S', 65, 43], ['D', 95, 43]].map(([k, x, y]) =>
    `<g class="camera-key"><rect x="${x}" y="${y}" width="25" height="26" rx="6"/><text x="${x + 12.5}" y="${y + 17.5}" text-anchor="middle">${k}</text></g>`).join('')}
    <circle class="stick-ring" cx="181" cy="43" r="27"/><circle class="stick-ring inner" cx="181" cy="43" r="18"/>
    <line class="stick-stem" x1="181" y1="43" x2="181" y2="43"/><circle class="stick-dot" cx="181" cy="43" r="6"/></svg>`;
  const players = [];

  function buildPlayer(root) {
    const category = root.dataset.category;
    const items = EXAMPLES.filter(x => x.category === category);
    if (!items.length) return;
    root.innerHTML = `
      <div class="player-main">
      <div class="viewer">
        <div class="compare">
          <video class="layer-main" muted playsinline preload="none"></video>
          <video class="layer-alt layer-overlay is-shown" muted playsinline preload="none"></video>
          <span class="compare-tag left">ForcingBooster</span><span class="compare-tag right">Step map<span class="tag-legend" aria-label="denoising steps per patch: green 1, blue 2, orange 3, red 4">${[1, 2, 3, 4].map(n => `<i style="background:${STEP_COLORS[n - 1]}"></i>${n}`).join('')}</span></span>
          <input class="split" type="range" min="0" max="100" value="50" step="1" aria-label="Divider position">
          <span class="divider" aria-hidden="true"><span class="knob">‹ ›</span></span>
          <span class="spinner" aria-hidden="true"></span>
          <p class="media-status" role="status" hidden></p>
        </div>
        <div class="transport">
          <button class="play" type="button" aria-label="Play">${ICON_PLAY}</button>
          <input class="seek" type="range" min="0" max="1000" value="0" step="1" aria-label="Seek">
          <span class="time">0:00 / 0:00</span>
          <div class="chunk-stats" title="Current chunk. Speedup = 4 ÷ mean steps: the four-step baseline's denoising steps divided by this chunk's average (denoising steps only, not wall-clock time).">
            <span class="cs-item"><span class="cs-label">Mean steps</span><strong class="cs-mean"></strong></span>
            <span class="cs-item"><span class="cs-label">Speedup</span><strong class="cs-speed"></strong></span>
          </div>
        </div>
      </div>
      <div class="details">
        <div class="details-head"><h3 class="details-title"></h3><span class="details-kicker"></span></div>
        <div class="details-body">
          <div class="ex-prompt">
            <p class="ex-label">Prompt</p>
            <p class="ex-prompt-text"></p>
            <button class="ex-more" type="button" aria-expanded="false" hidden>Show more</button>
          </div>
          <div class="camera-box" hidden><p class="ex-label">Camera</p><div class="camera-widget camera-widget-dark" role="img" aria-label="Camera input synchronized with the video">${cameraSVG()}</div></div>
        </div>
      </div>
      </div>
      <div class="player-side">
        <div class="rail" role="list" aria-label="Choose an example">${items.map(x => `
          <button class="thumb" type="button" role="listitem" data-id="${esc(x.id)}" aria-label="${esc(x.title)}${category === 'world' ? ', ' + esc(x.motion || '') : ''}">
            <span class="thumb-img"><img src="${esc(x.previewPoster)}" alt="" loading="lazy" decoding="async"></span>
            <span class="thumb-title">${esc(x.title)}</span>${category === 'world' ? `<span class="thumb-sub">${esc(x.motion || '')}</span>` : ''}
          </button>`).join('')}</div>
      </div>`;

    const viewer = $('.viewer', root), compare = $('.compare', root);
    const main = $('.layer-main', root), overlay = $('.layer-overlay', root), alt = overlay;
    const split = $('.split', root);
    const playBtn = $('.play', root), seek = $('.seek', root), clock = $('.time', root), status = $('.media-status', root);
    const stats = $('.chunk-stats', root), csMean = $('.cs-mean', root), csSpeed = $('.cs-speed', root);
    const kicker = $('.details-kicker', root), title = $('.details-title', root);
    const promptBox = $('.ex-prompt', root), promptText = $('.ex-prompt-text', root), moreBtn = $('.ex-more', root);
    const camBox = $('.camera-box', root), body = $('.details-body', root);
    moreBtn.addEventListener('click', () => {
      const open = moreBtn.getAttribute('aria-expanded') !== 'true';
      moreBtn.setAttribute('aria-expanded', String(open)); promptText.classList.toggle('is-open', open);
      moreBtn.textContent = open ? 'Show less' : 'Show more';
    });
    const fitPrompt = () => {
      promptText.classList.remove('is-open'); moreBtn.setAttribute('aria-expanded', 'false'); moreBtn.textContent = 'Show more';
      moreBtn.hidden = promptText.scrollHeight <= promptText.clientHeight + 2;
    };
    new ResizeObserver(() => { if (!promptText.classList.contains('is-open')) moreBtn.hidden = promptText.scrollHeight <= promptText.clientHeight + 2; }).observe(promptText);
    const camKeys = $$('.camera-key', camBox), camDot = $('.stick-dot', camBox), camStem = $('.stick-stem', camBox);
    const thumbs = $$('.thumb', root);

    let current = null, playing = false, userPaused = false, visible = false, raf = 0, op = 0;

    split.addEventListener('input', () => {
      compare.style.setProperty('--split', split.value + '%');
      split.setAttribute('aria-valuetext', `${split.value}% ForcingBooster, ${100 - split.value}% step map`);
    });

    const duration = () => (Number.isFinite(main.duration) ? main.duration : NaN);

    /* current chunk: mean denoising steps and speedup (4 / mean), where the comparison switch used to be */
    let shownChunk = -1;
    const totalFrames = () => (current && current.chunkFrames ? current.chunkFrames[current.chunkFrames.length - 1][1] + 1 : 0);
    function drawChart() {
      if (!current || !current.chunkSteps) return;
      const d = duration(), n = totalFrames();
      const f = Number.isFinite(d) && d > 0 ? Math.min(n - 1, Math.floor(main.currentTime / d * n)) : 0;
      const c = Math.max(0, current.chunkFrames.findIndex(([a, b]) => f >= a && f <= b));
      if (c === shownChunk) return;
      shownChunk = c;
      const mean = current.chunkSteps[c];
      csMean.textContent = mean.toFixed(2);
      csSpeed.textContent = `${(4 / mean).toFixed(2)}×`;
    }
    function buildChart() {
      shownChunk = -1;
      stats.hidden = !(current && current.chunkSteps && current.chunkSteps.length);
      drawChart();
    }
    const drawCamera = () => {
      const m = current && CAMERA[current.id];
      if (!m) return;
      const last = m.keys.length - 1;
      const pos = Math.min(last, Math.max(0, main.currentTime * m.fps)), i = Math.floor(pos), j = Math.min(last, i + 1), f = pos - i;
      camKeys.forEach((key, k) => key.classList.toggle('active', Boolean(m.keys[i][k])));
      const x = 181 + 21 * (m.stick[i][0] * (1 - f) + m.stick[j][0] * f), y = 43 + 21 * (m.stick[i][1] * (1 - f) + m.stick[j][1] * f);
      camDot.setAttribute('cx', x); camDot.setAttribute('cy', y); camStem.setAttribute('x2', x); camStem.setAttribute('y2', y);
    };
    const drawTime = () => {
      const d = duration();
      if (Number.isFinite(d) && d > 0) {
        const p = main.currentTime / d;
        seek.value = Math.round(p * 1000); seek.style.setProperty('--p', (p * 100).toFixed(2) + '%');
        clock.textContent = `${fmt(main.currentTime)} / ${fmt(d)}`;
      }
      drawCamera(); drawChart();
    };
    const setButton = () => { playBtn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY; playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play'); };
    const tick = () => {
      if (!playing) return;
      if (!alt.seeking && Math.abs(alt.currentTime - main.currentTime) > .06) alt.currentTime = main.currentTime;
      drawTime(); raf = requestAnimationFrame(tick);
    };
    const ready = v => (v.readyState >= 3 ? Promise.resolve() : new Promise((resolve, reject) => {
      const timer = setTimeout(() => done(new Error('timeout')), 20000);
      const ok = () => done(), bad = () => done(new Error('media'));
      function done(err) { clearTimeout(timer); v.removeEventListener('canplay', ok); v.removeEventListener('error', bad); err ? reject(err) : resolve(); }
      v.addEventListener('canplay', ok, { once: true }); v.addEventListener('error', bad, { once: true });
    }));
    const ensureSrc = v => { if (v.dataset.src && v.getAttribute('src') !== v.dataset.src) { v.src = v.dataset.src; v.load(); } };

    function pause() { op++; playing = false; [main, overlay].forEach(v => v.pause()); cancelAnimationFrame(raf); viewer.classList.remove('is-loading'); setButton(); }
    async function play() {
      players.forEach(p => { if (p.root !== root) p.pause(); });
      const token = ++op; playing = true; setButton(); status.hidden = true;
      ensureSrc(main); ensureSrc(alt);
      viewer.classList.add('is-loading');
      try {
        await Promise.all([ready(main), ready(alt)]);
        if (token !== op) return;
        alt.currentTime = main.currentTime;
        await Promise.all([main.play(), alt.play()]);
        if (token !== op) { main.pause(); alt.pause(); return; }
        viewer.classList.remove('is-loading');
        cancelAnimationFrame(raf); tick();
      } catch (err) {
        if (token !== op) return;
        pause(); status.textContent = 'This video could not start. Press play to try again.'; status.hidden = false;
      }
    }

    function select(id, { autoplay = true } = {}) {
      const item = items.find(x => x.id === id);
      if (!item || item === current) return;
      pause();
      current = item;
      thumbs.forEach(t => t.setAttribute('aria-current', String(t.dataset.id === id)));
      const set = (v, src, poster) => { v.removeAttribute('src'); v.dataset.src = src || ''; v.poster = poster || ''; v.load(); };
      set(main, item.preview, item.previewPoster);
      set(overlay, item.overlay, item.overlayPoster);
      buildChart();
      seek.value = 0; seek.style.setProperty('--p', '0%'); clock.textContent = '0:00 / 0:00';
      status.hidden = true;

      kicker.textContent = category === 'world' ? (item.motion || '') : '';
      title.textContent = item.title;
      promptBox.hidden = !item.prompt; promptText.textContent = item.prompt || '';
      fitPrompt();
      camBox.hidden = !CAMERA[item.id];
      body.classList.toggle('has-camera', !camBox.hidden);
      drawCamera();
      if (autoplay && visible && !userPaused) play();
    }

    main.addEventListener('loadedmetadata', () => { if (main.videoWidth) compare.style.aspectRatio = `${main.videoWidth} / ${main.videoHeight}`; drawTime(); });
    main.addEventListener('ended', () => { if (playing) { main.currentTime = 0; alt.currentTime = 0; main.play().catch(pause); alt.play().catch(() => {}); } });
    [main, overlay].forEach(v => v.addEventListener('error', () => {
      if (!v.getAttribute('src') || (v !== main && v !== alt)) return;
      pause(); status.textContent = 'This video could not load. Please refresh and try again.'; status.hidden = false;
    }));
    playBtn.addEventListener('click', () => { if (playing) { userPaused = true; pause(); } else { userPaused = false; play(); } });
    seek.addEventListener('input', () => {
      ensureSrc(main); ensureSrc(alt);
      const d = duration(); if (!Number.isFinite(d)) return;
      const t = Number(seek.value) / 1000 * d; main.currentTime = t; if (alt.readyState) alt.currentTime = t; drawTime();
    });
    thumbs.forEach(t => t.addEventListener('click', () => {
      userPaused = false; select(t.dataset.id);
      // on a single-column layout the player sits above the list; bring it back into view
      if (getComputedStyle(root).gridTemplateColumns.split(' ').length < 2) viewer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }));

    new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) { ensureSrc(main); ensureSrc(alt); if (!playing && !userPaused) play(); }
      else if (playing) pause();
    }, { threshold: .35 }).observe(viewer);

    document.addEventListener('visibilitychange', () => { if (!document.hidden && visible && !userPaused && !playing) play(); });
    players.push({ root, pause: () => { if (playing) pause(); } });
    select(items[0].id, { autoplay: false });
  }

  $$('.player').forEach(buildPlayer);
  document.addEventListener('visibilitychange', () => { if (document.hidden) players.forEach(p => p.pause()); });
})();
