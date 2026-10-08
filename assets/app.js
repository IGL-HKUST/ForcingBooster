/* Paired playback: one transport controls both the preview and its schedule. */
(() => {
  'use strict';
  const players = [];
  const cameraMarkup = `<div class="camera-controller" role="img" aria-label="Camera movement and look direction"><svg viewBox="0 0 240 88" aria-hidden="true">
    <rect x="2" y="2" width="236" height="84" rx="14" fill="#f8faf9" stroke="#aab7b0" stroke-width="1.5" stroke-dasharray="6 4"/>
    ${[['W',65,12],['A',35,43],['S',65,43],['D',95,43]].map(([key,x,y])=>`<g class="camera-key" data-key="${key}"><rect x="${x}" y="${y}" width="25" height="26" rx="4"/><text x="${x+12.5}" y="${y+17}" text-anchor="middle">${key}</text></g>`).join('')}
    <circle cx="181" cy="43" r="27" fill="none" stroke="#adc0c9"/><circle cx="181" cy="43" r="18" fill="none" stroke="#d8e2e4"/>
    <path d="M181 12v7m0 48v7M150 43h7m48 0h7" fill="none" stroke="#adc0c9"/>
    <line class="stick-stem" x1="181" y1="43" x2="181" y2="43" stroke="#008d91" stroke-width="3" stroke-linecap="round"/>
    <circle class="stick-dot" cx="181" cy="43" r="6" fill="#008d91"/>
  </svg></div>`;
  const stepLegend = '<span class="schedule-legend" aria-label="Denoising steps"><span>Steps</span>' + [1,2,3,4].map(n => `<span aria-label="${n} denoising ${n === 1 ? 'step' : 'steps'}"><i class="step${n}" aria-hidden="true"></i>${n}</span>`).join('') + '</span>';
  const format = t => `0:${String(Math.floor(t || 0)).padStart(2,'0')}`;
  for (const item of window.EXAMPLES || []) {
    const card = document.createElement('article');
    card.className = 'example'; card.id = item.id; card.setAttribute('aria-label', item.title);
    card.innerHTML = `      <div class="video-pair">${['preview','overlay'].map(kind => `<figure><figcaption>${kind === 'preview' ? '<span>Preview</span>' : '<span>Denoising Schedule</span>' + stepLegend}</figcaption><video muted playsinline preload="none" poster="${item[kind+'Poster']}" data-src="${item[kind]}" aria-label="${item.title}: ${kind === 'preview' ? 'generated video' : 'denoising schedule'}"></video></figure>`).join('')}</div>
      <div class="controls"><button class="play-toggle" type="button" aria-label="Play ${item.title}" aria-pressed="false">▶ Play</button><input class="seek" type="range" min="0" max="1000" value="0" step="1" aria-label="Seek ${item.title}"><span class="time">0:00 / 0:00</span><button class="restart" type="button" aria-label="Restart ${item.title}">↺ Restart</button></div>
      ${item.category === 'world' ? cameraMarkup : ''}<p class="media-status" role="status" hidden></p>`;
    document.querySelector(`[data-gallery="${item.category}"]`).append(card);
    const videos = [...card.querySelectorAll('video')], [master, follower] = videos;
    const toggle = card.querySelector('.play-toggle'), seek = card.querySelector('.seek'), clock = card.querySelector('.time'), status = card.querySelector('.media-status');
    const motion = (window.CAMERA_MOTION || {})[item.id];
    const animateCamera = () => {
      if (!motion) return;
      const position = Math.min(motion.keys.length - 1, master.currentTime * motion.fps);
      const i = Math.floor(position), j = Math.min(i + 1, motion.keys.length - 1), blend = position - i;
      card.querySelectorAll('.camera-key').forEach((key,k) => key.classList.toggle('active', Boolean(motion.keys[i][k])));
      const x = 181 + 21 * (motion.stick[i][0] * (1-blend) + motion.stick[j][0] * blend);
      const y = 43 + 21 * (motion.stick[i][1] * (1-blend) + motion.stick[j][1] * blend);
      const dot = card.querySelector('.stick-dot'), stem = card.querySelector('.stick-stem');
      dot.setAttribute('cx',x);dot.setAttribute('cy',y);stem.setAttribute('x2',x);stem.setAttribute('y2',y);
    };
    animateCamera();
    let playing = false, loaded = false, frame = 0, operation = 0;
    const duration = () => Math.min(...videos.map(v => Number.isFinite(v.duration) ? v.duration : Infinity));
    const load = () => { if (loaded) return; loaded = true; videos.forEach(v => {v.src = v.dataset.src; v.load();}); };
    const update = () => { animateCamera(); const d = duration(); if (Number.isFinite(d)) { seek.value = d ? Math.round(master.currentTime / d * 1000) : 0; clock.textContent = `${format(master.currentTime)} / ${format(d)}`; } };
    const button = () => { toggle.textContent = playing ? 'Ⅱ Pause' : '▶ Play'; toggle.setAttribute('aria-pressed',String(playing)); toggle.setAttribute('aria-label',`${playing ? 'Pause' : 'Play'} ${item.title}`); };
    const pause = () => { operation++; playing = false; videos.forEach(v => v.pause()); cancelAnimationFrame(frame); button(); };
    const setTime = t => { videos.forEach(v => { if (v.readyState) v.currentTime = t; }); update(); };
    const ready = v => v.readyState >= 3 ? Promise.resolve() : new Promise((resolve,reject) => {
      const timer = setTimeout(() => done(new Error('timeout')),20000);
      function done(error) { clearTimeout(timer); v.removeEventListener('canplay',ok); v.removeEventListener('error',bad); error ? reject(error) : resolve(); }
      const ok = () => done(), bad = () => done(new Error('media'));
      v.addEventListener('canplay',ok,{once:true});v.addEventListener('error',bad,{once:true});
    });
    const tick = () => {
      if (!playing) return;
      if (Math.abs(master.currentTime - follower.currentTime) > .09 && !follower.seeking) follower.currentTime = master.currentTime;
      if (master.currentTime >= duration() - .03) { setTime(0); videos.forEach(v => v.play().catch(pause)); }
      update(); frame = requestAnimationFrame(tick);
    };
    const play = async () => {
      players.forEach(p => {if (p.card !== card) p.pause();});
      load(); playing = true; button(); status.hidden = true;
      const token = ++operation;
      try {
        await Promise.all(videos.map(ready));
        if (token !== operation) return;
        if (master.currentTime >= duration() - .06) setTime(0);
        follower.currentTime = master.currentTime;
        await Promise.all(videos.map(v => v.play()));
        if (token !== operation) { videos.forEach(v => v.pause()); return; }
        cancelAnimationFrame(frame); tick();
      } catch (e) { if (token !== operation) return; pause(); status.textContent = 'Playback could not start. Please try Play again.';status.hidden = false; }
    };
    videos.forEach(v => {
      v.addEventListener('loadedmetadata',update);
      v.addEventListener('error',() => {pause();status.textContent='This video could not load. Please refresh to try again.';status.hidden=false;});
      v.addEventListener('ended',() => {if (playing) {setTime(0); videos.forEach(x => x.play().catch(pause));}});
    });
    toggle.addEventListener('click',() => playing ? pause() : play());
    card.querySelector('.restart').addEventListener('click',() => {pause();setTime(0);play();});
    seek.addEventListener('input',() => {load();const d = duration();if(Number.isFinite(d)) setTime(Number(seek.value)/1000*d);});
    players.push({card,pause,load});
  }
  const observer = new IntersectionObserver(entries => entries.forEach(entry => {
    const player = players.find(p => p.card === entry.target);
    if (entry.isIntersecting) player.load(); else player.pause();
  }),{rootMargin:'200px 0px'});
  players.forEach(p => observer.observe(p.card));
  document.addEventListener('visibilitychange',() => {if (document.hidden) players.forEach(p => p.pause());});
})();
