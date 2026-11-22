/* One video clock synchronizes the two examples, input transitions, and camera UI. */
(() => {
  const card = document.querySelector('.method-card');
  if (!card) return;
  const video = card.querySelector('#hero-denoising'), flow = card.querySelector('.method-flow');
  const typed = card.querySelector('.prompt-typed'), fullPrompt = typed.textContent;
  const cursor = card.querySelector('.typing-cursor');
  const textPanel = card.querySelector('.hero-text-panel'), cameraPanel = card.querySelector('.hero-camera-panel');
  const captions = card.querySelector('.hero-region-captions'), outputLabel = card.querySelector('.output-head .stage-label');
  const existing = card.querySelector('.existing-module'), router = card.querySelector('.router-module'), input = card.querySelector('.input-window');
  const motion = window.HERO_WORLD_MOTION;
  const keys = [...cameraPanel.querySelectorAll('.camera-key')], dot = cameraPanel.querySelector('.stick-dot'), stem = cameraPanel.querySelector('.stick-stem');
  const svg = card.querySelector('.flow-line'), path = svg.querySelector('.flow-connection'), pattern = svg.querySelector('pattern');
  let visible = false, raf = 0;
  const ease = x => {x=Math.max(0,Math.min(1,x));return x*x*(3-2*x);};
  // The card is stacked bottom to top (input → Existing Method → Router → video). One centred arrow runs up
  // through the gaps between them and ends with its tip on the bottom edge of the video.
  function placeLine() {
    const box = flow.getBoundingClientRect(), y = v => v - box.top;
    const from = input.getBoundingClientRect(), a = existing.getBoundingClientRect(), b = router.getBoundingClientRect(), output = video.getBoundingClientRect();
    svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
    const x=(output.left+output.right)/2-box.left;
    path.setAttribute('d',`M ${x} ${y(from.top)} L ${x} ${y(a.bottom)} M ${x} ${y(a.top)} L ${x} ${y(b.bottom)} M ${x} ${y(b.top)} L ${x} ${y(output.bottom)+2}`);
  }
  function panel(element,opacity,offset){
    element.style.opacity=opacity;
    element.style.transform=`translateX(${offset}px)`;
    element.style.visibility=opacity>0?'visible':'hidden';
    element.setAttribute('aria-hidden',String(opacity===0));
  }
  function update() {
    const clock=video.currentTime;
    const climberExplanation=clock>=3.2&&clock<6.2,worldExplanation=clock>=14.4&&clock<17.4;
    const explaining=climberExplanation||worldExplanation;
    // Collapse each three-second hold onto its completed first denoising frame.
    const t=clock<3.2?clock:clock<6.2?3.2:clock<14.4?clock-3:clock<17.4?11.4:clock-6;
    const elapsed=climberExplanation?clock-3.2:worldExplanation?clock-14.4:0;
    const explanationAlpha=explaining?Math.min(ease(elapsed/.45),ease((3-elapsed)/.45)):0;
    panel(captions,explanationAlpha,0);
    if(outputLabel)outputLabel.style.opacity=1-explanationAlpha;
    // Input panels switch horizontally: the outgoing panel slides out to the left, the incoming one enters from the right.
    const shift=48;
    let textAlpha=0,textX=0,cameraAlpha=0,cameraX=0;
    if(t<9)textAlpha=1;
    else if(t<9.4){const u=ease((t-9)/.4);textAlpha=1-u;textX=-shift*u;}
    else if(t<9.9){const u=ease((t-9.4)/.5);cameraAlpha=u;cameraX=shift*(1-u);}
    else if(t<20.9)cameraAlpha=1;
    else if(t<21.3){const u=ease((t-20.9)/.4);cameraAlpha=1-u;cameraX=-shift*u;}
    else{const u=ease((t-21.3)/.5);textAlpha=u;textX=shift*(1-u);}
    panel(textPanel,textAlpha,textX);panel(cameraPanel,cameraAlpha,cameraX);
    const textTime=t<9.4?t:0;
    typed.textContent=fullPrompt.slice(0,Math.min(fullPrompt.length,Math.floor(Math.max(0,textTime-.1)/.85*fullPrompt.length)));
    cursor.style.visibility=textTime<1.15?'visible':'hidden';
    card.dataset.example=t>=9.4&&t<21.3?'world':'video';
    card.dataset.phase=explaining?'explaining':t<1.2||(t>=9.4&&t<11.4)?'routing':'denoising';
    // negative x shift moves the 45° stripes upward along the vertical arrow (towards the video)
    pattern.setAttribute('patternTransform',`translate(${-((t*24)%12)} 0) rotate(45)`);
    if(motion){
      // start frame and playback rate come from hero-world-motion.js (the video plays the source from `start` at `rate` frames/s)
      const last=motion.keys.length-1,pos=Math.min(last,(motion.start??13)+Math.max(0,t-11.4)*(motion.rate||16)),i=Math.floor(pos),j=Math.min(last,i+1),f=pos-i;
      keys.forEach((key,k)=>key.classList.toggle('active',Boolean(motion.keys[i][k])));
      const x=181+21*(motion.stick[i][0]*(1-f)+motion.stick[j][0]*f),y=43+21*(motion.stick[i][1]*(1-f)+motion.stick[j][1]*f);
      dot.setAttribute('cx',x);dot.setAttribute('cy',y);stem.setAttribute('x2',x);stem.setAttribute('y2',y);
    }
  }
  function frame(){update();if(!video.paused)raf=requestAnimationFrame(frame);}
  function reflect(){card.classList.toggle('is-paused',video.paused);cancelAnimationFrame(raf);if(!video.paused)frame();}
  function resume(){if(visible&&!document.hidden)video.play().catch(reflect);}
  video.addEventListener('play',reflect);video.addEventListener('pause',reflect);
  video.addEventListener('seeked',update);video.addEventListener('loadedmetadata',()=>{placeLine();resume();});
  video.addEventListener('error',()=>{typed.textContent=fullPrompt;cursor.style.visibility='hidden';video.pause();});
  new ResizeObserver(placeLine).observe(flow);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)resume();else video.pause();},{threshold:.12}).observe(card);
  document.addEventListener('visibilitychange',()=>document.hidden?video.pause():resume());
  placeLine();update();reflect();
})();
