/* One video clock synchronizes the two examples, input transitions, and camera UI. */
(() => {
  const card = document.querySelector('.method-card');
  if (!card) return;
  const video = card.querySelector('#hero-denoising'), heatmap = card.querySelector('#hero-heatmap'), flow = card.querySelector('.method-flow');
  const comparison = card.querySelector('.hero-comparison'), slider = card.querySelector('.comparison-slider');
  const typed = card.querySelector('.prompt-typed'), fullPrompt = typed.textContent;
  const cursor = card.querySelector('.typing-cursor');
  const textPanel = card.querySelector('.hero-text-panel'), cameraPanel = card.querySelector('.hero-camera-panel');
  const motion = window.HERO_WORLD_MOTION;
  const keys = [...cameraPanel.querySelectorAll('.camera-key')], dot = cameraPanel.querySelector('.stick-dot'), stem = cameraPanel.querySelector('.stick-stem');
  const svg = card.querySelector('.flow-line'), path = svg.querySelector('.flow-connection'), pattern = svg.querySelector('pattern');
  let visible = false, raf = 0;
  const ease = x => {x=Math.max(0,Math.min(1,x));return x*x*(3-2*x);};
  function setSplit(){
    const value=Number(slider.value);
    comparison.style.setProperty('--split',`${value}%`);
    slider.setAttribute('aria-valuetext',`Preview ${value}%, heatmap ${100-value}%`);
  }
  slider.addEventListener('input',setSplit);setSplit();
  function syncHeatmap(force=false){
    if(heatmap.readyState<1)return;
    if(force||(!heatmap.seeking&&Math.abs(heatmap.currentTime-video.currentTime)>.06))heatmap.currentTime=video.currentTime;
  }
  function playHeatmap(){if(!video.paused)heatmap.play().catch(()=>{});}
  function placeLine() {
    const box = flow.getBoundingClientRect(), input = card.querySelector('.input-window').getBoundingClientRect(), output = video.getBoundingClientRect();
    svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
    if(window.innerWidth>900){
      const y=(input.top+input.bottom)/2-box.top;
      path.setAttribute('d',`M ${input.right-box.left} ${y} L ${output.left-box.left-2} ${y}`);
    }else{
      const x=input.left-box.left+8;
      path.setAttribute('d',`M ${x} ${input.bottom-box.top} L ${x} ${output.top-box.top-2}`);
    }
  }
  function panel(element,opacity,offset){
    element.style.opacity=opacity;
    element.style.transform=`translateY(${offset}px)`;
    element.style.visibility=opacity>0?'visible':'hidden';
    element.setAttribute('aria-hidden',String(opacity===0));
  }
  function update() {
    syncHeatmap();
    const t=video.currentTime;
    let textAlpha=0,textY=0,cameraAlpha=0,cameraY=0;
    if(t<9)textAlpha=1;
    else if(t<9.4){const u=ease((t-9)/.4);textAlpha=1-u;textY=-28*u;}
    else if(t<9.9){const u=ease((t-9.4)/.5);cameraAlpha=u;cameraY=28*(1-u);}
    else if(t<20.9)cameraAlpha=1;
    else if(t<21.3){const u=ease((t-20.9)/.4);cameraAlpha=1-u;cameraY=-28*u;}
    else{const u=ease((t-21.3)/.5);textAlpha=u;textY=28*(1-u);}
    panel(textPanel,textAlpha,textY);panel(cameraPanel,cameraAlpha,cameraY);
    const textTime=t<9.4?t:0;
    typed.textContent=fullPrompt.slice(0,Math.min(fullPrompt.length,Math.floor(Math.max(0,textTime-.1)/.85*fullPrompt.length)));
    cursor.style.visibility=textTime<1.15?'visible':'hidden';
    card.dataset.example=t>=9.4&&t<21.3?'world':'video';
    card.dataset.phase=t<1.2||(t>=9.4&&t<11.4)?'routing':'denoising';
    pattern.setAttribute('patternTransform',`translate(${(t*24)%12} 0) rotate(45)`);
    if(motion){
      const pos=Math.min(156,13+Math.max(0,t-11.4)*16),i=Math.floor(pos),j=Math.min(156,i+1),f=pos-i;
      keys.forEach((key,k)=>key.classList.toggle('active',Boolean(motion.keys[i][k])));
      const x=181+21*(motion.stick[i][0]*(1-f)+motion.stick[j][0]*f),y=43+21*(motion.stick[i][1]*(1-f)+motion.stick[j][1]*f);
      dot.setAttribute('cx',x);dot.setAttribute('cy',y);stem.setAttribute('x2',x);stem.setAttribute('y2',y);
    }
  }
  function frame(){update();if(!video.paused)raf=requestAnimationFrame(frame);}
  function reflect(){
    card.classList.toggle('is-paused',video.paused);cancelAnimationFrame(raf);
    if(video.paused)heatmap.pause();
    else{syncHeatmap(true);playHeatmap();frame();}
  }
  function resume(){if(visible&&!document.hidden)video.play().catch(reflect);}
  video.addEventListener('play',reflect);video.addEventListener('pause',reflect);
  video.addEventListener('seeking',()=>syncHeatmap(true));
  video.addEventListener('seeked',()=>{syncHeatmap(true);update();});
  video.addEventListener('waiting',()=>heatmap.pause());video.addEventListener('playing',playHeatmap);
  heatmap.addEventListener('loadedmetadata',()=>{syncHeatmap(true);playHeatmap();});
  video.addEventListener('loadedmetadata',()=>{placeLine();resume();});
  video.addEventListener('error',()=>{typed.textContent=fullPrompt;cursor.style.visibility='hidden';video.pause();});
  new ResizeObserver(placeLine).observe(flow);
  new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)resume();else video.pause();},{threshold:.12}).observe(card);
  document.addEventListener('visibilitychange',()=>document.hidden?video.pause():resume());
  placeLine();update();reflect();
})();
