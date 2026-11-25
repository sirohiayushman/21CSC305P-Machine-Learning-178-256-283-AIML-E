const sectionsNode=document.getElementById('formSections');
const form=document.getElementById('screeningForm');
const predictButton=document.getElementById('predictBtn');
let modelReady=false;
let questionnaire=[];

function el(tag,props={},children=[]){
  const node=document.createElement(tag);
  for(const [key,value] of Object.entries(props)){
    if(key==='className')node.className=value;
    else if(key==='textContent')node.textContent=value;
    else node.setAttribute(key,String(value));
  }
  for(const child of children)node.append(child);
  return node;
}
function notify(message){
  document.querySelectorAll('.error-toast').forEach(n=>n.remove());
  const box=el('div',{className:'error-toast',role:'alert',textContent:message});
  document.body.append(box);
  window.setTimeout(()=>box.remove(),8500);
}
function renderSections(data){
  sectionsNode.replaceChildren(); questionnaire=data;
  data.forEach((section,i)=>{
    const details=el('details',{className:'form-section'});
    if(i===0||i===1) details.open=true;
    const summary=el('summary');
    const left=el('span',{className:'left-title'});
    left.append(el('span',{className:'section-icon',textContent:['♙','♡','✚','◈'][i]||'◇'}),
                document.createTextNode(section.title));
    summary.append(left,el('span',{className:'chevron',textContent:'⌄'}));
    details.append(summary);
    const grid=el('div',{className:'field-grid'});
    section.fields.forEach(field=>{
      const block=el('div',{className:'field'});
      const label=el('label',{for:'f_'+field.name,textContent:field.label});
      if(field.required)label.append(el('span',{className:'star',textContent:' *'}));
      let input;
      if(field.type==='number'){
        input=el('input',{id:'f_'+field.name,name:field.name,type:'number',step:'any',placeholder:'Enter value (optional)'});
        if(field.min!==null)input.min=String(field.min);
        if(field.max!==null)input.max=String(field.max);
      }else{
        input=el('select',{id:'f_'+field.name,name:field.name});
        input.append(el('option',{value:'',textContent:'Select an option (or unknown)'}));
        field.options.forEach(opt=>input.append(el('option',{value:opt,textContent:opt})));
        if(field.name==='State'&&field.options.length===0){
          input.append(el('option',{value:'',textContent:'Available after training'}));
          input.disabled=true;
        }
      }
      if(field.required)input.required=true;
      block.append(label,input); grid.append(block);
    });
    details.append(grid);sectionsNode.append(details);
  });
}
function updateModel(meta){
  document.getElementById('selectedModel').textContent=meta.model_name+' classifier';
  document.getElementById('selectedVariant').textContent=meta.variant.replaceAll('_',' ');
  document.getElementById('featureCount').textContent=meta.engineered_features;
  document.getElementById('selectedFeatures').textContent=meta.input_features;
  const high=document.getElementById('highRecallOption');
  if(!Object.prototype.hasOwnProperty.call(meta.thresholds,'high_recall')){
    high.classList.add('hidden');
  }else high.classList.remove('hidden');
}
async function initialize(){
  try{
    const response=await fetch('/api/schema');if(!response.ok)throw new Error('Could not load schema');
    const data=await response.json();modelReady=data.model_ready;
    renderSections(data.sections);
    predictButton.disabled=!modelReady;
    const status=document.getElementById('modelStatus');
    status.replaceChildren(el('span',{className:'status-dot'+(modelReady?' ready':'')}),
      document.createTextNode(modelReady?' Model ready for local inference':' Model unavailable — see README'));
    if(modelReady)updateModel(data.model);
  }catch(error){sectionsNode.textContent='Could not load local questionnaire.';notify(error.message);}
}
form.addEventListener('submit',async event=>{
  event.preventDefault();
  if(!modelReady){notify('Prepare the model first using train.py or import_notebook.py.');return;}
  const answers={};
  questionnaire.forEach(s=>s.fields.forEach(f=>{
    const input=form.elements.namedItem(f.name);
    if(input&&input.value!=='')answers[f.name]=f.type==='number'?Number(input.value):input.value;
  }));
  const missingRequired=questionnaire.flatMap(s=>s.fields).filter(f=>f.required&&!Object.hasOwn(answers,f.name));
  if(missingRequired.length){notify('Complete the required fields: '+missingRequired.map(f=>f.label).join(', '));
    const field=form.elements.namedItem(missingRequired[0].name);
    if(field){field.closest('details').open=true;field.focus();}return;}
  const mode=document.querySelector('input[name="mode"]:checked').value;
  predictButton.disabled=true;
  predictButton.querySelector('span').textContent='Processing…';
  try{
    const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({answers,mode})});
    const result=await response.json();
    if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:JSON.stringify(result.detail));
    const score=(result.screening_score*100).toFixed(1);
    const isFlagged=result.flagged_for_follow_up;
    const panel=document.getElementById('resultPanel');panel.classList.remove('hidden');
    document.getElementById('scoreRing').style.setProperty('--progress',score+'%');
    document.getElementById('scoreValue').textContent=score+'%';
    const status=document.getElementById('resultStatus');
    status.textContent=isFlagged?'FOLLOW-UP FLAG RAISED':'NO FLAG AT THIS THRESHOLD';
    status.className='result-status'+(isFlagged?' flag':'');
    document.getElementById('resultHeadline').textContent=isFlagged?'Consider further medical evaluation':'Do not interpret this as medical clearance';
    document.getElementById('resultExplanation').textContent=result.message+' This is a calibrated score for the survey-history label, not a probability of a future attack.';
    document.getElementById('resultThreshold').textContent=(result.threshold*100).toFixed(2)+'%';
    document.getElementById('resultInputs').textContent=result.input_fields_provided+' / '+result.input_fields_total;
    const missing=document.getElementById('missingNotice');
    if(result.unanswered_fields.length){missing.classList.remove('hidden');
      missing.textContent=result.unanswered_fields.length+' survey questions were left blank and handled as missing. Consider completing them for a more meaningful demonstration.';
    }else missing.classList.add('hidden');
    panel.scrollIntoView({behavior:'smooth',block:'start'});
  }catch(error){notify(error.message)}finally{
    predictButton.disabled=!modelReady;
    predictButton.querySelector('span').textContent='Run screening';
  }
});
form.addEventListener('reset',()=>{document.getElementById('resultPanel').classList.add('hidden');});
initialize();
