
// site_info.json is written by the HW3 notebook (feature names, label maps, an example stock)
let siteInfo = null;

async function loadSiteInfo() {
  try {
    const resp = await fetch("./site_info.json?v=" + Date.now());
    siteInfo = await resp.json();
  } catch (e) {
    console.error("Could not load site_info.json:", e);
    alert("Error: could not load site_info.json");
    return;
  }

  // Sector and Industry drop-downs (the model expects their integer IDs)
  const sector = document.getElementById('sector');
  siteInfo.sector_names.forEach((name, i) => sector.add(new Option(name, i)));

  const industry = document.getElementById('industry');
  siteInfo.industry_names.forEach((name, i) => industry.add(new Option(name, i)));

  // one text box for every numeric feature, pre-filled with an example test stock
  const ex = siteInfo.example_values;
  let html = "";
  for (let i = 2; i < siteInfo.feature_columns.length; i++) {
    html += `<label>${siteInfo.feature_columns[i]}</label>
             <input type="text" id="box${i}" value="${ex[i].toFixed(4)}"><br/>`;
  }
  document.getElementById('numericInputs').innerHTML = html;

  sector.value   = Math.round(ex[0]);
  industry.value = Math.round(ex[1]);

  document.getElementById('example').innerHTML =
    `<p>Pre-filled with test stock <b>${siteInfo.example_symbol}</b></p>`;
}


function readInputs() {
  const n = siteInfo.feature_columns.length;
  const x = new Float32Array(n);
  x[0] = parseFloat(document.getElementById('sector').value)   || 0;
  x[1] = parseFloat(document.getElementById('industry').value) || 0;
  for (let i = 2; i < n; i++) {
    x[i] = parseFloat(document.getElementById('box' + i).value) || 0;
  }
  return new ort.Tensor('float32', x, [1, n]);
}


function renderTable(divId, probs) {
  let best = 0;
  for (let k = 1; k < probs.length; k++) { if (probs[k] > probs[best]) best = k; }

  let rows = "";
  for (let k = 0; k < probs.length; k++) {
    const cls = (k === best) ? ' class="best"' : '';
    rows += `<tr${cls}><td>${siteInfo.strategy_names[k]}</td><td>${(probs[k] * 100).toFixed(1)}%</td></tr>`;
  }

  document.getElementById(divId).innerHTML = `
    <hr>Predicted strategy: <b>${siteInfo.strategy_names[best]}</b><br/><br/>
    <table>
      <tr><th>Strategy</th><th>Probability</th></tr>
      ${rows}
    </table>`;
}


async function runDL() {
  const tensorX = readInputs();
  try {
    const session = await ort.InferenceSession.create("./DLnet_StockStrategy.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    renderTable('predictionsDL', results.output1.data);   // softmax probabilities, length 3
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}


async function runXGB() {
  const tensorX = readInputs();
  try {
    const session = await ort.InferenceSession.create("./xgboost_StockStrategy_ort.onnx?v=" + Date.now());
    const results = await session.run({ float_input: tensorX });
    renderTable('predictionsXGB', results.probabilities.data);
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}


async function runBoth() {
  await runDL();
  await runXGB();
}
