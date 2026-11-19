
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


// same idea as the Iris example:  vals, indeces = torch.max( y_pred, dim=1 )
// -> the index of the largest network output is the predicted INTEGER class
function argmax(arr) {
  let best = 0;
  for (let k = 1; k < arr.length; k++) { if (arr[k] > arr[best]) best = k; }
  return best;
}


function renderClass(divId, predClass) {
  let rows = "";
  for (let k = 0; k < siteInfo.strategy_names.length; k++) {
    const cls = (k === predClass) ? ' class="best"' : '';
    rows += `<tr${cls}><td>${k}</td><td>${siteInfo.strategy_names[k]}</td></tr>`;
  }

  document.getElementById(divId).innerHTML = `
    <hr>Predicted class (integer): <b style="font-size:1.6em">${predClass}</b><br/>
    Strategy: <b>${siteInfo.strategy_names[predClass]}</b><br/><br/>
    <table>
      <tr><th>Class</th><th>Strategy</th></tr>
      ${rows}
    </table>`;
}


async function runDL() {
  const tensorX = readInputs();
  try {
    const session = await ort.InferenceSession.create("./DLnet_StockStrategy.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const predClass = argmax(results.output1.data);        // integer class 0, 1 or 2
    renderClass('predictionsDL', predClass);
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
    const predClass = Number(results.label.data[0]);       // XGBoost 'label' output is already an integer class
    renderClass('predictionsXGB', predClass);
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }
}


async function runBoth() {
  await runDL();
  await runXGB();
}
