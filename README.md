# HW3 – Stock Strategy Classifier (ONNX on GitHub Pages)

Predicts the best of three investment strategies (DCA_Buy_Hold, Drawdown60, Year_Sept_April)
for a stock from its sector, industry and risk/return ratios.
The page outputs an **integer class**: 0 = DCA_Buy_Hold, 1 = Drawdown60, 2 = Year_Sept_April.

Files:
- `index.html`, `index.js` – web page, runs the models in the browser with onnxruntime-web
- `DLnet_StockStrategy.onnx` – PyTorch deep learning model (2 hidden layers), input `input1` [1,10], output `output1`; the page converts it to an integer class with argmax (like `torch.max(y_pred, dim=1)`)
- `xgboost_StockStrategy_ort.onnx` – XGBoost model, input `float_input` [1,10], the page uses the integer `label` output
- `site_info.json` – feature names, Sector/Industry/strategy label maps, example stock
- `HW3_classification.ipynb` – full training / evaluation notebook
