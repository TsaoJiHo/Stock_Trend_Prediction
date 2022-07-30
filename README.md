# Taiwan Stock Trend Prediction Using Deep Learning with Technical Indicators

## Requirements :

`pip install -r requirements.txt`

[ta-lib installation](https://markdownlivepreview.com/)



## Training :

Follow the steps in `training.ipynb` before `Testing` block, finally will get trained models.

## Application :

Apply trained models in recent days' data to show results.

Set `{"LOCAL_MODE": true}` in `config.json` and command

`streamlit run application.py`

The application will be like [this](https://tsaojiho-stock-trend-prediction-application-qe9dmo.streamlitapp.com/). (May take some time to run, suggest to run on local side)
