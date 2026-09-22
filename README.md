
# EGX Smart Stock Assistant

Mobile-first Streamlit app for Egyptian Exchange (EGX) stock analysis.

## What it does
- Enter an EGX ticker such as `HBCO`, `COMI`, `SWDY`.
- Automatically resolves it to Yahoo Finance's `.CA` format.
- Pulls market history.
- Calculates RSI, MACD, ADX, Stochastic, MFI, OBV, ATR, Bollinger Bands, EMA/SMA and volume signals.
- Produces a transparent Smart Score and explains the drivers.
- Estimates support/resistance, staged entry zones, stop-loss and 3 targets.
- Shows available fundamentals and an indicative fair-value range.
- Searches current news through Google News RSS.
- Responsive UI designed for phone screens.

## Important data limitation
EGX data availability varies by symbol and provider. Yahoo Finance may be delayed, incomplete, or temporarily unavailable for some Egyptian securities. The app does not pretend missing data is accurate.

The Fair Value model is deliberately simple and transparent: when EPS/BPS are available it uses indicative P/E and P/B assumptions. It is not a guaranteed target price.

## Deploy to mobile
The app is a web app, so it runs in Safari/Chrome on iPhone or Android. Streamlit Community Cloud can host it and gives it a `streamlit.app` URL.

1. Create a GitHub repository.
2. Upload `app.py` and `requirements.txt`.
3. Open https://share.streamlit.io
4. Sign in and choose the repository and `app.py`.
5. Deploy.
6. Open the generated URL on the phone and add it to the Home Screen.

No API key is required for the default version.

## Local test
```bash
pip install -r requirements.txt
streamlit run app.py
```
