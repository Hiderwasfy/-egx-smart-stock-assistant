
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import re
import json
import requests
import feedparser
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import plotly.graph_objects as go

st.set_page_config(
    page_title="EGX Smart Stock Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Mobile-first styling
# ----------------------------
st.markdown("""
<style>
.block-container {padding: 1rem 0.8rem 3rem 0.8rem; max-width: 1200px;}
h1 {font-size: 1.8rem !important;}
h2 {font-size: 1.35rem !important;}
h3 {font-size: 1.1rem !important;}
.metric-card {
    padding: 14px; border-radius: 16px; border: 1px solid rgba(128,128,128,.18);
    background: rgba(128,128,128,.06); margin-bottom: 8px;
}
.score {
    font-size: 2.2rem; font-weight: 800; line-height: 1.0;
}
.small {font-size: .82rem; opacity: .72;}
.good {color:#0a9f55;}
.bad {color:#d83b3b;}
.warn {color:#c88900;}
.newsbox {padding:10px 0; border-bottom:1px solid rgba(128,128,128,.18);}
div[data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>
""", unsafe_allow_html=True)

COMMON = {
    "COMI":"COMI.CA","SWDY":"SWDY.CA","EFIH":"EFIH.CA","EAST":"EAST.CA",
    "FWRY":"FWRY.CA","TMGH":"TMGH.CA","ORAS":"ORAS.CA","ETEL":"ETEL.CA",
    "ABUK":"ABUK.CA","MFPC":"MFPC.CA","JUFO":"JUFO.CA","CIEB":"CIEB.CA",
    "HDBK":"HDBK.CA","ADIB":"ADIB.CA","SAUD":"SAUD.CA","QNBA":"QNBA.CA",
    "FAIT":"FAIT.CA","HELI":"HELI.CA","OCDI":"OCDI.CA","MNHD":"MNHD.CA",
    "PHDC":"PHDC.CA","EMFD":"EMFD.CA","TAQA":"TAQA.CA","CCAP":"CCAP.CA",
    "SKPC":"SKPC.CA","ORWE":"ORWE.CA","ARCC":"ARCC.CA","ALCN":"ALCN.CA",
    "HDBO":"HDBO.CA","HBCO":"HBCO.CA",
}

def resolve_symbol(raw):
    s = raw.strip().upper().replace(" ", "")
    if s in COMMON:
        return COMMON[s]
    if s.endswith(".CA"):
        return s
    return s + ".CA"

@st.cache_data(ttl=300, show_spinner=False)
def _investing_search(symbol):
    """Resolve an EGX ticker to an Investing.com pair id.
    This is a fallback provider; Yahoo remains the first provider when available.
    """
    q = symbol.replace(".CA", "").upper()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        r = requests.post(
            "https://www.investing.com/search/service/searchTopBar",
            data={"search_text": q}, headers=headers, timeout=12
        )
        if r.ok:
            data = r.json()
            for item in data.get("quotes", []):
                exch = str(item.get("exchange", "")).upper()
                sym = str(item.get("symbol", "")).upper()
                if sym == q and ("CAIRO" in exch or "EGX" in exch or "EGYPT" in exch):
                    return item
            for item in data.get("quotes", []):
                if str(item.get("symbol", "")).upper() == q:
                    return item
    except Exception:
        pass
    return None


def _clean_num(v):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, str):
            v = v.replace(",", "").replace("%", "").strip()
            # Investing may return volume strings such as 3.15M / 501.09K
            m = re.fullmatch(r"([-+]?\d*\.?\d+)\s*([KMB])?", v, flags=re.I)
            if m:
                x = float(m.group(1)); u = (m.group(2) or "").upper()
                return x * {"K":1e3,"M":1e6,"B":1e9}.get(u,1)
        return float(v)
    except Exception:
        return np.nan


def _parse_investing_json(payload):
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return pd.DataFrame()
    out=[]
    for r in rows:
        if not isinstance(r, dict):
            continue
        date = r.get("date") or r.get("rowDate") or r.get("rowDateRaw") or r.get("timestamp")
        o = r.get("price_open", r.get("open", r.get("rowOpen")))
        h = r.get("price_high", r.get("high", r.get("rowHigh")))
        l = r.get("price_low", r.get("low", r.get("rowLow")))
        c = r.get("price_close", r.get("close", r.get("rowClose")))
        v = r.get("volume", r.get("rowVolume", r.get("volumeRaw")))
        if date is None or c is None:
            continue
        out.append({"Date":date,"Open":_clean_num(o),"High":_clean_num(h),"Low":_clean_num(l),"Close":_clean_num(c),"Volume":_clean_num(v)})
    if not out:
        return pd.DataFrame()
    df=pd.DataFrame(out)
    df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
    for c in ["Open","High","Low","Close","Volume"]:
        df[c]=pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Date","Close"]).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def _get_investing_history(symbol, period="2y"):
    found = _investing_search(symbol)
    if not found:
        return pd.DataFrame()
    pair_id = found.get("pairId") or found.get("pair_id")
    if not pair_id:
        return pd.DataFrame()
    days = {"1y":370,"2y":740,"5y":1850}.get(period,740)
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/128 Safari/537.36",
        "Accept":"application/json, text/plain, */*",
        "domain-id":"www",
        "Referer":"https://www.investing.com/",
    }
    # Current Investing historical-data JSON endpoint.
    try:
        url=f"https://api.investing.com/api/financialdata/historical/{int(pair_id)}"
        params={"start-date":start.isoformat(),"end-date":end.isoformat(),"time-frame":"Daily","add-missing-rows":"false"}
        r=requests.get(url,params=params,headers=headers,timeout=15)
        if r.ok:
            df=_parse_investing_json(r.json())
            if len(df)>=60:
                return df
    except Exception:
        pass
    # Legacy browser endpoint fallback.
    try:
        page_url="https://www.investing.com" + str(found.get("url", ""))
        sess=requests.Session()
        sess.headers.update(headers)
        page=sess.get(page_url,timeout=15)
        payload={
            "curr_id":str(pair_id), "smlID":str(np.random.randint(1000000,99999999)),
            "header":str(found.get("description") or symbol)+" Historical Data",
            "st_date":start.strftime("%m/%d/%Y"), "end_date":end.strftime("%m/%d/%Y"),
            "interval_sec":"Daily", "sort_col":"date", "sort_ord":"DESC", "action":"historical_data"
        }
        h={"User-Agent":headers["User-Agent"],"X-Requested-With":"XMLHttpRequest","Referer":page_url}
        r=sess.post("https://www.investing.com/instruments/HistoricalDataAjax",data=payload,headers=h,timeout=15)
        if r.ok:
            tables=pd.read_html(r.text)
            if tables:
                t=tables[0]
                rename={"Date":"Date","Price":"Close","Open":"Open","High":"High","Low":"Low","Vol.":"Volume"}
                t=t.rename(columns=rename)
                needed=[c for c in ["Date","Open","High","Low","Close","Volume"] if c in t.columns]
                t=t[needed]
                t["Date"]=pd.to_datetime(t["Date"],errors="coerce")
                for c in ["Open","High","Low","Close"]:
                    if c in t: t[c]=t[c].map(_clean_num)
                if "Volume" in t: t["Volume"]=t["Volume"].map(_clean_num)
                return t.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def _egx_api_history(symbol, period="2y"):
    code=symbol.replace(".CA","").upper()
    base="https://ticker.egidegypt.com/api/Feed"
    headers={"User-Agent":"Mozilla/5.0","Accept":"application/json, text/plain, */*"}
    days={"1y":370,"2y":740,"5y":1850}.get(period,740)
    end=datetime.utcnow().date(); start=end-timedelta(days=days)
    def norm(p):
        if isinstance(p,dict):
            for k in ("data","result","items","history","prices","bars","values"):
                if k in p:
                    z=norm(p[k])
                    if not z.empty: return z
            if all(k in p for k in ("t","o","h","l","c")):
                rows=[]
                for i,t in enumerate(p.get("t",[])):
                    try: rows.append({"Date":pd.to_datetime(float(t),unit="s"),"Open":_clean_num(p["o"][i]),"High":_clean_num(p["h"][i]),"Low":_clean_num(p["l"][i]),"Close":_clean_num(p["c"][i]),"Volume":_clean_num(p.get("v",[np.nan]*len(p["t"]))[i])})
                    except Exception: pass
                return pd.DataFrame(rows)
            return pd.DataFrame()
        if isinstance(p,list):
            rows=[]
            for r in p:
                if isinstance(r,dict):
                    d=r.get("date") or r.get("Date") or r.get("tradingDate") or r.get("time") or r.get("timestamp")
                    o=r.get("open") or r.get("Open") or r.get("openPrice"); h=r.get("high") or r.get("High") or r.get("highPrice"); l=r.get("low") or r.get("Low") or r.get("lowPrice"); c=r.get("close") or r.get("Close") or r.get("closePrice") or r.get("price"); v=r.get("volume") or r.get("Volume") or r.get("volumeValue")
                    if d is not None and c is not None: rows.append({"Date":d,"Open":_clean_num(o),"High":_clean_num(h),"Low":_clean_num(l),"Close":_clean_num(c),"Volume":_clean_num(v)})
                elif isinstance(r,(list,tuple)) and len(r)>=5:
                    d=r[0]
                    if isinstance(d,(int,float)): d=pd.to_datetime(float(d),unit="s" if float(d)<2e10 else "ms")
                    rows.append({"Date":d,"Open":_clean_num(r[1]),"High":_clean_num(r[2]),"Low":_clean_num(r[3]),"Close":_clean_num(r[4]),"Volume":_clean_num(r[5]) if len(r)>5 else np.nan})
            if rows:
                df=pd.DataFrame(rows); df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
                for c in ["Open","High","Low","Close","Volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
                return df.dropna(subset=["Date","Close"]).sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
        return pd.DataFrame()
    attempts=[("GET",f"{base}/GetSymbolChart",{"symbol":code}),("GET",f"{base}/GetSymbolChart",{"symbolName":code}),("GET",f"{base}/GetSymbolHistories",{"symbol":code}),("GET",f"{base}/GetSymbolHistories",{"symbolName":code}),("POST",f"{base}/GetSymbolHistory",{"symbol":code,"fromDate":start.isoformat(),"toDate":end.isoformat()}),("POST",f"{base}/GetSymbolHistory",{"symbolName":code,"fromDate":start.isoformat(),"toDate":end.isoformat()})]
    for method,url,payload in attempts:
        try:
            r=requests.get(url,params=payload,headers=headers,timeout=12) if method=="GET" else requests.post(url,json=payload,headers=headers,timeout=12)
            if r.ok:
                df=norm(r.json())
                if len(df)>=60: return df
        except Exception: pass
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def get_history(symbol, period="2y"):
    df=_egx_api_history(symbol,period)
    if df is not None and len(df)>=60: return df
    try:
        t=yf.Ticker(symbol); df=t.history(period=period,interval="1d",auto_adjust=False)
        if df is not None and not df.empty:
            df=df.reset_index(); df.columns=[str(c).strip().title() for c in df.columns]
            if "Datetime" in df.columns: df.rename(columns={"Datetime":"Date"},inplace=True)
            df["Date"]=pd.to_datetime(df["Date"],errors="coerce"); df=df.dropna(subset=["Date"]).copy()
            if len(df)>=60: return df[[c for c in ["Date","Open","High","Low","Close","Volume"] if c in df.columns]]
    except Exception: pass
    return _get_investing_history(symbol,period)

@st.cache_data(ttl=900, show_spinner=False)
def get_info(symbol):
    try:
        info = yf.Ticker(symbol).info
        if isinstance(info, dict) and info:
            return info
    except Exception:
        pass
    return {}

def sma(s, n): return s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False).mean()
    ad = down.ewm(alpha=1/n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - (100/(1+rs))

def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"]-df["Low"],
        (df["High"]-pc).abs(),
        (df["Low"]-pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def macd(s):
    m12 = ema(s, 12)
    m26 = ema(s, 26)
    line = m12 - m26
    sig = ema(line, 9)
    return line, sig, line-sig

def stochastic(df, n=14, d=3):
    lo = df["Low"].rolling(n).min()
    hi = df["High"].rolling(n).max()
    k = 100*(df["Close"]-lo)/(hi-lo).replace(0,np.nan)
    return k, k.rolling(d).mean()

def bollinger(s, n=20, k=2):
    mid=sma(s,n); sd=s.rolling(n).std()
    return mid, mid+k*sd, mid-k*sd

def adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([
        df["High"]-df["Low"],
        (df["High"]-df["Close"].shift()).abs(),
        (df["Low"]-df["Close"].shift()).abs()
    ],axis=1).max(axis=1)
    atrv = tr.rolling(n).mean()
    pdi = 100*pd.Series(plus,index=df.index).rolling(n).mean()/atrv.replace(0,np.nan)
    mdi = 100*pd.Series(minus,index=df.index).rolling(n).mean()/atrv.replace(0,np.nan)
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.rolling(n).mean()

def add_indicators(df):
    x=df.copy()
    c=x["Close"]
    x["SMA20"]=sma(c,20); x["SMA50"]=sma(c,50); x["SMA200"]=sma(c,200)
    x["EMA20"]=ema(c,20); x["EMA50"]=ema(c,50); x["EMA200"]=ema(c,200)
    x["RSI14"]=rsi(c,14)
    x["MACD"],x["MACD_SIGNAL"],x["MACD_HIST"]=macd(c)
    x["ATR14"]=atr(x,14)
    x["ADX14"]=adx(x,14)
    x["STO_K"],x["STO_D"]=stochastic(x)
    x["BB_MID"],x["BB_UP"],x["BB_LOW"]=bollinger(c)
    vol=x["Volume"].replace(0,np.nan)
    x["VOL20"]=vol.rolling(20).mean()
    x["OBV"]=(np.sign(c.diff()).fillna(0)*vol.fillna(0)).cumsum()
    x["MFI14"]=money_flow_index(x,14)
    return x

def money_flow_index(df, n=14):
    tp=(df["High"]+df["Low"]+df["Close"])/3
    mf=tp*df["Volume"].fillna(0)
    direction=tp.diff()
    pos=mf.where(direction>0,0).rolling(n).sum()
    neg=mf.where(direction<0,0).rolling(n).sum().abs()
    ratio=pos/neg.replace(0,np.nan)
    return 100-(100/(1+ratio))

def safe(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default

def signal(v, pos="Bullish", neg="Bearish", neutral="Neutral"):
    if pd.isna(v): return neutral
    return pos if v else neg

def analyze(x):
    last=x.iloc[-1]; prev=x.iloc[-2]
    score=50.0
    reasons=[]
    # trend
    trend_points=0
    for fast, slow in [("EMA20","EMA50"),("EMA50","EMA200"),("SMA50","SMA200")]:
        if pd.notna(last[fast]) and pd.notna(last[slow]):
            if last[fast]>last[slow]: trend_points += 1
            else: trend_points -= 1
    score += trend_points*6
    trend = "Bullish" if trend_points>=2 else ("Bearish" if trend_points<=-2 else "Neutral")
    reasons.append(f"Trend structure: {trend}.")
    # RSI
    rv=safe(last["RSI14"])
    if not np.isnan(rv):
        if 50<=rv<=70: score+=6; reasons.append(f"RSI {rv:.1f} supports momentum without being extreme.")
        elif rv>70: score-=3; reasons.append(f"RSI {rv:.1f} is overbought; chasing is higher risk.")
        elif rv<30: score+=2; reasons.append(f"RSI {rv:.1f} is oversold; reversal confirmation is needed.")
        else: score-=1
    # MACD
    if pd.notna(last["MACD"]) and pd.notna(last["MACD_SIGNAL"]):
        if last["MACD"]>last["MACD_SIGNAL"]: score+=6; reasons.append("MACD is above its signal line.")
        else: score-=6; reasons.append("MACD is below its signal line.")
    # ADX
    av=safe(last["ADX14"])
    if not np.isnan(av):
        if av>=25: score+=4; reasons.append(f"ADX {av:.1f} indicates a meaningful trend.")
        else: reasons.append(f"ADX {av:.1f} indicates limited trend strength.")
    # volume
    if pd.notna(last["Volume"]) and pd.notna(last["VOL20"]) and last["VOL20"]>0:
        vr=last["Volume"]/last["VOL20"]
        if vr>=1.5: score+=5; reasons.append(f"Volume is {vr:.1f}x the 20-day average.")
        elif vr<0.7: score-=2
    score=float(np.clip(score,0,100))
    return {"score":score,"trend":trend,"reasons":reasons}

def levels(x):
    last=x.iloc[-1]
    price=safe(last["Close"])
    atrv=safe(last["ATR14"])
    recent=x.tail(60)
    # swing levels; use conservative quantiles to avoid one noisy candle
    support=float(recent["Low"].quantile(.15))
    resistance=float(recent["High"].quantile(.85))
    if np.isnan(atrv) or atrv<=0:
        atrv=price*0.03
    # two staged zones around support and current price
    z1_low=max(0.01, support-0.35*atrv)
    z1_high=max(z1_low, support+0.20*atrv)
    z2_low=max(0.01, support-1.00*atrv)
    z2_high=max(z2_low, support-0.35*atrv)
    stop=max(0.01, z2_low-0.65*atrv)
    r1=max(price, resistance)
    r2=r1+1.25*atrv
    r3=r1+2.5*atrv
    return dict(price=price, support=support, resistance=resistance,
                z1=(z1_low,z1_high), z2=(z2_low,z2_high),
                stop=stop, t1=r1, t2=r2, t3=r3, atr=atrv)

def fair_value(info, price):
    # Transparent valuation estimate using available EPS/BPS and sector-neutral multiples.
    vals=[]
    eps=safe(info.get("trailingEps"))
    bps=safe(info.get("bookValue"))
    if not np.isnan(eps) and eps>0:
        vals.append(eps*10.0)
    if not np.isnan(bps) and bps>0:
        vals.append(bps*1.15)
    if not vals:
        return None
    low=float(np.percentile(vals,25))
    high=float(np.percentile(vals,75))
    mid=(low+high)/2
    return {"low":low,"mid":mid,"high":high,
            "method":"Indicative P/E 10x + P/B 1.15x using available company data"}

def get_news(company, symbol):
    q=quote_plus(f'"{company}" OR "{symbol.replace(".CA","")}" Egypt stock')
    url=f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        r=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        feed=feedparser.parse(r.content)
        items=[]
        for e in feed.entries[:10]:
            items.append({
                "title":e.get("title",""),
                "link":e.get("link",""),
                "published":e.get("published",""),
                "source":e.get("source",{}).get("title","") if isinstance(e.get("source"),dict) else ""
            })
        return items
    except Exception:
        return []

def company_name(info, symbol):
    return info.get("longName") or info.get("shortName") or symbol.replace(".CA","")

# ----------------------------
# UI
# ----------------------------
st.title("📈 EGX Smart Stock Assistant")
st.caption("اكتب كود السهم فقط — التطبيق يجمع السعر، المؤشرات، مناطق الدخول، المخاطر، التقييم والأخبار.")

with st.form("search"):
    col1,col2=st.columns([3,1])
    with col1:
        raw=st.text_input("Stock symbol", value="HBCO", placeholder="مثال: COMI أو HBCO")
    with col2:
        period=st.selectbox("History", ["1y","2y","5y"], index=1)
    submitted=st.form_submit_button("🔎 Analyze", use_container_width=True)

if submitted or raw:
    symbol=resolve_symbol(raw)
    with st.spinner(f"Analyzing {symbol}..."):
        df=get_history(symbol,period)
        info=get_info(symbol)
    if df.empty or len(df)<60:
        st.error(f"لم أستطع الحصول على بيانات كافية للسهم {symbol}. جرّب الرمز مرة أخرى.")
        st.stop()

    x=add_indicators(df)
    a=analyze(x)
    lv=levels(x)
    name=company_name(info,symbol)
    price=lv["price"]
    change=((price/safe(x.iloc[-2]["Close"]))-1)*100 if safe(x.iloc[-2]["Close"]) else np.nan
    fv=fair_value(info,price)
    news=get_news(name,symbol)

    # Header
    st.markdown(f"## {symbol.replace('.CA','')} — {name}")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Price", f"{price:,.2f}")
    c2.metric("Change", f"{change:+.2f}%")
    c3.metric("Smart Score", f"{a['score']:.0f}/100")
    c4.metric("Trend", a["trend"])

    # Score bars
    st.subheader("🧠 Smart Analysis")
    b1,b2,b3=st.columns(3)
    rsi_v=safe(x.iloc[-1]["RSI14"])
    macd_bull=x.iloc[-1]["MACD"]>x.iloc[-1]["MACD_SIGNAL"]
    vol_ratio=safe(x.iloc[-1]["Volume"])/safe(x.iloc[-1]["VOL20"]) if safe(x.iloc[-1]["VOL20"]) else np.nan
    b1.metric("RSI 14", "—" if np.isnan(rsi_v) else f"{rsi_v:.1f}")
    b2.metric("MACD", "Bullish" if macd_bull else "Bearish")
    b3.metric("Volume vs Avg", "—" if np.isnan(vol_ratio) else f"{vol_ratio:.1f}x")

    with st.expander("لماذا حصل السهم على هذه القراءة؟", expanded=True):
        for r in a["reasons"]:
            st.write("• "+r)

    # Chart
    st.subheader("📊 Price & Trend")
    chart=x.tail(180)
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=chart["Date"],open=chart["Open"],high=chart["High"],low=chart["Low"],close=chart["Close"],name="Price"))
    for col,label in [("EMA20","EMA 20"),("EMA50","EMA 50"),("SMA200","SMA 200")]:
        if chart[col].notna().any():
            fig.add_trace(go.Scatter(x=chart["Date"],y=chart[col],name=label,mode="lines"))
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=20,b=10),xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

    # Entry / risk
    st.subheader("🎯 Entry Zones & Risk")
    e1,e2,e3=st.columns(3)
    e1.metric("Entry Zone 1", f"{lv['z1'][0]:.2f} – {lv['z1'][1]:.2f}")
    e2.metric("Entry Zone 2", f"{lv['z2'][0]:.2f} – {lv['z2'][1]:.2f}")
    e3.metric("Stop Loss", f"{lv['stop']:.2f}")
    t1,t2,t3=st.columns(3)
    t1.metric("Target 1", f"{lv['t1']:.2f}")
    t2.metric("Target 2", f"{lv['t2']:.2f}")
    t3.metric("Target 3", f"{lv['t3']:.2f}")

    risk=max(price-lv["stop"],0.01)
    rr=max(lv["t1"]-price,0)/risk
    st.info(f"Support: {lv['support']:.2f} | Resistance: {lv['resistance']:.2f} | Indicative R/R to T1: {rr:.2f}:1")

    # Fundamentals
    st.subheader("💰 Fundamentals & Fair Value")
    f1,f2,f3,f4=st.columns(4)
    f1.metric("Market Cap", "—" if not info.get("marketCap") else f"{info['marketCap']/1e9:.2f}B")
    f2.metric("P/E", "—" if safe(info.get("trailingPE"))!=safe(info.get("trailingPE")) else f"{safe(info.get('trailingPE')):.2f}")
    f3.metric("P/B", "—" if safe(info.get("priceToBook"))!=safe(info.get("priceToBook")) else f"{safe(info.get('priceToBook')):.2f}")
    f4.metric("Dividend Yield", "—" if safe(info.get("dividendYield"))!=safe(info.get("dividendYield")) else f"{safe(info.get('dividendYield'))*100:.2f}%")
    if fv:
        st.write(f"**Indicative fair-value range:** {fv['low']:.2f} – {fv['high']:.2f} EGP (mid {fv['mid']:.2f})")
        st.caption("هذا نموذج تقديري شفاف وليس سعرًا مستهدفًا مضمونًا؛ يتغير حسب البيانات والافتراضات.")
    else:
        st.warning("لا توجد بيانات Fundamental كافية لبناء Fair Value آلي موثوق لهذا السهم من المصدر الحالي.")

    # News
    st.subheader("📰 Latest News")
    if news:
        for n in news:
            st.markdown(f"<div class='newsbox'><b>{n['title']}</b><br><span class='small'>{n['published']} — {n['source']}</span><br><a href='{n['link']}' target='_blank'>فتح الخبر ↗</a></div>",unsafe_allow_html=True)
    else:
        st.info("لم يتم العثور على أخبار عبر المصدر الحالي.")

    # Data
    with st.expander("📋 Indicator table"):
        cols=["Date","Close","Volume","RSI14","MACD","ADX14","STO_K","MFI14","EMA20","EMA50","SMA200","ATR14"]
        st.dataframe(x[cols].tail(20).sort_values("Date",ascending=False),use_container_width=True,hide_index=True)

    st.caption("⚠️ البيانات قد تكون متأخرة أو غير مكتملة. هذا التطبيق أداة تحليل معلوماتية وليس توصية استثمارية.")
else:
    st.info("اكتب كود السهم ثم اضغط Analyze.")
    st.markdown("""
    **أمثلة:** `HBCO` · `COMI` · `SWDY` · `TMGH` · `EFIH`

    **المخرجات:** السعر • Smart Score • Trend • RSI/MACD/ADX/MFI • مناطق دخول • Stop Loss • Targets • Support/Resistance • Fair Value • الأخبار.
    """)
