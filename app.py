
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests, re, feedparser
from datetime import datetime, timedelta
from urllib.parse import quote_plus
import plotly.graph_objects as go

st.set_page_config(page_title="EGX Smart Stock Assistant", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container{padding:1rem .8rem 3rem .8rem;max-width:1200px}
h1{font-size:1.8rem!important}.metric-card{padding:14px;border-radius:16px}
.small{font-size:.82rem;opacity:.72}.newsbox{padding:10px 0;border-bottom:1px solid rgba(128,128,128,.18)}
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

# Known Investing.com historical pages for EGX symbols that are missing from Yahoo.
INVESTING_SLUGS = {
    "HBCO": "heibco-for-commercial-investments",
}

def resolve_symbol(raw):
    s = raw.strip().upper().replace(" ","")
    if s in COMMON: return COMMON[s]
    return s if s.endswith(".CA") else s + ".CA"

def clean_num(v):
    try:
        if v is None: return np.nan
        if isinstance(v,float) and np.isnan(v): return np.nan
        if isinstance(v,str):
            v=v.replace(",","").replace("%","").strip()
            m=re.fullmatch(r"([-+]?\d*\.?\d+)\s*([KMB])?",v,re.I)
            if m:
                x=float(m.group(1)); u=(m.group(2) or "").upper()
                return x*{"K":1e3,"M":1e6,"B":1e9}.get(u,1)
        return float(v)
    except: return np.nan

@st.cache_data(ttl=300, show_spinner=False)
def investing_page_history(symbol, period):
    code=symbol.replace(".CA","").upper()
    slug=INVESTING_SLUGS.get(code)
    if not slug: return pd.DataFrame()
    days={"1y":370,"2y":740,"5y":1850}.get(period,740)
    end=datetime.utcnow().date()
    start=end-timedelta(days=days)
    url=f"https://www.investing.com/equities/{slug}-historical-data"
    headers={
        "User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 Version/17.6 Mobile/15E148 Safari/604.1",
        "Accept-Language":"en-US,en;q=0.9",
        "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        r=requests.get(url,headers=headers,timeout=20)
        if not r.ok: return pd.DataFrame()
        tables=pd.read_html(r.text)
        for t in tables:
            cols={str(c).strip().lower():c for c in t.columns}
            dc=cols.get("date"); pc=cols.get("price") or cols.get("last")
            oc=cols.get("open"); hc=cols.get("high"); lc=cols.get("low")
            vc=next((cols[k] for k in cols if k.startswith("vol")),None)
            if not dc or not pc: continue
            out=pd.DataFrame({
                "Date":pd.to_datetime(t[dc],errors="coerce"),
                "Close":t[pc].map(clean_num),
                "Open":t[oc].map(clean_num) if oc else np.nan,
                "High":t[hc].map(clean_num) if hc else np.nan,
                "Low":t[lc].map(clean_num) if lc else np.nan,
                "Volume":t[vc].map(clean_num) if vc else np.nan
            })
            out=out.dropna(subset=["Date","Close"])
            out=out[(out.Date.dt.date>=start)&(out.Date.dt.date<=end)]
            if len(out)>=60:
                return out.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def yahoo_history(symbol, period):
    try:
        df=yf.Ticker(symbol).history(period=period,interval="1d",auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        df=df.reset_index()
        df.columns=[str(c).strip().title() for c in df.columns]
        if "Datetime" in df: df.rename(columns={"Datetime":"Date"},inplace=True)
        df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
        cols=[c for c in ["Date","Open","High","Low","Close","Volume"] if c in df]
        df=df[cols].dropna(subset=["Date","Close"])
        return df if len(df)>=60 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def get_info(symbol):
    try:
        x=yf.Ticker(symbol).info
        return x if isinstance(x,dict) else {}
    except: return {}

def get_history(symbol,period):
    # 1) Direct Investing page for known EGX stocks such as HBCO.
    df=investing_page_history(symbol,period)
    if len(df)>=60: return df,"Investing.com"
    # 2) Yahoo for symbols supported there.
    df=yahoo_history(symbol,period)
    if len(df)>=60: return df,"Yahoo Finance"
    return pd.DataFrame(),"No provider"

def sma(s,n): return s.rolling(n).mean()
def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    au=up.ewm(alpha=1/n,adjust=False).mean()
    ad=dn.ewm(alpha=1/n,adjust=False).mean()
    rs=au/ad.replace(0,np.nan)
    return 100-(100/(1+rs))

def atr(df,n=14):
    pc=df.Close.shift()
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def macd(s):
    a=ema(s,12); b=ema(s,26); line=a-b; sig=ema(line,9)
    return line,sig,line-sig

def stochastic(df,n=14):
    lo=df.Low.rolling(n).min(); hi=df.High.rolling(n).max()
    k=100*(df.Close-lo)/(hi-lo).replace(0,np.nan)
    return k,k.rolling(3).mean()

def adx(df,n=14):
    up=df.High.diff(); dn=-df.Low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0)
    minus=np.where((dn>up)&(dn>0),dn,0.0)
    tr=pd.concat([df.High-df.Low,(df.High-df.Close.shift()).abs(),
                  (df.Low-df.Close.shift()).abs()],axis=1).max(axis=1)
    av=tr.rolling(n).mean()
    pdi=100*pd.Series(plus,index=df.index).rolling(n).mean()/av.replace(0,np.nan)
    mdi=100*pd.Series(minus,index=df.index).rolling(n).mean()/av.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    return dx.rolling(n).mean()

def mfi(df,n=14):
    tp=(df.High+df.Low+df.Close)/3
    mf=tp*df.Volume.fillna(0)
    pos=mf.where(tp.diff()>0,0).rolling(n).sum()
    neg=mf.where(tp.diff()<0,0).rolling(n).sum().abs()
    return 100-(100/(1+pos/neg.replace(0,np.nan)))

def indicators(df):
    x=df.copy(); c=x.Close
    x["SMA20"]=sma(c,20); x["SMA50"]=sma(c,50); x["SMA200"]=sma(c,200)
    x["EMA20"]=ema(c,20); x["EMA50"]=ema(c,50); x["EMA200"]=ema(c,200)
    x["RSI14"]=rsi(c); x["MACD"],x["MACD_SIGNAL"],x["MACD_HIST"]=macd(c)
    x["ATR14"]=atr(x); x["ADX14"]=adx(x); x["STO_K"],x["STO_D"]=stochastic(x)
    mid=sma(c,20); sd=c.rolling(20).std()
    x["BB_MID"]=mid; x["BB_UP"]=mid+2*sd; x["BB_LOW"]=mid-2*sd
    x["VOL20"]=x.Volume.replace(0,np.nan).rolling(20).mean()
    x["OBV"]=(np.sign(c.diff()).fillna(0)*x.Volume.fillna(0)).cumsum()
    x["MFI14"]=mfi(x)
    return x

def safe(v):
    try: return float(v)
    except: return np.nan

def analyze(x):
    last=x.iloc[-1]; score=50; reasons=[]; tp=0
    for a,b in [("EMA20","EMA50"),("EMA50","EMA200"),("SMA50","SMA200")]:
        if pd.notna(last[a]) and pd.notna(last[b]): tp += 1 if last[a]>last[b] else -1
    score+=tp*6
    trend="Bullish" if tp>=2 else "Bearish" if tp<=-2 else "Neutral"
    reasons.append(f"Trend structure: {trend}.")
    rv=safe(last.RSI14)
    if not np.isnan(rv):
        if 50<=rv<=70: score+=6; reasons.append(f"RSI {rv:.1f} supports momentum.")
        elif rv>70: score-=3; reasons.append(f"RSI {rv:.1f} is overbought.")
        elif rv<30: score+=2; reasons.append(f"RSI {rv:.1f} is oversold; confirmation is needed.")
    if pd.notna(last.MACD) and pd.notna(last.MACD_SIGNAL):
        if last.MACD>last.MACD_SIGNAL: score+=6; reasons.append("MACD is above signal.")
        else: score-=6; reasons.append("MACD is below signal.")
    av=safe(last.ADX14)
    if not np.isnan(av):
        if av>=25: score+=4; reasons.append(f"ADX {av:.1f} indicates a meaningful trend.")
        else: reasons.append(f"ADX {av:.1f} indicates limited trend strength.")
    if safe(last.VOL20)>0 and pd.notna(last.Volume):
        vr=last.Volume/last.VOL20
        if vr>=1.5: score+=5; reasons.append(f"Volume is {vr:.1f}x the 20-day average.")
        elif vr<.7: score-=2
    return {"score":float(np.clip(score,0,100)),"trend":trend,"reasons":reasons}

def levels(x):
    last=x.iloc[-1]; price=safe(last.Close); av=safe(last.ATR14)
    recent=x.tail(60)
    support=float(recent.Low.quantile(.15)); resistance=float(recent.High.quantile(.85))
    if np.isnan(av) or av<=0: av=price*.03
    z1=(max(.01,support-.35*av),max(.01,support+.20*av))
    z2=(max(.01,support-1.00*av),max(.01,support-.35*av))
    stop=max(.01,z2[0]-.65*av)
    t1=max(price,resistance); t2=t1+1.25*av; t3=t1+2.5*av
    return dict(price=price,support=support,resistance=resistance,z1=z1,z2=z2,
                stop=stop,t1=t1,t2=t2,t3=t3,atr=av)

def fair_value(info):
    vals=[]; eps=safe(info.get("trailingEps")); bps=safe(info.get("bookValue"))
    if not np.isnan(eps) and eps>0: vals.append(eps*10)
    if not np.isnan(bps) and bps>0: vals.append(bps*1.15)
    if not vals: return None
    return {"low":float(np.percentile(vals,25)),"mid":float(np.mean(vals)),
            "high":float(np.percentile(vals,75))}

def news(company,symbol):
    q=quote_plus(f'"{company}" OR "{symbol.replace(".CA","")}" Egypt stock')
    url=f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        r=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        feed=feedparser.parse(r.content)
        return [{"title":e.get("title",""),"link":e.get("link",""),
                 "published":e.get("published","")} for e in feed.entries[:10]]
    except: return []

st.title("📈 EGX Smart Stock Assistant")
st.caption("تحليل الأسهم المصرية: السعر، المؤشرات، مناطق الدخول، المخاطر، التقييم والأخبار.")

with st.form("search"):
    c1,c2=st.columns([3,1])
    with c1: raw=st.text_input("Stock symbol",value="HBCO",placeholder="مثال: HBCO أو COMI")
    with c2: period=st.selectbox("History",["1y","2y","5y"],index=1)
    submitted=st.form_submit_button("🔎 Analyze",use_container_width=True)

if submitted or raw:
    symbol=resolve_symbol(raw)
    with st.spinner(f"Analyzing {symbol}..."):
        df,source=get_history(symbol,period)
        info=get_info(symbol)
    if df.empty or len(df)<60:
        st.error(f"لم أستطع الحصول على بيانات كافية للسهم {symbol}.")
        if symbol.replace(".CA","").upper() in INVESTING_SLUGS:
            st.info("مصدر Investing.com المباشر لم يُرجع الجدول في هذه المحاولة. جرّب Analyze مرة أخرى.")
        st.stop()

    x=indicators(df); a=analyze(x); lv=levels(x)
    name=info.get("longName") or info.get("shortName") or symbol.replace(".CA","")
    price=lv["price"]; prev=safe(x.iloc[-2].Close)
    change=(price/prev-1)*100 if prev else np.nan
    fv=fair_value(info); ns=news(name,symbol)

    st.success(f"Data source: {source}")
    st.markdown(f"## {symbol.replace('.CA','')} — {name}")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Price",f"{price:,.2f}")
    c2.metric("Change",f"{change:+.2f}%")
    c3.metric("Smart Score",f"{a['score']:.0f}/100")
    c4.metric("Trend",a["trend"])

    st.subheader("🧠 Smart Analysis")
    b1,b2,b3=st.columns(3)
    rv=safe(x.iloc[-1].RSI14); vb=safe(x.iloc[-1].Volume)/safe(x.iloc[-1].VOL20) if safe(x.iloc[-1].VOL20) else np.nan
    b1.metric("RSI 14","—" if np.isnan(rv) else f"{rv:.1f}")
    b2.metric("MACD","Bullish" if x.iloc[-1].MACD>x.iloc[-1].MACD_SIGNAL else "Bearish")
    b3.metric("Volume vs Avg","—" if np.isnan(vb) else f"{vb:.1f}x")
    with st.expander("لماذا حصل السهم على هذه القراءة؟",expanded=True):
        for r in a["reasons"]: st.write("• "+r)

    st.subheader("📊 Price & Trend")
    chart=x.tail(180); fig=go.Figure()
    fig.add_trace(go.Candlestick(x=chart.Date,open=chart.Open,high=chart.High,low=chart.Low,close=chart.Close,name="Price"))
    for col,label in [("EMA20","EMA 20"),("EMA50","EMA 50"),("SMA200","SMA 200")]:
        if chart[col].notna().any(): fig.add_trace(go.Scatter(x=chart.Date,y=chart[col],name=label))
    fig.update_layout(height=430,margin=dict(l=10,r=10,t=20,b=10),xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("🎯 Entry Zones & Risk")
    e1,e2,e3=st.columns(3)
    e1.metric("Entry Zone 1",f"{lv['z1'][0]:.2f} – {lv['z1'][1]:.2f}")
    e2.metric("Entry Zone 2",f"{lv['z2'][0]:.2f} – {lv['z2'][1]:.2f}")
    e3.metric("Stop Loss",f"{lv['stop']:.2f}")
    t1,t2,t3=st.columns(3)
    t1.metric("Target 1",f"{lv['t1']:.2f}"); t2.metric("Target 2",f"{lv['t2']:.2f}"); t3.metric("Target 3",f"{lv['t3']:.2f}")
    risk=max(price-lv["stop"],.01); rr=max(lv["t1"]-price,0)/risk
    st.info(f"Support: {lv['support']:.2f} | Resistance: {lv['resistance']:.2f} | Indicative R/R to T1: {rr:.2f}:1")

    st.subheader("💰 Fundamentals & Fair Value")
    f1,f2,f3,f4=st.columns(4)
    mc=info.get("marketCap"); pe=safe(info.get("trailingPE")); pb=safe(info.get("priceToBook")); dy=safe(info.get("dividendYield"))
    f1.metric("Market Cap","—" if not mc else f"{mc/1e9:.2f}B")
    f2.metric("P/E","—" if np.isnan(pe) else f"{pe:.2f}")
    f3.metric("P/B","—" if np.isnan(pb) else f"{pb:.2f}")
    f4.metric("Dividend Yield","—" if np.isnan(dy) else f"{dy*100:.2f}%")
    if fv: st.write(f"**Indicative fair-value range:** {fv['low']:.2f} – {fv['high']:.2f} EGP (mid {fv['mid']:.2f})")
    else: st.warning("لا توجد بيانات Fundamental كافية لبناء Fair Value آلي من المصدر الحالي.")

    st.subheader("📰 Latest News")
    if ns:
        for n in ns:
            st.markdown(f"<div class='newsbox'><b>{n['title']}</b><br><span class='small'>{n['published']}</span><br><a href='{n['link']}' target='_blank'>فتح الخبر ↗</a></div>",unsafe_allow_html=True)
    else: st.info("لم يتم العثور على أخبار عبر المصدر الحالي.")

    with st.expander("📋 Indicator table"):
        cols=["Date","Close","Volume","RSI14","MACD","ADX14","STO_K","MFI14","EMA20","EMA50","SMA200","ATR14"]
        st.dataframe(x[cols].tail(20).sort_values("Date",ascending=False),use_container_width=True,hide_index=True)

    st.caption("⚠️ البيانات قد تكون متأخرة أو غير مكتملة. هذا التطبيق أداة تحليل معلوماتية وليس توصية استثمارية.")
