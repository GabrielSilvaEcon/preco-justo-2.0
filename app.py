"""
Painel de Preço Justo – B3
Rodar com:  streamlit run app.py
Dependências: pip install streamlit yfinance plotly pandas
"""
from datetime import datetime
from math import sqrt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Preço Justo B3", page_icon="📈", layout="wide")

# ---------------------------------------------------------------- estilo
st.markdown(
    """
<style>
.stApp {background: linear-gradient(180deg,#0b1020 0%,#111936 100%);}
h1,h2,h3,p,label,span,div {color:#e8ecf8;}
section[data-testid="stSidebar"] {background:#0e1428;}

/* Cabeçalho (barra superior do Streamlit) escuro */
header[data-testid="stHeader"] {
    background: rgba(11,16,32,.96) !important;
    border-bottom: 1px solid rgba(255,255,255,.08);
    backdrop-filter: blur(8px);
}
[data-testid="stDecoration"] {display:none;}
header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] span,
header[data-testid="stHeader"] svg {color:#e8ecf8 !important; fill:#e8ecf8 !important;}
[data-testid="stToolbar"] {background: transparent !important;}

/* Abas */
button[data-baseweb="tab"] {color:#cbd5e1;}
button[data-baseweb="tab"][aria-selected="true"] {color:#ffffff; font-weight:700;}

.card {background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.09);
       border-radius:20px; padding:18px 20px; margin-bottom:14px;}
.tk {font-size:1.5rem; font-weight:800; letter-spacing:.5px;}
.nm {font-size:.8rem; opacity:.6; margin-bottom:8px;}
.badge {display:inline-block; padding:4px 12px; border-radius:999px;
        font-size:.75rem; font-weight:700; float:right;}
.good {background:rgba(34,197,94,.18); color:#4ade80;}
.fair {background:rgba(250,204,21,.18); color:#facc15;}
.bad  {background:rgba(239,68,68,.18); color:#f87171;}
.na   {background:rgba(148,163,184,.18); color:#cbd5e1;}
.lbl {font-size:.72rem; opacity:.6; text-transform:uppercase; letter-spacing:.8px;}
.val {font-size:1.25rem; font-weight:700;}
.hero {font-size:2rem; font-weight:800;}
</style>
""",
    unsafe_allow_html=True,
)

LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              font={"color": "#e8ecf8"})

# ---------------------------------------------------------------- estado
if "ativos" not in st.session_state:
    st.session_state.ativos = ["PETR4", "ITSA4", "BBAS3", "VALE3"]


def normaliza(t: str) -> str:
    return t.upper().strip().replace(".SA", "")


# ---------------------------------------------------------------- dados
def _cambio(moeda):
    """Cotação da moeda de reporte em reais (1,0 se já estiver em BRL)."""
    if not moeda or moeda == "BRL":
        return 1.0
    try:
        h = yf.Ticker(f"{moeda}BRL=X").history(period="5d")
        return float(h["Close"].iloc[-1]) if not h.empty else None
    except Exception:
        return None


def _serie_fcl(t) -> pd.Series:
    """Fluxo de caixa livre anual (até 4 anos), indexado pelo ano."""
    try:
        cf = t.cashflow
        if cf is not None and not cf.empty and "Free Cash Flow" in cf.index:
            s = cf.loc["Free Cash Flow"].dropna().astype(float)
            s.index = pd.to_datetime(s.index).year
            s = s[~s.index.duplicated(keep="last")]
            return s.sort_index().tail(4)
    except Exception:
        pass
    return pd.Series(dtype=float)


@st.cache_data(ttl=900, show_spinner=False)
def buscar(ticker: str) -> dict:
    """Busca cotação, LPA, VPA, dividendos e fluxo de caixa (Yahoo Finance, tickers .SA da B3)."""
    t = yf.Ticker(f"{ticker}.SA")
    try:
        info = t.info or {}
    except Exception:
        info = {}

    preco = info.get("currentPrice") or info.get("regularMarketPrice")
    if not preco:
        h = t.history(period="5d")
        preco = float(h["Close"].iloc[-1]) if not h.empty else None

    divs = t.dividends
    por_ano = pd.Series(dtype=float)
    if divs is not None and not divs.empty:
        idx = divs.index.tz_localize(None) if divs.index.tz is not None else divs.index
        por_ano = divs.groupby(idx.year).sum()
        por_ano = por_ano[por_ano.index < datetime.now().year].tail(5)

    # Fluxo de caixa livre, dívida e caixa (convertidos para reais, se necessário)
    fx = _cambio(info.get("financialCurrency"))
    fcl_hist = _serie_fcl(t)
    if fx is None:  # não foi possível converter a moeda: o DCF fica indisponível
        fcl_hist, fx = pd.Series(dtype=float), 1.0
    fcl_hist = fcl_hist * fx
    fcl_base = float(fcl_hist.tail(3).mean()) if len(fcl_hist) else None

    return {
        "ticker": ticker,
        "nome": info.get("shortName") or info.get("longName") or ticker,
        "preco": preco,
        "lpa": info.get("trailingEps"),
        "vpa": info.get("bookValue"),
        "div_ano": por_ano,
        "div_medio": float(por_ano.mean()) if len(por_ano) else None,
        "fcl_hist": fcl_hist,
        "fcl_base": fcl_base,
        "divida": (info.get("totalDebt") or 0) * fx,
        "caixa": (info.get("totalCash") or 0) * fx,
        "acoes": info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"),
    }


def calcular_dcf(d: dict, taxa: float, cresc: float, anos: int, g_perp: float):
    """
    Fluxo de caixa descontado (taxas em fração: 0,13 = 13%).
    Projeta o FCL por `anos` anos com crescimento `cresc`, soma a perpetuidade
    (crescimento `g_perp`), desconta tudo por `taxa` e ajusta pela dívida líquida.
    Retorna (preço justo por ação, detalhes) ou (None, None) se não for aplicável.
    """
    base, acoes = d.get("fcl_base"), d.get("acoes")
    if not base or base <= 0 or not acoes or taxa <= g_perp:
        return None, None

    fluxos = [base * (1 + cresc) ** t for t in range(1, anos + 1)]
    vps = [f / (1 + taxa) ** t for t, f in enumerate(fluxos, start=1)]
    perp = fluxos[-1] * (1 + g_perp) / (taxa - g_perp)
    vp_perp = perp / (1 + taxa) ** anos
    ev = sum(vps) + vp_perp
    patrimonio = ev - d["divida"] + d["caixa"]
    if patrimonio <= 0:
        return None, None

    return patrimonio / acoes, {
        "fluxos": fluxos, "vps": vps, "vp_perp": vp_perp, "ev": ev,
        "divida_liq": d["divida"] - d["caixa"], "patrimonio": patrimonio,
    }


def analisar(d: dict, metodo: str, yield_alvo: float, margem_alvo: float, dcf_cfg: dict) -> dict:
    graham = None
    if d["lpa"] and d["vpa"] and d["lpa"] > 0 and d["vpa"] > 0:
        graham = sqrt(22.5 * d["lpa"] * d["vpa"])

    bazin = None
    if d["div_medio"] and d["div_medio"] > 0:
        bazin = d["div_medio"] / (yield_alvo / 100)

    dcf, dcf_det = calcular_dcf(d, **dcf_cfg)

    if metodo == "Graham":
        justo = graham
    elif metodo == "Bazin":
        justo = bazin
    elif metodo == "DCF":
        justo = dcf
    else:
        vals = [v for v in (graham, bazin, dcf) if v]
        justo = sum(vals) / len(vals) if vals else None

    margem = None
    if justo and d["preco"]:
        margem = (justo - d["preco"]) / justo * 100

    if margem is None:
        status, cls = "Sem dados", "na"
    elif margem >= margem_alvo:
        status, cls = "Com margem", "good"
    elif margem >= 0:
        status, cls = "Preço justo", "fair"
    else:
        status, cls = "Acima do justo", "bad"

    pl = d["preco"] / d["lpa"] if d["preco"] and d["lpa"] and d["lpa"] > 0 else None
    pvp = d["preco"] / d["vpa"] if d["preco"] and d["vpa"] and d["vpa"] > 0 else None
    roe = d["lpa"] / d["vpa"] * 100 if d["lpa"] and d["vpa"] and d["vpa"] > 0 else None
    dy = d["div_medio"] / d["preco"] * 100 if d["div_medio"] and d["preco"] else None

    return {**d, "graham": graham, "bazin": bazin, "dcf": dcf, "dcf_det": dcf_det,
            "justo": justo, "margem": margem, "status": status, "cls": cls,
            "pl": pl, "pvp": pvp, "roe": roe, "dy": dy}


def brl(v):
    return "—" if v is None else f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def bi(v):
    """Valores grandes em bilhões/milhões de reais."""
    if v is None:
        return "—"
    s = f"R$ {abs(v) / 1e9:,.2f} bi" if abs(v) >= 1e9 else f"R$ {abs(v) / 1e6:,.1f} mi"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"-{s}" if v < 0 else s


def pct(v):
    return "—" if v is None else f"{v:.1f}%".replace(".", ",")


def gauge(margem):
    v = 0 if margem is None else max(-60, min(60, margem))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        number={"suffix": "%", "font": {"size": 30, "color": "#e8ecf8"}},
        gauge={
            "axis": {"range": [-60, 60], "tickcolor": "#64748b"},
            "bar": {"color": "#e8ecf8", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-60, 0], "color": "rgba(239,68,68,.45)"},
                {"range": [0, 20], "color": "rgba(250,204,21,.45)"},
                {"range": [20, 60], "color": "rgba(34,197,94,.45)"},
            ],
        },
    ))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=0), **LAYOUT)
    return fig


# ---------------------------------------------------------------- barra lateral
METODOS = {
    "Média dos métodos": "Média",
    "Graham": "Graham",
    "Bazin": "Bazin",
    "DCF (fluxo de caixa descontado)": "DCF",
}

with st.sidebar:
    st.markdown("## ⚙️ Configurações")
    metodo_label = st.radio("Método do preço justo", list(METODOS))
    metodo = METODOS[metodo_label]
    yield_alvo = st.slider("Dividend yield desejado (Bazin)", 4.0, 12.0, 6.0, 0.5, format="%.1f%%")
    margem_alvo = st.slider("Margem de segurança alvo", 5, 50, 20, 5, format="%d%%")

    with st.expander("📐 Premissas do DCF", expanded=(metodo == "DCF")):
        taxa_desc = st.slider("Taxa de desconto (a.a.)", 8.0, 20.0, 13.0, 0.5, format="%.1f%%")
        cresc_fcl = st.slider("Crescimento do FCL nos anos projetados", -5.0, 15.0, 5.0, 0.5, format="%.1f%%")
        anos_proj = st.slider("Anos de projeção", 5, 10, 5)
        g_perp = st.slider("Crescimento na perpetuidade", 0.0, 6.0, 3.0, 0.5, format="%.1f%%")

    st.caption(
        "Graham = √(22,5 × LPA × VPA)  ·  Bazin = dividendo médio de 5 anos ÷ yield desejado  ·  "
        "DCF = valor presente dos fluxos de caixa livres projetados, ajustado pela dívida líquida"
    )

    st.markdown("## ➕ Adicionar ativos")
    with st.form("add", clear_on_submit=True):
        novo = st.text_input("Ticker(s)", placeholder="Ex.: WEGE3, TAEE11")
        if st.form_submit_button("Adicionar", use_container_width=True):
            for t in novo.replace(";", ",").replace(" ", ",").split(","):
                t = normaliza(t)
                if t and t not in st.session_state.ativos:
                    st.session_state.ativos.append(t)

    st.session_state.ativos = st.multiselect(
        "Ativos na análise", options=st.session_state.ativos, default=st.session_state.ativos
    )
    if st.button("🔄 Atualizar cotações", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

dcf_cfg = dict(taxa=taxa_desc / 100, cresc=cresc_fcl / 100, anos=anos_proj, g_perp=g_perp / 100)

# ---------------------------------------------------------------- corpo
st.markdown('<div class="hero">📈 Preço Justo · B3</div>', unsafe_allow_html=True)
st.caption(f"Atualizado em {datetime.now():%d/%m/%Y %H:%M} · fonte: dados de mercado da B3 via Yahoo Finance")

if not st.session_state.ativos:
    st.info("Adicione um ticker na barra lateral para começar.")
    st.stop()

resultados = []
with st.spinner("Buscando dados…"):
    for tk in st.session_state.ativos:
        try:
            resultados.append(analisar(buscar(tk), metodo, yield_alvo, margem_alvo, dcf_cfg))
        except Exception as e:
            st.warning(f"Não foi possível carregar {tk}: {e}")

if not resultados:
    st.error("Nenhum ativo foi carregado. Verifique os tickers e a conexão.")
    st.stop()

tab_painel, tab_comp, tab_met, tab_dcf, tab_div, tab_fund = st.tabs([
    "🃏 Painel", "📊 Comparativo geral", "⚖️ Métodos", "🧮 Fluxo de caixa descontado",
    "💰 Dividendos", "🔬 Fundamentos",
])

# ------------------------------------------------ Painel
with tab_painel:
    cols_por_linha = 3
    for i in range(0, len(resultados), cols_por_linha):
        cols = st.columns(cols_por_linha)
        for col, r in zip(cols, resultados[i:i + cols_por_linha]):
            with col:
                st.markdown(
                    f"""<div class="card">
                    <span class="badge {r['cls']}">{r['status']}</span>
                    <div class="tk">{r['ticker']}</div>
                    <div class="nm">{r['nome']}</div>
                    <div style="display:flex; gap:22px;">
                      <div><div class="lbl">Preço atual</div><div class="val">{brl(r['preco'])}</div></div>
                      <div><div class="lbl">Preço justo</div><div class="val">{brl(r['justo'])}</div></div>
                    </div>
                    <div style="margin-top:10px;">
                      <div class="lbl">Dividendo médio (5 anos)</div>
                      <div class="val">{brl(r['div_medio'])} <span style="font-size:.75rem;opacity:.6">/ ano</span></div>
                    </div></div>""",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(gauge(r["margem"]), use_container_width=True,
                                key=f"g_{r['ticker']}", config={"displayModeBar": False})
                st.caption(f"Margem de segurança · Graham {brl(r['graham'])} · "
                           f"Bazin {brl(r['bazin'])} · DCF {brl(r['dcf'])}")

# ------------------------------------------------ Comparativo geral
with tab_comp:
    df = pd.DataFrame([{
        "Ticker": r["ticker"], "Preço": r["preco"], "Preço justo": r["justo"],
        "Margem (%)": r["margem"], "P/L": r["pl"], "P/VP": r["pvp"],
        "Status": r["status"],
    } for r in resultados])

    st.subheader(f"Preço atual × preço justo ({metodo_label})")
    fig = go.Figure()
    fig.add_bar(x=df["Ticker"], y=df["Preço"], name="Preço atual", marker_color="#60a5fa")
    fig.add_bar(x=df["Ticker"], y=df["Preço justo"], name="Preço justo", marker_color="#34d399")
    fig.update_layout(barmode="group", height=360, legend=dict(orientation="h", y=1.1),
                      margin=dict(t=30), **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    d2 = df.dropna(subset=["Margem (%)"]).sort_values("Margem (%)")
    if len(d2):
        fig2 = go.Figure(go.Bar(
            x=d2["Margem (%)"], y=d2["Ticker"], orientation="h",
            marker_color=["#f87171" if m < 0 else "#4ade80" if m >= margem_alvo else "#facc15"
                          for m in d2["Margem (%)"]],
            text=[f"{m:.1f}%" for m in d2["Margem (%)"]], textposition="outside"))
        fig2.update_layout(title="Margem de segurança", height=60 + 50 * len(d2), **LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df.style.format({"Preço": "R$ {:.2f}", "Preço justo": "R$ {:.2f}",
                                  "Margem (%)": "{:.1f}%", "P/L": "{:.1f}", "P/VP": "{:.2f}"},
                                 na_rep="—"),
                 use_container_width=True, hide_index=True)

# ------------------------------------------------ Métodos
with tab_met:
    st.subheader("Comparação entre os métodos de valuation")
    tickers = [r["ticker"] for r in resultados]
    fig = go.Figure()
    fig.add_bar(x=tickers, y=[r["preco"] for r in resultados], name="Preço atual", marker_color="#60a5fa")
    fig.add_bar(x=tickers, y=[r["graham"] for r in resultados], name="Graham", marker_color="#a78bfa")
    fig.add_bar(x=tickers, y=[r["bazin"] for r in resultados], name="Bazin", marker_color="#34d399")
    fig.add_bar(x=tickers, y=[r["dcf"] for r in resultados], name="DCF", marker_color="#fbbf24")
    fig.update_layout(barmode="group", height=400, yaxis_title="R$ por ação",
                      legend=dict(orientation="h", y=1.1), margin=dict(t=30), **LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    def potencial(valor, preco):
        return (valor / preco - 1) * 100 if valor and preco else None

    dm = pd.DataFrame([{
        "Ticker": r["ticker"], "Preço": r["preco"],
        "Graham": r["graham"], "Potencial Graham (%)": potencial(r["graham"], r["preco"]),
        "Bazin": r["bazin"], "Potencial Bazin (%)": potencial(r["bazin"], r["preco"]),
        "DCF": r["dcf"], "Potencial DCF (%)": potencial(r["dcf"], r["preco"]),
    } for r in resultados])
    st.dataframe(
        dm.style.format({c: "R$ {:.2f}" for c in ["Preço", "Graham", "Bazin", "DCF"]}
                        | {c: "{:+.1f}%" for c in dm.columns if c.startswith("Potencial")},
                        na_rep="—"),
        use_container_width=True, hide_index=True)
    st.caption("Potencial = quanto o valor do método está acima (+) ou abaixo (−) do preço atual. "
               "Divergências grandes entre os métodos indicam que o ativo merece uma análise mais cuidadosa.")

# ------------------------------------------------ Fluxo de caixa descontado
with tab_dcf:
    st.subheader("Detalhamento do fluxo de caixa descontado")
    tk_sel = st.selectbox("Ativo", [r["ticker"] for r in resultados], key="dcf_ativo")
    r = next(x for x in resultados if x["ticker"] == tk_sel)
    det = r["dcf_det"]

    if det is None:
        st.info(
            f"O DCF não está disponível para {tk_sel}. Isso costuma acontecer quando o fluxo de caixa "
            "livre histórico é negativo ou ausente (comum em bancos e seguradoras, cujo fluxo de caixa "
            "não representa o resultado distribuível), quando não há número de ações disponível ou "
            "quando o patrimônio resultante é negativo. Para essas empresas, Graham e Bazin tendem a "
            "ser mais adequados."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Preço justo (DCF)", brl(r["dcf"]))
        c2.metric("Preço atual", brl(r["preco"]))
        c3.metric("Valor da firma (EV)", bi(det["ev"]))
        c4.metric("Dívida líquida", bi(det["divida_liq"]))

        anos_lbl = [f"Ano {i}" for i in range(1, len(det["fluxos"]) + 1)]
        proj = pd.DataFrame({
            "Ano": anos_lbl,
            "FCL projetado": det["fluxos"],
            "Valor presente": det["vps"],
        })
        cL, cR = st.columns([3, 2])
        with cL:
            fig = go.Figure()
            fig.add_bar(x=anos_lbl, y=det["fluxos"], name="FCL projetado", marker_color="#60a5fa")
            fig.add_bar(x=anos_lbl, y=det["vps"], name="Valor presente", marker_color="#34d399")
            fig.update_layout(barmode="group", height=320, legend=dict(orientation="h", y=1.15),
                              margin=dict(t=30), **LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with cR:
            vp_fluxos = sum(det["vps"])
            peso_perp = det["vp_perp"] / det["ev"] * 100
            st.markdown(
                f"**FCL-base (média dos últimos 3 anos):** {bi(r['fcl_base'])}  \n"
                f"**VP dos fluxos projetados:** {bi(vp_fluxos)}  \n"
                f"**VP da perpetuidade:** {bi(det['vp_perp'])} ({peso_perp:.0f}% do EV)  \n"
                f"**(−) Dívida líquida:** {bi(det['divida_liq'])}  \n"
                f"**Valor para o acionista:** {bi(det['patrimonio'])}  \n"
                f"**Ações em circulação:** {r['acoes']:,.0f}".replace(",", ".")
            )
            if peso_perp > 70:
                st.warning("Mais de 70% do valor vem da perpetuidade: o resultado é muito sensível às premissas.")

        st.dataframe(
            proj.style.format({"FCL projetado": lambda v: bi(v), "Valor presente": lambda v: bi(v)}),
            use_container_width=True, hide_index=True)

        st.markdown("#### Sensibilidade do preço justo (taxa de desconto × crescimento na perpetuidade)")
        taxas = [taxa_desc + k for k in (-2, -1, 0, 1, 2)]
        gps = [g_perp + k for k in (-1, -0.5, 0, 0.5, 1)]
        z = [[calcular_dcf(r, t / 100, cresc_fcl / 100, anos_proj, g / 100)[0] for g in gps] for t in taxas]
        texto = [[brl(v) for v in linha] for linha in z]
        fig_s = go.Figure(go.Heatmap(
            z=z, x=[f"{g:.1f}%" for g in gps], y=[f"{t:.1f}%" for t in taxas],
            text=texto, texttemplate="%{text}", colorscale="RdYlGn",
            zmid=r["preco"] if r["preco"] else None, showscale=False))
        fig_s.update_layout(height=330, xaxis_title="Crescimento na perpetuidade",
                            yaxis_title="Taxa de desconto", margin=dict(t=10), **LAYOUT)
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("Verde = acima do preço atual · Vermelho = abaixo do preço atual. "
                   "O FCL de empresas que reportam em dólar (como a Vale) é convertido para reais pelo câmbio atual.")

# ------------------------------------------------ Dividendos
with tab_div:
    st.subheader("Dividendos por ação (R$/ano)")
    fig3 = go.Figure()
    for r in resultados:
        if len(r["div_ano"]):
            fig3.add_bar(x=[str(a) for a in r["div_ano"].index], y=r["div_ano"].values, name=r["ticker"])
    fig3.update_layout(barmode="group", height=380, **LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    dd = pd.DataFrame([{
        "Ticker": r["ticker"], "Preço": r["preco"], "Dividendo médio (R$/ano)": r["div_medio"],
        "Yield sobre o preço atual (%)": r["dy"], "Anos com dados": len(r["div_ano"]),
        "Preço-teto Bazin": r["bazin"],
    } for r in resultados])
    st.dataframe(
        dd.style.format({"Preço": "R$ {:.2f}", "Dividendo médio (R$/ano)": "R$ {:.2f}",
                         "Yield sobre o preço atual (%)": "{:.1f}%", "Preço-teto Bazin": "R$ {:.2f}"},
                        na_rep="—"),
        use_container_width=True, hide_index=True)

    hist = pd.DataFrame({r["ticker"]: r["div_ano"] for r in resultados if len(r["div_ano"])})
    if not hist.empty:
        hist = hist.sort_index(ascending=False)
        hist.index = hist.index.astype(str)
        hist.index.name = "Ano"
        st.markdown("#### Histórico anual por ação")
        st.dataframe(hist.style.format("R$ {:.2f}", na_rep="—"), use_container_width=True)

# ------------------------------------------------ Fundamentos
with tab_fund:
    st.subheader("Indicadores fundamentalistas")
    dfu = pd.DataFrame([{
        "Ticker": r["ticker"], "LPA": r["lpa"], "VPA": r["vpa"], "P/L": r["pl"], "P/VP": r["pvp"],
        "ROE (LPA/VPA, %)": r["roe"],
        "FCL-base": r["fcl_base"], "Dívida líquida": (r["divida"] - r["caixa"]) if r["acoes"] else None,
    } for r in resultados])
    st.dataframe(
        dfu.style.format({"LPA": "R$ {:.2f}", "VPA": "R$ {:.2f}", "P/L": "{:.1f}", "P/VP": "{:.2f}",
                          "ROE (LPA/VPA, %)": "{:.1f}%", "FCL-base": lambda v: bi(v),
                          "Dívida líquida": lambda v: bi(v)}, na_rep="—"),
        use_container_width=True, hide_index=True)

    fig4 = go.Figure()
    for r in resultados:
        if len(r["fcl_hist"]):
            fig4.add_bar(x=[str(a) for a in r["fcl_hist"].index], y=r["fcl_hist"].values / 1e9, name=r["ticker"])
    if fig4.data:
        fig4.update_layout(barmode="group", title="Fluxo de caixa livre anual (R$ bilhões)", height=380, **LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)
    st.caption("ROE aproximado = LPA ÷ VPA. Os indicadores vêm do Yahoo Finance e podem ter defasagem ou lacunas.")

st.caption("⚠️ Ferramenta educacional. Graham, Bazin e DCF são modelos simplificados e não constituem recomendação de investimento.")
