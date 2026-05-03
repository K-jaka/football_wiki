"""
Nogometni Wiki -- RAG Knowledge Base App
FAMNIT AI Workshop
"""

import streamlit as st
import numpy as np
import re

st.set_page_config(page_title="Nogometni Wiki", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, .stApp { background-color: #f4f6fb; font-family: 'Inter', sans-serif; color: #1e1e2e; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a237e 0%, #283593 60%, #1565c0 100%); border-right: none; padding-top: 1rem; }
    section[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; color: #ffffff !important; }
    h1 { font-family: 'Inter', sans-serif !important; font-weight: 700 !important; font-size: 2.1rem !important; color: #1a237e !important; -webkit-text-fill-color: #1a237e !important; letter-spacing: -0.5px; margin-bottom: 0.2rem !important; }
    h1::after { content: ''; display: block; width: 52px; height: 4px; background: linear-gradient(90deg, #1a237e, #42a5f5); border-radius: 2px; margin-top: 10px; }
    h2, h3 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; color: #1a237e !important; }
    h4 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; color: #1e1e2e !important; }
    .hero-banner { background: linear-gradient(135deg, #1a237e 0%, #1565c0 55%, #0288d1 100%); border-radius: 20px; padding: 52px 44px; margin-bottom: 36px; color: white; position: relative; overflow: hidden; box-shadow: 0 8px 32px rgba(26,35,126,0.22); }
    .hero-banner::before { content: "⚽"; position: absolute; right: 48px; top: 50%; transform: translateY(-50%); font-size: 8rem; opacity: 0.12; pointer-events: none; }
    .hero-banner h1 { color: white !important; -webkit-text-fill-color: white !important; font-size: 2.7rem !important; margin-bottom: 0.6rem !important; }
    .hero-banner h1::after { display: none; }
    .hero-banner p { color: rgba(255,255,255,0.88) !important; font-size: 1.05rem; max-width: 600px; margin: 0; line-height: 1.6; }
    .hero-badge { display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.3); border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; color: white !important; margin-bottom: 18px; font-weight: 500; }
    .feature-card { background: #ffffff; border: 1px solid #e3e8f0; border-radius: 16px; padding: 28px 20px; margin: 4px 0; text-align: center; box-shadow: 0 2px 8px rgba(26,35,126,0.06); transition: transform 0.2s, box-shadow 0.2s; height: 100%; }
    .feature-card:hover { transform: translateY(-4px); box-shadow: 0 10px 28px rgba(26,35,126,0.13); }
    .feature-card .icon { font-size: 2.5rem; margin-bottom: 14px; }
    .feature-card h4 { font-size: 0.95rem; font-weight: 600; color: #1a237e !important; margin-bottom: 8px; }
    .feature-card p { font-size: 0.82rem; color: #5c6370 !important; line-height: 1.5; margin: 0; }
    .topic-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
    .topic-pill { background: #e8eaf6; color: #1a237e !important; border-radius: 20px; padding: 6px 16px; font-size: 0.84rem; font-weight: 500; border: 1px solid #c5cae9; display: inline-block; }
    .compare-card { background: #ffffff; border: 1px solid #e3e8f0; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; }
    .compare-card.active { border-left: 4px solid #1a237e; background: #f0f3ff; }
    .result-card { background: #ffffff; border: 1px solid #e3e8f0; border-left: 4px solid #1a237e; border-radius: 12px; padding: 22px 26px; margin: 14px 0; box-shadow: 0 2px 8px rgba(26,35,126,0.06); }
    .result-rank { display: inline-flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #1a237e, #1565c0); color: white !important; border-radius: 50%; width: 30px; height: 30px; font-size: 0.85rem; font-weight: 700; margin-right: 10px; vertical-align: middle; }
    .relevance-label { font-size: 0.78rem; font-weight: 600; color: #1a237e !important; text-transform: uppercase; letter-spacing: 0.5px; }
    .relevance-bar-wrap { background: #e8eaf6; border-radius: 6px; height: 7px; margin-top: 8px; margin-bottom: 14px; overflow: hidden; }
    .relevance-bar { height: 7px; border-radius: 6px; background: linear-gradient(90deg, #1a237e, #42a5f5); }
    .result-text { font-size: 0.92rem; line-height: 1.75; color: #2c2c3e !important; margin: 0; }
    div[data-testid="metric-container"] { background-color: #ffffff; border: 1px solid #e3e8f0; border-radius: 12px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }
    .stButton > button { background: linear-gradient(135deg, #1a237e, #1565c0); color: white !important; border: none; border-radius: 10px; font-weight: 600; font-family: 'Inter', sans-serif; padding: 0.55rem 1.8rem; font-size: 0.95rem; box-shadow: 0 3px 10px rgba(26,35,126,0.25); transition: opacity 0.2s, transform 0.15s; }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { background-color: #ffffff; border: 1.5px solid #c5cae9; border-radius: 10px; font-family: 'Inter', sans-serif; color: #1e1e2e; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background-color: #e8eaf6; border-radius: 12px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { border-radius: 9px; padding: 8px 20px; font-family: 'Inter', sans-serif; font-weight: 500; color: #5c6370; }
    .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #1a237e !important; box-shadow: 0 2px 6px rgba(26,35,126,0.1); font-weight: 600 !important; }
    .stAlert { border-radius: 12px; font-family: 'Inter', sans-serif; }
    hr { border-color: #e3e8f0; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Nalaganje modela...")
def load_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4}
    )

LONG_SAMPLE_TEXT = """Football (nogomet) je ekipni sport, ki ga igrata dve ekipi z enajstimi igralci z okroglo zogo. Je najbolj priljubljen sport na svetu z vec kot 250 milijoni igralcev v vec kot 200 drzavah. Igra poteka na pravokotnem igrišcu s travo ali umetno travo, z goloma na vsakem koncu.

Svetovno prvenstvo FIFA je najprestiznejs mednarodni nogometni turnir, ki poteka vsaka stiri leta. Brazilija je najuspesnejsa drzava s petimi naslovi, sledita ji Nemcija in Italija s stirimi. Svetovno prvenstvo 2022 v Katarju je dobila Argentina, Lionel Messi pa je koncno osvojil edini pokal, ki mu je manjkal v zbirki.

UEFA Liga prvakov je najprestiznejs klubsko tekmovanje na svetu. Real Madrid ima rekordnih 15 naslovov. Domace lige kot Premier liga, La Liga, Bundesliga in Serie A privabljajo milijarde gledalcev po vsem svetu."""

DOCUMENTS = [
    "Football oziroma nogomet je ekipni sport, ki ga igrata dve ekipi z enajstimi igralci z okroglo zogo. Je najbolj priljubljen sport na svetu z vec kot 250 milijoni igralcev v vec kot 200 drzavah. Igra poteka na pravokotnem igriscu s travo ali umetno travo, z goloma na vsakem koncu. Cilj je zadeti zogo v nasprotnikova vrata. Samo vratar sme uporabljati roke; ostali igralci uporabljajo noge, glavo in telo. Standardna tekma traja 90 minut, razdeljenih na dva polcasa po 45 minut, s 15-minutnim odmorom vmes.",

    "Zgodovina nogometa sega v stare civilizacije, vendar je bila moderna igra kodificirana v Angliji leta 1863, ko je bila ustanovljena Nogometna zveza in dolocena Pravila igre. Pred tem so se po vsej Evropi stoletja igrale razlicne oblike ljudskega nogometa. Cambridgeskа pravila iz leta 1848 so bila pomemben korak k standardizaciji. Prva mednarodna tekma je bila odigrana med Skotsko in Anglijo leta 1872 in se je koncala z izidom 0:0. Sport se je hitro razsiril prek britanskih mornarjev, trgovcev in vojakov ter do zacetka 20. stoletja dosegel Juzno Ameriko, Evropo in ostali svet.",

    "FIFA je mednarodna krovna organizacija nogometa, ustanovljena v Parizu leta 1904. Trenutno ima 211 clanic, vec kot Zdruzeni narodi. FIFA nadzoruje Svetovno prvenstvo FIFA, ki se od leta 1930 odvija vsaka stiri leta, z izjemo let 1942 in 1946 zaradi druge svetovne vojne. Prvo svetovno prvenstvo je gostila Urugvaj, ki je bila tudi zmagovalka. Brazilija ima rekordnih pet naslovov prvaka, sledita ji Nemcija in Italija s stirimi. Turnir je najbolj gledan sportni dogodek na svetu.",

    "Svetovno prvenstvo FIFA je najprestiznejs mednarodni nogometni turnir. Poteka vsaka stiri leta in zajema kvalifikacijsko fazo ter zakljucni turnir. Svetovno prvenstvo 2022 je potekalo v Katarju in ga je dobila Argentina, ki je po 36 letih znova postala prvakinja sveta. Lionel Messi je koncno osvojil pokal, ki mu je manjkal v zbirki. Svetovno prvenstvo 2026 bodo skupaj gostile ZDA, Kanada in Mehika, prvic pa bo nastopilo 48 ekip namesto dosedanjih 32.",

    "UEFA Liga prvakov je najprestiznejs klubsko nogometno tekmovanje na svetu, ki ga organizira UEFA. Ustanovljena je bila leta 1955 kot Evropski pokal in preimenovana leta 1992. Tekmovanje zdruzuje najboljse klube iz evropskih domacih lig. Real Madrid ima rekordnih 15 naslovov. Finale je eden od najbolj gledanih letnih sportnih dogodkov na svetu z vec kot 300 milijoni gledalci. Odmeven finale je bil leta 2005 med Liverpoolom in AC Milanom, ko je Liverpool obrnil rezultat s 3:0 in zmagal po enajstmetrovkah.",

    "Lionel Messi in Cristiano Ronaldo sta splosno priznana kot dva najvecja nogometaša moderne dobe in sta dominirala v sportu vec kot 15 let. Messi, rojen v Rossariu v Argentini leta 1987, je vecino kariere prezivel pri FC Barceloni, kjer je osvojil deset naslovov La Lige in stiri Lige prvakov. Osvojil je rekordnih osem nagrad Zlata zoga. Ronaldo, rojen na Madeiri leta 1985, je igral za Sporting CP, Manchester United, Real Madrid in Juventus. Je najboljsi strelec vseh casov v Ligi prvakov in je prav tako osvojil pet nagrad Zlata zoga. Oba sta dosegla vec kot 800 zadetkov.",

    "Taktika in postavitve v nogometu opisujejo, kako so igralci razporejeni in kako ekipa igra. Najpogostejsa postavitev je 4-3-3, ki jo uporabljajo klubi kot FC Barcelona, s stirimi branilci, tremi vezisti in tremi napadalci. Postavitev 4-4-2 je bila dominantna v angleškem nogometu desetletja. Sistem 3-5-2 uporablja tri centralne branilce in krilne branilce. Totalni nogomet, ki sta ga razvila Ajax in Nizozemska v sedemdesetih letih pod vodstvom Rinusa Michelsa, je bila revolucionarna filozofija, kjer je vsak zunanji igralec lahko prevzel vlogo kateregakoli drugega.",

    "Angleskа Premier liga je najvisjа raven angleškega nogometa in velja za najbolj konkurencno domaco ligo na svetu. Ustanovljena je bila leta 1992. Dvajset klubov tekmuje vsako sezono, pri cemer se tri najslabse ekipe izselijo. Manchester United ima rekordnih 13 naslovov. Liga je znana po hitrem tempu, fizicnosti in mednarodnem talentu. V sezoni 2023/24 je nastopilo vec kot 700 igralcev iz vec kot 100 narodnosti. Globalne pogodbe o prenosu jo omogocajo v 188 drzavah in ustvarjajo milijarde prihodkov.",

    "Pravilo prepovedanega polozaja je eno od temeljnih in najbolj spornih pravil v nogometu. Igralec je v prepovedanem polozaju, ce je blizje nasprotnikovi golovi crti kot zoga in predzadnji nasprotnik v trenutku podaje. Biti v prepovedanem polozaju samo po sebi ni prekrsek — igralec je kaznovan le, ce je aktivno vkljucen v igro. Pravilo je bilo uvedeno, da bi preprecilo cakanje pri nasprotnikovih vratih. Od leta 2019 se v mnogih tekmovanjih uporablja tehnologija VAR za natancnejse preverjanje odlocitev.",

    "Nogometni stadioni so ikonične zgradbe, ki sluzijo kot domaci tereni klubov in reprezentanc. Najvecji stadion na svetu po zmogljivosti je stadion Rungrado 1. maja v Pjongjangu s 114.000 sedzei. V Evropi je Camp Nou v Barceloni sprejel vec kot 99.000 gledalcev. Stadion Wembley v Londonu, obnovljen leta 2007, sprejme 90.000 in je znan kot dom angleškega nogometa. Maracana v Rio de Janeiru je bil zgrajen za svetovno prvenstvo 1950 in je nekoc sprejel skoraj 200.000 ljudi.",
]

st.sidebar.markdown("""
<div style='text-align:center; padding: 12px 0 24px 0;'>
    <div style='font-size:2.8rem;'>&#9917;</div>
    <div style='font-size:1.15rem; font-weight:700; letter-spacing:0.3px;'>Nogometni Wiki</div>
    <div style='font-size:0.75rem; opacity:0.7; margin-top:4px;'>Semanticni iskalnik</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigacija", ["Domov", "Embeddings", "Razdruževanje", "Iskanje"])

st.sidebar.markdown("""
<div style='margin-top:40px; text-align:center; font-size:0.72rem; opacity:0.55; padding: 0 16px;'>
    paraphrase-multilingual<br>MiniLM-L12-v2<br>
    <span style='opacity:0.7;'>FAMNIT AI Workshop</span>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# HOME
# =========================================================================
if page == "Domov":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">&#127757; Vecjezicni &nbsp;&middot;&nbsp; Slovenscina &nbsp;&middot;&nbsp; AI-powered</div>
        <h1>&#9917; Nogometni Wiki</h1>
        <p>Semanticni iskalnik po nogometnem znanju &mdash; poisci odgovore po <strong style='color:white;'>pomenu</strong>, ne le po kljucnih besedah. Zgrajen na 10 slovenskih clankih iz Wikipedije.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("&#128196;", "10 Dokumentov", "Clanki o nogometu iz Wikipedije v slovenscini"),
        ("&#9986;&#65039;", "Razdruževanje", "Besedilo se razdeli na manjse koscke za boljse iskanje"),
        ("&#128290;", "Embeddings", "Besedilo se pretvori v vektorje, ki zajamejo pomen"),
        ("&#128269;", "Semanticno iskanje", "Iskanje po pomenu z vecjezicnim AI modelom"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="icon">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### O aplikaciji")
        st.markdown("""
**Nogometni Wiki** je semanticni iskalnik, zgrajen na osnovi **10 dokumentov iz Wikipedije**,
ki pokrivajo najpomembnejse teme v svetu nogometa — od zgodovine igre in FIFA do taktike,
slavnih igralcev in ikonicnih stadionov.

Namesto ujemanja kljucnih besed razume **pomen** vasega vprasanja in poisce
najpomembnejse znanje — omogoceno z vecjezicnim modelom
`paraphrase-multilingual-MiniLM-L12-v2` in vektorsko bazo ChromaDB.
        """)

    with right:
        st.markdown("### Teme")
        st.markdown("""
        <div class="topic-grid">
            <span class="topic-pill">&#9917; Pravila igre</span>
            <span class="topic-pill">&#128220; Zgodovina</span>
            <span class="topic-pill">&#127757; FIFA &amp; SP</span>
            <span class="topic-pill">&#127942; Liga prvakov</span>
            <span class="topic-pill">&#127775; Messi &amp; Ronaldo</span>
            <span class="topic-pill">&#129504; Taktika</span>
            <span class="topic-pill">&#127959; Premier liga</span>
            <span class="topic-pill">&#128208; Prepovedani polozaj</span>
            <span class="topic-pill">&#127967; Stadioni</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Uporabite stransko vrstico za navigacijo. Pojdite na Iskanje za postavljanje vprasanj!")

# =========================================================================
# EMBEDDINGS
# =========================================================================
elif page == "Embeddings":
    st.title("Kaj so Embeddings?")
    st.markdown("""
    **Embedding** pretvori besedilo v seznam stevil (vektor), ki zajame njegov **pomen**.
    Podobna besedila dobijo podobne vektorje — tudi ce uporabljajo popolnoma razlicne besede.

    ```
    "Messi je zmagal na SP"          --> [0.12, -0.45, 0.78, ...] (384 stevilk)
    "Argentina je postala prvakinja" --> [0.11, -0.43, 0.76, ...] (podobno!)
    "Pravilo prepovedanega polozaja" --> [-0.67, 0.22, -0.11, ...] (zelo razlicno)
    ```
    """)

    st.divider()
    st.subheader("Del 1: Primerjaj dve besedili")

    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_input("Besedilo A", "Messi je najboljsi nogometaš vseh casov")
    with col2:
        text_b = st.text_input("Besedilo B", "Ronaldo je najboljsi igralec v zgodovini nogometa")

    if st.button("Primerjaj", type="primary"):
        model = load_model()
        vectors = np.array(model.embed_documents([text_a, text_b]))
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(vectors)[0, 1]

        col_metric, col_interp = st.columns(2)
        with col_metric:
            st.metric("Kosinusna podobnost", f"{sim:.3f}")
        with col_interp:
            if sim > 0.7:
                st.success("Ti besedili sta si zelo podobni po pomenu!")
            elif sim > 0.4:
                st.info("Ti besedili delita nekaj pomena.")
            else:
                st.warning("Ti besedili sta si precej razlicni.")

        st.write(f"Vsako besedilo je bilo pretvorjeno v vektor **{len(vectors[0])}** stevilk.")
        with st.expander("Prikaži surove vektorje"):
            st.code(f"Besedilo A: {vectors[0][:10].tolist()}... ({len(vectors[0])} dimenzij)")
            st.code(f"Besedilo B: {vectors[1][:10].tolist()}... ({len(vectors[1])} dimenzij)")

    st.divider()
    st.subheader("Del 2: Matrika podobnosti")
    st.markdown("Vnesite vec nogometnih stavkov (en na vrstico) in preverite, kako so si med seboj podobni.")

    extra_texts = st.text_area(
        "Besedila za primerjavo (en stavek na vrstico)",
        "Messi je osvojil Zlato zogo osemkrat\n"
        "Ronaldo je dosegel vec kot 800 zadetkov\n"
        "Pravilo prepovedanega polozaja preprecuje cakanje pri vratih\n"
        "VAR se uporablja za preverjanje sodniskih odlocitev\n"
        "Real Madrid ima 15 naslovov Lige prvakov\n"
        "Stadion Wembley sprejme 90.000 gledalcev",
        height=150
    )

    if st.button("Zgradi matriko podobnosti", type="primary"):
        model = load_model()
        all_texts = [t.strip() for t in extra_texts.strip().split("\n") if t.strip()]
        if len(all_texts) < 2:
            st.error("Vnesite vsaj 2 besedili.")
        else:
            with st.spinner("Racunam embeddings..."):
                vectors = np.array(model.embed_documents(all_texts))
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(vectors)
            import plotly.express as px
            labels = [t[:40] + ("..." if len(t) > 40 else "") for t in all_texts]
            fig = px.imshow(sim_matrix, x=labels, y=labels, text_auto=".2f",
                            color_continuous_scale="Blues",
                            title="Matrika kosinusne podobnosti — Nogometni stavki", aspect="auto")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Vgrajenih {len(all_texts)} besedil v {vectors.shape[1]}-dimenzionalne vektorje.")

# =========================================================================
# CHUNKING
# =========================================================================
elif page == "Razdruževanje":
    st.title("Strategije razdruževanja")
    st.markdown("""
    Nogometni clanki so predolgi za neposredno vgrajevanje. Razdelimo jih na
    **koscke** — manjse dele, ki vsak dobi svoj vektor.

    Velikost koscka vpliva na kakovost iskanja:
    - **Premajhni** = koscki izgubijo kontekst
    - **Preveliki** = pomen se razredci
    - **Ravno prav** = vsak koscek zajame eno koherentno idejo
    """)

    sample_text = st.text_area("Besedilo za razdruževanje:", value=LONG_SAMPLE_TEXT, height=200)
    st.write(f"**Skupna dolzina:** {len(sample_text)} znakov")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Fiksna velikost", "Na osnovi stavkov", "LangChain Rekurzivno"])

    with tab1:
        st.markdown("**Fiksna velikost** razdeli besedilo vsakih N znakov, ne glede na vsebino.")
        chunk_size = st.slider("Velikost koscka (znaki)", 50, 500, 200, key="fixed_size")
        overlap = st.slider("Prekrivanje (znaki)", 0, 100, 30, key="fixed_overlap")
        chunks = []
        start = 0
        while start < len(sample_text):
            chunks.append(sample_text[start:start + chunk_size])
            start += chunk_size - overlap
            if overlap >= chunk_size:
                break
        st.write(f"**{len(chunks)} kosckov** ustvarjenih")
        for i, chunk in enumerate(chunks):
            st.text_area(f"Koscek {i+1} ({len(chunk)} znakov)", chunk, height=80, key=f"fixed_{i}", disabled=True)

    with tab2:
        st.markdown("**Na osnovi stavkov** razdeli pri mejah stavkov. Vsak koscek je cel stavek.")
        sentences = re.split(r'(?<=[.!?])\s+', sample_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        st.write(f"**{len(sentences)} stavkov** najdenih")
        for i, sent in enumerate(sentences):
            st.text_area(f"Stavek {i+1} ({len(sent)} znakov)", sent, height=60, key=f"sent_{i}", disabled=True)

    with tab3:
        st.markdown("**LangChain RecursiveCharacterTextSplitter** najprej razdeli pri odstavkih, nato stavkih, nato besedah.")
        rc_size = st.slider("Velikost koscka", 50, 500, 200, key="rc_size")
        rc_overlap = st.slider("Prekrivanje", 0, 100, 30, key="rc_overlap")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=rc_size, chunk_overlap=rc_overlap)
        rc_chunks = splitter.split_text(sample_text)
        st.write(f"**{len(rc_chunks)} kosckov** ustvarjenih")
        for i, chunk in enumerate(rc_chunks):
            st.text_area(f"Koscek {i+1} ({len(chunk)} znakov)", chunk, height=80, key=f"rc_{i}", disabled=True)

# =========================================================================
# SEARCH
# =========================================================================
elif page == "Iskanje":
    st.title("Iskanje po nogometnem znanju")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="compare-card">
            <strong style="color:#888;">Klasicno iskanje</strong><br>
            <span style="font-size:0.85rem; color:#5c6370;">Išce tocno ujemanje besed v besedilu</span>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="compare-card active">
            <strong style="color:#1a237e;">Semanticno iskanje (aktivno)</strong><br>
            <span style="font-size:0.85rem; color:#1a237e;">Razume pomen — najde relevantne odgovore brez tocnih besed</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Prikaži vseh 10 dokumentov"):
        for i, doc in enumerate(DOCUMENTS, 1):
            st.markdown(f"**{i}.** {doc.strip()}")
            st.markdown("---")

    st.divider()

    query = st.text_input(
        "Zastavi vprašanje o nogometu",
        placeholder="npr. 'Kdo je najboljsi nogometaš vseh casov?' ali 'Kako deluje pravilo prepovedanega polozaja?'"
    )

    col_slider, col_settings = st.columns([1, 1])
    with col_slider:
        num_results = st.slider("Stevilo rezultatov", 1, 5, 3, key="num_results")
    with col_settings:
        with st.expander("Nastavitve razdruževanja"):
            chunk_size = st.slider("Velikost koscka (znaki)", 100, 1000, 500, key="search_chunk_size")
            chunk_overlap = st.slider("Prekrivanje (znaki)", 0, 200, 50, key="search_chunk_overlap")

    if st.button("Išci", type="primary") and query:
        model = load_model()

        with st.spinner("Iscemo po bazi znanja..."):
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.create_documents(DOCUMENTS)
            chunk_texts = [c.page_content.strip() for c in chunks]

            vectorstore = Chroma.from_texts(texts=chunk_texts, embedding=model)

            # similarity_search_with_score returns L2 distance (lower = better)
            raw_results = vectorstore.similarity_search_with_score(query, k=num_results * 5)

            seen_texts = set()
            results = []
            for doc, distance in raw_results:
                key = doc.page_content.strip()[:120]
                if key not in seen_texts:
                    seen_texts.add(key)
                    # Convert distance to similarity: closer to 1 = more relevant
                    similarity = 1 / (1 + distance)
                    results.append((doc, similarity))
                if len(results) == num_results:
                    break

            # Normalise so best result = 100%
            if results:
                max_sim = max(s for _, s in results)
                if max_sim > 0:
                    results = [(doc, sim / max_sim) for doc, sim in results]

        st.caption(f"Razdeljeno {len(DOCUMENTS)} dokumentov na **{len(chunk_texts)} kosckov** "
                   f"(velikost={chunk_size}, prekrivanje={chunk_overlap})")

        st.markdown(f"### Rezultati za: *\"{query}\"*")

        for i, (doc, score) in enumerate(results, 1):
            pct = round(score * 100)
            bar_width = pct
            st.markdown(f"""
            <div class="result-card">
                <span class="result-rank">{i}</span>
                <span class="relevance-label">Ustreznost: {pct:.0f}%</span>
                <div class="relevance-bar-wrap">
                    <div class="relevance-bar" style="width:{bar_width}%;"></div>
                </div>
                <p class="result-text">{doc.page_content.strip()}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.info("""
Preizkusite ta vprašanja:
- "Kdo je najboljsi nogometaš vseh casov?" — Messi in Ronaldo
- "Kako deluje pravilo prepovedanega polozaja?" — VAR in pravila
- "Katera drzava je najveckrat zmagala na SP?" — FIFA in SP
- "Kateri je najvecji stadion?" — stadioni
- "Kako se je nogomet razsiril po svetu?" — zgodovina
- "Katero je najprestiznejs klubsko tekmovanje?" — Liga prvakov
        """)
