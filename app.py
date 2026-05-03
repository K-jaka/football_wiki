"""
Nogometni Wiki -- RAG Knowledge Base App
FAMNIT AI Workshop
"""

import streamlit as st
import numpy as np
import re

st.set_page_config(page_title="Football Wiki", page_icon="⚽", layout="wide")

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
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { background-color: #ffffff; border: 1.5px solid #c5cae9; border-radius: 10px; font-family: 'Inter', sans-serif; color: #1e1e2e !important; font-size: 0.95rem; }
    .stTextArea textarea { color: #1e1e2e !important; background-color: #ffffff !important; }
    .stTextArea textarea:disabled { color: #1e1e2e !important; -webkit-text-fill-color: #1e1e2e !important; opacity: 1 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; background-color: #e8eaf6; border-radius: 12px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { border-radius: 9px; padding: 8px 20px; font-family: 'Inter', sans-serif; font-weight: 500; color: #5c6370; }
    .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #1a237e !important; box-shadow: 0 2px 6px rgba(26,35,126,0.1); font-weight: 600 !important; }
    .stAlert { border-radius: 12px; font-family: 'Inter', sans-serif; }
    hr { border-color: #e3e8f0; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

LONG_SAMPLE_TEXT = """Football is the world's most popular sport, played by over 250 million players across more than 200 countries. The game is governed internationally by FIFA, which was founded in Paris in 1904 and currently has 211 member associations. The modern rules of the game were codified in England in 1863 when the Football Association was established.

The FIFA World Cup is the most prestigious football tournament in the world, held every four years since 1930. Brazil is the most successful nation with five titles, followed by Germany and Italy with four each. The 2022 tournament in Qatar was won by Argentina, with Lionel Messi finally claiming the one trophy missing from his collection.

At club level, the UEFA Champions League is the most watched annual club competition in the world. Real Madrid holds the record with 15 titles. Domestically, leagues such as the English Premier League, La Liga, Bundesliga, and Serie A attract billions of viewers worldwide."""

DOCUMENTS = [
    """Football (also known as association football or soccer) is a team sport played between two teams
    of eleven players using a spherical ball. It is the world's most popular sport, with over 250 million
    players in more than 200 countries. The game is played on a rectangular grass or artificial turf pitch
    with a goal at each end. The objective is to score by getting the ball into the opposing team's goal.
    Only the goalkeeper may use their hands; outfield players use their feet, head, and body. A standard
    match lasts 90 minutes divided into two 45-minute halves, with a 15-minute break in between.""",

    """The history of football dates back to ancient civilizations, but the modern game was codified in
    England in 1863 when the Football Association was founded and the Laws of the Game were established.
    The Cambridge Rules of 1848 were an important step toward standardization. The first international
    match was played between Scotland and England in 1872, ending 0-0. The sport spread rapidly through
    British sailors, traders, and soldiers, reaching South America, Europe, and beyond by the early 20th century.""",

    """FIFA (Federation Internationale de Football Association) is the international governing body of
    football, founded in Paris in 1904. It currently has 211 member associations, more than the United
    Nations. FIFA oversees the FIFA World Cup, held every four years since 1930. Brazil holds the record
    for the most World Cup titles with five, followed by Germany and Italy with four each.
    The tournament is the most-watched sporting event in the world.""",

    """The FIFA World Cup is the most prestigious international football tournament. The 2022 World Cup was
    held in Qatar and was won by Argentina, ending a 36-year wait for the nation's third title. Lionel
    Messi finally won his first World Cup trophy, cementing his legacy. The 2026 World Cup will be jointly
    hosted by the United States, Canada, and Mexico, and will be the first tournament to feature 48 teams
    instead of the previous 32, with 104 matches across 16 host cities.""",

    """The UEFA Champions League is the most prestigious club football competition in the world, organized
    by UEFA. It was founded in 1955 as the European Cup and rebranded in 1992. Real Madrid holds the record
    for the most Champions League titles with 15. The final regularly attracts over 300 million television
    viewers. A famous final was Liverpool vs AC Milan in 2005, when Liverpool overturned a 3-0 deficit
    to win on penalties in one of the greatest comebacks in football history.""",

    """Lionel Messi and Cristiano Ronaldo are widely considered the two greatest footballers of the modern
    era, dominating the sport for over 15 years. Messi, born in Rosario, Argentina in 1987, spent most of
    his career at FC Barcelona, winning ten La Liga titles and four Champions League trophies. He has won
    the Ballon d'Or a record eight times. Ronaldo, born in Madeira, Portugal in 1985, played for Sporting CP,
    Manchester United, Real Madrid, and Juventus. He is the all-time top scorer in Champions League history
    and has won five Ballon d'Or awards. Both players have scored over 800 career goals.""",

    """Football tactics and formations describe how players are positioned on the pitch. The most common
    formation is 4-3-3, used by clubs like FC Barcelona, featuring four defenders, three midfielders, and
    three forwards. The 4-4-2 was dominant for decades in English football. Total Football, developed by
    Ajax and the Netherlands in the 1970s under Rinus Michels, was a revolutionary philosophy where any
    outfield player could take over the role of any other player.""",

    """The English Premier League is the top tier of English football and the most-watched domestic league
    in the world, founded in 1992. Twenty clubs compete each season with the bottom three relegated.
    Manchester United holds the record with 13 titles. In the 2023-24 season, over 700 players from more
    than 100 nationalities participated. Global broadcast deals make it visible in 188 countries,
    generating billions in revenue each season.""",

    """The offside rule is one of the most debated rules in football. A player is in an offside position
    if they are nearer to the opponent's goal line than both the ball and the second-last opponent when
    the ball is played to them. Since 2019, VAR (Video Assistant Referee) technology has been used in
    many competitions to check offside decisions with greater precision, though this has sparked controversy
    due to marginal calls decided by millimeters.""",

    """Football stadiums are iconic structures serving as homes for clubs and national teams. The largest
    stadium by capacity is Rungrado 1st of May Stadium in Pyongyang, North Korea, holding 114,000.
    Camp Nou in Barcelona held over 99,000. Wembley Stadium in London holds 90,000 and is known as
    the home of English football. The Maracana in Rio de Janeiro was built for the 1950 World Cup and
    once held nearly 200,000 people.""",
]

st.sidebar.markdown("""
<div style='text-align:center; padding: 12px 0 24px 0;'>
    <div style='font-size:2.8rem;'>&#9917;</div>
    <div style='font-size:1.15rem; font-weight:700; letter-spacing:0.3px;'>Football Wiki</div>
    <div style='font-size:0.75rem; opacity:0.7; margin-top:4px;'>Semantic Search Engine</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", ["Home", "Embeddings", "Chunking", "Search"])

st.sidebar.markdown("""
<div style='margin-top:40px; text-align:center; font-size:0.72rem; opacity:0.55; padding: 0 16px;'>
    all-MiniLM-L6-v2<br>
    <span style='opacity:0.7;'>FAMNIT AI Workshop</span>
</div>
""", unsafe_allow_html=True)

# =========================================================================
# HOME
# =========================================================================
if page == "Home":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">&#9917; Football Knowledge &nbsp;&middot;&nbsp; Semantic Search &nbsp;&middot;&nbsp; AI-powered</div>
        <h1>&#9917; Football Wiki</h1>
        <p>A semantic search engine for football knowledge &mdash; find answers by <strong style='color:white;'>meaning</strong>, not just keywords. Built on 30 Wikipedia articles about football.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("&#128196;", "10 Documents", "Football articles sourced from Wikipedia"),
        ("&#9986;&#65039;", "Chunking", "Documents are split into smaller pieces for better search"),
        ("&#128290;", "Embeddings", "Text is converted into vectors that capture meaning"),
        ("&#128269;", "Semantic Search", "Search by meaning using an AI embedding model"),
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
        st.markdown("### About")
        st.markdown("""
**Football Wiki** is a semantic search engine built on **30 Wikipedia articles** covering the most important topics in football — from the history of the game and FIFA, to tactics, famous players, and iconic stadiums.

Instead of matching keywords, it understands the **meaning** of your question and finds the most relevant football knowledge — powered by `all-MiniLM-L6-v2` and ChromaDB.
        """)

    with right:
        st.markdown("### Topics")
        st.markdown("""
        <div class="topic-grid">
            <span class="topic-pill">&#9917; Rules of the game</span>
            <span class="topic-pill">&#128220; History</span>
            <span class="topic-pill">&#127757; FIFA &amp; World Cup</span>
            <span class="topic-pill">&#127942; Champions League</span>
            <span class="topic-pill">&#127775; Messi &amp; Ronaldo</span>
            <span class="topic-pill">&#129504; Tactics</span>
            <span class="topic-pill">&#127959; Premier League</span>
            <span class="topic-pill">&#128208; Offside rule</span>
            <span class="topic-pill">&#127967; Stadiums</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Use the sidebar to navigate. Head to Search to ask football questions!")

# =========================================================================
# EMBEDDINGS
# =========================================================================
elif page == "Embeddings":
    st.title("What Are Embeddings?")
    st.markdown("""
    **Embedding** pretvori besedilo v seznam stevil (vektor), ki zajame njegov **pomen**.
    Podobna besedila dobijo podobne vektorje — tudi ce uporabljajo popolnoma razlicne besede.

    ```
    "Messi won the World Cup"        --> [0.12, -0.45, 0.78, ...] (384 numbers)
    "Argentina became champions"     --> [0.11, -0.43, 0.76, ...] (similar!)
    "The offside rule explained"     --> [-0.67, 0.22, -0.11, ...] (very different)
    ```
    """)

    st.divider()
    st.subheader("Part 1: Compare two texts")

    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_input("Text A", "Messi is the greatest footballer of all time")
    with col2:
        text_b = st.text_input("Text B", "Ronaldo is the best player in football history")

    if st.button("Compare", type="primary"):
        model = load_model()
        vectors = np.array(model.embed_documents([text_a, text_b]))
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(vectors)[0, 1]

        col_metric, col_interp = st.columns(2)
        with col_metric:
            st.metric("Cosine Similarity", f"{sim:.3f}")
        with col_interp:
            if sim > 0.7:
                st.success("These texts are very similar in meaning!")
            elif sim > 0.4:
                st.info("These texts share some meaning.")
            else:
                st.warning("These texts are quite different.")

        st.write(f"Each text was converted to a vector of **{len(vectors[0])}** numbers.")
        with st.expander("Show raw vectors"):
            st.code(f"Text A: {vectors[0][:10].tolist()}... ({len(vectors[0])} dims)")
            st.code(f"Text B: {vectors[1][:10].tolist()}... ({len(vectors[1])} dims)")

    st.divider()
    st.subheader("Part 2: Similarity Matrix")
    st.markdown("Enter multiple football sentences (one per line) to see how they relate to each other.")

    extra_texts = st.text_area(
        "Texts to compare (one per line)",
        "Messi won the Ballon d'Or eight times\n"
        "Ronaldo scored over 800 career goals\n"
        "The offside rule prevents goal-hanging\n"
        "VAR is used to check referee decisions\n"
        "Real Madrid has won 15 Champions League titles\n"
        "Wembley Stadium holds 90,000 spectators",
        height=150
    )

    if st.button("Build Similarity Matrix", type="primary"):
        model = load_model()
        all_texts = [t.strip() for t in extra_texts.strip().split("\n") if t.strip()]
        if len(all_texts) < 2:
            st.error("Please enter at least 2 texts.")
        else:
            with st.spinner("Computing embeddings..."):
                vectors = np.array(model.embed_documents(all_texts))
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(vectors)
            import plotly.express as px
            labels = [t[:40] + ("..." if len(t) > 40 else "") for t in all_texts]
            fig = px.imshow(sim_matrix, x=labels, y=labels, text_auto=".2f",
                            color_continuous_scale="Blues",
                            title="Cosine Similarity Matrix — Football Sentences", aspect="auto")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Embedded {len(all_texts)} texts into {vectors.shape[1]}-dimensional vectors.")

# =========================================================================
# CHUNKING
# =========================================================================
elif page == "Chunking":
    st.title("Chunking Strategies")
    st.markdown("""
    Football articles are too long to embed as a single block. We split them into
    **chunks** — smaller pieces that each get their own vector.

    Chunk size affects search quality:
    - **Too small** = chunks lose context
    - **Too big** = meaning gets diluted
    - **Just right** = each chunk captures one coherent football topic
    """)

    sample_text = st.text_area("Text to chunk:", value=LONG_SAMPLE_TEXT, height=200)
    st.write(f"**Total length:** {len(sample_text)} characters")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Fixed-Size", "Sentence-Based", "LangChain Recursive"])

    with tab1:
        st.markdown("**Fixed-size chunking** splits text every N characters regardless of content. Simple but may cut mid-word or mid-sentence.")
        chunk_size = st.slider("Chunk size (characters)", 50, 500, 200, key="fixed_size")
        overlap = st.slider("Overlap (characters)", 0, 100, 30, key="fixed_overlap")
        chunks = []
        start = 0
        while start < len(sample_text):
            chunks.append(sample_text[start:start + chunk_size])
            start += chunk_size - overlap
            if overlap >= chunk_size:
                break
        st.write(f"**{len(chunks)} chunks** created")
        for i, chunk in enumerate(chunks):
            st.text_area(f"Chunk {i+1} ({len(chunk)} chars)", chunk, height=80, key=f"fixed_{i}", disabled=True)

    with tab2:
        st.markdown("**Sentence-based chunking** splits at sentence boundaries. Each chunk is a complete sentence, preserving grammatical structure.")
        sentences = re.split(r'(?<=[.!?])\s+', sample_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        st.write(f"**{len(sentences)} sentences** found")
        for i, sent in enumerate(sentences):
            st.text_area(f"Sentence {i+1} ({len(sent)} chars)", sent, height=60, key=f"sent_{i}", disabled=True)

    with tab3:
        st.markdown("**LangChain RecursiveCharacterTextSplitter** first tries to split at paragraph boundaries, then sentences, then words — keeping football context intact.")
        rc_size = st.slider("Chunk size", 50, 500, 200, key="rc_size")
        rc_overlap = st.slider("Overlap", 0, 100, 30, key="rc_overlap")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=rc_size, chunk_overlap=rc_overlap)
        rc_chunks = splitter.split_text(sample_text)
        st.write(f"**{len(rc_chunks)} chunks** created")
        for i, chunk in enumerate(rc_chunks):
            st.text_area(f"Chunk {i+1} ({len(chunk)} chars)", chunk, height=80, key=f"rc_{i}", disabled=True)

# =========================================================================
# SEARCH
# =========================================================================
elif page == "Search":
    st.title("Football Knowledge Search")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="compare-card">
            <strong style="color:#888;">Keyword Search</strong><br>
            <span style="font-size:0.85rem; color:#5c6370;">Finds exact word matches in the text</span>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="compare-card active">
            <strong style="color:#1a237e;">Semantic Search (aktivno)</strong><br>
            <span style="font-size:0.85rem; color:#1a237e;">Understands meaning — finds relevant answers without exact words</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("Show all 30 football documents"):
        for i, doc in enumerate(DOCUMENTS, 1):
            st.markdown(f"**{i}.** {doc.strip()}")
            st.markdown("---")

    st.divider()

    query = st.text_input(
        "Ask a football question",
        placeholder="e.g. 'Who is the best player ever?' or 'How does the offside rule work?'"
    )

    col_slider, col_settings = st.columns([1, 1])
    with col_slider:
        num_results = st.slider("Number of results", 1, 5, 3, key="num_results")
    with col_settings:
        with st.expander("Chunking settings"):
            chunk_size = st.slider("Chunk size (characters)", 100, 1000, 500, key="search_chunk_size")
            chunk_overlap = st.slider("Overlap (characters)", 0, 200, 50, key="search_chunk_overlap")

    if st.button("Search", type="primary") and query:
        model = load_model()

        with st.spinner("Searching the knowledge base..."):
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = splitter.create_documents(DOCUMENTS)
            chunk_texts = [c.page_content.strip() for c in chunks]

            # Build ChromaDB vector store using LangChain
            from langchain_community.vectorstores import Chroma
            vectorstore = Chroma.from_texts(texts=chunk_texts, embedding=model)

            # Get all candidate docs from ChromaDB
            candidate_docs = vectorstore.similarity_search(query, k=len(chunk_texts))

            # Re-rank using cosine similarity manually for accurate scores
            query_vec = np.array(model.embed_query(query))
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

            scored = []
            for doc in candidate_docs:
                text = doc.page_content.strip()
                if len(text) < 80:
                    continue
                vec = np.array(model.embed_query(text))
                vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
                cosine_score = float(np.dot(query_norm, vec_norm))
                scored.append((doc, cosine_score))

            # Sort by cosine similarity descending
            scored.sort(key=lambda x: x[1], reverse=True)

            # Deduplicate
            seen_texts = set()
            results_raw = []
            for doc, score in scored:
                key = doc.page_content.strip()[:120]
                if key not in seen_texts:
                    seen_texts.add(key)
                    results_raw.append((doc, score))
                if len(results_raw) == num_results:
                    break

            # Scale scores so best = 100% and worst shown result = 60%
            if results_raw:
                max_s = results_raw[0][1]
                min_s = results_raw[-1][1]
                rng = max_s - min_s if max_s != min_s else 1.0
                results = [(doc, 0.6 + 0.4 * (s - min_s) / rng) for doc, s in results_raw]
            else:
                results = []

        st.caption(f"Split {len(DOCUMENTS)} documents into **{len(chunk_texts)} chunks** (chunk_size={chunk_size}, overlap={chunk_overlap})")

        st.markdown(f"### Results for: *\"{query}\"*")

        for i, (doc, score) in enumerate(results, 1):
            pct = round(max(0, score) * 100)
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
Try these queries:
- "Who is the best player ever?" — Messi & Ronaldo
- "How does the offside rule work?" — offside & VAR
- "Which country won the most World Cups?" — FIFA doc
- "What is the biggest stadium?" — stadiums
- "How did football spread around the world?" — history
- "What is the most prestigious club competition?" — Champions League
        """)
