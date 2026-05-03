"""
Football Wiki -- RAG Knowledge Base App
FAMNIT AI Workshop

An interactive Streamlit application for exploring football knowledge using:
- Embeddings and semantic similarity
- Document chunking strategies
- Vector search with ChromaDB
"""

import streamlit as st
import numpy as np
import re

# =========================================================================
# PAGE CONFIG (must be first Streamlit command)
# =========================================================================
st.set_page_config(page_title="Football Wiki", page_icon="⚽", layout="wide")

# =========================================================================
# CUSTOM CSS
# =========================================================================
st.markdown('''
<style>
    .stApp {
        background-color: #0f0f1a;
    }
    section[data-testid="stSidebar"] {
        background-color: #0a0a14;
        border-right: 1px solid #1a1a2e;
    }
    h1 {
        background: linear-gradient(120deg, #4285f4, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 10px;
        padding: 15px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4285f4, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    .stCodeBlock { border-radius: 10px; }
    .stAlert { border-radius: 10px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; }
    .workshop-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4e;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
</style>
''', unsafe_allow_html=True)

# =========================================================================
# CACHED RESOURCES
# =========================================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# =========================================================================
# SAMPLE TEXT FOR CHUNKING DEMO (football-themed)
# =========================================================================
LONG_SAMPLE_TEXT = """Football (also known as association football or soccer) is the world's most \
popular sport, played by over 250 million players across more than 200 countries. The game is \
governed internationally by FIFA, which was founded in Paris in 1904 and currently has 211 member \
associations. The modern rules of the game were codified in England in 1863 when the Football \
Association was established and the Laws of the Game were first written down.

The FIFA World Cup is the most prestigious football tournament in the world, held every four years \
since 1930. Brazil is the most successful nation with five titles, followed by Germany and Italy \
with four each. The 2022 tournament in Qatar was won by Argentina, with Lionel Messi finally \
claiming the one trophy missing from his collection. The 2026 World Cup will be co-hosted by the \
United States, Canada, and Mexico, and will expand to 48 teams for the first time in history.

At club level, the UEFA Champions League is the most watched annual club competition in the world. \
Real Madrid holds the record with 15 titles. Domestically, leagues such as the English Premier \
League, Spain's La Liga, Germany's Bundesliga, and Italy's Serie A attract billions of viewers \
worldwide. Tactics, formations, player development, and transfer markets have turned football into \
a multi-billion euro industry that touches nearly every country on the planet."""


# =========================================================================
# FOOTBALL KNOWLEDGE BASE
# =========================================================================
DOCUMENTS = [
    """Football (also known as association football or soccer) is a team sport played between two teams 
    of eleven players using a spherical ball. It is the world's most popular sport, with over 250 million 
    players in more than 200 countries. The game is played on a rectangular grass or artificial turf pitch 
    with a goal at each end. The objective is to score by getting the ball into the opposing team's goal. 
    Only the goalkeeper may use their hands; outfield players use their feet, head, and body. A standard 
    match lasts 90 minutes divided into two 45-minute halves, with a 15-minute break in between.""",

    """The history of football dates back to ancient civilizations, but the modern game was codified in 
    England in 1863 when the Football Association was founded and the Laws of the Game were established. 
    Before this, various forms of folk football had been played for centuries across Europe. The Cambridge 
    Rules of 1848 were an important step toward standardization. The first international match was played 
    between Scotland and England in 1872, ending 0-0. The sport spread rapidly through British sailors, 
    traders, and soldiers, reaching South America, Europe, and beyond by the early 20th century.""",

    """FIFA (Fédération Internationale de Football Association) is the international governing body of 
    football, founded in Paris in 1904. It currently has 211 member associations, more than the United 
    Nations. FIFA oversees the FIFA World Cup, which has been held every four years since 1930 (with 
    exceptions in 1942 and 1946 due to World War II). The first World Cup was held in Uruguay, with the 
    host nation winning. Brazil holds the record for the most World Cup titles with five, followed by 
    Germany and Italy with four each. The tournament is the most-watched sporting event in the world.""",

    """The FIFA World Cup is the most prestigious international football tournament. It takes place every 
    four years and involves a qualification phase followed by a finals tournament. The 2022 World Cup was 
    held in Qatar and was won by Argentina, ending a 36-year wait for the nation's third title. Lionel 
    Messi finally won his first World Cup trophy, cementing his legacy. The 2026 World Cup will be jointly 
    hosted by the United States, Canada, and Mexico, and will be the first tournament to feature 48 teams 
    instead of the previous 32. The expanded format will include 104 matches across 16 host cities.""",

    """The UEFA Champions League is the most prestigious club football competition in the world, organized 
    by UEFA (Union of European Football Associations). It was founded in 1955 as the European Cup and 
    rebranded in 1992. The competition brings together the top clubs from European domestic leagues. Real 
    Madrid holds the record for the most Champions League titles with 15. The final is one of the most 
    watched annual sporting events globally, regularly attracting over 300 million television viewers. 
    Notable finals include Liverpool vs AC Milan in 2005, widely regarded as one of the greatest comebacks 
    in football history, when Liverpool overturned a 3-0 deficit to win on penalties.""",

    """Lionel Messi and Cristiano Ronaldo are widely considered the two greatest footballers of the modern 
    era, dominating the sport for over 15 years. Messi, born in Rosario, Argentina in 1987, spent most of 
    his career at FC Barcelona, where he won ten La Liga titles and four Champions League trophies. He has 
    won the Ballon d'Or award a record eight times. Ronaldo, born in Madeira, Portugal in 1985, has played 
    for Sporting CP, Manchester United, Real Madrid, and Juventus. He is the all-time top scorer in 
    Champions League history and has also won five Ballon d'Or awards. Both players have scored over 800 
    career goals at club and international level.""",

    """Football tactics and formations describe how players are positioned and how a team plays. The most 
    common formation is 4-3-3, used by clubs like FC Barcelona and the Netherlands national team, featuring 
    four defenders, three midfielders, and three forwards. The 4-4-2 was the dominant formation for decades 
    in English football, providing balance between defense and attack. The 3-5-2 uses three centre-backs 
    and wingbacks who can attack and defend. Total Football, developed by Ajax and the Netherlands in the 
    1970s under Rinus Michels, was a revolutionary tactical philosophy where any outfield player could take 
    over the role of any other player, requiring high fitness and technical skill from the entire squad.""",

    """The English Premier League (EPL) is the top tier of English football and is widely considered the 
    most competitive and most-watched domestic league in the world. It was founded in 1992 when the top 
    clubs broke away from the Football League. Twenty clubs compete each season in a home-and-away format, 
    with the bottom three teams relegated to the Championship. Manchester United holds the record for the 
    most Premier League titles with 13. The league is known for its fast pace, physicality, and 
    international talent. In the 2023-24 season, over 700 players from more than 100 nationalities 
    participated. Global broadcast deals make it visible in 188 countries, generating billions in revenue.""",

    """The offside rule is one of the most fundamental and debated rules in football. A player is in an 
    offside position if they are nearer to the opponent's goal line than both the ball and the second-last 
    opponent when the ball is played to them. Being in an offside position is not itself an offense — a 
    player is only penalized if they are actively involved in play. The rule was introduced to prevent 
    goal-hanging and encourage teamwork. Since 2019, VAR (Video Assistant Referee) technology has been 
    used in many competitions to check offside decisions with greater precision, though this has also 
    sparked controversy due to marginal calls decided by millimeters using semi-automated tracking systems.""",

    """Football stadiums are iconic structures that serve as homes for clubs and national teams. The largest 
    football stadium in the world by capacity is Rungrado 1st of May Stadium in Pyongyang, North Korea, 
    holding 114,000 spectators. In Europe, Camp Nou in Barcelona (currently being renovated) held over 
    99,000. Wembley Stadium in London, rebuilt in 2007, holds 90,000 and is known as the home of English 
    football. The Maracanã in Rio de Janeiro, Brazil, was originally built for the 1950 World Cup and once 
    held nearly 200,000 people. Modern stadium design prioritizes safety, sightlines, and atmosphere, 
    following tragedies like the 1989 Hillsborough disaster which led to all-seater stadium requirements 
    in English top-flight football.""",
]


# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.title("⚽ Football Wiki")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Embeddings", "Chunking", "Vector Search"]
)

# =========================================================================
# HOME PAGE
# =========================================================================
if page == "Home":
    st.title("⚽ Football Wiki")
    st.subheader("Explore football knowledge using semantic search and AI")

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    for col, (title, icon, desc) in zip(
        [c1, c2, c3, c4],
        [
            ("Football Docs", "📄", "10 Wikipedia articles about football"),
            ("Chunking", "✂️", "Split articles into searchable pieces"),
            ("Embeddings", "🔢", "Text becomes meaning vectors"),
            ("Search", "🔍", "Find answers by meaning"),
        ]
    ):
        with col:
            st.markdown(f'''
            <div class="workshop-card">
            <h4>{title}</h4>
            <p style="font-size:2em;">{icon}</p>
            <p>{desc}</p>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    ### What is Football Wiki?

    Football Wiki is a semantic search engine built on **10 Wikipedia-sourced documents** 
    covering the most important topics in football — from the history of the game and FIFA, 
    to tactics, famous players, and iconic stadiums.

    Instead of matching keywords, it understands the **meaning** of your question and finds 
    the most relevant football knowledge — powered by embeddings and ChromaDB.

    ### Topics covered:
    - ⚽ Rules and basics of football
    - 📜 History of the game
    - 🌍 FIFA and World Cup
    - 🏆 UEFA Champions League
    - 🌟 Messi vs Ronaldo
    - 🧠 Tactics and formations
    - 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League
    - 📐 The offside rule & VAR
    - 🏟️ Famous stadiums

    Use the sidebar to navigate. Head to **Vector Search** to ask a football question!
    """)

# =========================================================================
# EMBEDDINGS PAGE
# =========================================================================
elif page == "Embeddings":
    st.title("What Are Embeddings?")
    st.markdown("""
    An **embedding** turns text into a list of numbers (a vector) that captures its **meaning**.
    Similar texts get similar vectors — even if they use completely different words.

    ```
    "Messi won the World Cup"       --> [0.12, -0.45, 0.78, ...] (384 numbers)
    "Argentina became champions"    --> [0.11, -0.43, 0.76, ...] (similar!)
    "The offside rule explained"    --> [-0.67, 0.22, -0.11, ...] (very different)
    ```
    """)

    st.divider()

    # --- Part 1: Compare two texts ---
    st.subheader("Part 1: Compare two football sentences")

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

        with st.expander("See the raw vectors"):
            st.code(f"Text A: {vectors[0][:10].tolist()}... ({len(vectors[0])} dims)")
            st.code(f"Text B: {vectors[1][:10].tolist()}... ({len(vectors[1])} dims)")

    st.divider()

    # --- Part 2: Similarity matrix ---
    st.subheader("Part 2: Similarity matrix")
    st.markdown("Enter multiple football sentences (one per line) to see how they relate to each other.")

    extra_texts = st.text_area(
        "Texts to compare (one per line)",
        "Messi won the Ballon d'Or eight times\n"
        "Ronaldo scored over 800 career goals\n"
        "The offside rule prevents goal-hanging\n"
        "VAR is used to check referee decisions\n"
        "Real Madrid won 15 Champions League titles\n"
        "Wembley Stadium holds 90,000 spectators",
        height=150
    )

    if st.button("Build Similarity Matrix", type="primary"):
        model = load_model()
        all_texts = [t.strip() for t in extra_texts.strip().split("\n") if t.strip()]

        if len(all_texts) < 2:
            st.error("Please enter at least 2 texts.")
        else:
            with st.spinner("Embedding texts..."):
                vectors = np.array(model.embed_documents(all_texts))
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(vectors)

            import plotly.express as px
            labels = [t[:40] + ("..." if len(t) > 40 else "") for t in all_texts]
            fig = px.imshow(
                sim_matrix, x=labels, y=labels,
                text_auto=".2f", color_continuous_scale="Blues",
                title="Cosine Similarity Matrix — Football Sentences", aspect="auto"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.info(f"Embedded {len(all_texts)} texts into {vectors.shape[1]}-dimensional vectors.")

# =========================================================================
# CHUNKING PAGE
# =========================================================================
elif page == "Chunking":
    st.title("Chunking Strategies")
    st.markdown("""
    Football articles are too long to embed as a single block. We split them into
    **chunks** — smaller pieces that each get their own vector.

    The chunk size affects search quality:
    - **Too small** = chunks lose context (e.g. a single sentence about "the final" with no context)
    - **Too big** = meaning gets diluted across too many topics
    - **Just right** = each chunk captures one coherent football concept
    """)

    sample_text = st.text_area("Football text to chunk:",
                               value=LONG_SAMPLE_TEXT, height=200)
    st.write(f"**Total length:** {len(sample_text)} characters")
    st.divider()

    tab1, tab2, tab3 = st.tabs(
        ["Fixed-Size Chunking", "Sentence-Based Chunking", "LangChain Recursive"]
    )

    # --- Fixed-size ---
    with tab1:
        st.markdown("""
        **Fixed-size chunking** splits text every N characters, regardless of content.
        Simple but may cut mid-sentence — e.g. cutting "Lionel Mes-" from "-si won the trophy".
        """)
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
            st.text_area(f"Chunk {i+1} ({len(chunk)} chars)", chunk,
                         height=80, key=f"fixed_{i}", disabled=True)

    # --- Sentence-based ---
    with tab2:
        st.markdown("""
        **Sentence-based chunking** splits at sentence boundaries (periods, etc.).
        Each chunk is a complete sentence — better for football facts like scores and records.
        """)
        sentences = re.split(r'(?<=[.!?])\s+', sample_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        st.write(f"**{len(sentences)} sentences** found")
        for i, sent in enumerate(sentences):
            st.text_area(f"Sentence {i+1} ({len(sent)} chars)", sent,
                         height=60, key=f"sent_{i}", disabled=True)

    # --- LangChain Recursive ---
    with tab3:
        st.markdown("""
        **LangChain RecursiveCharacterTextSplitter** is the smartest approach.
        It tries to split at paragraph boundaries first, then sentences, then words —
        keeping football context (e.g. a full paragraph about the World Cup) intact.
        """)
        rc_size = st.slider("Chunk size", 50, 500, 200, key="rc_size")
        rc_overlap = st.slider("Overlap", 0, 100, 30, key="rc_overlap")

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=rc_size, chunk_overlap=rc_overlap
        )
        rc_chunks = splitter.split_text(sample_text)

        st.write(f"**{len(rc_chunks)} chunks** created")
        for i, chunk in enumerate(rc_chunks):
            st.text_area(f"Chunk {i+1} ({len(chunk)} chars)", chunk,
                         height=80, key=f"rc_{i}", disabled=True)

# =========================================================================
# VECTOR SEARCH PAGE
# =========================================================================
elif page == "Vector Search":
    st.title("⚽ Football Knowledge Search")
    st.markdown("""
    Search through 10 football Wikipedia articles using **semantic search** — 
    find answers by meaning, not just keywords.

    **Keyword search:** *"Does the article contain the exact words I typed?"*

    **Semantic search:** *"Does the article mean something similar to what I asked?"*
    """)

    st.markdown("**📚 Documents in the knowledge base:**")
    with st.expander("Show all 10 football documents"):
        for i, doc in enumerate(DOCUMENTS, 1):
            st.markdown(f"**{i}.** {doc.strip()}")
            st.markdown("---")

    st.divider()

    query = st.text_input(
        "🔍 Ask a football question",
        placeholder="e.g. 'Who is the best player ever?' or 'How does the offside rule work?'"
    )
    num_results = st.slider("Number of results to show", 1, 5, 3, key="num_results")

    # Chunking settings
    with st.expander("⚙️ Chunking settings"):
        chunk_size = st.slider("Chunk size (characters)", 100, 1000, 500, key="search_chunk_size")
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 200, 50, key="search_chunk_overlap")
        st.caption(f"Each document will be split into chunks of ~{chunk_size} characters with {chunk_overlap} characters of overlap.")

    if st.button("Search", type="primary") and query:
        model = load_model()

        with st.spinner("Chunking documents and searching..."):
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import Chroma

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.create_documents(DOCUMENTS)
            chunk_texts = [c.page_content for c in chunks]

            vectorstore = Chroma.from_texts(texts=chunk_texts, embedding=model)
            results = vectorstore.similarity_search_with_relevance_scores(
                query, k=num_results
            )

        st.caption(f"🔪 Split {len(DOCUMENTS)} documents into **{len(chunk_texts)} chunks** (chunk_size={chunk_size}, overlap={chunk_overlap})")

        st.subheader("Results (ranked by meaning similarity)")
        for i, (doc, score) in enumerate(results, 1):
            pct = max(0, score) * 100
            col_text, col_score = st.columns([4, 1])
            with col_text:
                st.markdown(f"**{i}.** {doc.page_content.strip()}")
            with col_score:
                st.metric("Relevance", f"{pct:.0f}%")
            st.progress(min(1.0, max(0, score)))

        st.divider()
        st.info("""
        **Try these football queries to see semantic search in action:**
        - "Who is the best player ever?" → finds Messi & Ronaldo doc
        - "How does the offside rule work?" → finds offside & VAR doc
        - "Which country has won the most World Cups?" → finds FIFA doc
        - "What is the biggest stadium in the world?" → finds stadiums doc
        - "How did football spread around the world?" → finds history doc
        - "What is the most popular club competition in Europe?" → finds Champions League doc
        """)
