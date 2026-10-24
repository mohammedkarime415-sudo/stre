"""
Assistant RAG — Diagnostic de pannes d'équipements industriels
YaneCode Academy — AI Générative

Lancer :  streamlit run streamlit_app.py
"""

import io
import re

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Client Gemini : si le paquet manque, l'application s'ouvre quand même
# et affiche la commande d'installation dans la barre latérale.
try:
    from google import genai
    GENAI_DISPONIBLE = True
except ImportError:
    genai = None
    GENAI_DISPONIBLE = False

# FAISS est utilisé s'il est installé ; sinon on bascule automatiquement sur
# une recherche NumPy équivalente (même résultat : similarité cosinus).
try:
    import faiss
    FAISS_DISPONIBLE = True
except ImportError:
    faiss = None
    FAISS_DISPONIBLE = False


class IndexNumpy:
    """Remplace faiss.IndexFlatIP : produit scalaire sur vecteurs normalisés."""

    def __init__(self, dimension):
        self.vecteurs = np.empty((0, dimension), dtype="float32")

    def add(self, vecteurs):
        self.vecteurs = np.vstack([self.vecteurs, vecteurs])

    def search(self, requete, k):
        scores = requete @ self.vecteurs.T
        k = min(k, scores.shape[1])
        indices = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, indices, axis=1), indices


def creer_index(dimension):
    return faiss.IndexFlatIP(dimension) if FAISS_DISPONIBLE else IndexNumpy(dimension)

# ----------------------------------------------------------------------------
# Configuration de la page
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Assistant Maintenance RAG",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Design (couleurs et style inspirés de l'image : dégradé bleu, cercles marine,
# accents bleu clair, bordure pointillée lumineuse)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root{
  --bleu-fond-1:#0a2fc4;
  --bleu-fond-2:#0a4de0;
  --bleu-fond-3:#0a6cff;
  --marine:#050d3a;
  --marine-2:#0a1a6b;
  --bleu-clair:#7cc7ff;
  --bleu-texte:#4d8dff;
  --blanc:#ffffff;
}

html, body, [class*="css"], .stApp {
  font-family: 'Poppins', sans-serif;
}

/* Fond général : dégradé bleu + grands arcs translucides */
.stApp{
  background:
    radial-gradient(circle at 92% 12%, rgba(124,199,255,.22) 0, rgba(124,199,255,0) 28%),
    radial-gradient(circle at 5% 88%, rgba(124,199,255,.25) 0, rgba(124,199,255,0) 30%),
    linear-gradient(135deg, var(--bleu-fond-1) 0%, var(--bleu-fond-2) 55%, var(--bleu-fond-3) 100%);
  background-attachment: fixed;
}
header[data-testid="stHeader"]{ background: transparent; }
.block-container{ padding-top: 1.5rem; max-width: 1100px; }

/* Sidebar marine */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, var(--marine) 0%, var(--marine-2) 100%);
  border-right: 2px solid rgba(124,199,255,.35);
}
[data-testid="stSidebar"] *{ color: #e8f1ff; }
[data-testid="stSidebar"] .stButton>button{
  width:100%;
}

/* Titre dégradé façon "RAG" */
.titre-rag{
  text-align:center;
  font-size: 4.2rem;
  font-weight: 800;
  line-height: 1;
  margin: .2rem 0 .2rem 0;
  background: linear-gradient(180deg, #bfe6ff 0%, #5db4ff 60%, rgba(93,180,255,.15) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  letter-spacing: 4px;
}
.sous-titre{
  text-align:center;
  color:#d8e9ff;
  font-size:1.05rem;
  margin-bottom: 1.2rem;
}

/* Cadre pointillé lumineux (comme l'image) */
.cadre{
  border: 4px dashed rgba(124,199,255,.85);
  border-radius: 28px;
  padding: 1.6rem 1rem 1.2rem 1rem;
  background: linear-gradient(160deg, rgba(5,13,58,.55), rgba(10,26,107,.35));
  box-shadow: 0 0 30px rgba(124,199,255,.25), inset 0 0 40px rgba(5,13,58,.45);
  margin-bottom: 1.4rem;
}

/* Pipeline : cercles marine reliés par des flèches */
.pipeline{
  display:flex; align-items:center; justify-content:center;
  gap: .6rem; flex-wrap: wrap;
}
.noeud{ text-align:center; width: 120px; }
.cercle{
  width: 84px; height: 84px; margin: 0 auto .45rem auto;
  border-radius: 50%;
  background: radial-gradient(circle at 30% 25%, #0e1f7a, var(--marine) 70%);
  border: 3px solid rgba(124,199,255,.75);
  display:flex; align-items:center; justify-content:center;
  font-size: 2.1rem;
  box-shadow: 0 0 18px rgba(124,199,255,.35);
}
.noeud .nom{ color:#fff; font-weight:600; font-size:.95rem; }
.noeud .desc{ color: var(--bleu-clair); font-size:.72rem; }
.fleche{ color: var(--bleu-clair); font-size: 1.7rem; font-weight:300; }

/* Cartes de statistiques */
.stat{
  background: linear-gradient(160deg, var(--marine), var(--marine-2));
  border: 2px solid rgba(124,199,255,.5);
  border-radius: 20px;
  padding: .8rem 1rem;
  text-align:center;
}
.stat .val{ color:#fff; font-size:1.6rem; font-weight:700; }
.stat .lib{ color: var(--bleu-clair); font-size:.8rem; }

/* Titres de section */
.section{
  color:#fff; font-weight:700; font-size:1.25rem;
  margin: .4rem 0 .6rem 0;
}

/* Chat */
[data-testid="stChatMessage"]{
  background: linear-gradient(160deg, rgba(5,13,58,.85), rgba(10,26,107,.75));
  border: 2px solid rgba(124,199,255,.45);
  border-radius: 22px;
  padding: .9rem 1.1rem;
  margin-bottom: .8rem;
  color:#eef5ff;
}
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li{ color:#eef5ff; }
[data-testid="stChatMessageAvatarUser"]{
  background: linear-gradient(135deg,#7cc7ff,#4d8dff) !important;
}
[data-testid="stChatMessageAvatarAssistant"]{
  background: radial-gradient(circle at 30% 25%, #0e1f7a, #050d3a) !important;
  border: 2px solid rgba(124,199,255,.8);
}

/* Zone de saisie */
[data-testid="stChatInput"]{
  background: var(--marine);
  border: 2px solid rgba(124,199,255,.8);
  border-radius: 26px;
}
[data-testid="stChatInput"] textarea{ color:#fff !important; }
[data-testid="stBottom"] > div{ background: transparent; }

/* Boutons */
.stButton>button{
  background: linear-gradient(135deg, #0e1f7a, var(--marine));
  color:#fff; font-weight:600;
  border: 2px solid rgba(124,199,255,.8);
  border-radius: 999px;
  padding: .45rem 1.1rem;
  transition: all .2s ease;
}
.stButton>button:hover{
  border-color:#fff; color:#bfe6ff;
  box-shadow: 0 0 16px rgba(124,199,255,.65);
  transform: translateY(-1px);
}

/* Expanders (sources) */
[data-testid="stExpander"]{
  background: rgba(5,13,58,.6);
  border: 1px solid rgba(124,199,255,.45);
  border-radius: 14px;
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] p{ color:#d8e9ff; }

/* Divers */
.stAlert{ border-radius: 16px; }
label, .stMarkdown, .stCaption{ color:#e8f1ff; }
hr{ border-color: rgba(124,199,255,.35); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
MODELE_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MESSAGE_INCONNU = "Information non précisée dans la documentation."

QUESTIONS_EXEMPLES = [
    "Quelles sont les causes possibles d'une surchauffe du moteur ?",
    "Quelles actions correctives pour une vibration anormale ?",
    "Quelles précautions de sécurité avant d'intervenir sur la machine ?",
    "À quelle fréquence faut-il lubrifier les roulements ?",
]


# ----------------------------------------------------------------------------
# Fonctions RAG (reprises du notebook, adaptées à plusieurs PDF)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle d'embeddings…")
def charger_modele():
    return SentenceTransformer(MODELE_EMBEDDING)


def extraire_pages(nom_fichier, contenu_octets):
    """Extrait le texte page par page en conservant le numéro de page."""
    lecteur = PdfReader(io.BytesIO(contenu_octets))
    pages = []
    for numero, page in enumerate(lecteur.pages, start=1):
        texte = page.extract_text() or ""
        texte = re.sub(r"\s+", " ", texte).strip()
        if texte:
            pages.append({"source": nom_fichier, "page": numero, "texte": texte})
    return pages


def decouper_texte(texte, taille=900, chevauchement=150):
    """Découpe par caractères, sans couper au milieu d'un mot si possible."""
    morceaux = []
    debut = 0
    while debut < len(texte):
        fin = min(debut + taille, len(texte))
        if fin < len(texte):
            espace = texte.rfind(" ", debut, fin)
            if espace > debut:
                fin = espace
        morceau = texte[debut:fin].strip()
        if morceau:
            morceaux.append(morceau)
        if fin == len(texte):
            break
        debut = max(fin - chevauchement, debut + 1)
    return morceaux


def construire_index(fichiers, taille, chevauchement):
    """PDF -> pages -> chunks -> embeddings -> index FAISS."""
    modele = charger_modele()

    pages = []
    for f in fichiers:
        pages.extend(extraire_pages(f.name, f.getvalue()))

    chunks = []
    for p in pages:
        for i, t in enumerate(decouper_texte(p["texte"], taille, chevauchement), start=1):
            chunks.append({"source": p["source"], "page": p["page"], "chunk": i, "texte": t})

    if not chunks:
        return None, [], 0

    vecteurs = modele.encode(
        [c["texte"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = creer_index(vecteurs.shape[1])  # produit scalaire + vecteurs normalisés = cosinus
    index.add(vecteurs)
    return index, chunks, len(pages)


def rechercher(question, k):
    modele = charger_modele()
    vecteur = modele.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, indices = st.session_state.index.search(vecteur, k)
    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            item = st.session_state.chunks[int(idx)].copy()
            item["score"] = float(score)
            resultats.append(item)
    return resultats


def construire_prompt(question, passages, historique):
    contexte = "\n\n".join(
        f'[Source : {p["source"]}, page {p["page"]}]\n{p["texte"]}' for p in passages
    )
    conv = ""
    if historique:
        lignes = []
        for m in historique[-6:]:
            role = "Technicien" if m["role"] == "user" else "Assistant"
            lignes.append(f"{role} : {m['content']}")
        conv = "HISTORIQUE DE LA CONVERSATION :\n" + "\n".join(lignes) + "\n\n"

    return f"""Tu es un assistant d'aide au diagnostic pour les équipes de maintenance industrielle.
Ta mission : aider à comprendre les pannes des équipements à partir de la documentation fournie
(manuels, fiches techniques, historiques de pannes, procédures).

Règles :
- Réponds uniquement à partir du CONTEXTE fourni, sans compléter avec tes connaissances.
- Structure la réponse si pertinent : Causes probables, Vérifications à faire, Actions correctives, Précautions de sécurité.
- Sois court, clair et fidèle aux documents.
- Cite les sources sous la forme (nom du fichier, page X).
- N'invente jamais une valeur numérique (seuil, couple, fréquence, pression...).
- Si la réponse exacte n'est pas présente, écris : « {MESSAGE_INCONNU} »

{conv}CONTEXTE :
{contexte}

QUESTION : {question}

RÉPONSE :"""


def repondre(question, k, seuil, modele_llm, api_key, historique):
    passages = rechercher(question, k)
    if not passages or passages[0]["score"] < seuil:
        return MESSAGE_INCONNU, passages

    if not GENAI_DISPONIBLE:
        raise RuntimeError(
            "Le paquet google-genai n'est pas installé. Lancez : pip install google-genai"
        )
    client = genai.Client(api_key=api_key)
    prompt = construire_prompt(question, passages, historique)
    reponse = client.models.generate_content(model=modele_llm, contents=prompt)
    return (reponse.text or MESSAGE_INCONNU), passages


# ----------------------------------------------------------------------------
# État de session
# ----------------------------------------------------------------------------
st.session_state.setdefault("index", None)
st.session_state.setdefault("chunks", [])
st.session_state.setdefault("nb_pages", 0)
st.session_state.setdefault("nb_docs", 0)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("question_en_attente", None)

# ----------------------------------------------------------------------------
# Barre latérale : paramètres + documents
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    if not GENAI_DISPONIBLE:
        st.error(
            "Paquet manquant : google-genai. Dans le terminal, lancez :\n\n"
            "pip install google-genai"
        )

    api_key = st.text_input(
        "Clé API Gemini",
        type="password",
        help="Créez une clé gratuite sur Google AI Studio : https://aistudio.google.com/app/apikey",
    )
    modele_llm = st.text_input("Modèle Gemini", value="gemini-3.6-flash")

    st.markdown("---")
    st.markdown("### 📄 Documents techniques")
    fichiers = st.file_uploader(
        "Manuels, fiches pannes, procédures (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    with st.expander("Paramètres avancés"):
        taille = st.slider("Taille des chunks (caractères)", 300, 1800, 900, 50)
        chevauchement = st.slider("Chevauchement", 0, 400, 150, 10)
        k = st.slider("k : passages récupérés", 1, 10, 4)
        seuil = st.slider("Seuil de pertinence minimal", 0.0, 0.6, 0.20, 0.01)

    if st.button("🔧 Indexer les documents"):
        if not fichiers:
            st.warning("Ajoutez au moins un PDF.")
        else:
            with st.spinner("Extraction, découpage et indexation…"):
                index, chunks, nb_pages = construire_index(fichiers, taille, chevauchement)
            if index is None:
                st.error("Aucun texte extractible (PDF scanné ?).")
            else:
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.nb_pages = nb_pages
                st.session_state.nb_docs = len(fichiers)
                st.success("Base de connaissances prête ✅")

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# valeurs par défaut si l'expander n'a pas encore été ouvert (sécurité)
taille = locals().get("taille", 900)
k = locals().get("k", 4)
seuil = locals().get("seuil", 0.20)

# ----------------------------------------------------------------------------
# En-tête + schéma du pipeline
# ----------------------------------------------------------------------------
st.markdown('<div class="titre-rag">RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sous-titre">Assistant de diagnostic des pannes pour les équipes de maintenance industrielle</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="cadre">
  <div class="pipeline">
    <div class="noeud"><div class="cercle">👷</div><div class="nom">Technicien</div><div class="desc">Question</div></div>
    <div class="fleche">⇄</div>
    <div class="noeud"><div class="cercle">🤖</div><div class="nom">Framework</div><div class="desc">Recherche + prompt</div></div>
    <div class="fleche">⇄</div>
    <div class="noeud"><div class="cercle">🧠</div><div class="nom">LLM</div><div class="desc">Réponse sourcée</div></div>
    <div class="fleche">⇅</div>
    <div class="noeud"><div class="cercle">🗄️</div><div class="nom">Base vectorielle</div><div class="desc">FAISS</div></div>
    <div class="fleche">⇠</div>
    <div class="noeud"><div class="cercle">📚</div><div class="nom">Documents</div><div class="desc">Manuels + pannes</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Statistiques
c1, c2, c3 = st.columns(3)
for col, val, lib in [
    (c1, st.session_state.nb_docs, "Documents"),
    (c2, st.session_state.nb_pages, "Pages extraites"),
    (c3, len(st.session_state.chunks), "Chunks indexés"),
]:
    col.markdown(
        f'<div class="stat"><div class="val">{val}</div><div class="lib">{lib}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Partie chat / conversation
# ----------------------------------------------------------------------------
st.markdown('<div class="section">💬 Conversation avec l\'assistant</div>', unsafe_allow_html=True)

if st.session_state.index is None:
    st.info(
        "Pour commencer : ajoutez vos PDF dans la barre latérale, "
        "collez votre clé Gemini, puis cliquez sur « Indexer les documents »."
    )

# Questions d'exemple
st.caption("Exemples de questions :")
cols = st.columns(2)
for i, q in enumerate(QUESTIONS_EXEMPLES):
    if cols[i % 2].button(q, key=f"ex_{i}"):
        st.session_state.question_en_attente = q

# Historique
for m in st.session_state.messages:
    avatar = "🧑‍🔧" if m["role"] == "user" else "🤖"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📎 Sources utilisées"):
                for s in m["sources"]:
                    st.markdown(f"**{s['source']} — page {s['page']}** · score {s['score']:.2f}")
                    st.caption(s["texte"][:400] + ("…" if len(s["texte"]) > 400 else ""))

# Saisie
saisie = st.chat_input("Décrivez la panne ou posez votre question…")
question = saisie or st.session_state.question_en_attente
st.session_state.question_en_attente = None

if question:
    historique = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🔧"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        if st.session_state.index is None:
            texte = "Veuillez d'abord indexer vos documents dans la barre latérale."
            passages = []
            st.markdown(texte)
        elif not api_key:
            texte = "Veuillez saisir votre clé API Gemini dans la barre latérale."
            passages = []
            st.markdown(texte)
        else:
            with st.spinner("Analyse de la documentation…"):
                try:
                    texte, passages = repondre(question, k, seuil, modele_llm, api_key, historique)
                except Exception as e:  # erreur API, quota, modèle introuvable…
                    texte = f"Une erreur est survenue lors de la génération : `{e}`"
                    passages = []
            st.markdown(texte)
            if passages:
                with st.expander("📎 Sources utilisées"):
                    for s in passages:
                        st.markdown(f"**{s['source']} — page {s['page']}** · score {s['score']:.2f}")
                        st.caption(s["texte"][:400] + ("…" if len(s["texte"]) > 400 else ""))

    st.session_state.messages.append(
        {"role": "assistant", "content": texte, "sources": passages}
    )
