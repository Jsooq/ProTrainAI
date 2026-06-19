"""
🏀 ProTrainAI — Streamlit UI
A kid-friendly interface for generating NBA-inspired workout plans.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import os
import shutil
import datetime
import csv

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ─────────────────────────────────────────────────────────────────
# 📊 USAGE LOGGING
# Appends one row per generated workout to a local CSV.
# NOTE: Streamlit Cloud's filesystem resets on every app restart/
# redeploy, so this log does NOT persist long-term in production.
# It's great for local testing or short stretches between restarts.
# For permanent tracking, this can be swapped for a Google Sheet
# or external database later.
# ─────────────────────────────────────────────────────────────────
LOG_PATH = "usage_log.csv"


def log_usage(player: str, age: str, notes: str):
    file_exists = os.path.exists(LOG_PATH)
    try:
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "player", "age", "notes"])
            writer.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                player,
                age or "",
                notes or "",
            ])
    except Exception:
        # Logging should never break the actual workout generation
        pass


# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit command
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProTrainAI 🏀",
    page_icon="🏀",
    layout="centered",
)


# ─────────────────────────────────────────────────────────────────
# 📚 KNOWLEDGE BASE — same players as your notebook
# Add more players here anytime, following the same format
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_player_data():
    return [
        Document(
            page_content="""
            Player: Steph Curry | Team: Golden State Warriors | Position: Point Guard

            SIGNATURE DRILLS:
            - Mikan Drill (both hands, at game speed): Builds finishing at the rim with either hand.
            - Ball Handling Circuit: Two-ball dribbling, figure-8s, behind-the-back, crossover combo. 5 minutes continuous.
            - Catch-and-Shoot off screens: Coach or partner sets a screen, player curls, catches, and fires. 50 makes.
            - 3-point shooting from 5 spots: Corners, wings, top of the key. Must make 10 from each spot before moving.
            - Off-dribble pull-up: Dribble hard left, stop on a dime, pull-up mid-range. Repeat right. 40 makes total.
            - Steph's 'runway' shooting drill: Start at half court, dribble full speed, pull up for 3 at the arc.

            TRAINING QUOTES:
            - 'Success is not an accident. Success is actually a choice.'
            - "I've worked too hard and too long to let anything stand in the way of my goals."
            - 'Shooting is about muscle memory. You have to do it 1,000 times to do it once in a game.'
            - 'I try to be efficient in everything I do — every rep, every move has a purpose.'

            KEY SKILLS: Ball handling, off-screen shooting, quick release, footwork, shooting off the dribble, conditioning.

            YOUTUBE SEARCH TERMS:
            - 'Steph Curry full workout training'
            - 'Steph Curry shooting drill routine NBA'
            - 'Steph Curry ball handling workout'
            """,
            metadata={"player": "Steph Curry", "position": "PG", "skills": "shooting, ball handling"}
        ),
        Document(
            page_content="""
            Player: Kobe Bryant | Team: LA Lakers | Position: Shooting Guard

            SIGNATURE DRILLS:
            - Post footwork series: Drop step, up-and-under, turnaround fadeaway. 20 makes from each side.
            - The Kobe 'triangle' shooting drill: Mid-range shots from elbow, baseline, and top. Must make 50 mid-range before 3s.
            - Mamba mentality conditioning: 5 minutes of full-court sprints at game pace with ball in hand.
            - 1-on-1 iso footwork: Jab step series — jab and shoot, jab and drive, jab and crossover. 20 reps each move.
            - Film study + replication: Watch a scoring sequence on film, then replicate every move on the court.
            - Early morning shooting: 800 shots before anyone else arrives. Pure volume and consistency.

            TRAINING QUOTES:
            - "The mindset isn't about seeking a result — it's about the process of getting to that result."
            - 'Everything negative — pressure, challenges — is all an opportunity for me to rise.'
            - "If you're afraid to fail, then you're probably going to fail."
            - "Every day. That's when the work starts."

            KEY SKILLS: Post play, mid-range shooting, footwork, iso scoring, mental toughness, fadeaway.

            YOUTUBE SEARCH TERMS:
            - 'Kobe Bryant workout training routine'
            - 'Kobe Bryant footwork drills post moves'
            - 'Kobe Bryant Mamba mentality practice'
            """,
            metadata={"player": "Kobe Bryant", "position": "SG", "skills": "post play, mid-range, footwork"}
        ),
        Document(
            page_content="""
            Player: Kevin Durant | Team: Phoenix Suns | Position: Small Forward

            SIGNATURE DRILLS:
            - Shoot over contact: Coach holds a foam pad above shooter's release point. Forces high arc.
            - KD's mid-range series: Step-back from the elbow, lean-in from baseline, turnaround from the block. 30 makes each.
            - Ball handling for bigs: Full dribbling circuits normally run by guards.
            - Catch-and-shoot off movement: Sprint, come off pin-down, catch, shoot. 60 makes total.
            - Post-to-perimeter combo: Start in the post, face up, pump fake, drive — or pop to the 3.
            - Free throw focus: 100 free throws, tracking make percentage. Nothing leaves the gym under 85%.

            TRAINING QUOTES:
            - 'Hard work beats talent when talent fails to work hard.'
            - "I just want to get better every single day. That's all I think about."
            - 'The most important thing is to stay patient and trust your process.'
            - 'Shooting is my gift. But I put in work every single day to keep it sharp.'

            KEY SKILLS: Shooting over defenders, mid-range mastery, scoring in traffic, versatility, free throws.

            YOUTUBE SEARCH TERMS:
            - 'Kevin Durant workout training drills'
            - 'KD shooting drill mid-range routine'
            - 'Kevin Durant skill development session'
            """,
            metadata={"player": "Kevin Durant", "position": "SF", "skills": "scoring, shooting, mid-range"}
        ),
        Document(
            page_content="""
            Player: LeBron James | Team: LA Lakers | Position: Small Forward / PG

            SIGNATURE DRILLS:
            - Full-court transition finishing: Push pace full court, finish with either hand at rim. 20 reps.
            - Playmaking vision drill: 3-on-2 fast break reads — coach calls out scenarios mid-play.
            - Body conditioning: Mix of plyometrics, resistance band work, and court sprints. 30-minute block.
            - Post-up and kick-out: Back down defender in post, read double team, kick to shooter.
            - Ball handling to layup: Crossover, behind-back, hesitation combo ending in a strong layup or euro step.
            - Recovery and flexibility: 30-minute stretching and recovery routine.

            TRAINING QUOTES:
            - 'I treat every practice like it's a championship game.'
            - 'You have to be able to accept failure to get better.'
            - 'My body is my business. I invest in it every single day.'
            - 'The only way to get better is to surround yourself with people who make you work harder.'

            KEY SKILLS: Athleticism, playmaking, finishing at rim, transition, court vision, conditioning.

            YOUTUBE SEARCH TERMS:
            - 'LeBron James workout training routine'
            - 'LeBron James full court drills athleticism'
            - 'LeBron James skill training session'
            """,
            metadata={"player": "LeBron James", "position": "SF/PG", "skills": "athleticism, playmaking, finishing"}
        ),
        Document(
            page_content="""
            Player: Nikola Jokic | Team: Denver Nuggets | Position: Center

            SIGNATURE DRILLS:
            - Passing from the post: From high post and low post, make 10 pinpoint passes to cutters. Then reverse.
            - Big man ball handling: Full guard-level dribbling circuits.
            - Shooting off movement at center range: Floaters, short mid-range, face-up jumpers to 15 feet. 40 makes.
            - Pick-and-roll reads: Set the screen, roll, read whether to catch-and-finish or kick back out.
            - IQ simulation: Watch film of a possession, pause it, explain where every player should move.
            - Conditioning: Side shuffles, drop steps under fatigue. Work at game speed not just rest speed.

            TRAINING QUOTES:
            - 'I just want to win. The stats will come if you focus on winning.'
            - 'Basketball is simple. See the open man. Make the right play.'
            - 'The highest IQ play is usually the right play.'

            KEY SKILLS: Passing, IQ, post play, pick-and-roll, floater, vision, unselfishness.

            YOUTUBE SEARCH TERMS:
            - 'Nikola Jokic skill training drills'
            - 'Jokic passing drill post play workout'
            - 'Nikola Jokic full workout session'
            """,
            metadata={"player": "Nikola Jokic", "position": "C", "skills": "passing, IQ, post play"}
        ),
    ]


# ─────────────────────────────────────────────────────────────────
# 🔧 BUILD THE RAG PIPELINE (cached so it only runs once per session)
# Note: this takes 30-90 seconds the FIRST time only (downloading the
# embedding model + building the vector DB). After that it's instant
# for the rest of the session because @st.cache_resource holds onto it.
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_rag_chain(api_key: str):
    os.environ["GOOGLE_API_KEY"] = api_key

    player_data = load_player_data()

    # This line downloads a ~90MB model the very first time it ever
    # runs on a given server. That download is what causes the long
    # first-run wait — it's not actually hanging.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db_path = "./coachbot_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    vectordb = Chroma.from_documents(
        documents=player_data,
        embedding=embeddings,
        persist_directory=db_path,
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=0.7,
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are ProTrainAI, an elite basketball training assistant. You help coaches
design 1-on-1 training sessions for youth players (ages 8-18) inspired by
NBA players' real training methods.

Use ONLY the player information provided below to design the workout.
Include real drills, motivational quotes from the player, and YouTube
search terms so the coach can find videos to show their athlete.

PLAYER KNOWLEDGE BASE:
{context}

COACH'S REQUEST:
{question}

Respond with a complete, structured 60-minute training session in this format:

🏀 PROTRAINAI SESSION PLAN
Player Inspiration: [Player Name]
Theme: [Core skill focus]

💬 OPENING QUOTE (Read this to your athlete at the start):
[Quote from the player]

📋 SESSION BREAKDOWN:

⏱️ WARMUP (10 min)
[2-3 warmup activities]

🔧 SKILL BLOCK 1 (15 min) — [Skill name]
[Drill name]: [Instructions]
[Coaching cue from the player's style]

🔧 SKILL BLOCK 2 (15 min) — [Skill name]
[Drill name]: [Instructions]
[Coaching cue]

🔧 SKILL BLOCK 3 (10 min) — [Skill name]
[Drill name]: [Instructions]

🏁 COMPETITIVE FINISH (8 min)
[1v1 or scoring challenge tied to session theme]

🧘 COOLDOWN & MINDSET (2 min)
[Closing quote + reflection question for the athlete]

🎥 VIDEOS TO SHOW YOUR ATHLETE:
Search YouTube for:
- [Search term 1]
- [Search term 2]

💡 COACH NOTES:
[2-3 tips for adapting this session to a youth player]
""",
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain, player_data


# ─────────────────────────────────────────────────────────────────
# 🔑 API KEY HANDLING
# Priority: Streamlit secrets (for deployment) > sidebar input (for local testing)
# ─────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


# ─────────────────────────────────────────────────────────────────
# 🔒 ACCESS GATE
# A simple shared password so random visitors with the link can't
# burn through your Google API quota. Not bank-grade security —
# just enough friction to keep this to your athletes.
# Set APP_PASSWORD in Streamlit Cloud's Secrets to enable this.
# If no APP_PASSWORD secret is set, the gate is skipped entirely
# (useful for local testing without typing a password every time).
# ─────────────────────────────────────────────────────────────────
def check_password():
    correct_password = st.secrets.get("APP_PASSWORD", None)

    # No password configured at all -> skip the gate (local dev mode)
    if not correct_password:
        return True

    if st.session_state.get("authenticated", False):
        return True

    st.title("🏀 ProTrainAI")
    st.markdown("##### Enter your access code to continue")

    entered = st.text_input("Access code", type="password", key="password_input")

    if st.button("Enter", type="primary"):
        if entered == correct_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("That code isn't right — check with your coach and try again.")

    return False


if not check_password():
    st.stop()


# ─────────────────────────────────────────────────────────────────
# 🎨 UI
# ─────────────────────────────────────────────────────────────────
st.title("🏀 ProTrainAI")
st.markdown("##### Build a workout inspired by your favorite NBA player")

api_key = get_api_key()

if not api_key:
    with st.sidebar:
        st.markdown("### 🔑 Setup")
        api_key = st.text_input(
            "Google API Key",
            type="password",
            help="Get a free key at aistudio.google.com",
        )
        st.caption("This stays on your device — it's not stored anywhere.")

    if not api_key:
        st.info("👈 Enter your Google API key in the sidebar to get started.")
        st.stop()

# Build the chain (cached — only runs once per server, not once per user)
is_first_load = "coachbot_warmed_up" not in st.session_state

if is_first_load:
    status_box = st.status("Starting up ProTrainAI for the first time...", expanded=True)
    status_box.write("📦 Downloading skill model (only happens once — ~30-60 sec)...")
    try:
        rag_chain, player_data = build_rag_chain(api_key)
        st.session_state["coachbot_warmed_up"] = True
        status_box.write("✅ Ready!")
        status_box.update(label="ProTrainAI is ready!", state="complete", expanded=False)
    except Exception as e:
        status_box.update(label="Setup failed", state="error")
        st.error(f"Something went wrong setting up ProTrainAI: {e}")
        st.stop()
else:
    try:
        rag_chain, player_data = build_rag_chain(api_key)
    except Exception as e:
        st.error(f"Something went wrong setting up ProTrainAI: {e}")
        st.stop()

player_names = [doc.metadata["player"] for doc in player_data]

st.divider()

# ─── INPUT FORM ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    selected_player = st.selectbox(
        "Pick your inspiration 🌟",
        options=player_names,
    )

with col2:
    age = st.text_input("Player age (optional)", placeholder="e.g. 12")

notes = st.text_area(
    "Anything specific you want to work on? (optional)",
    placeholder="e.g. I want to work on my left hand, or I struggle with free throws",
    height=80,
)

generate_clicked = st.button("🏀 Generate My Workout", type="primary", use_container_width=True)

st.divider()

# ─── GENERATE & DISPLAY OUTPUT ──────────────────────────────────
if generate_clicked:
    # Build the query from the dropdown + optional notes
    query_parts = [f"Build me a 60-minute workout inspired by {selected_player}."]
    if age:
        query_parts.append(f"The player is {age} years old.")
    if notes:
        query_parts.append(f"Specific focus: {notes}")
    query = " ".join(query_parts)

    with st.spinner(f"ProTrainAI is building your {selected_player}-inspired workout... 🤖"):
        try:
            response = rag_chain.invoke(query)
        except Exception as e:
            st.error(f"ProTrainAI hit an error generating your workout: {e}")
            st.stop()

    # Save to session state so it persists across reruns (e.g. download button clicks)
    st.session_state["last_response"] = response
    st.session_state["last_player"] = selected_player

    # Log this generation (best-effort, never blocks the user)
    log_usage(selected_player, age, notes)

# Display the most recent workout, if one exists
if "last_response" in st.session_state:
    st.markdown(f"### Your {st.session_state['last_player']}-Inspired Workout")
    st.markdown(st.session_state["last_response"])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    st.download_button(
        label="💾 Download this workout",
        data=st.session_state["last_response"],
        file_name=f"protrainai_{st.session_state['last_player'].replace(' ', '_')}_{timestamp}.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────
# 📊 COACH-ONLY USAGE PANEL
# Only visible since you're already past the password gate to be
# running this code at all. Shows which players/skills get
# requested most, pulled from the local CSV log.
# Remember: this resets when the app restarts/redeploys.
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    with st.expander("📊 Coach: Usage Stats"):
        if os.path.exists(LOG_PATH):
            import pandas as pd
            log_df = pd.read_csv(LOG_PATH)
            st.caption(f"{len(log_df)} workouts generated this session")
            st.bar_chart(log_df["player"].value_counts())
            st.dataframe(log_df.tail(20), use_container_width=True, hide_index=True)
            st.download_button(
                "Download full log (CSV)",
                data=log_df.to_csv(index=False),
                file_name="protrainai_usage_log.csv",
                mime="text/csv",
            )
        else:
            st.caption("No workouts generated yet.")

