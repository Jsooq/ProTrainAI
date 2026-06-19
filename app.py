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


def log_usage(athlete_name: str, player: str, age: str, notes: str):
    file_exists = os.path.exists(LOG_PATH)
    try:
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "athlete_name", "player", "age", "notes"])
            writer.writerow([
                datetime.datetime.now().isoformat(timespec="seconds"),
                athlete_name,
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

        Document(
            page_content="""
            Player: Jalen Brunson | Team: New York Knicks | Position: Point Guard

            SIGNATURE DRILLS:
            - Snatch-back to pull-up: Drive hard at the defender, snatch the ball back to create space, rise into a midrange jumper. 30 makes each side.
            - Catch-and-shoot reps: Off screens, off relocations, and in transition. Hundreds of reps per session to sharpen timing and release speed.
            - Pivot and footwork series: Pivot out of the post, up-fake into a jab step, counter off the catch. Builds the ability to operate in tight spaces.
            - The 'broom drill': A partner holds a long pole/broom overhead to contest the shot, forcing a higher, quicker release arc.
            - Touch shot reps: Floaters and short pull-ups around the paint, focused on soft touch over bigger defenders.
            - 1-on-1 to a make: Live one-on-one work where the defender plays full speed, building decision-making under real pressure.

            TRAINING QUOTES:
            - 'Our training sessions are very fundamental.' (Brunson's trainer, Dave Williams)
            - 'We definitely work on pivots, footwork, and touch shots. We get into all of that.' (Dave Williams)

            KEY SKILLS: Footwork, pivoting, midrange shooting, catch-and-shoot mechanics, scoring in tight spaces, decision-making.

            YOUTUBE SEARCH TERMS:
            - 'Jalen Brunson workout training drills'
            - 'Jalen Brunson footwork pivot drills'
            - 'Jalen Brunson signature moves snatch back'
            """,
            metadata={"player": "Jalen Brunson", "position": "PG", "skills": "footwork, midrange, pivoting"}
        ),

        Document(
            page_content="""
            Player: Shai Gilgeous-Alexander | Team: Oklahoma City Thunder | Position: Point Guard

            SIGNATURE DRILLS:
            - Deceleration drill: Sprint at full speed, then stop on a dime under control. Builds the strength to absorb high-impact stops without losing balance.
            - Balance and stability work: Single-leg holds, pivots, and contorting movements while staying grounded — trains body control for drawing fouls and changing pace.
            - Two-ball dribbling under pressure: Combine two-ball dribbling with rapid direction changes to simulate defensive pressure in confined spaces.
            - Agility ladder footwork: Fast-feet ladder patterns to build quickness for elite guard movement.
            - Change-of-pace drive: Practice driving at one speed, then exploding to a completely different speed mid-drive to throw off defenders' timing.
            - Strength foundation: Bulgarian split squats and RDLs to build the strength needed to absorb contact at full speed.

            TRAINING QUOTES:
            - 'He already had a routine. He was already working... He had a paper that he wrote drills on.' (on SGA's youth training habits, via his former coach)

            KEY SKILLS: Change of pace, balance, body control, deceleration, footwork, drawing contact.

            YOUTUBE SEARCH TERMS:
            - 'Shai Gilgeous-Alexander workout training drills'
            - 'SGA footwork balance training'
            - 'Shai Gilgeous-Alexander signature moves'
            """,
            metadata={"player": "Shai Gilgeous-Alexander", "position": "PG", "skills": "change of pace, balance, footwork"}
        ),

        Document(
            page_content="""
            Player: Jayson Tatum | Team: Boston Celtics | Position: Forward

            SIGNATURE DRILLS:
            - Floater off the catch: Catch, take 1-2 dribbles, pick the ball up low, and rise for a floater over a defender. Repeat from both sides.
            - In-and-out side step: Practice the in-and-out dribble move into a side-step jumper, used to create separation from a closing defender.
            - Step-back into 'punch drag': Combine a hard dribble with a step-back to create space for a clean jumper.
            - Post-up footwork: Get solid post positioning, work counters from the elbow, using size advantage with a high-release jumpshot.
            - Deadlift-to-court transition: Heavy lower body strength work immediately followed by on-court dribbling, shooting, and running drills to train power under fatigue.

            TRAINING QUOTES:
            - On his approach to recovery and training: relentless, methodical work translated his rehab into renewed explosiveness on the floor.

            KEY SKILLS: Scoring versatility, floaters, step-back shooting, post footwork, size and strength utilization.

            YOUTUBE SEARCH TERMS:
            - 'Jayson Tatum workout training drills'
            - 'Jayson Tatum signature moves step back'
            - 'Jayson Tatum floater post move drill'
            """,
            metadata={"player": "Jayson Tatum", "position": "F", "skills": "scoring versatility, floaters, step-back"}
        ),

        Document(
            page_content="""
            Player: Klay Thompson | Team: Golden State Warriors / Dallas Mavericks | Position: Shooting Guard

            SIGNATURE DRILLS:
            - Zero-dribble catch-and-shoot: Catch the pass and shoot immediately with no dribble, repeated from 5 spots around the arc. Builds quick-release mechanics.
            - Quick catch into a single move: Catch on the move, make one quick decision (shoot, one-dribble pull-up, or hesitation), and finish. No wasted motion.
            - Transition three reps: Run the floor in transition and spot up immediately behind the arc instead of attacking the rim — trains floor spacing and conditioning together.
            - Pump fake jab into one-dribble shot: Catch at the three-point line, pump fake, jab to move the defender, then rise into a one-dribble jumper.
            - Footwork off the catch: Plant the inside foot before catching while moving left; anchor and step into the shot while stationary. Repeat both directions.
            - Conditioning + free throws: Burpees paired immediately with free throws to practice shooting under fatigue.

            TRAINING QUOTES:
            - 'I don't adjust my routine to the opponent. I try to make the defense adjust to me, rather than adjust to them.'
            - 'I used to shoot a lot more before the game... I cut my routine in half, and my shooting percentage went up.'

            KEY SKILLS: Catch-and-shoot, off-ball movement, quick release, shooting footwork, conditioning while shooting.

            YOUTUBE SEARCH TERMS:
            - 'Klay Thompson catch and shoot drills'
            - 'Klay Thompson shooting form workout'
            - 'Klay Thompson transition shooting drill'
            """,
            metadata={"player": "Klay Thompson", "position": "SG", "skills": "catch-and-shoot, quick release, off-ball movement"}
        ),

        Document(
            page_content="""
            Player: Kyrie Irving | Team: Dallas Mavericks | Position: Point Guard

            SIGNATURE DRILLS:
            - Two-ball partner awareness drill: Dribble one ball while a partner randomly tosses a second ball at you from 10 feet away. Catch it one-handed, pass back, and keep your original dribble alive. Builds split focus and ball control under chaos.
            - The 'Kyrie Irving drill': Start at half court, dribble at speed to the 3-point line while passing back and forth with a coach using your weak hand 3 times, then finish at the rim or with a short pull-up jumper.
            - Ambidextrous handle work: Off-hand dribbling drills specifically to make the off hand as active and protective as the dominant hand.
            - Finishing with either hand: Practice scripted finishing sequences within 8 feet of the rim, using different releases and spins off the backboard.
            - Scripted combo moves: Choreograph a dribble sequence (behind-the-back into crossover, etc.) and drill it repeatedly until it becomes second nature — Kyrie treats his in-game creativity as rehearsed, not improvised.

            TRAINING QUOTES:
            - 'What I want people to realize is that when I make a move, it's really a simple move.'
            - 'I have counters to every move.'

            KEY SKILLS: Ball handling, ambidextrous control, finishing at the rim, footwork, creative scoring.

            YOUTUBE SEARCH TERMS:
            - 'Kyrie Irving ball handling drills'
            - 'Kyrie Irving finishing moves layup drills'
            - 'Kyrie Irving signature dribble moves'
            """,
            metadata={"player": "Kyrie Irving", "position": "PG", "skills": "ball handling, finishing, creative scoring"}
        ),

        Document(
            page_content="""
            Player: James Harden | Team: LA Clippers | Position: Guard

            SIGNATURE DRILLS:
            - Step-back footwork progression: Drill the footwork slowly first — forward dribble, plant, step back, shoot — before adding speed.
            - Step-back off the between-the-legs dribble: Once the basic footwork is clean, add a between-the-legs dribble into the step-back to make the move feel natural in motion.
            - Hesitation into step-back: Use a hesitation dribble to freeze the defender, then explode backward into space for the jumper.
            - Euro step progression: Jab step into a side step (footwork only), then add a dribble, then add full speed and distance from the 3-point line — a layered progression for the Euro step finish.
            - Balance and release drill: Practice the step-back while focused purely on not leaning back too far, keeping the shot's balance and rhythm clean.

            TRAINING QUOTES:
            - 'Are they shaky? Are they moving while I'm dribbling?' (on reading a defender's balance before attacking)

            KEY SKILLS: Step-back shooting, creating space, hesitation moves, footwork, one-on-one scoring.

            YOUTUBE SEARCH TERMS:
            - 'James Harden step back drill tutorial'
            - 'James Harden Euro step footwork drill'
            - 'James Harden workout training routine'
            """,
            metadata={"player": "James Harden", "position": "G", "skills": "step-back, creating space, footwork"}
        ),

        Document(
            page_content="""
            Player: Steve Nash | Team: Phoenix Suns / Dallas Mavericks | Position: Point Guard

            SIGNATURE DRILLS (from Nash's iconic 20-Minute Workout):
            - 50x close to the rim: Start with easy baskets from the lane line under the rim, switching sides with each shot to build a rhythm without stopping.
            - 10x pull-up jumpers: From the top of the key, take a few quick steps toward the basket before a pull-up jumper. Alternate direction with each rebound — pure footwork emphasis.
            - 10x spin and jump shot: From the same starting position, take a few quick steps and add a spin move before each jump shot. Alternate directions with each rebound.
            - 10x college three-point shots: Work around the arc at the NCAA three-point distance (19'9"), keeping the heart rate up between makes.
            - Final series — NBA three-point line: Finish the workout at the regulation NBA three-point distance (23'9"), shooting fatigued to simulate late-game shot-making.
            - The whole routine runs continuously for 20 minutes with no stoppages, building shooting touch under cardio fatigue — exactly the workout in Nash's own training video.

            TRAINING QUOTES:
            - 'Get to it, and keep those eyes up.'

            KEY SKILLS: Shooting under fatigue, footwork into jumpers, conditioning, range progression, rhythm shooting.

            YOUTUBE SEARCH TERMS:
            - 'Steve Nash 20 Minute Workout' (full session: https://www.youtube.com/watch?v=D3cO9c7RgAE)
            - 'Steve Nash shooting drill routine'
            - 'Steve Nash hesitation workout'
            """,
            metadata={"player": "Steve Nash", "position": "PG", "skills": "shooting under fatigue, conditioning, footwork"}
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
You are ProTrainAI, an elite basketball training assistant. You speak
directly to youth players (ages 8-18) and give them a personal training
session inspired by an NBA player's real training methods.

Use ONLY the player information provided below to design the workout.
Include real drills, motivational quotes from the player, and YouTube
search terms so the athlete can find videos themselves.

Write directly to the athlete, second person, like a personal trainer
talking to them one-on-one. Use "you" and "your" throughout — never
"the athlete" or "your player." This workout is being delivered straight
to the person doing it, not relayed through a coach.

PLAYER KNOWLEDGE BASE:
{context}

ATHLETE'S REQUEST:
{question}

Respond with a complete, structured 60-minute training session in this format:

🏀 YOUR PROTRAINAI SESSION
Inspired By: [Player Name]
Today's Focus: [Core skill focus]

💬 GET LOCKED IN:
[Quote from the player, framed as something to read to yourself before you start]

📋 YOUR SESSION:

⏱️ WARMUP (10 min)
[2-3 warmup activities, written as direct instructions: "Start with...", "Then do..."]

🔧 SKILL BLOCK 1 (15 min) — [Skill name]
[Drill name]: [Instructions written directly to the athlete: "Dribble to...", "Take 10 shots from..."]
[A coaching cue in the player's style]

🔧 SKILL BLOCK 2 (15 min) — [Skill name]
[Drill name]: [Instructions]
[Coaching cue]

🔧 SKILL BLOCK 3 (10 min) — [Skill name]
[Drill name]: [Instructions]

🏁 FINISH STRONG (8 min)
[A challenge or competitive finish, written as a direct dare/goal: "See if you can..."]

🧘 COOL DOWN (2 min)
[Closing quote + a reflection question asked directly: "Ask yourself..."]

🎥 WATCH THESE NEXT:
Search YouTube for:
- [Search term 1]
- [Search term 2]

💡 ONE THING TO REMEMBER:
[1-2 direct, motivating tips for the athlete to carry into their next session]
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
player_name_input = st.text_input(
    "What's your name?",
    placeholder="e.g. Jordan",
    help="So your coach knows who trained today.",
)

col1, col2 = st.columns(2)

with col1:
    selected_player = st.selectbox(
        "Pick your inspiration 🌟",
        options=player_names,
    )

with col2:
    age = st.text_input("Your age (optional)", placeholder="e.g. 12")

notes = st.text_area(
    "Anything specific you want to work on? (optional)",
    placeholder="e.g. I want to work on my left hand, or I struggle with free throws",
    height=80,
)

generate_clicked = st.button("🏀 Generate My Workout", type="primary", use_container_width=True)

st.divider()

# ─── GENERATE & DISPLAY OUTPUT ──────────────────────────────────
if generate_clicked:
    if not player_name_input.strip():
        st.warning("👋 Type your name above first, so your coach knows who this workout is for!")
        st.stop()

    # Build the query from the dropdown + optional notes
    query_parts = [
        f"My name is {player_name_input.strip()}. "
        f"Build me a 60-minute workout inspired by {selected_player}."
    ]
    if age:
        query_parts.append(f"I am {age} years old.")
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
    log_usage(player_name_input.strip(), selected_player, age, notes)

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

