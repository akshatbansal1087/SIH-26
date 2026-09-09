import streamlit as st
from PIL import Image
from pathlib import Path
from datetime import datetime
import sys
import base64
import urllib.request
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Tropical Cyclone Intelligence",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# BACKEND
# =========================================================

sys.path.append(str(Path(__file__).parent))

from backend import predict_cyclone


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #08111f;
    color: #e8eef7;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #edf4ff !important;
}


/* =====================================================
   NAVIGATION
   ===================================================== */

.stButton button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #263c55;
    background-color: #101e30;
    color: #dcecff;
    min-height: 42px;
    font-weight: 600;
}

.stButton button:hover {
    border-color: #4e86bd;
    background-color: #152941;
}


/* =====================================================
   UPLOADERS
   ===================================================== */

[data-testid="stFileUploader"] {
    background-color: #0c1828;
    border-radius: 12px;
    padding: 8px;
}


/* =====================================================
   METRICS
   ===================================================== */

[data-testid="stMetric"] {
    background-color: #0d1a2b;
    border: 1px solid #20354d;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricLabel"] {
    color: #8294aa;
}

[data-testid="stMetricValue"] {
    color: #edf5ff;
}


/* =====================================================
   INFO BOXES
   ===================================================== */

.info-box {
    background-color: #0d1a2b;
    border: 1px solid #20354d;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.small-text {
    color: #8798ad;
    font-size: 14px;
}

.big-number {
    font-size: 48px;
    font-weight: 700;
    color: #edf5ff;
}

.classification-box {
    background-color: #132a42;
    border: 1px solid #31577b;
    border-radius: 12px;
    padding: 14px;
    font-size: 18px;
    font-weight: 600;
    color: #b9dcff;
}


/* =====================================================
   COMPACT TRACKING METRICS
   ===================================================== */

.compact-metric {
    background: #0c1828;
    border: 1px solid #1b3854;
    border-radius: 10px;
    padding: 10px 13px;
    min-height: 72px;
}

.compact-label {
    color: #7f96ad;
    font-size: 11px;
    margin-bottom: 5px;
}

.compact-value {
    color: #edf5ff;
    font-size: 18px;
    font-weight: 700;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# IMAGE DOWNLOADER
# =========================================================

@st.cache_data(show_spinner=False)
def download_image(url):

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/140 Safari/537.36"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:

            data = response.read()

        lower_url = url.lower()

        if ".svg" in lower_url:
            mime = "image/svg+xml"

        elif ".gif" in lower_url:
            mime = "image/gif"

        elif ".png" in lower_url:
            mime = "image/png"

        else:
            mime = "image/jpeg"

        encoded = base64.b64encode(data).decode("utf-8")

        return f"data:{mime};base64,{encoded}"

    except Exception:
        return ""


# =========================================================
# IMAGE SOURCES
# =========================================================

NASA_GITA = (
    "https://eoimages.gsfc.nasa.gov/images/"
    "imagerecords/91000/91735/"
    "gita_vir_2018047_lrg.jpg"
)

NASA_LAILA = (
    "https://eoimages.gsfc.nasa.gov/images/"
    "imagerecords/44000/44045/"
    "laila_amo_2010138_lrg.jpg"
)

NASA_SHANSHAN = (
    "https://eoimages.gsfc.nasa.gov/images/"
    "imagerecords/153000/153266/"
    "shanshan_virs2_20240828_lrg.jpg"
)


# =========================================================
# LOAD DASHBOARD IMAGES
# =========================================================

hero_img = download_image(NASA_GITA)

laila_img = download_image(NASA_LAILA)

shanshan_img = download_image(NASA_SHANSHAN)


# =========================================================
# HEADER
# =========================================================

head_col1, head_col2 = st.columns([4, 1])

with head_col1:

    st.title(
        "🌀 N.E.T.R.A."
    )

    st.caption(
        "Neural Engine for Tropical Recognition and Analysis"
    )


with head_col2:

    st.markdown("###")

    st.markdown(
        """
<div style="
text-align:right;
color:#00ff88;
font-weight:600;
font-size:14px;
">
● Model Online
</div>
""",
        unsafe_allow_html=True
    )


st.markdown("###")


# =========================================================
# NAVIGATION
# =========================================================

page1, page2, page3, page4 = st.tabs(
    [
        "⌂ Dashboard",
        "◈ Cyclone Analysis",
        "◎ Tracking & Alerts",
        "◇ Intelligence"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

with page1:

    dashboard_html = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="utf-8">

<style>

* {
    box-sizing: border-box;
}

html,
body {

    margin: 0;
    padding: 0;

    background: #050b14;

    color: #eaf4ff;

    font-family:
        Arial,
        Helvetica,
        sans-serif;
}

body {
    padding: 2px;
}

.wrap {

    max-width: 1450px;

    margin: 0 auto;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {

    height: 275px;

    border:
        1px solid #155080;

    border-radius:
        17px;

    overflow:
        hidden;

    position:
        relative;

    background:
        linear-gradient(
            90deg,
            rgba(3,13,27,.98) 0%,
            rgba(5,17,34,.94) 34%,
            rgba(5,12,23,.30) 72%,
            rgba(5,12,23,.05) 100%
        ),
        url("__HERO_IMAGE__")
        center 44% / cover no-repeat;

    box-shadow:
        0 0 28px
        rgba(0,143,255,.08);
}

.hero-content {

    position:
        absolute;

    left:
        30px;

    top:
        25px;

    width:
        55%;
}

.kicker {

    font-size:
        12px;

    letter-spacing:
        1.8px;

    font-weight:
        800;

    color:
        #19d9ef;

    margin-bottom:
        9px;
}

.hero h1 {

    font-size:
        42px;

    line-height:
        1;

    margin:
        0 0 14px;

    font-weight:
        800;
}

.hero p {

    font-size:
        15px;

    line-height:
        1.45;

    color:
        #c2d1e3;

    max-width:
        590px;

    margin:
        0 0 25px;
}

.features {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        0;

    max-width:
        690px;
}

.feature {

    padding:
        0 15px;

    border-right:
        1px solid
        rgba(90,145,190,.35);

    min-height:
        55px;
}

.feature:first-child {
    padding-left:
        0;
}

.feature:last-child {
    border-right:
        0;
}

.fi {

    font-size:
        23px;

    color:
        #16d9ff;

    margin-bottom:
        4px;
}

.ft {

    font-size:
        11px;

    line-height:
        1.35;

    color:
        #c7d8ea;
}

.source {

    position:
        absolute;

    right:
        22px;

    bottom:
        16px;

    text-align:
        right;

    color:
        #c5d2e0;

    font-size:
        10px;

    text-shadow:
        0 1px 4px #000;
}


/* =====================================================
   ROWS
   ===================================================== */

.row {

    display:
        grid;

    grid-template-columns:
        1.72fr 1fr;

    gap:
        13px;

    margin-top:
        13px;
}

.bottom {

    display:
        grid;

    grid-template-columns:
        1.72fr 1fr;

    gap:
        13px;

    margin-top:
        13px;
}


/* =====================================================
   CARDS
   ===================================================== */

.card {

    background:
        linear-gradient(
            145deg,
            #07172a,
            #06111f
        );

    border:
        1px solid #17456c;

    border-radius:
        15px;

    overflow:
        hidden;

    box-shadow:
        0 5px 20px
        rgba(0,0,0,.16);
}

.card-head {

    min-height:
        68px;

    padding:
        14px 18px;

    border-bottom:
        1px solid #17344f;

    display:
        flex;

    justify-content:
        space-between;

    align-items:
        center;
}

.card-title {

    font-size:
        17px;

    font-weight:
        800;

    color:
        #eef7ff;
}

.card-sub {

    font-size:
        11px;

    color:
        #7f98b3;

    margin-top:
        4px;
}


/* =====================================================
   BADGES
   ===================================================== */

.badges {

    display:
        flex;

    gap:
        8px;
}

.badge {

    padding:
        7px 11px;

    border-radius:
        18px;

    background:
        #092744;

    color:
        #61c8ff;

    border:
        1px solid #155a86;

    font-size:
        10px;

    font-weight:
        700;

    white-space:
        nowrap;
}

.badge.orange {

    background:
        #2c1808;

    color:
        #ffad52;

    border-color:
        #8e4b10;
}


/* =====================================================
   GLOBAL MAP
   ===================================================== */

.map {

    height:
        280px;

    position:
        relative;

    overflow:
        hidden;

    background:
        #07182a;
}


/* -----------------------------------------------------
   WORLD MAP - ALWAYS VISIBLE
   ----------------------------------------------------- */

.world-svg {

    position:
        absolute;

    inset:
        0;

    width:
        100%;

    height:
        100%;

    z-index:
        1;
}


/* -----------------------------------------------------
   MAP OCEAN GRID
   ----------------------------------------------------- */

.map-grid {

    position:
        absolute;

    inset:
        0;

    z-index:
        2;

    pointer-events:
        none;

    opacity:
        .18;

    background-image:
        linear-gradient(
            rgba(67,153,206,.18) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(67,153,206,.18) 1px,
            transparent 1px
        );

    background-size:
        50px 35px;
}


/* -----------------------------------------------------
   GLOBAL CYCLONE INDICATORS
   ----------------------------------------------------- */

.global-hotspot {

    position:
        absolute;

    width:
        14px;

    height:
        14px;

    border-radius:
        50%;

    background:
        #18d8ff;

    border:
        2px solid
        rgba(255,255,255,.85);

    box-shadow:
        0 0 0 5px
        rgba(24,216,255,.15),
        0 0 20px
        rgba(24,216,255,.95);

    z-index:
        8;
}


/* North Atlantic */

.g1 {
    left:
        34%;

    top:
        42%;
}


/* East Pacific */

.g2 {
    left:
        22%;

    top:
        53%;
}


/* Caribbean / Atlantic */

.g3 {
    left:
        39%;

    top:
        51%;
}


/* West Pacific */

.g4 {
    right:
        13%;

    top:
        48%;
}


/* North Pacific */

.g5 {
    right:
        24%;

    top:
        31%;
}


/* South Pacific */

.g6 {
    right:
        20%;

    bottom:
        23%;
}


/* North Indian Ocean */

.g7 {
    left:
        61%;

    top:
        54%;
}


/* Arabian Sea */

.g8 {
    left:
        55%;

    top:
        56%;
}


/* South Indian Ocean */

.g9 {
    left:
        57%;

    bottom:
        22%;
}


/* Australian region */

.g10 {
    right:
        17%;

    bottom:
        32%;
}


/* -----------------------------------------------------
   CYCLONE BASIN LABELS
   ----------------------------------------------------- */

.basin-label {

    position:
        absolute;

    z-index:
        7;

    color:
        #a9c9df;

    font-size:
        9px;

    font-weight:
        700;

    letter-spacing:
        .4px;

    text-shadow:
        0 1px 5px #00101c;

    opacity:
        .85;
}

.basin-atlantic {
    left:
        30%;
    top:
        35%;
}

.basin-pacific {
    right:
        9%;
    top:
        41%;
}

.basin-indian {
    left:
        53%;
    top:
        61%;
}

.basin-australia {
    right:
        12%;
    bottom:
        27%;
}


/* =====================================================
   INDIA
   ===================================================== */

.map-india {

    position:
        absolute;

    z-index:
        9;

    left:
        55.5%;

    top:
        38%;

    width:
        105px;

    height:
        125px;

    object-fit:
        contain;

    opacity:
        .98;

    filter:
        invert(71%)
        sepia(97%)
        saturate(1600%)
        hue-rotate(347deg)
        brightness(104%)
        contrast(101%);
}


/* =====================================================
   INDIA HOTSPOTS
   ===================================================== */

.hotspot {

    position:
        absolute;

    width:
        16px;

    height:
        16px;

    border-radius:
        50%;

    background:
        #ff8a18;

    box-shadow:
        0 0 0 6px
        rgba(255,138,24,.12),
        0 0 18px
        #ff8a18;

    z-index:
        10;
}

.h1 {
    left:
        56%;
    top:
        49%;
}

.h2 {
    left:
        60%;
    top:
        56%;
}

.h3 {
    left:
        64%;
    top:
        48%;
}

.h4 {
    left:
        59%;
    top:
        63%;
}


/* =====================================================
   MAP LABELS
   ===================================================== */

.map-label {

    position:
        absolute;

    z-index:
        6;

    color:
        #d0e1f2;

    font-style:
        italic;

    font-size:
        12px;

    text-shadow:
        0 1px 5px #00101c;
}

.atlantic {

    left:
        31%;

    top:
        37%;
}

.indian {

    left:
        55%;

    bottom:
        20px;
}

.pacific {

    right:
        11%;

    top:
        36%;
}


/* =====================================================
   MAP BOTTOM
   ===================================================== */

.map-bottom {

    height:
        64px;

    padding:
        10px 18px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    border-top:
        1px solid #122d47;
}

.stats {

    display:
        flex;

    gap:
        38px;
}

.stat-label {

    font-size:
        10px;

    color:
        #7188a2;
}

.stat-value {

    font-size:
        20px;

    font-weight:
        800;

    margin-top:
        3px;
}

.orange-t {
    color:
        #ff9b39;
}

.realtime {

    font-size:
        10px;

    color:
        #7188a2;
}


/* =====================================================
   STORMS
   ===================================================== */

.storms {

    padding:
        0 15px 12px;
}

.storm {

    display:
        grid;

    grid-template-columns:
        64px 1fr auto;

    gap:
        12px;

    align-items:
        center;

    padding:
        10px 0;

    border-bottom:
        1px solid #17314c;
}

.storm:last-child {
    border-bottom:
        0;
}

.storm img {

    width:
        64px;

    height:
        55px;

    border-radius:
        9px;

    object-fit:
        cover;

    background:
        #0c1e31;
}

.name {

    font-size:
        13px;

    font-weight:
        800;
}

.meta {

    font-size:
        10px;

    line-height:
        1.5;

    color:
        #829ab4;

    margin-top:
        3px;
}

.cat {

    font-size:
        10px;

    font-weight:
        800;

    border-radius:
        18px;

    padding:
        5px 10px;

    white-space:
        nowrap;
}

.c1 {

    color:
        #ffb23e;

    border:
        1px solid #b87917;

    background:
        #2c1d07;
}

.c2 {

    color:
        #ffd24a;

    border:
        1px solid #b39117;

    background:
        #302605;
}

.c3 {

    color:
        #ff527e;

    border:
        1px solid #b51d45;

    background:
        #300914;
}

.c5 {

    color:
        #c66dff;

    border:
        1px solid #8b2db9;

    background:
        #260a35;
}

.view {

    color:
        #14bfff;

    font-size:
        10px;

    font-weight:
        800;
}


/* =====================================================
   FACTS
   ===================================================== */

.facts {

    padding:
        0 18px 16px;
}

.fact-grid {

    display:
        grid;

    grid-template-columns:
        repeat(4,1fr);

    gap:
        0;
}

.fact {

    text-align:
        center;

    padding:
        13px 12px;

    border-right:
        1px solid #183650;
}

.fact:last-child {
    border-right:
        0;
}

.fact-icon {

    font-size:
        26px;

    color:
        #13d6ff;
}

.fact-num {

    font-size:
        21px;

    font-weight:
        800;

    margin-top:
        8px;
}

.fact-desc {

    font-size:
        10px;

    color:
        #8ba0b8;

    line-height:
        1.35;

    margin-top:
        4px;
}


/* =====================================================
   INDIAN OCEAN
   ===================================================== */

.region {

    min-height:
        230px;

    padding:
        19px;

    position:
        relative;

    overflow:
        hidden;

    background:
        linear-gradient(
            90deg,
            rgba(5,18,34,.97),
            rgba(5,18,34,.72)
        ),
        url("__LAILA_IMAGE__")
        center / cover no-repeat;
}

.region:after {

    content:
        "";

    position:
        absolute;

    inset:
        0;

    background:
        linear-gradient(
            90deg,
            rgba(4,15,28,.20),
            rgba(4,15,28,.05)
        );
}

.region-content {

    position:
        relative;

    z-index:
        2;

    max-width:
        58%;
}

.region-k {

    font-size:
        10px;

    letter-spacing:
        1.5px;

    font-weight:
        800;

    color:
        #13d8f2;
}

.region h2 {

    font-size:
        22px;

    line-height:
        1.1;

    margin:
        6px 0 12px;
}

.region p {

    font-size:
        11px;

    line-height:
        1.5;

    color:
        #c1d0df;
}

.cta {

    display:
        inline-block;

    margin-top:
        8px;

    border:
        1px solid #0e78a9;

    border-radius:
        18px;

    padding:
        8px 13px;

    color:
        #13d8f2;

    font-size:
        10px;

    font-weight:
        800;

    background:
        rgba(3,28,48,.6);
}


/* =====================================================
   RESPONSIVE
   ===================================================== */

@media(max-width:900px) {

    .row,
    .bottom {

        grid-template-columns:
            1fr;
    }

    .hero {

        height:
            390px;
    }

    .hero-content {

        width:
            88%;
    }

    .features {

        grid-template-columns:
            repeat(2,1fr);

        gap:
            12px;
    }

    .feature {

        border-right:
            0;
    }

}

</style>

</head>


<body>

<div class="wrap">


<!-- =====================================================
     HERO
     ===================================================== -->

<div class="hero">

    <div class="hero-content">

        <div class="kicker">
            NATURE'S MOST POWERFUL STORMS
        </div>

        <h1>
            Tropical Cyclones
        </h1>

        <p>
            From warm ocean waters to massive storm systems —
            tropical cyclones are nature's way of moving heat
            around the planet.
        </p>


        <div class="features">

            <div class="feature">

                <div class="fi">
                    🌀
                </div>

                <div class="ft">
                    Form over warm oceans
                    <br>
                    (≥ 26.5°C)
                </div>

            </div>


            <div class="feature">

                <div class="fi">
                    🌪
                </div>

                <div class="ft">
                    Can reach wind speeds
                    <br>
                    of &gt; 200 km/h
                </div>

            </div>


            <div class="feature">

                <div class="fi">
                    🌧
                </div>

                <div class="ft">
                    Bring heavy rain,
                    <br>
                    storm surge and flooding
                </div>

            </div>


            <div class="feature">

                <div class="fi">
                    ⚡
                </div>

                <div class="ft">
                    Early detection
                    <br>
                    saves lives
                </div>

            </div>

        </div>

    </div>


    <div class="source">

        Satellite image: Tropical Cyclone Gita (2018)
        <br>
        Source: NASA Earth Observatory

    </div>

</div>


<!-- =====================================================
     GLOBAL + INDIAN ACTIVITY / RECENT CYCLONES
     ===================================================== -->

<div class="row">


<!-- =====================================================
     MAP
     ===================================================== -->

<div class="card">

    <div class="card-head">

        <div>

            <div class="card-title">
                Global &amp; Indian Cyclone Activity
            </div>

            <div class="card-sub">
                Major tropical cyclone basins and Indian Ocean focus
            </div>

        </div>


        <div class="badges">

            <div class="badge">
                Global
            </div>

            <div class="badge orange">
                Indian Region
            </div>

        </div>

    </div>


    <div class="map">


        <!-- =================================================
             ALWAYS-VISIBLE WORLD MAP
             ================================================= -->

        <svg
            class="world-svg"
            viewBox="0 0 1000 500"
            preserveAspectRatio="none"
            xmlns="http://www.w3.org/2000/svg"
        >

            <!-- Ocean -->

            <rect
                x="0"
                y="0"
                width="1000"
                height="500"
                fill="#071c31"
            />


            <!-- Longitude lines -->

            <g
                stroke="#1c4765"
                stroke-width="1"
                opacity=".35"
                fill="none"
            >

                <path d="M100 0 C120 130 120 370 100 500"/>
                <path d="M200 0 C220 130 220 370 200 500"/>
                <path d="M300 0 C320 130 320 370 300 500"/>
                <path d="M400 0 C420 130 420 370 400 500"/>
                <path d="M500 0 C520 130 520 370 500 500"/>
                <path d="M600 0 C620 130 620 370 600 500"/>
                <path d="M700 0 C720 130 720 370 700 500"/>
                <path d="M800 0 C820 130 820 370 800 500"/>
                <path d="M900 0 C920 130 920 370 900 500"/>

            </g>


            <!-- Latitude lines -->

            <g
                stroke="#1c4765"
                stroke-width="1"
                opacity=".35"
                fill="none"
            >

                <path d="M0 100 H1000"/>
                <path d="M0 200 H1000"/>
                <path d="M0 300 H1000"/>
                <path d="M0 400 H1000"/>

            </g>


            <!-- =================================================
                 NORTH AMERICA
                 ================================================= -->

            <path
                d="
                M75 92
                L105 55
                L155 45
                L205 65
                L245 95
                L230 125
                L205 142
                L190 170
                L160 180
                L145 155
                L115 148
                L95 125
                L75 92
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- Central America -->

            <path
                d="
                M190 170
                L215 180
                L225 205
                L213 220
                L200 205
                L190 170
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- South America -->

            <path
                d="
                M255 215
                L295 225
                L315 260
                L305 300
                L285 340
                L270 390
                L245 430
                L225 405
                L230 365
                L210 330
                L225 295
                L215 260
                L255 215
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- Greenland -->

            <path
                d="
                M185 25
                L225 12
                L265 30
                L250 65
                L215 75
                L185 55
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- Europe -->

            <path
                d="
                M475 78
                L505 58
                L545 60
                L575 78
                L605 80
                L625 100
                L605 118
                L575 112
                L555 125
                L525 113
                L500 120
                L478 105
                L475 78
                Z
                "
                fill="#1a4057"
                stroke="#4a7b94"
                stroke-width="2"
            />


            <!-- Scandinavia -->

            <path
                d="
                M520 25
                L550 20
                L575 45
                L560 75
                L535 62
                Z
                "
                fill="#1a4057"
                stroke="#4a7b94"
                stroke-width="2"
            />


            <!-- Asia -->

            <path
                d="
                M575 95
                L620 65
                L680 60
                L745 75
                L820 105
                L885 135
                L900 175
                L865 195
                L820 180
                L780 205
                L735 190
                L700 205
                L665 185
                L625 180
                L595 155
                L575 95
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- Arabian Peninsula -->

            <path
                d="
                M570 165
                L610 170
                L630 200
                L600 225
                L565 205
                L570 165
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- India -->

            <path
                d="
                M650 175
                L675 180
                L690 205
                L675 245
                L650 280
                L635 245
                L642 215
                L630 195
                Z
                "
                fill="#b75d17"
                stroke="#ff9b39"
                stroke-width="3"
            />


            <!-- Southeast Asia -->

            <path
                d="
                M700 190
                L735 205
                L750 235
                L735 260
                L710 245
                L700 215
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- Japan -->

            <path
                d="
                M825 135
                L840 145
                L835 165
                L820 155
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- Africa -->

            <path
                d="
                M470 155
                L515 150
                L550 180
                L555 230
                L540 285
                L515 335
                L480 370
                L450 340
                L435 295
                L445 250
                L425 210
                L440 175
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- Madagascar -->

            <path
                d="
                M570 315
                L580 345
                L570 380
                L558 350
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- Australia -->

            <path
                d="
                M755 310
                L810 295
                L865 315
                L885 355
                L860 390
                L815 400
                L770 380
                L745 345
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />


            <!-- New Zealand -->

            <path
                d="
                M900 390
                L920 405
                L910 430
                L895 415
                Z
                "
                fill="#163a52"
                stroke="#39708f"
                stroke-width="2"
            />

        </svg>


        <!-- GRID -->

        <div class="map-grid"></div>


        <!-- =================================================
             GLOBAL CYCLONE MARKERS
             ================================================= -->

        <div class="global-hotspot g1"></div>
        <div class="global-hotspot g2"></div>
        <div class="global-hotspot g3"></div>
        <div class="global-hotspot g4"></div>
        <div class="global-hotspot g5"></div>
        <div class="global-hotspot g6"></div>
        <div class="global-hotspot g7"></div>
        <div class="global-hotspot g8"></div>
        <div class="global-hotspot g9"></div>
        <div class="global-hotspot g10"></div>


        <!-- =================================================
             BASIN LABELS
             ================================================= -->

        <div class="basin-label basin-atlantic">
            NORTH ATLANTIC
        </div>

        <div class="basin-label basin-pacific">
            WEST PACIFIC
        </div>

        <div class="basin-label basin-indian">
            INDIAN OCEAN
        </div>

        <div class="basin-label basin-australia">
            AUSTRALIAN REGION
        </div>


        <!-- INDIA HOTSPOTS -->

        <div class="hotspot h1"></div>
        <div class="hotspot h2"></div>
        <div class="hotspot h3"></div>
        <div class="hotspot h4"></div>


        <div class="map-label atlantic">
            Atlantic
        </div>

        <div class="map-label indian">
            Indian Ocean
        </div>

        <div class="map-label pacific">
            Pacific
        </div>

    </div>


    <!-- MAP FOOTER -->

    <div class="map-bottom">

        <div class="stats">

            <div>

                <div class="stat-label">
                    Global Active Systems
                </div>

                <div class="stat-value">
                    12
                </div>

            </div>


            <div>

                <div class="stat-label">
                    Indian Ocean Systems
                </div>

                <div class="stat-value orange-t">
                    5
                </div>

            </div>

        </div>


        <div class="realtime">
            Cyclone-prone basin overview
        </div>

    </div>

</div>


<!-- =====================================================
     RECENT NOTABLE CYCLONES
     ===================================================== -->

<div class="card">

    <div class="card-head">

        <div>

            <div class="card-title">
                Recent Notable Cyclones
            </div>

            <div class="card-sub">
                Selected tropical cyclone observations
            </div>

        </div>

        <div class="view">
            VIEW ALL →
        </div>

    </div>


    <div class="storms">


        <div class="storm">

            <img
                src="__HERO_IMAGE__"
                alt="Cyclone"
            >

            <div>

                <div class="name">
                    Cyclone Gita
                </div>

                <div class="meta">
                    South Pacific
                    <br>
                    2018
                </div>

            </div>

            <div class="cat c1">
                Severe
            </div>

        </div>


        <div class="storm">

            <img
                src="__LAILA_IMAGE__"
                alt="Cyclone"
            >

            <div>

                <div class="name">
                    Cyclone Laila
                </div>

                <div class="meta">
                    Bay of Bengal
                    <br>
                    2010
                </div>

            </div>

            <div class="cat c2">
                Cyclonic
            </div>

        </div>


        <div class="storm">

            <img
                src="__SHANSHAN_IMAGE__"
                alt="Typhoon"
            >

            <div>

                <div class="name">
                    Typhoon Shanshan
                </div>

                <div class="meta">
                    Western Pacific
                    <br>
                    2024
                </div>

            </div>

            <div class="cat c3">
                Severe
            </div>

        </div>


        <div class="storm">

            <img
                src="__HERO_IMAGE__"
                alt="Cyclone"
            >

            <div>

                <div class="name">
                    Tropical Cyclone
                </div>

                <div class="meta">
                    Global Basin
                    <br>
                    Satellite observation
                </div>

            </div>

            <div class="cat c5">
                Monitored
            </div>

        </div>


    </div>

</div>

</div>


<!-- =====================================================
     LOWER SECTION
     ===================================================== -->

<div class="bottom">


<!-- =====================================================
     FACTS
     ===================================================== -->

<div class="card">

    <div class="card-head">

        <div>

            <div class="card-title">
                Tropical Cyclone Facts
            </div>

            <div class="card-sub">
                Key scientific characteristics
            </div>

        </div>

    </div>


    <div class="facts">

        <div class="fact-grid">


            <div class="fact">

                <div class="fact-icon">
                    🌡
                </div>

                <div class="fact-num">
                    26.5°C+
                </div>

                <div class="fact-desc">
                    Typical minimum ocean
                    temperature for formation
                </div>

            </div>


            <div class="fact">

                <div class="fact-icon">
                    🌀
                </div>

                <div class="fact-num">
                    5
                </div>

                <div class="fact-desc">
                    Major global tropical
                    cyclone basins
                </div>

            </div>


            <div class="fact">

                <div class="fact-icon">
                    💨
                </div>

                <div class="fact-num">
                    &gt;200 km/h
                </div>

                <div class="fact-desc">
                    Wind speeds possible in
                    intense tropical cyclones
                </div>

            </div>


            <div class="fact">

                <div class="fact-icon">
                    🌍
                </div>

                <div class="fact-num">
                    ~80
                </div>

                <div class="fact-desc">
                    Tropical cyclones occur
                    globally in a typical year
                </div>

            </div>


        </div>

    </div>

</div>


<!-- =====================================================
     INDIAN OCEAN
     ===================================================== -->

<div class="card">

    <div class="region">

        <div class="region-content">

            <div class="region-k">
                INDIAN OCEAN
            </div>

            <h2>
                A Region Prone to Cyclones
            </h2>

            <p>
                The North Indian Ocean contains the
                Arabian Sea and Bay of Bengal — two
                important cyclone-producing regions
                affecting the Indian subcontinent.
            </p>

            <div class="cta">
                EXPLORE CYCLONE ANALYSIS →
            </div>

        </div>

    </div>

</div>


</div>


</div>

</body>

</html>
"""


    dashboard_html = dashboard_html.replace(
        "__HERO_IMAGE__",
        hero_img
    )

    dashboard_html = dashboard_html.replace(
        "__LAILA_IMAGE__",
        laila_img
    )

    dashboard_html = dashboard_html.replace(
        "__SHANSHAN_IMAGE__",
        shanshan_img
    )


    components.html(
        dashboard_html,
        height=1040,
        scrolling=False
    )


# =========================================================
# CYCLONE ANALYSIS
# =========================================================

with page2:

    st.header("Cyclone Analysis")

    st.write(
        "Detailed view of the two satellite input streams "
        "used for intensity estimation."
    )

    c1, c2 = st.columns(2)


    # =====================================================
    # BT STREAM
    # =====================================================

    with c1:

        st.subheader("BT Stream")

        st.info(
            "Brightness Temperature imagery is processed "
            "through the Swin Transformer."
        )

        bt_analysis = st.file_uploader(
            "Choose BT image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ],
            key="analysis_bt"
        )

        if bt_analysis:

            image = Image.open(bt_analysis)

            # COMPACT MEDIUM-SIZED PREVIEW
            image.thumbnail((360, 360))

            st.image(
                image,
                caption="BT Input",
                width=360
            )


    # =====================================================
    # RAW STREAM
    # =====================================================

    with c2:

        st.subheader("RAW Stream")

        st.info(
            "RAW satellite imagery is processed "
            "through EfficientNet-B0."
        )

        raw_analysis = st.file_uploader(
            "Choose RAW image",
            type=[
                "png",
                "jpg",
                "jpeg",
                "tif",
                "tiff"
            ],
            key="analysis_raw"
        )

        if raw_analysis:

            image = Image.open(raw_analysis)

            # COMPACT MEDIUM-SIZED PREVIEW
            image.thumbnail((360, 360))

            st.image(
                image,
                caption="RAW Input",
                width=360
            )


    st.markdown("###")


    # =====================================================
    # RUN ESTIMATION
    # =====================================================

    if st.button(
        "◉ RUN INTENSITY ESTIMATION",
        use_container_width=True
    ):

        if bt_analysis is None or raw_analysis is None:

            st.warning(
                "Both BT and RAW images are required."
            )

        else:

            with st.spinner(
                "Running the trained two-stream model..."
            ):

                wind, category = predict_cyclone(
                    Image.open(
                        bt_analysis
                    ).convert("RGB"),

                    Image.open(
                        raw_analysis
                    ).convert("RGB")
                )

                current_time = datetime.now().strftime(
                    "%H:%M:%S"
                )

                result = {
                    "wind": wind,
                    "category": category,
                    "time": current_time
                }

                st.session_state.prediction = result

                st.session_state.history.insert(
                    0,
                    result
                )

            st.success(
                "Intensity estimation completed."
            )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    if st.session_state.prediction:

        result = st.session_state.prediction

        st.divider()

        st.subheader("Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Wind Speed",
                f"{result['wind']:.1f} kt"
            )

        with r2:

            st.metric(
                "Cyclone Classification",
                result["category"]
            )

        with r3:

            st.metric(
                "Observation",
                result["time"]
            )


# =========================================================
# TRACKING & ALERTS
# =========================================================

with page3:

    st.header("Cyclone Tracking & Alerts")

    st.write(
        "Observation history and intensity-change intelligence."
    )


    # =====================================================
    # CURRENT STATUS
    # =====================================================

    if st.session_state.prediction:

        latest = st.session_state.prediction

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "Latest Intensity",
                f"{latest['wind']:.1f} kt"
            )

        with b:

            st.metric(
                "Current Classification",
                latest["category"]
            )

        with c:

            st.metric(
                "Observations",
                len(st.session_state.history)
            )

    else:

        st.info(
            "Run a cyclone analysis to begin building "
            "the observation history."
        )


    st.divider()


    # =====================================================
    # OBSERVATION HISTORY
    # =====================================================

    st.subheader("Observation Timeline")

    if st.session_state.history:

        for i, item in enumerate(
            st.session_state.history
        ):

            st.write(
                f"**Observation {i + 1}**"
            )

            h1, h2, h3 = st.columns(3)

            with h1:

                st.metric(
                    "Wind",
                    f"{item['wind']:.1f} kt"
                )

            with h2:

                st.metric(
                    "Classification",
                    item["category"]
                )

            with h3:

                st.metric(
                    "Time",
                    item["time"]
                )

            st.divider()

    else:

        st.info(
            "No observations recorded yet."
        )


    # =====================================================
    # EARLY WARNING
    # =====================================================

    st.subheader("Early Warning Layer")

    if len(st.session_state.history) >= 2:

        latest = st.session_state.history[0]["wind"]

        previous = st.session_state.history[1]["wind"]

        change = latest - previous


        if change > 5:

            st.warning(
                f"Strengthening observed: "
                f"latest estimated intensity is "
                f"{change:.1f} kt higher than the previous observation."
            )


        elif change < -5:

            st.info(
                f"Intensity decrease observed: "
                f"latest estimated intensity is "
                f"{abs(change):.1f} kt lower than the previous observation."
            )


        else:

            st.success(
                "Intensity is relatively stable between "
                "the latest observations."
            )


    else:

        st.info(
            "Run at least two analyses to evaluate "
            "changes in estimated intensity."
        )


    st.divider()


    # =====================================================
    # TRACKING SCOPE
    # =====================================================

    st.subheader("Tracking Scope")

    st.write(
        "The current trained model estimates cyclone intensity "
        "from paired satellite imagery. Geographic coordinates "
        "and future cyclone trajectories require separate "
        "geospatial track data or a dedicated forecasting model."
    )


# =========================================================
# INTELLIGENCE / THEORY
# =========================================================

with page4:

    st.header("Cyclone Intelligence")

    st.write(
        "Technical overview of the system and its trained model."
    )


    # =====================================================
    # SYSTEM OVERVIEW
    # =====================================================

    st.subheader("1. System Overview")

    st.info(
        "The system processes two satellite image streams. "
        "Each stream is handled by a dedicated deep-learning "
        "model. Their learned representations are combined "
        "to estimate maximum sustained wind speed."
    )


    # =====================================================
    # TWO STREAMS
    # =====================================================

    st.subheader("2. Two-Stream Architecture")

    a, b = st.columns(2)


    with a:

        st.markdown("### BT Image")

        st.write(
            "Brightness Temperature imagery"
        )

        st.success(
            "Swin Transformer"
        )

        st.caption(
            "Extracts spatial features from the BT observation."
        )


    with b:

        st.markdown("### RAW Image")

        st.write(
            "RAW satellite observation"
        )

        st.success(
            "EfficientNet-B0"
        )

        st.caption(
            "Extracts a second feature representation "
            "of the cyclone structure."
        )


    # =====================================================
    # PIPELINE
    # =====================================================

    st.subheader("3. Prediction Pipeline")

    st.write("")

    p1, p2, p3, p4 = st.columns(4)


    with p1:

        st.metric(
            "INPUT 01",
            "BT"
        )

        st.caption(
            "Swin Transformer"
        )


    with p2:

        st.metric(
            "INPUT 02",
            "RAW"
        )

        st.caption(
            "EfficientNet-B0"
        )


    with p3:

        st.metric(
            "STAGE",
            "FUSION"
        )

        st.caption(
            "Feature combination"
        )


    with p4:

        st.metric(
            "OUTPUT",
            "WIND SPEED"
        )

        st.caption(
            "Maximum sustained wind"
        )


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.subheader("4. Model Performance")

    m1, m2 = st.columns(2)


    with m1:

        st.metric(
            "Validation MAE",
            "2.91 kt"
        )

        st.caption(
            "Mean Absolute Error"
        )


    with m2:

        st.metric(
            "Validation RMSE",
            "4.75 kt"
        )

        st.caption(
            "Root Mean Square Error"
        )


    # =====================================================
    # CLASSIFICATION
    # =====================================================

    st.subheader("5. Cyclone Classification")

    st.write(
        "Classification is derived from the predicted "
        "maximum sustained wind speed."
    )


    classification_data = [

        ["< 31 kt", "Low Pressure"],

        ["31 – 49 kt", "Depression"],

        ["50 – 61 kt", "Deep Depression"],

        ["62 – 88 kt", "Cyclonic Storm"],

        ["89 – 117 kt", "Severe Cyclonic Storm"],

        ["> 117 kt", "Very Severe Cyclonic Storm"],

    ]


    st.table(
        {
            "Wind Speed":
                [x[0] for x in classification_data],

            "Classification":
                [x[1] for x in classification_data]
        }
    )


    # =====================================================
    # SCOPE
    # =====================================================

    st.subheader("6. Current System Scope")

    st.write(
        "The trained model directly provides cyclone intensity "
        "estimation from paired BT and RAW satellite imagery. "
        "Classification is then derived from the estimated wind "
        "speed. Geographic tracking and future trajectory "
        "prediction require additional geospatial data or a "
        "dedicated forecasting component."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Tropical Cyclone Intelligence System • "
    "AI/ML-based satellite analysis"
)