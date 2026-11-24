"""Indian sites and scenarios used by the synthetic generators.

Coordinates are real places (district towns / river reaches); what is
generated at each one is a SIMULATION parameterised by the place's broad
character (river style, terrain, field size, settlement density, season),
not a reconstruction of any specific event or of the actual ground there.

`region` follows the task's grouping: North, West, East, South, Central,
North-East.
"""
from __future__ import annotations

MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December")


def season_of(month: int) -> str:
    """India Meteorological Department's four seasons."""
    if month in (1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "pre-monsoon"
    if month in (6, 7, 8, 9):
        return "south-west monsoon"
    return "post-monsoon"


# flood `kind`: meandering | braided | urban | coastal | agricultural | flash
# parcel_m: typical field size; villages: settlements per 5 km chip (min, max)
FLOOD_SITES = [
    # ---- North ----
    dict(state="Uttar Pradesh", region="North", place="Prayagraj (Ganga-Yamuna confluence)", lat=25.43, lon=81.88, feature="Ganga / Yamuna",
         scenario="Ganga and Yamuna river flooding", kind="meandering", months=(8, 9), river_m=(500, 1100), parcel_m=(35, 70), villages=(2, 5)),
    dict(state="Uttar Pradesh", region="North", place="Varanasi", lat=25.32, lon=83.01, feature="Ganga",
         scenario="Ganga and Yamuna river flooding", kind="meandering", months=(8, 9), river_m=(450, 900), parcel_m=(35, 70), villages=(3, 6)),
    dict(state="Uttar Pradesh", region="North", place="Gorakhpur (Rapti floodplain)", lat=26.76, lon=83.37, feature="Rapti",
         scenario="Bihar and eastern Uttar Pradesh monsoon flooding", kind="meandering", months=(7, 8, 9), river_m=(120, 300), parcel_m=(30, 60), villages=(3, 6)),
    dict(state="Delhi", region="North", place="Yamuna floodplain, Delhi", lat=28.66, lon=77.25, feature="Yamuna",
         scenario="Ganga and Yamuna river flooding", kind="meandering", months=(7, 8), river_m=(250, 500), parcel_m=(40, 80), villages=(5, 8)),
    dict(state="Delhi", region="North", place="Central Delhi (ITO - Minto Road)", lat=28.63, lon=77.24, feature="storm drains",
         scenario="Urban waterlogging", kind="urban", months=(7, 8, 9)),
    dict(state="Uttarakhand", region="North", place="Rudraprayag (Mandakini valley)", lat=30.28, lon=78.98, feature="Mandakini / Alaknanda",
         scenario="Himalayan flash flood", kind="flash", months=(6, 7, 8)),
    dict(state="Himachal Pradesh", region="North", place="Kullu (Beas valley)", lat=31.96, lon=77.11, feature="Beas",
         scenario="Himalayan flash flood", kind="flash", months=(7, 8)),
    dict(state="Punjab", region="North", place="Phillaur (Sutlej)", lat=31.02, lon=75.79, feature="Sutlej",
         scenario="Agricultural flooding", kind="agricultural", months=(7, 8), river_m=(150, 350), parcel_m=(90, 180), villages=(1, 3)),
    dict(state="Punjab", region="North", place="Patiala (Ghaggar)", lat=30.20, lon=76.40, feature="Ghaggar",
         scenario="Agricultural flooding", kind="agricultural", months=(7, 8), river_m=(60, 140), parcel_m=(90, 180), villages=(1, 3)),
    # ---- West ----
    dict(state="Rajasthan", region="West", place="Kota (Chambal)", lat=25.18, lon=75.83, feature="Chambal",
         scenario="Monsoon river flooding", kind="meandering", months=(8,), river_m=(200, 400), parcel_m=(80, 160), villages=(1, 3)),
    dict(state="Gujarat", region="West", place="Vadodara (Vishwamitri)", lat=22.31, lon=73.18, feature="Vishwamitri",
         scenario="Urban waterlogging", kind="urban", months=(7, 8)),
    dict(state="Gujarat", region="West", place="Bharuch (Narmada)", lat=21.70, lon=72.99, feature="Narmada",
         scenario="Monsoon river flooding", kind="meandering", months=(8, 9), river_m=(500, 1000), parcel_m=(70, 140), villages=(1, 3)),
    dict(state="Maharashtra", region="West", place="Mumbai (Kurla - Sion, Mithi river)", lat=19.07, lon=72.88, feature="Mithi river",
         scenario="Urban waterlogging", kind="urban", months=(6, 7, 8)),
    dict(state="Maharashtra", region="West", place="Kolhapur (Panchganga)", lat=16.70, lon=74.24, feature="Panchganga",
         scenario="Monsoon river flooding", kind="meandering", months=(7, 8), river_m=(100, 220), parcel_m=(50, 100), villages=(2, 4)),
    # ---- East ----
    dict(state="Bihar", region="East", place="Darbhanga (Bagmati - Kamla)", lat=26.15, lon=85.90, feature="Bagmati / Kamla",
         scenario="Bihar and eastern Uttar Pradesh monsoon flooding", kind="meandering", months=(7, 8), river_m=(100, 260), parcel_m=(25, 55), villages=(4, 7)),
    dict(state="Bihar", region="East", place="Supaul (Kosi)", lat=26.12, lon=86.60, feature="Kosi",
         scenario="Bihar and eastern Uttar Pradesh monsoon flooding", kind="braided", months=(7, 8), parcel_m=(25, 55), villages=(2, 5)),
    dict(state="Bihar", region="East", place="Patna", lat=25.61, lon=85.16, feature="Ganga",
         scenario="Ganga and Yamuna river flooding", kind="meandering", months=(8, 9), river_m=(700, 1400), parcel_m=(30, 60), villages=(3, 6)),
    dict(state="West Bengal", region="East", place="Gosaba, Sundarbans", lat=22.16, lon=88.80, feature="Bay of Bengal tidal creeks",
         scenario="Coastal flooding (cyclone storm surge)", kind="coastal", months=(5, 10, 11), parcel_m=(30, 60), villages=(2, 5)),
    dict(state="West Bengal", region="East", place="Malda", lat=25.01, lon=88.14, feature="Ganga",
         scenario="Ganga and Yamuna river flooding", kind="meandering", months=(8, 9), river_m=(600, 1300), parcel_m=(25, 55), villages=(3, 6)),
    dict(state="Assam", region="East", place="Dibrugarh", lat=27.48, lon=94.91, feature="Brahmaputra",
         scenario="Brahmaputra flooding", kind="braided", months=(6, 7, 8), parcel_m=(30, 60), villages=(2, 4)),
    dict(state="Assam", region="East", place="Majuli", lat=26.95, lon=94.17, feature="Brahmaputra",
         scenario="Brahmaputra flooding", kind="braided", months=(6, 7, 8), parcel_m=(30, 60), villages=(2, 4)),
    dict(state="Assam", region="East", place="Morigaon", lat=26.25, lon=92.34, feature="Brahmaputra / Kopili",
         scenario="Brahmaputra flooding", kind="meandering", months=(6, 7, 8), river_m=(200, 450), parcel_m=(30, 60), villages=(3, 5)),
    dict(state="Odisha", region="East", place="Kendrapara (Mahanadi delta)", lat=20.50, lon=86.42, feature="Bay of Bengal",
         scenario="Coastal flooding (cyclone storm surge)", kind="coastal", months=(5, 10, 11), parcel_m=(30, 60), villages=(2, 5)),
    dict(state="Odisha", region="East", place="Cuttack (Mahanadi)", lat=20.46, lon=85.88, feature="Mahanadi",
         scenario="Monsoon river flooding", kind="meandering", months=(8, 9), river_m=(500, 1100), parcel_m=(30, 60), villages=(3, 5)),
    # ---- South ----
    dict(state="Karnataka", region="South", place="Bengaluru (Bellandur - Mahadevapura)", lat=12.93, lon=77.68, feature="lake overflow / storm drains",
         scenario="Urban waterlogging", kind="urban", months=(9, 10)),
    dict(state="Karnataka", region="South", place="Chikkodi, Belagavi (Krishna)", lat=16.43, lon=74.60, feature="Krishna",
         scenario="Monsoon river flooding", kind="meandering", months=(8,), river_m=(150, 320), parcel_m=(50, 110), villages=(2, 4)),
    dict(state="Kerala", region="South", place="Kuttanad, Alappuzha", lat=9.42, lon=76.42, feature="Vembanad backwaters / Arabian Sea",
         scenario="Coastal flooding (backwater and sea)", kind="coastal", months=(7, 8), parcel_m=(40, 80), villages=(3, 6)),
    dict(state="Kerala", region="South", place="Aluva (Periyar)", lat=10.11, lon=76.35, feature="Periyar",
         scenario="Monsoon river flooding", kind="meandering", months=(8,), river_m=(150, 320), parcel_m=(25, 50), villages=(4, 7)),
    dict(state="Tamil Nadu", region="South", place="Chennai (Velachery - Pallikaranai)", lat=12.98, lon=80.22, feature="marshland / storm drains",
         scenario="Urban waterlogging", kind="urban", months=(11, 12)),
    dict(state="Tamil Nadu", region="South", place="Thanjavur (Cauvery delta)", lat=10.79, lon=79.14, feature="Cauvery distributaries",
         scenario="Agricultural flooding", kind="agricultural", months=(11, 12), river_m=(50, 120), parcel_m=(35, 70), villages=(2, 4)),
    dict(state="Andhra Pradesh", region="South", place="Vijayawada (Krishna)", lat=16.51, lon=80.62, feature="Krishna",
         scenario="Monsoon river flooding", kind="meandering", months=(8, 9), river_m=(600, 1200), parcel_m=(40, 80), villages=(3, 5)),
    dict(state="Andhra Pradesh", region="South", place="Konaseema (Godavari delta)", lat=16.58, lon=82.00, feature="Godavari distributaries",
         scenario="Agricultural flooding", kind="agricultural", months=(7, 8), river_m=(80, 200), parcel_m=(35, 70), villages=(2, 4)),
    dict(state="Telangana", region="South", place="Hyderabad (Musi)", lat=17.37, lon=78.48, feature="Musi / storm drains",
         scenario="Urban waterlogging", kind="urban", months=(9, 10)),
    dict(state="Telangana", region="South", place="Bhadrachalam (Godavari)", lat=17.67, lon=80.89, feature="Godavari",
         scenario="Monsoon river flooding", kind="meandering", months=(7, 8), river_m=(600, 1200), parcel_m=(50, 100), villages=(1, 3)),
    # ---- Central ----
    dict(state="Madhya Pradesh", region="Central", place="Narmadapuram (Narmada)", lat=22.75, lon=77.72, feature="Narmada",
         scenario="Monsoon river flooding", kind="meandering", months=(8, 9), river_m=(350, 700), parcel_m=(70, 140), villages=(1, 3)),
    dict(state="Chhattisgarh", region="Central", place="Janjgir-Champa (Mahanadi)", lat=21.97, lon=82.58, feature="Mahanadi",
         scenario="Agricultural flooding", kind="agricultural", months=(8, 9), river_m=(200, 450), parcel_m=(35, 70), villages=(2, 4)),
    # ---- North-East ----
    dict(state="Meghalaya", region="North-East", place="Garo Hills foothills (Tura)", lat=25.57, lon=90.22, feature="hill streams",
         scenario="Himalayan flash flood", kind="flash", months=(6, 7)),
    dict(state="Sikkim", region="North-East", place="Chungthang (Teesta)", lat=27.60, lon=88.65, feature="Teesta",
         scenario="Himalayan flash flood", kind="flash", months=(7, 10)),
    dict(state="Arunachal Pradesh", region="North-East", place="Pasighat (Siang)", lat=28.07, lon=95.33, feature="Siang",
         scenario="Brahmaputra flooding", kind="braided", months=(6, 7, 8), parcel_m=(35, 70), villages=(1, 2)),
    dict(state="Nagaland", region="North-East", place="Dimapur (Dhansiri)", lat=25.91, lon=93.73, feature="Dhansiri",
         scenario="Monsoon river flooding", kind="meandering", months=(7, 8), river_m=(80, 180), parcel_m=(30, 60), villages=(2, 4)),
]

# lightning `kind` (the simulator's storm recipe):
#   isolated   scattered afternoon cells           squall     a line of cells moving together
#   mcs        one large long-lived cluster        orographic cells anchored on the hills
#   coastal    cells forming along a coast          wd         western-disturbance cloud band, few embedded cells
# `cape`: relative convective vigour 0-1 (drives how cold the tops get and how much lightning forms)
LIGHTNING_SITES = [
    # ---- Pre-monsoon thunderstorms, East and North-East (Kalbaishakhi / Nor'westers) ----
    dict(state="West Bengal", region="East", place="Kolkata", lat=22.57, lon=88.36, scenario="Pre-monsoon thunderstorm (Kalbaishakhi)", kind="squall", months=(4, 5), cape=0.85),
    dict(state="Odisha", region="East", place="Bhubaneswar", lat=20.30, lon=85.82, scenario="Pre-monsoon thunderstorm (Kalbaishakhi)", kind="squall", months=(4, 5), cape=0.8),
    dict(state="Assam", region="East", place="Guwahati", lat=26.14, lon=91.74, scenario="Pre-monsoon thunderstorm (Nor'wester)", kind="squall", months=(4, 5), cape=0.8),
    dict(state="Meghalaya", region="North-East", place="Shillong", lat=25.57, lon=91.88, scenario="Pre-monsoon thunderstorm (Nor'wester)", kind="orographic", months=(4, 5), cape=0.75),
    dict(state="Nagaland", region="North-East", place="Kohima", lat=25.67, lon=94.11, scenario="Pre-monsoon thunderstorm (Nor'wester)", kind="orographic", months=(4, 5), cape=0.7),
    # ---- Monsoon thunderstorms, Uttar Pradesh and Bihar ----
    dict(state="Uttar Pradesh", region="North", place="Lucknow", lat=26.85, lon=80.95, scenario="Monsoon thunderstorm", kind="isolated", months=(6, 7, 8, 9), cape=0.7),
    dict(state="Uttar Pradesh", region="North", place="Prayagraj", lat=25.43, lon=81.85, scenario="Monsoon thunderstorm", kind="mcs", months=(6, 7, 8, 9), cape=0.7),
    dict(state="Bihar", region="East", place="Patna", lat=25.61, lon=85.14, scenario="Monsoon thunderstorm", kind="isolated", months=(6, 7, 8, 9), cape=0.75),
    dict(state="Bihar", region="East", place="Gaya", lat=24.79, lon=85.00, scenario="Monsoon thunderstorm", kind="mcs", months=(6, 7, 9), cape=0.75),
    # ---- Severe convective storms, West Bengal and Assam ----
    dict(state="West Bengal", region="East", place="Siliguri", lat=26.73, lon=88.40, scenario="Severe convective storm", kind="mcs", months=(4, 5, 6), cape=1.0),
    dict(state="Assam", region="East", place="Dibrugarh", lat=27.48, lon=94.91, scenario="Severe convective storm", kind="mcs", months=(4, 5, 6), cape=0.95),
    # ---- Maharashtra and Madhya Pradesh ----
    dict(state="Maharashtra", region="West", place="Nagpur (Vidarbha)", lat=21.15, lon=79.09, scenario="Thunderstorm over central India", kind="isolated", months=(6, 9, 10), cape=0.75),
    dict(state="Maharashtra", region="West", place="Pune", lat=18.52, lon=73.86, scenario="Thunderstorm over central India", kind="isolated", months=(5, 6, 10), cape=0.7),
    dict(state="Madhya Pradesh", region="Central", place="Bhopal", lat=23.26, lon=77.41, scenario="Thunderstorm over central India", kind="isolated", months=(6, 7, 9), cape=0.75),
    dict(state="Madhya Pradesh", region="Central", place="Jabalpur", lat=23.18, lon=79.99, scenario="Thunderstorm over central India", kind="mcs", months=(6, 7, 9), cape=0.75),
    dict(state="Chhattisgarh", region="Central", place="Raipur", lat=21.25, lon=81.63, scenario="Thunderstorm over central India", kind="isolated", months=(5, 6, 9), cape=0.8),
    # ---- Coastal thunderstorms: Bay of Bengal ----
    dict(state="Odisha", region="East", place="Puri coast", lat=19.81, lon=85.83, scenario="Coastal thunderstorm (Bay of Bengal)", kind="coastal", months=(5, 10), cape=0.75, coast="east"),
    dict(state="Andhra Pradesh", region="South", place="Visakhapatnam", lat=17.69, lon=83.22, scenario="Coastal thunderstorm (Bay of Bengal)", kind="coastal", months=(5, 9, 10), cape=0.75, coast="east"),
    dict(state="Tamil Nadu", region="South", place="Chennai", lat=13.08, lon=80.27, scenario="Coastal thunderstorm (Bay of Bengal)", kind="coastal", months=(10, 11), cape=0.7, coast="east"),
    # ---- Coastal thunderstorms: Arabian Sea ----
    dict(state="Kerala", region="South", place="Kochi", lat=9.93, lon=76.27, scenario="Coastal thunderstorm (Arabian Sea)", kind="coastal", months=(4, 5, 10), cape=0.8, coast="west"),
    dict(state="Karnataka", region="South", place="Mangaluru", lat=12.91, lon=74.86, scenario="Coastal thunderstorm (Arabian Sea)", kind="coastal", months=(5, 10), cape=0.75, coast="west"),
    dict(state="Gujarat", region="West", place="Surat", lat=21.17, lon=72.83, scenario="Coastal thunderstorm (Arabian Sea)", kind="coastal", months=(6, 9), cape=0.65, coast="west"),
    dict(state="Maharashtra", region="West", place="Mumbai", lat=19.08, lon=72.88, scenario="Coastal thunderstorm (Arabian Sea)", kind="coastal", months=(6, 9, 10), cape=0.7, coast="west"),
    # ---- Inland south ----
    dict(state="Karnataka", region="South", place="Bengaluru", lat=12.97, lon=77.59, scenario="Pre-monsoon thunderstorm (inland south)", kind="isolated", months=(4, 5), cape=0.75),
    dict(state="Telangana", region="South", place="Hyderabad", lat=17.38, lon=78.49, scenario="Pre-monsoon thunderstorm (inland south)", kind="isolated", months=(4, 5, 10), cape=0.75),
    dict(state="Rajasthan", region="West", place="Jaipur", lat=26.91, lon=75.79, scenario="Pre-monsoon dust-thunderstorm", kind="squall", months=(5, 6), cape=0.6),
    # ---- Himalayan thunderstorms ----
    dict(state="Uttarakhand", region="North", place="Dehradun", lat=30.32, lon=78.03, scenario="Himalayan thunderstorm", kind="orographic", months=(4, 5, 6), cape=0.7),
    dict(state="Himachal Pradesh", region="North", place="Shimla", lat=31.10, lon=77.17, scenario="Himalayan thunderstorm", kind="orographic", months=(4, 5, 6), cape=0.65),
    dict(state="Sikkim", region="North-East", place="Gangtok", lat=27.33, lon=88.61, scenario="Himalayan thunderstorm", kind="orographic", months=(4, 5), cape=0.7),
    dict(state="Arunachal Pradesh", region="North-East", place="Itanagar", lat=27.08, lon=93.61, scenario="Himalayan thunderstorm", kind="orographic", months=(4, 5, 6), cape=0.75),
    # ---- Western disturbances, northern India (winter) ----
    dict(state="Punjab", region="North", place="Amritsar", lat=31.63, lon=74.87, scenario="Western disturbance", kind="wd", months=(12, 1, 2, 3), cape=0.3),
    dict(state="Delhi", region="North", place="New Delhi", lat=28.61, lon=77.21, scenario="Western disturbance", kind="wd", months=(12, 1, 2, 3), cape=0.3),
    dict(state="Uttar Pradesh", region="North", place="Meerut", lat=28.98, lon=77.71, scenario="Western disturbance", kind="wd", months=(1, 2, 3), cape=0.3),
    dict(state="Himachal Pradesh", region="North", place="Dharamshala", lat=32.22, lon=76.32, scenario="Western disturbance", kind="wd", months=(1, 2, 3), cape=0.35),
]

# Lightning-free weather, generated at the same sites as negatives.
LIGHTNING_NEGATIVE_KINDS = ("clear", "stratiform", "cirrus", "fog")
