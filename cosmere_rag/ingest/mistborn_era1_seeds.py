"""Hand-curated Mistborn Era 1 article titles known to exist in the
Malthemester/CoppermindScraper mirror.

These are unioned with infobox-`Featured In`-derived matches to form the
Era 1 corpus. Every name here matches a file under
`data/coppermind-mirror/Cosmere/<title>.md`. The CLI warns if any seed
title is missing from the corpus, so this list is self-checking on each run.

Coppermind redirects mean some characters appear under their real names:
Ham -> Hammond, The Lord Ruler -> Rashek. Known gaps in this snapshot of
the mirror (no file present, even via redirect): Breeze, Clubs, Spook
(Lestibournes), Mistborn: Secret History, the standalone "The Final Empire"
book article.
"""

SEED_TITLES: list[str] = [
    # Books and series
    "Mistborn (series)",
    "The Well of Ascension",
    "The Hero of Ages",
    # Crew
    "Vin",
    "Kelsier",
    "Sazed",
    "Marsh",
    "Elend Venture",
    "Hammond",
    "Dockson",
    "Yeden",
    "Cladent",
    "Demoux",
    # Antagonists
    "Rashek",
    "Steel Inquisitor",
    "Steel Ministry",
    "Straff Venture",
    "Ashweather Cett",
    "Ruin",
    # Kandra / koloss
    "TenSoon",
    "OreSeur",
    "Kandra",
    "Koloss",
    # Magic systems and metals
    "Allomancy",
    "Feruchemy",
    "Hemalurgy",
    "Atium",
    "Lerasium",
    "Mistcloak",
    # Locations
    "Luthadel",
    "Pits of Hathsin",
    "Final Empire",
    "Scadrial",
    # Cosmere context
    "Preservation",
    "Sliver",
    "Cognitive Shadow",
    # Peoples
    "Skaa",
    "Terris",
]
