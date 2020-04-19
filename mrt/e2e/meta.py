import re


LEXICON = {
    'name': [
        'Alimentum',
        'Aromi',
        'Bibimbap House',
        'Blue Spice',
        'Browns Cambridge',
        'Clowns',
        'Cocum',
        'Cotto',
        'Fitzbillies',
        'Giraffe',
        'Green Man',
        'Loch Fyne',
        'Midsummer House',
        'Strada',
        'Taste of Cambridge',
        'The Cambridge Blue',
        'The Cricketers',
        'The Dumpling Tree',
        'The Eagle',
        'The Golden Curry',
        'The Golden Palace',
        'The Mill',
        'The Olive Grove',
        'The Phoenix',
        'The Plough',
        'The Punter',
        'The Rice Boat',
        'The Twenty Two',
        'The Vaults',
        'The Waterman',
        'The Wrestlers',
        'Travellers Rest Beefeater',
        'Wildwood',
        'Zizzi',
    ],
    'near': [
        'All Bar One',
        'Avalon',
        'Burger King',
        'Café Adriatic',
        'Café Brazil',
        'Café Rouge',
        'Café Sicilia',
        'Clare Hall',
        'Crowne Plaza Hotel',
        'Express by Holiday Inn',
        'Rainbow Vegetarian Café',
        'Raja Indian Cuisine',
        'Ranch',
        'The Bakers',
        'The Portland Arms',
        'The Rice Boat',
        'The Six Bells',
        'The Sorrento',
        'Yippee Noodle Bar',
    ],
    'food': [
        'Chinese',
        'English',
        'Fast food',
        'French',
        'Indian',
        'Italian',
        'Japanese',
    ], 
}

NAME_RE = "|".join([re.escape(name) for name in LEXICON['name']])
NEAR_RE = "|".join([re.escape(near) for near in LEXICON['near']])
for name in LEXICON['name']:
    if "The " in name:
        NAME_RE = NAME_RE + "|" + re.escape(name.replace("The ", ""))
for near in LEXICON['near']:
    if "The " in near:
        NEAR_RE = NEAR_RE + "|" + re.escape(near.replace("The ", ""))

NAMES_TOKENIZED = [
    [name, name.replace("The ", "").lower().split()]
    for name in LEXICON["name"]
]

NEARS_TOKENIZED = [
    [near, near.replace("The ", "").lower().split()]
    for near in LEXICON["near"]
]
