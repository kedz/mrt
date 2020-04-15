from nltk import word_tokenize


LIST_SLOTS = ['genres', 'player_perspective', 'platforms']

CATEGORICAL_SLOTS = [
    "name", "developer", "rating", "esrb", "has_linux_release", 
    "has_mac_release", "available_on_steam", "release_year",
    "exp_release_date", 'has_multiplayer',
]

LEXICON = {
    "name": [
        'Age of Empires II: The Age of Kings',
        'God of War',
        'Stronghold 2',
        'Rollcage',
        "Alan Wake's American Nightmare",
        'Never Alone',
        'Superhot',
        'Mass Effect 2',
        'The Last of Us',
        'The Witcher 3: Wild Hunt',
        'Portal 2',
        'Warcraft III: Reign of Chaos',
        'Tomb Raider: The Angel of Darkness',
        'The Wolf Among Us',
        'Ancient Cities',
        'Shadow of the Tomb Raider',
        "Assassin's Creed II",
        'RollerCoaster Tycoon',
        'Need for Speed: Most Wanted',
        'StarCraft',
        "Sid Meier's Civilization V",
        "Tom Clancy's Splinter Cell: Chaos Theory",
        'Grand Theft Auto V',
        "Tom Clancy's The Division",
        "Assassin's Creed Chronicles: India",
        'Football Manager 2015',
        'Call of Duty: Advanced Warfare',
        'Anthem',
        'Metro 2033',
        'Need for Speed: Shift',
        'Transport Tycoon',
        'The Vanishing of Ethan Carter',
        'World of Warcraft',
        'Undertale',
        'Rise of the Tomb Raider',
        'Diablo II',
        'Worms: Reloaded',
        "Uncharted 4: A Thief's End",
        'Battlefield V',
        "Mirror's Edge Catalyst",
        'F1 2014',
        "Hellblade: Senua's Sacrifice",
        'Rocket League',
        'Small World 2',
        'Tomb Raider: The Last Revelation',
        'Might & Magic: Heroes VI',
        'Driver',
        'NBA 2K19',
        'The Sims',
        'Heroes of Might and Magic III: The Restoration of Erathia',
        'Spider-Man',
        'Tetris',
        'Final Fantasy VII',
        'Lara Croft and the Temple of Osiris',
        'Little Nightmares',
        'The Forest of Doom',
        'Dirt: Showdown',
        'Super Bomberman',
        'A Way Out',
        'Need for Speed: The Run',
        'Metal Gear Solid 3: Snake Eater',
        'The Legend of Zelda: Ocarina of Time',
        'Metroid Prime Pinball',
        'MotorStorm: Apocalypse',
        'Horizon: Zero Dawn',
        'Assetto Corsa',
        'Payday 2',
        'Mafia',
        'Half-Life 2',
        'Skyforge',
        'Metro Exodus',
        'Bus Driver',
        'Commandos: Behind Enemy Lines',
        'Hitman 2',
        'Max Payne',
        'SpellForce 3',
        "Tony Hawk's Pro Skater 3",
        'Dance Dance Revolution Universe 3',
        'World of Warcraft: Battle for Azeroth',
        'Life is Strange',
        'Need for Speed: Payback',
        'Guitar Hero: Smash Hits',
        'Crysis',
        'The Elder Scrolls V: Skyrim',
        'Ori and the Blind Forest',
        'Far Cry 3',
        'BioShock',
        'Euro Truck Simulator',
        'Silent Hill 2',
        'Madden NFL 15',
        "Age of Wonders II: The Wizard's Throne",
        'Trivial Pursuit',
        'Layers of Fear',
        'The Room',
        'Outlast II',
        'FIFA 12',
        'Little Big Adventure',
        'NBA 2K16',
        'The Crew 2',
        'Nightshade',
        'Resident Evil 4',
        'Quantum Break',
        'The Elder Scrolls Online',
        'NHL 15',
        'Super Mario World',
        'TrackMania Turbo',
    ],
    "developer": [
        "Ivory Tower",
        "Adeline Software International",
        "EA Canada",
        "CD Projekt RED",
        "Fuse Games",
        "Ghost Games",
        "Upper One Games",
        "4A Games",
        "Beenox",
        "Blizzard Entertainment",
        "Bethesda Game Studios",
        "Ensemble Studios",
        "Blizzard North",
        "Neversoft Entertainment",
        "Visual Concepts",
        "Naughty Dog",
        "Nadeo",
        "Core Design",
        "EA Digital Illusions CE",
        "Crytek Frankfurt",
        "Spectrum HoloByte, Inc.",
        "SCS Software",
        "Black Hole Entertainment",
        "Dontnod Entertainment",
        "Nintendo EAD",
        "Konami Computer Entertainment Japan",
        "Konami Computer Entertainment Tokyo",
        "Red Entertainment Corporation",
        "Rockstar North",
        "Firaxis Games",
        "Allods Online",
        "Chris Sawyer",
        "Capcom Production Studio 4",
        "Kunos Simulazioni",
        "SIE Santa Monica Studio",
        "Konami",
        "Uncasual Games",
        "Loki Software",
        "Firebrand Games",
        "Massive Entertainment",
        "Telltale Games",
        "Electronic Arts",
        "Overkill Software",
        "Sports Interactive",
        "Firefly Studios",
        "Eidos Montréal",
        "Spectrum HoloByte, Inc.",
        "Sledgehammer Games",
        "Fireproof Games",
        "Days of Wonder",
        "Codemasters Birmingham",
        "Ninja Theory",
        "Crystal Dynamics",
        "Team17 Digital Ltd",
        "Codemasters Southam",
        "Tarsier Studios",
        "Psyonix",
        "Hazelight Studios",
        "Hudson Soft",
        "EA Redwood Shores",
        "Valve Corporation",
        "Slightly Mad Studios",
        "Ubisoft Montreal",
        "Tin Man Games",
        "Illusion Softworks",
        "Reflections Interactive",
        "tobyfox",
        "Moon Studios",
        "MicroProse",
        "Red Barrels",
        "SUPERHOT Team",
        "Evolution Studios",
        "Pyro Studios",
        "Square",
        "Climax Studios",
        "ZeniMax Online Studios",
        "2K Boston",
        "The Astronauts",
        "Guerrilla Games",
        "Triumph Studios",
        "Grimlore Games",
        "Bloober Team",
        "EA Tiburon",
        "Remedy Entertainment",
        "IO Interactive",
        "BioWare",
        "Insomniac Games",
        "Attention To Detail",
        "Maxis",
        "Ubisoft Massive",

    ],
    'platforms': ['Xbox', 'PlayStation', 'PC', 'Nintendo', 'Nintendo Switch'],
    'other': ['Mac', 'Linux', 'Steam', 'Windows'],

}

DEVELOPER_TOKENS = [
    [dev, word_tokenize(dev.lower())]
    for dev in LEXICON['developer']
]

NAME_TOKENS = [
    [name, word_tokenize(name.lower())]
    for name in LEXICON['name']
]

SRT_NAME_TOKENS = sorted(NAME_TOKENS, key=lambda x: len(x[1]), reverse=True)

MONTHS = "|".join(
    ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 
     'september', 'october', 'november', 'december'])


