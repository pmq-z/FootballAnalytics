"""Conector de datos. Mocks deterministas con la misma forma que API-Football / FBref."""
import hashlib
import numpy as np
import pandas as pd

QUEST_BOARD_ENDPOINT = "https://v3.football.api-sports.io"  # listo para produccion
HUNTER_RANKS = {  # perfil base por seleccion: ataque, defensa, xg, posesion, tiros, elo
    "ESPAÑA":     dict(atk=2.05, dfs=0.85, pos=63.0, sht=15.8, elo=2065),
    "FRANCIA":    dict(atk=2.00, dfs=0.90, pos=54.0, sht=14.6, elo=2058),
    "INGLATERRA": dict(atk=1.85, dfs=0.95, pos=57.0, sht=14.0, elo=2010),
    "ARGENTINA":  dict(atk=1.90, dfs=0.80, pos=55.0, sht=13.5, elo=2040),
    "BRASIL":     dict(atk=1.95, dfs=0.95, pos=58.0, sht=15.0, elo=2015),
    "ALEMANIA":   dict(atk=1.88, dfs=1.10, pos=59.0, sht=15.2, elo=1975),
    "PORTUGAL":   dict(atk=1.92, dfs=1.05, pos=57.5, sht=14.8, elo=1990),
    "ITALIA":     dict(atk=1.55, dfs=0.75, pos=53.0, sht=12.9, elo=1955),
    "PAISES BAJOS": dict(atk=1.70, dfs=1.00, pos=56.0, sht=13.2, elo=1950),
    "MEXICO":     dict(atk=1.35, dfs=1.30, pos=52.0, sht=11.5, elo=1810),
}

SQUAD_SCROLLS = {  # plantilla: nombre, posicion, minutos temporada, goles recientes
    "ESPAÑA": [("Lamine Yamal", "EXT", 2450, 11), ("Nico Williams", "EXT", 2100, 7),
               ("Álvaro Morata", "DEL", 1780, 6), ("Mikel Oyarzabal", "DEL", 1520, 8),
               ("Pedri", "MC", 2600, 4), ("Fabián Ruiz", "MC", 2200, 3),
               ("Dani Olmo", "MP", 1900, 6), ("Rodri", "MCD", 2700, 2)],
    "FRANCIA": [("Kylian Mbappé", "DEL", 2800, 14), ("Ousmane Dembélé", "EXT", 2300, 9),
                ("Michael Olise", "EXT", 2150, 7), ("Randal Kolo Muani", "DEL", 1400, 4),
                ("Aurélien Tchouaméni", "MCD", 2600, 1), ("Eduardo Camavinga", "MC", 2050, 2),
                ("Warren Zaïre-Emery", "MC", 1900, 3), ("William Saliba", "DFC", 2900, 1)],
    "INGLATERRA": [("Harry Kane", "DEL", 2750, 15), ("Bukayo Saka", "EXT", 2400, 9),
                   ("Phil Foden", "MP", 2300, 8), ("Jude Bellingham", "MP", 2500, 7),
                   ("Cole Palmer", "MP", 2200, 10), ("Declan Rice", "MCD", 2800, 3)],
    "ARGENTINA": [("Lionel Messi", "MP", 1900, 10), ("Julián Álvarez", "DEL", 2400, 11),
                  ("Lautaro Martínez", "DEL", 2100, 9), ("Enzo Fernández", "MC", 2600, 4),
                  ("Alexis Mac Allister", "MC", 2500, 5)],
    "BRASIL": [("Vinícius Jr.", "EXT", 2600, 12), ("Rodrygo", "EXT", 2200, 7),
               ("Raphinha", "EXT", 2400, 10), ("Estêvão", "EXT", 1600, 5),
               ("Bruno Guimarães", "MC", 2700, 3)],
    "ALEMANIA": [("Florian Wirtz", "MP", 2500, 9), ("Jamal Musiala", "MP", 2300, 10),
                 ("Kai Havertz", "DEL", 2000, 6), ("Niclas Füllkrug", "DEL", 1300, 5)],
    "PORTUGAL": [("Cristiano Ronaldo", "DEL", 1800, 9), ("Rafael Leão", "EXT", 2200, 7),
                 ("Bruno Fernandes", "MP", 2700, 8), ("Bernardo Silva", "MP", 2400, 4)],
    "ITALIA": [("Mateo Retegui", "DEL", 2100, 9), ("Federico Chiesa", "EXT", 1700, 4),
               ("Nicolò Barella", "MC", 2600, 3)],
    "PAISES BAJOS": [("Cody Gakpo", "EXT", 2200, 8), ("Memphis Depay", "DEL", 1600, 5),
                     ("Xavi Simons", "MP", 2300, 6), ("Frenkie de Jong", "MC", 2500, 2)],
    "MEXICO": [("Santiago Giménez", "DEL", 2100, 8), ("Hirving Lozano", "EXT", 1800, 4),
               ("Edson Álvarez", "MCD", 2600, 1)],
}


def _monster_seed(*keys):
    raw = "|".join(str(k) for k in keys).encode()
    return int(hashlib.md5(raw).hexdigest()[:8], 16) % (2**31)


def scoutfly_roster():
    return sorted(HUNTER_RANKS.keys())


def _forge_match(rng, local, visita):
    """Genera un registro de partido coherente con los perfiles de ambos equipos."""
    a, b = HUNTER_RANKS[local], HUNTER_RANKS[visita]
    xg_l = max(0.15, rng.normal(a["atk"] * (b["dfs"] / 1.0) * 1.08, 0.42))
    xg_v = max(0.15, rng.normal(b["atk"] * (a["dfs"] / 1.0) * 0.92, 0.42))
    gl, gv = int(rng.poisson(xg_l)), int(rng.poisson(xg_v))
    pos_l = float(np.clip(rng.normal(50 + (a["pos"] - b["pos"]) * 0.6, 4.5), 28, 74))
    sht_l = max(3, int(rng.normal(a["sht"] * (0.6 + xg_l / 3), 2.6)))
    sht_v = max(3, int(rng.normal(b["sht"] * (0.6 + xg_v / 3), 2.6)))
    return dict(
        local=local, visita=visita, goles_local=gl, goles_visita=gv,
        xg_local=round(xg_l, 2), xg_visita=round(xg_v, 2),
        posesion_local=round(pos_l, 1), posesion_visita=round(100 - pos_l, 1),
        tiros_local=sht_l, tiros_visita=sht_v,
        tiros_puerta_local=max(1, int(sht_l * rng.uniform(0.30, 0.48))),
        tiros_puerta_visita=max(1, int(sht_v * rng.uniform(0.30, 0.48))),
        corners_local=max(0, int(rng.normal(5.5, 2))),
        corners_visita=max(0, int(rng.normal(4.8, 2))),
        faltas_local=int(rng.normal(11, 3)), faltas_visita=int(rng.normal(12, 3)),
        elo_local=HUNTER_RANKS[local]["elo"], elo_visita=HUNTER_RANKS[visita]["elo"],
    )


def hunt_log(equipo, n=12):
    """Ultimos n partidos individuales del equipo."""
    rng = np.random.default_rng(_monster_seed("log", equipo))
    rivales = [r for r in HUNTER_RANKS if r != equipo]
    filas = []
    for i in range(n):
        rival = rivales[rng.integers(0, len(rivales))]
        en_casa = bool(rng.integers(0, 2))
        local, visita = (equipo, rival) if en_casa else (rival, equipo)
        m = _forge_match(rng, local, visita)
        m["fecha"] = str(pd.Timestamp("2026-07-01") - pd.Timedelta(days=9 * (i + 1)))[:10]
        m["origen"] = f"FORM_{equipo}"
        filas.append(m)
    return pd.DataFrame(filas)


def zinogre_h2h(equipo_a, equipo_b, n=10):
    """Historial de enfrentamientos directos."""
    rng = np.random.default_rng(_monster_seed("h2h", *sorted([equipo_a, equipo_b])))
    filas = []
    for i in range(n):
        local, visita = (equipo_a, equipo_b) if i % 2 == 0 else (equipo_b, equipo_a)
        m = _forge_match(rng, local, visita)
        m["fecha"] = str(pd.Timestamp("2026-06-01") - pd.Timedelta(days=210 * (i + 1)))[:10]
        m["origen"] = "H2H"
        filas.append(m)
    return pd.DataFrame(filas)


def palico_scouts(equipo):
    """Rendimiento de jugadores: minutos y goles recientes."""
    rng = np.random.default_rng(_monster_seed("squad", equipo))
    filas = []
    for nombre, pos, mins, goles in SQUAD_SCROLLS.get(equipo, []):
        p90 = round(goles / max(1, mins / 90), 3)
        filas.append(dict(equipo=equipo, jugador=nombre, posicion=pos, minutos=mins,
                          goles_recientes=goles, goles_p90=p90,
                          forma=round(float(np.clip(rng.normal(7.0, 0.7), 5.0, 9.5)), 1)))
    return pd.DataFrame(filas)


def gathering_hall_dataset(equipo_a, equipo_b):
    """Pool de entrenamiento: forma de ambos + H2H + ruido de contexto del resto de la liga."""
    piezas = [hunt_log(equipo_a), hunt_log(equipo_b), zinogre_h2h(equipo_a, equipo_b)]
    for otro in scoutfly_roster():
        if otro not in (equipo_a, equipo_b):
            piezas.append(hunt_log(otro, n=8))
    rathalos_data = pd.concat(piezas, ignore_index=True)
    return rathalos_data.drop_duplicates().reset_index(drop=True)
