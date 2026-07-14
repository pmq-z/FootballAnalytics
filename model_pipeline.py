"""Pipeline hibrido: K-Means (perfiles de partido) -> inyeccion de feature -> ID3 (ganador)."""
import io
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import data_source as qb

MONSTER_METRICS = ["goles_totales", "xg_total", "desbalance_posesion",
                   "tiros_totales", "precision_tiro", "diferencia_xg"]
ID3_FEATURES = ["felyne_cluster", "elo_diff", "forma_local", "forma_visita",
                "h2h_dominio", "gf_local", "gc_visita"]
VERDICTS = ["LOCAL", "EMPATE", "VISITANTE"]

CLUSTER_LABELS = {  # se re-etiqueta dinamicamente segun centroides
    "cerrado": "PARTIDO CERRADO — pocos goles, xG bajo, juego trabado",
    "abierto": "PARTIDO ABIERTO — muchos goles y transiciones rapidas",
    "dominio": "DOMINIO UNILATERAL — un equipo monopoliza balon y tiros",
    "tactico": "DUELO TACTICO — volumen medio, definicion escasa",
}


def carve_features(rathalos_data):
    """Fase 0: deriva metricas continuas de partido, sin tocar la etiqueta de ganador."""
    df = rathalos_data.copy()
    df["goles_totales"] = df.goles_local + df.goles_visita
    df["xg_total"] = df.xg_local + df.xg_visita
    df["desbalance_posesion"] = (df.posesion_local - 50).abs()
    df["tiros_totales"] = df.tiros_local + df.tiros_visita
    df["precision_tiro"] = (df.tiros_puerta_local + df.tiros_puerta_visita) / df.tiros_totales.clip(lower=1)
    df["diferencia_xg"] = (df.xg_local - df.xg_visita).abs()
    df["ganador"] = np.select(
        [df.goles_local > df.goles_visita, df.goles_local < df.goles_visita],
        ["LOCAL", "VISITANTE"], default="EMPATE")
    return df


def felyne_clusters(df, k=4):
    """Fase 1: K-Means no supervisado sobre metricas normalizadas."""
    great_sword_scaler = StandardScaler()
    x = great_sword_scaler.fit_transform(df[MONSTER_METRICS])
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=14)
    df["felyne_cluster"] = kmeans.fit_predict(x)
    centros = pd.DataFrame(great_sword_scaler.inverse_transform(kmeans.cluster_centers_),
                           columns=MONSTER_METRICS)
    centros["nombre"] = [_name_cluster(r) for _, r in centros.iterrows()]
    centros.insert(0, "cluster", range(k))
    return df, great_sword_scaler, kmeans, centros


def _name_cluster(row):
    if row.desbalance_posesion > 9 and row.diferencia_xg > 1.0:
        return CLUSTER_LABELS["dominio"]
    if row.goles_totales >= 3.2 or row.xg_total >= 3.4:
        return CLUSTER_LABELS["abierto"]
    if row.goles_totales <= 1.8 and row.xg_total <= 2.4:
        return CLUSTER_LABELS["cerrado"]
    return CLUSTER_LABELS["tactico"]


def _team_profile(df, equipo):
    como_local = df[df.local == equipo]
    como_visita = df[df.visita == equipo]
    gf = pd.concat([como_local.goles_local, como_visita.goles_visita]).mean()
    gc = pd.concat([como_local.goles_visita, como_visita.goles_local]).mean()
    xgf = pd.concat([como_local.xg_local, como_visita.xg_visita]).mean()
    pos = pd.concat([como_local.posesion_local, como_visita.posesion_visita]).mean()
    tiros = pd.concat([como_local.tiros_local, como_visita.tiros_visita]).mean()
    dianas = pd.concat([como_local.tiros_puerta_local, como_visita.tiros_puerta_visita]).mean()
    pts = np.where(pd.concat([como_local.goles_local - como_local.goles_visita,
                              como_visita.goles_visita - como_visita.goles_local]) > 0, 3, 0)
    return dict(gf=gf, gc=gc, xgf=xgf, pos=pos, tiros=tiros, dianas=dianas,
                forma=float(np.mean(pts)) if len(pts) else 1.0,
                elo=qb.HUNTER_RANKS[equipo]["elo"])


def _h2h_dominio(df, local, visita):
    h = df[(df.origen == "H2H")]
    if h.empty:
        return 0.0
    win_l = ((h.local == local) & (h.goles_local > h.goles_visita)).sum() + \
            ((h.visita == local) & (h.goles_visita > h.goles_local)).sum()
    win_v = ((h.local == visita) & (h.goles_local > h.goles_visita)).sum() + \
            ((h.visita == visita) & (h.goles_visita > h.goles_local)).sum()
    return float((win_l - win_v) / len(h))


def inject_context(df):
    """Fase 2: contexto historico por partido, junto a la etiqueta de cluster."""
    perfiles = {t: _team_profile(df, t) for t in set(df.local) | set(df.visita)}
    df["elo_diff"] = df.elo_local - df.elo_visita
    df["forma_local"] = df.local.map(lambda t: perfiles[t]["forma"])
    df["forma_visita"] = df.visita.map(lambda t: perfiles[t]["forma"])
    df["gf_local"] = df.local.map(lambda t: perfiles[t]["gf"])
    df["gc_visita"] = df.visita.map(lambda t: perfiles[t]["gc"])
    df["h2h_dominio"] = [_h2h_dominio(df, l, v) for l, v in zip(df.local, df.visita)]
    return df, perfiles


def rathian_tree(df):
    """Fase 3: ID3 (entropia) sobre cluster + contexto -> Ganador."""
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=5,
                                  min_samples_leaf=6, random_state=14)
    tree.fit(df[ID3_FEATURES], df["ganador"])
    return tree


def _trace_decision(tree, fila):
    """Reconstruye la ruta de decision del arbol para un solo partido."""
    t, pasos = tree.tree_, []
    nodo, x = 0, fila[ID3_FEATURES].values.astype(float)
    while t.children_left[nodo] != -1:
        f, thr = ID3_FEATURES[t.feature[nodo]], t.threshold[nodo]
        val = x[t.feature[nodo]]
        izq = val <= thr
        pasos.append(f"{f} = {val:.2f} {'<=' if izq else '>'} {thr:.2f}")
        nodo = t.children_left[nodo] if izq else t.children_right[nodo]
    dist = t.value[nodo][0]
    pasos.append(f"HOJA -> {tree.classes_[int(np.argmax(dist))]} "
                 f"(pureza {dist.max() / dist.sum():.0%}, n={int(t.n_node_samples[nodo])})")
    return pasos


def _scoreline(lam_l, lam_v, veredicto, maxg=6):
    m = np.outer(poisson.pmf(range(maxg + 1), lam_l), poisson.pmf(range(maxg + 1), lam_v))
    mask = np.zeros_like(m, dtype=bool)
    for i in range(maxg + 1):
        for j in range(maxg + 1):
            r = "LOCAL" if i > j else ("VISITANTE" if j > i else "EMPATE")
            mask[i, j] = (r == veredicto)
    m = np.where(mask, m, 0)
    i, j = np.unravel_index(np.argmax(m), m.shape)
    top = sorted([((a, b), m[a, b]) for a in range(maxg + 1) for b in range(maxg + 1) if m[a, b] > 0],
                 key=lambda z: -z[1])[:3]
    return int(i), int(j), [dict(marcador=f"{a}-{b}", prob=round(float(p) * 100, 1)) for (a, b), p in top]


def _likely_slayers(equipo, goles_esperados, cupo=3):
    plantilla = qb.palico_scouts(equipo)
    if plantilla.empty or goles_esperados < 1:
        return []
    peso = plantilla.goles_p90 * (plantilla.minutos / plantilla.minutos.max()) * (plantilla.forma / 10)
    plantilla = plantilla.assign(peso=peso).sort_values("peso", ascending=False)
    total = plantilla.peso.sum() or 1
    out = []
    for _, r in plantilla.head(cupo).iterrows():
        p = 1 - np.exp(-(r.peso / total) * goles_esperados)
        out.append(dict(jugador=r.jugador, equipo=equipo, posicion=r.posicion,
                        prob=round(float(p) * 100, 1)))
    return out


def execute_prediction(equipo_local, equipo_visita, k=4):
    """Orquesta el pipeline completo y devuelve la prediccion."""
    rathalos_data = qb.gathering_hall_dataset(equipo_local, equipo_visita)
    df = carve_features(rathalos_data)
    df, great_sword_scaler, kmeans, centros = felyne_clusters(df, k=k)
    df, perfiles = inject_context(df)
    tree = rathian_tree(df)

    pl, pv = perfiles[equipo_local], perfiles[equipo_visita]
    lam_l = max(0.25, (pl["gf"] + pv["gc"]) / 2 * 1.06)   # ventaja de localia
    lam_v = max(0.25, (pv["gf"] + pl["gc"]) / 2 * 0.96)
    tiros_tot = pl["tiros"] + pv["tiros"]
    fixture = pd.Series({
        "goles_totales": lam_l + lam_v,
        "xg_total": lam_l + lam_v,
        "desbalance_posesion": abs(pl["pos"] - pv["pos"]) / 2,
        "tiros_totales": tiros_tot,
        "precision_tiro": (pl["dianas"] + pv["dianas"]) / max(1, tiros_tot),
        "diferencia_xg": abs(lam_l - lam_v),
    })
    fixture_x = pd.DataFrame([fixture[MONSTER_METRICS]], columns=MONSTER_METRICS)
    cluster = int(kmeans.predict(great_sword_scaler.transform(fixture_x))[0])
    fixture["felyne_cluster"] = cluster
    fixture["elo_diff"] = pl["elo"] - pv["elo"]
    fixture["forma_local"], fixture["forma_visita"] = pl["forma"], pv["forma"]
    fixture["h2h_dominio"] = _h2h_dominio(df, equipo_local, equipo_visita)
    fixture["gf_local"], fixture["gc_visita"] = pl["gf"], pv["gc"]

    probs = tree.predict_proba(pd.DataFrame([fixture[ID3_FEATURES]]))[0]
    veredicto = tree.classes_[int(np.argmax(probs))]
    gl, gv, alternativos = _scoreline(lam_l, lam_v, veredicto)
    ganador = {"LOCAL": equipo_local, "VISITANTE": equipo_visita, "EMPATE": "EMPATE"}[veredicto]

    anotadores = _likely_slayers(equipo_local, max(gl, lam_l)) + _likely_slayers(equipo_visita, max(gv, lam_v))
    anotadores = sorted(anotadores, key=lambda a: -a["prob"])[:5]

    tabla = df[["fecha", "local", "visita", "goles_local", "goles_visita", "xg_local",
                "xg_visita", "posesion_local", "tiros_local", "tiros_visita",
                "felyne_cluster", "ganador", "origen"]].sort_values("fecha", ascending=False)

    return dict(
        local=equipo_local, visita=equipo_visita, ganador=ganador, veredicto=veredicto,
        marcador=f"{gl}-{gv}", goles_local=gl, goles_visita=gv,
        probabilidades={c: round(float(p) * 100, 1) for c, p in zip(tree.classes_, probs)},
        marcadores_alternativos=alternativos, anotadores=anotadores,
        cluster_activo=cluster,
        cluster_nombre=centros.loc[centros.cluster == cluster, "nombre"].iloc[0],
        centroides=centros.round(2).to_dict("records"),
        ruta_id3=_trace_decision(tree, fixture),
        reglas_id3=export_text(tree, feature_names=ID3_FEATURES, max_depth=3),
        xg_estimado=dict(local=round(lam_l, 2), visita=round(lam_v, 2)),
        muestras=int(len(df)),
        tabla=tabla.head(40).round(2).to_dict("records"),
        _dataset=df,
    )


def export_dataset(equipo_local, equipo_visita):
    """Exporta datasets limpios + clusters a .xlsx en memoria."""
    caza = execute_prediction(equipo_local, equipo_visita)
    df = caza.pop("_dataset")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as w:
        df.round(3).rename(columns={"felyne_cluster": "cluster"}) \
            .to_excel(w, sheet_name="PARTIDOS_CLUSTERS", index=False)
        pd.DataFrame(caza["centroides"]).to_excel(w, sheet_name="CENTROIDES_KMEANS", index=False)
        pd.concat([qb.palico_scouts(equipo_local), qb.palico_scouts(equipo_visita)]) \
            .to_excel(w, sheet_name="PLANTILLAS", index=False)
        pd.DataFrame([dict(local=equipo_local, visita=equipo_visita, ganador=caza["ganador"],
                           marcador=caza["marcador"], cluster=caza["cluster_activo"],
                           perfil=caza["cluster_nombre"], **caza["probabilidades"])]) \
            .to_excel(w, sheet_name="PREDICCION", index=False)
    buffer.seek(0)
    return buffer
