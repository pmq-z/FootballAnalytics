"""Servidor Flask del predictor hibrido K-Means + ID3."""
from datetime import datetime
from flask import Flask, jsonify, render_template, request, send_file

import data_source as qb
import model_pipeline as mp

gathering_hall = Flask(__name__)
MATCH_DATE = "2026-07-14"


@gathering_hall.route("/")
def village():
    return render_template("index.html", roster=qb.scoutfly_roster(), fecha=MATCH_DATE)


@gathering_hall.route("/api/teams")
def roster():
    return jsonify(qb.scoutfly_roster())


@gathering_hall.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    local, visita = body.get("local", "ESPAÑA"), body.get("visita", "FRANCIA")
    if local == visita:
        return jsonify(error="Un equipo no puede jugar contra si mismo."), 400
    if local not in qb.HUNTER_RANKS or visita not in qb.HUNTER_RANKS:
        return jsonify(error="Equipo no registrado."), 404
    caza = mp.execute_prediction(local, visita, k=int(body.get("k", 4)))
    caza.pop("_dataset")
    caza["fecha"] = MATCH_DATE
    caza["timestamp"] = datetime.now().isoformat(timespec="seconds")
    return jsonify(caza)


@gathering_hall.route("/api/export")
def export():
    local = request.args.get("local", "ESPAÑA")
    visita = request.args.get("visita", "FRANCIA")
    buffer = mp.export_dataset(local, visita)
    nombre = f"dataset_{local}_vs_{visita}_{MATCH_DATE}.xlsx".replace(" ", "_")
    return send_file(buffer, as_attachment=True, download_name=nombre,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    gathering_hall.run(debug=True, port=5000)
