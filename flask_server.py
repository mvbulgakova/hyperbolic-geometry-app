import json
import os

import numpy as np
from flask import Flask, Response, render_template, request

_base = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_base, "templates"),
    static_folder=os.path.join(_base, "static"),
)


class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


def _surface(x, y, z, **kw):
    d = {"type": "surface", "x": x, "y": y, "z": z, "showscale": False, "hoverinfo": "none"}
    d.update(kw)
    return d


def _line3d(x, y, z, color, width, dash=None):
    line = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    return {
        "type": "scatter3d",
        "x": list(x), "y": list(y), "z": list(z),
        "mode": "lines",
        "line": line,
        "showlegend": False,
        "hoverinfo": "none",
    }


def _text3d(x, y, z, text, color):
    return {
        "type": "scatter3d",
        "x": list(x), "y": list(y), "z": list(z),
        "mode": "text",
        "text": text,
        "textfont": {"color": color, "size": 14},
        "showlegend": False,
        "hoverinfo": "none",
    }


def compute_figure(radius_hs, cx, cy, cz, show_axes=True):
    r = 1.0
    center = np.array([cx, cy, cz])
    traces = []

    phi = np.linspace(0, 2 * np.pi, 50)
    theta = np.linspace(0, np.pi, 50)

    # Абсолют (граничная сфера)
    xa = r * np.outer(np.cos(phi), np.sin(theta))
    ya = r * np.outer(np.sin(phi), np.sin(theta))
    za = r * np.outer(np.ones_like(phi), np.cos(theta))
    traces.append(_surface(xa, ya, za, colorscale="Blues", opacity=0.15, name="Абсолют"))

    dist = np.linalg.norm(center)

    if dist < 1e-6:
        xh = center[0] + radius_hs * np.outer(np.cos(phi), np.sin(theta))
        yh = center[1] + radius_hs * np.outer(np.sin(phi), np.sin(theta))
        zh = center[2] + radius_hs * np.outer(np.ones_like(phi), np.cos(theta))
        R = np.eye(3)
        rpar = rperp = radius_hs
    else:
        sf = np.sqrt(max(1.0 - dist ** 2, 1e-9))
        rpar = radius_hs * sf
        rperp = radius_hs

        xs = np.outer(np.cos(phi), np.sin(theta))
        ys = np.outer(np.sin(phi), np.sin(theta))
        zs = np.outer(np.ones_like(phi), np.cos(theta))

        xe, ye, ze = rperp * xs, rperp * ys, rpar * zs

        uz = np.array([0.0, 0.0, 1.0])
        uzp = center / dist
        v = np.cross(uz, uzp)
        s = np.linalg.norm(v)
        c = np.dot(uz, uzp)

        if s < 1e-9:
            R = np.sign(c) * np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

        coords = np.vstack([xe.ravel(), ye.ravel(), ze.ravel()])
        rot = R @ coords
        xh = rot[0].reshape(xs.shape) + center[0]
        yh = rot[1].reshape(ys.shape) + center[1]
        zh = rot[2].reshape(zs.shape) + center[2]

    traces.append(_surface(xh, yh, zh, colorscale="Greens", opacity=0.6, name="Гиперболическая сфера"))
    traces.append({
        "type": "scatter3d",
        "x": [center[0]], "y": [center[1]], "z": [center[2]],
        "mode": "markers",
        "marker": {"color": "black", "size": 5, "symbol": "diamond"},
        "name": "Центр",
        "showlegend": True,
        "hoverinfo": "none",
    })

    # Геодезические
    idx = np.arange(50, dtype=float) + 0.5
    phi_d = np.arccos(1 - 2 * idx / 50)
    theta_d = np.pi * (1 + 5 ** 0.5) * idx

    D_inv = np.diag([1 / rperp ** 2, 1 / rperp ** 2, 1 / rpar ** 2])
    M_inv = R @ D_inv @ R.T

    for i in range(50):
        ud = np.array([
            np.cos(theta_d[i]) * np.sin(phi_d[i]),
            np.sin(theta_d[i]) * np.sin(phi_d[i]),
            np.cos(phi_d[i]),
        ])
        b = 2 * np.dot(center, ud)
        cc = np.dot(center, center) - r ** 2
        disc = b ** 2 - 4 * cc
        if disc < 0:
            continue

        t_m = (-b - np.sqrt(disc)) / 2
        t_p = (-b + np.sqrt(disc)) / 2
        p1 = center + t_m * ud
        p2 = center + t_p * ud
        chord = p2 - p1
        if np.linalg.norm(chord) < 1e-6:
            continue

        oc = p1 - center
        ae = chord @ M_inv @ chord
        be = 2 * (chord @ M_inv @ oc)
        ce = oc @ M_inv @ oc - 1
        de = be ** 2 - 4 * ae * ce

        if de < 0:
            traces.append(_line3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "#C80000", 2))
            continue

        ten = (-be - np.sqrt(de)) / (2 * ae)
        tex = (-be + np.sqrt(de)) / (2 * ae)
        pen = p1 + ten * chord
        pex = p1 + tex * chord

        if ten > 1e-6:
            traces.append(_line3d([p1[0], pen[0]], [p1[1], pen[1]], [p1[2], pen[2]], "#C80000", 2))
        if (tex - ten) * np.linalg.norm(chord) > 1e-6:
            traces.append(_line3d([pen[0], pex[0]], [pen[1], pex[1]], [pen[2], pex[2]], "#C80000", 1.5, "dash"))
        if 1 - tex > 1e-6:
            traces.append(_line3d([pex[0], p2[0]], [pex[1], p2[1]], [pex[2], p2[2]], "#C80000", 2))

    if show_axes:
        al = r * 1.1
        for xs, ys, zs, col, lbl in [
            ([-al, al], [0, 0], [0, 0], "red", "X"),
            ([0, 0], [-al, al], [0, 0], "blue", "Y"),
            ([0, 0], [0, 0], [-al, al], "green", "Z"),
        ]:
            traces.append(_line3d(xs, ys, zs, col, 2))
            tx = [al * 1.05 if lbl == "X" else 0]
            ty = [al * 1.05 if lbl == "Y" else 0]
            tz = [al * 1.05 if lbl == "Z" else 0]
            traces.append(_text3d(tx, ty, tz, [lbl], col))

    layout = {
        "scene": {
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}},
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 0},
        "legend": {"x": 0.8, "y": 0.9},
        "paper_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "size": 12, "color": "black"},
    }

    return json.dumps({"data": traces, "layout": layout}, cls=_NpEncoder)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/figure")
def api_figure():
    try:
        radius = float(request.args.get("radius", 0.4))
        cx = float(request.args.get("cx", 0.2))
        cy = float(request.args.get("cy", -0.1))
        cz = float(request.args.get("cz", 0.3))
        show_axes = request.args.get("show_axes", "1") == "1"
        return Response(compute_figure(radius, cx, cy, cz, show_axes), mimetype="application/json")
    except Exception as exc:
        return Response(json.dumps({"error": str(exc)}), status=500, mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
