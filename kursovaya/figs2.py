import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif"],
    "mathtext.fontset": "stix", "font.size": 12,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})
OUT = "fig/"
T = np.linspace(0, 2 * np.pi, 400)
BLUE, RED, OCHRE, GREEN, GREY = "#2243c4", "#b8204f", "#a86a10", "#23734f", "0.55"


def disk(ax):
    ax.plot(np.cos(T), np.sin(T), color="0.25", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")


def ends(p, d):
    d = np.asarray(d, float) / np.linalg.norm(d)
    p = np.asarray(p, float)
    b = p @ d
    disc = b * b - (p @ p - 1)
    s = np.sqrt(disc)
    return p + (-b - s) * d, p + (-b + s) * d


def chord(ax, p, d, **kw):
    a, c = ends(p, d)
    ax.plot([a[0], c[0]], [a[1], c[1]], **kw)
    return a, c


def boost_to(q):
    q = np.asarray(q, float)
    r = np.linalg.norm(q)
    M = np.eye(3)
    if r < 1e-14:
        return M
    u = q / r
    t = np.arctanh(r)
    ch, sh = np.cosh(t), np.sinh(t)
    M[:2, :2] += (ch - 1) * np.outer(u, u)
    M[:2, 2] = sh * u
    M[2, :2] = sh * u
    M[2, 2] = ch
    return M


def app(M, k):
    X = M @ np.array([k[0], k[1], 1.0])
    return X[:2] / X[2]


def along(P, Q, s):
    T_ = boost_to(-np.asarray(P))
    q = app(T_, Q)
    d = q / np.linalg.norm(q)
    return app(boost_to(P), np.tanh(s) * d)


def dist(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.arccosh((1 - a @ b) / np.sqrt((1 - a @ a) * (1 - b @ b)))


# Рисунок: линейка в модели (равные отрезки длины 0,5)
def fig_lineyka():
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    disk(ax)
    A = np.array([-0.2, -0.25])
    U, V = ends(A, [1, 0.45])
    ax.plot([U[0], V[0]], [U[1], V[1]], color=BLUE, lw=1.6)
    for E in (U, V):
        for j in range(1, 14):
            q = along(A, E, 0.5 * j)
            if np.linalg.norm(q - E) < 0.01:
                break
            n = np.array([-(V - U)[1], (V - U)[0]])
            n /= np.linalg.norm(n)
            ax.plot([q[0] - 0.025 * n[0], q[0] + 0.025 * n[0]], [q[1] - 0.025 * n[1], q[1] + 0.025 * n[1]], color="k", lw=1)
    ax.plot(*A, "ko", ms=4)
    ax.text(A[0] - 0.06, A[1] + 0.07, "$A$", fontsize=13)
    ax.text(U[0] - 0.02, U[1] - 0.12, "$U$", fontsize=13)
    ax.text(V[0] - 0.08, V[1] + 0.05, "$V$", fontsize=13)
    fig.savefig(OUT + "lineyka.pdf")


# Рисунок: перпендикуляр через полюс
def fig_polyus():
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    disk(ax)
    n, s = np.array([np.cos(0.35), np.sin(0.35)]), 0.55
    m = np.array([-n[1], n[0]])
    U, V = ends(s * n, m)
    ax.plot([U[0], V[0]], [U[1], V[1]], color=BLUE, lw=2)
    P = n / s
    for E in (U, V):
        ax.plot([E[0], P[0]], [E[1], P[1]], color=GREY, lw=0.8, ls="--")
    ax.plot(*P, "ko", ms=4)
    ax.text(P[0] + 0.04, P[1] - 0.02, "$P$", fontsize=13)
    for X in [np.array([-0.5, 0.35]), np.array([-0.3, -0.55]), np.array([0.05, 0.05])]:
        a, c = chord(ax, X, P - X, color=RED, lw=1.3)
        ext = c if np.linalg.norm(c - P) < np.linalg.norm(a - P) else a
        ax.plot([ext[0], P[0]], [ext[1], P[1]], color=RED, lw=0.6, ls=":")
    ax.text(*(s * n + 0.35 * m + np.array([0.04, 0.02])), "$\\ell$", fontsize=14, color=BLUE)
    ax.set_xlim(-1.08, P[0] + 0.15)
    fig.savefig(OUT + "polyus.pdf")


# Рисунок: угол параллельности с вершиной в центре круга
def fig_pi():
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    disk(ax)
    h = 0.55
    w = np.sqrt(1 - h * h)
    ax.plot([-w, w], [-h, -h], color=BLUE, lw=2.2)
    ax.plot([0, 0], [0, -h], color="k", lw=1.6)
    for sgn in (1, -1):
        ax.plot([0, sgn * w], [0, -h], color=RED, lw=1.8)
        ax.plot([0, -sgn * w], [0, h], color=RED, lw=1.8)
    a1, a2 = -np.pi / 2, np.arctan2(-h, w)
    tt = np.linspace(a1, a2, 30)
    ax.plot(0.16 * np.cos(tt), 0.16 * np.sin(tt), color=RED, lw=1.2)
    ax.plot(0, 0, "ko", ms=4)
    ax.plot(0, -h, "ko", ms=4)
    ax.text(-0.1, 0.06, "$B$", fontsize=13)
    ax.text(-0.11, -h - 0.11, "$H$", fontsize=13)
    ax.text(0.1, -0.27, "$\\Pi(x)$", fontsize=12, color=RED)
    ax.text(w - 0.12, -h - 0.12, "$a$", fontsize=13, color=BLUE)
    fig.savefig(OUT + "pi.pdf")


# Рисунок: расстояние от точки прямой b до прямой a (диаметр): параллельные и расходящиеся
def fig_rasstoyanie():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.linspace(-0.999, 0.999, 800)
    k = 0.55
    d_par = np.arctanh(np.clip(k * np.sqrt((1 - x) / (1 + x)), 0, 0.999999))
    c = 0.4
    xs = np.linspace(-np.sqrt(1 - c * c) + 1e-4, np.sqrt(1 - c * c) - 1e-4, 800)
    d_div = np.arctanh(np.clip(c / np.sqrt(1 - xs ** 2), 0, 0.999999))
    for ax, X, Dd, title in [(axs[0], x, d_par, "параллельные: $y=k(1-x)$, $k=0{,}55$"),
                             (axs[1], xs, d_div, "расходящиеся: $y=c$, $c=0{,}4$")]:
        ax.plot(X, Dd, color=OCHRE, lw=2)
        ax.set_ylim(0, 3.2)
        ax.set_xlabel("абсцисса точки $M$ на прямой $b$")
        ax.set_ylabel("$d(M, a)$")
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=12)
    fig.savefig(OUT + "rasstoyanie.pdf")


# Рисунок: четырёхугольник Саккери в модели
def fig_sakkeri():
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    disk(ax)
    p, h = np.tanh(0.6), 1.1
    q = np.sqrt(1 - p * p) * np.tanh(h)
    A, B, C, Dd = [-p, -0.2], [p, -0.2], [p, -0.2], [-p, -0.2]
    A, B = np.array([-p, 0.0]), np.array([p, 0.0])
    C, Dd = np.array([p, q]), np.array([-p, q])
    M = boost_to([0, -0.45])
    A, B, C, Dd = [app(M, z) for z in (A, B, C, Dd)]
    for X, Y in [(A, B), (B, C), (C, Dd), (Dd, A)]:
        chord(ax, X, Y - X, color="0.8", lw=0.8)
    xs = [A[0], B[0], C[0], Dd[0], A[0]]
    ys = [A[1], B[1], C[1], Dd[1], A[1]]
    ax.fill(xs, ys, color=BLUE, alpha=0.08)
    ax.plot(xs, ys, color=BLUE, lw=2)
    for P_, n_, off in [(A, "A", (-0.1, -0.1)), (B, "B", (0.03, -0.1)), (C, "C", (0.03, 0.03)), (Dd, "D", (-0.1, 0.03))]:
        ax.plot(*P_, "ko", ms=3.5)
        ax.text(P_[0] + off[0], P_[1] + off[1], f"${n_}$", fontsize=13)
    fig.savefig(OUT + "sakkeri.pdf")


# Рисунок: эквидистанта и хорда (отрицание предложения Посидония)
def fig_ekvidistanta():
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    disk(ax)
    ax.plot([-1, 1], [0, 0], color=BLUE, lw=2)
    h = 0.6
    x = np.linspace(-1, 1, 400)
    y = np.tanh(h) * np.sqrt(1 - x ** 2)
    ax.plot(x, y, color=OCHRE, lw=2.2)
    for x0 in [-0.75, -0.35, 0.0, 0.35, 0.75]:
        y0 = np.tanh(h) * np.sqrt(1 - x0 ** 2)
        ax.plot([x0, x0], [0, y0], color="k", lw=0.9, ls="--")
        ax.plot(x0, y0, "ko", ms=3)
    x1, x2 = -0.75, 0.75
    y1 = np.tanh(h) * np.sqrt(1 - x1 ** 2)
    ax.plot([x1, x2], [y1, y1], color=GREY, lw=1.2, ls=":")
    ax.text(0.8, -0.1, "$a$", fontsize=13, color=BLUE)
    fig.savefig(OUT + "ekvidistanta.pdf")


# Рисунок: острый угол с вершиной в центре и перпендикуляры к стороне
def fig_lezhandr():
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    disk(ax)
    phi = 0.75
    ax.plot([0, 1], [0, 0], color=BLUE, lw=2.2)
    ax.plot([0, np.cos(phi)], [0, np.sin(phi)], color=BLUE, lw=2.2)
    for pnt in [0.2, 0.4, 0.6]:
        ax.plot([pnt, pnt], [-np.sqrt(1 - pnt ** 2), np.sqrt(1 - pnt ** 2)], color=GREEN, lw=1.2)
    for pnt in [np.cos(phi), 0.86]:
        ax.plot([pnt, pnt], [-np.sqrt(1 - pnt ** 2), np.sqrt(1 - pnt ** 2)], color=RED, lw=1.4, ls="--" if pnt > np.cos(phi) + 1e-9 else "-")
    ax.plot(0, 0, "ko", ms=4)
    ax.text(-0.1, -0.1, "$O$", fontsize=13)
    ax.text(np.cos(phi) - 0.04, -0.12, "$P^*$", fontsize=12, color=RED)
    ax.text(1.02, -0.04, "$E_1$", fontsize=12)
    ax.text(np.cos(phi) + 0.02, np.sin(phi) + 0.03, "$E_2$", fontsize=12)
    ax.plot([np.cos(phi), 1], [np.sin(phi), 0], color=RED, lw=0.0)
    fig.savefig(OUT + "lezhandr.pdf")


# Рисунок: пример треугольника без описанной окружности (по построению из лекции)
def fig_bez_okruzhnosti():
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    disk(ax)

    def refl(L, k):
        X = np.array([k[0], k[1], 1.0])
        N = np.array([L[0], L[1], -L[2]])
        f = 2 * (L @ X) / (N[0] ** 2 + N[1] ** 2 - N[2] ** 2)
        Y = X - f * N
        return Y[:2] / Y[2]
    a = np.array([0.0, 1.0, 0.0])                 # прямая a: y = 0
    A = np.array([0.0, 0.55])                     # A на перпендикуляре h (ось y)
    d = np.array([1.0, 0.52])
    b = np.cross([A[0], A[1], 1.0], [d[0], d[1], 0.0])   # прямая b через A, не перпендикулярна h
    ax.plot([0, 0], [-1, 1], color=GREY, lw=0.8, ls=":")
    ax.plot([-1, 1], [0, 0], color=OCHRE, lw=1.4, ls="--")
    chord(ax, A, d, color=OCHRE, lw=1.4, ls="--")
    M = np.array([0.0, 0.12])
    Nn, P = refl(a, M), refl(b, M)
    tri = np.array([M, Nn, P, M])
    ax.fill(tri[:, 0], tri[:, 1], color=BLUE, alpha=0.08)
    ax.plot(tri[:, 0], tri[:, 1], color=BLUE, lw=2)
    for X, n_, off in [(M, "M", (-0.12, -0.02)), (Nn, "N", (-0.12, -0.04)), (P, "P", (0.04, 0.02)), (A, "A", (-0.12, 0.03))]:
        ax.plot(*X, "ko", ms=3.5)
        ax.text(X[0] + off[0], X[1] + off[1], f"${n_}$", fontsize=13)
    ax.text(0.85, -0.1, "$a$", fontsize=13, color=OCHRE)
    e = ends(A, d)[1]
    ax.text(e[0] - 0.12, e[1] - 0.12, "$b$", fontsize=13, color=OCHRE)
    ax.text(0.03, -0.92, "$h$", fontsize=13, color=GREY)
    fig.savefig(OUT + "bez_okruzhnosti.pdf")


if __name__ == "__main__":
    fig_lineyka()
    fig_polyus()
    fig_pi()
    fig_rasstoyanie()
    fig_sakkeri()
    fig_ekvidistanta()
    fig_lezhandr()
    fig_bez_okruzhnosti()
    print("ok")
