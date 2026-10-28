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
GRID = np.linspace(-1, 1, 801)
KX, KY = np.meshgrid(GRID, GRID)
INSIDE = KX ** 2 + KY ** 2 < 1


def disk(ax):
    ax.plot(np.cos(T), np.sin(T), color="0.3", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")


def chord(ax, p, d, **kw):
    # хорда через точку p (внутреннюю, на абсолюте или внешнюю) в направлении d
    d = np.asarray(d, float) / np.linalg.norm(d)
    b = p @ d
    disc = b * b - (p @ p - 1)
    if disc <= 0:
        return None
    t1, t2 = -b - np.sqrt(disc), -b + np.sqrt(disc)
    a, c = p + t1 * d, p + t2 * d
    ax.plot([a[0], c[0]], [a[1], c[1]], **kw)
    return a, c


def level(ax, f, values, **kw):
    # кривая уровня f = value внутри круга
    F = np.where(INSIDE, f(KX, KY), np.nan)
    ax.contour(KX, KY, F, levels=sorted(values), **kw)


# Рисунок: классификация прямых
def fig_klassifikaciya():
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    disk(ax)
    A, B = np.array([-0.96, -0.28]), np.array([0.96, -0.28])
    ax.plot(*zip(A, B), color="k", lw=1.8)
    ax.text(0.62, -0.37, "$a$", fontsize=14)
    P = np.array([-0.1, 0.45])
    for E, name, xy in [(A, "$p_1$", (-0.72, 0.03)), (B, "$p_2$", (0.55, 0.05))]:
        chord(ax, P, E - P, color="tab:orange", lw=1.6)
        ax.text(*xy, name, fontsize=13, color="tab:orange")
    chord(ax, P, [0.25, -1], color="tab:blue", lw=1.4)
    ax.text(0.07, -0.62, "$r$", fontsize=13, color="tab:blue")
    chord(ax, P, [1, 0.12], color="tab:green", lw=1.4)
    ax.text(0.78, 0.6, "$q$", fontsize=13, color="tab:green")
    ax.plot(*P, "ko", ms=4)
    ax.text(P[0] - 0.12, P[1] + 0.06, "$P$", fontsize=13)
    fig.savefig(OUT + "klassifikaciya.pdf")


# Рисунок: три пучка и их траектории
def fig_puchki():
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    # эллиптический: центр c
    ax = axs[0]
    disk(ax)
    c = np.array([0.35, 0.15])
    for th in np.linspace(0, np.pi, 12, endpoint=False):
        chord(ax, c, [np.cos(th), np.sin(th)], color="tab:red", lw=0.8)
    f = lambda x, y: (1 - x * c[0] - y * c[1]) / np.sqrt((1 - c @ c) * (1 - x * x - y * y))
    level(ax, f, np.cosh([0.6, 1.2]), colors="tab:blue", linewidths=1.8)
    ax.plot(*c, "ko", ms=4)
    ax.set_title("а) пучок пересекающихся прямых", fontsize=12)
    # параболический: идеальная точка w
    ax = axs[1]
    disk(ax)
    w = np.array([np.cos(2.3), np.sin(2.3)])
    for th in np.linspace(2.3 - 1.45, 2.3 + 1.45, 13):
        chord(ax, w * 0.9999, [np.cos(th + np.pi), np.sin(th + np.pi)], color="tab:red", lw=0.8)
    f = lambda x, y: (1 - x * w[0] - y * w[1]) / np.sqrt(1 - x * x - y * y)
    level(ax, f, [0.6, 1.5], colors="tab:blue", linewidths=1.8)
    ax.plot(*w, "ko", ms=4)
    ax.set_title("б) пучок параллельных прямых", fontsize=12)
    # гиперболический: ось n.k = s, полюс n/s
    ax = axs[2]
    disk(ax)
    n, s = np.array([np.cos(-0.5), np.sin(-0.5)]), 0.62
    pole = n / s
    m = np.array([-n[1], n[0]])
    ends = chord(ax, s * n, m, color="k", lw=1.8)
    half = np.arcsin(1 / np.linalg.norm(pole))
    base = np.arctan2(-pole[1], -pole[0])
    for th in np.linspace(base - half * 0.93, base + half * 0.93, 11):
        seg = chord(ax, pole, [np.cos(th), np.sin(th)], color="tab:red", lw=0.8)
    for e in ends:
        ax.plot([e[0], pole[0]], [e[1], pole[1]], color="0.55", lw=0.7, ls="--")
    ax.plot(*pole, "ko", ms=4)
    ax.text(pole[0] + 0.03, pole[1] - 0.08, "$S$", fontsize=13)
    f = lambda x, y: (x * n[0] + y * n[1] - s) / np.sqrt((1 - s * s) * (1 - x * x - y * y))
    level(ax, f, [np.sinh(0.5), -np.sinh(0.5)], colors="tab:blue", linewidths=1.8, linestyles="solid")
    ax.set_xlim(-1.08, 1.6)
    ax.set_ylim(-1.08, 1.08)
    ax.set_title("в) пучок расходящихся прямых", fontsize=12)
    fig.savefig(OUT + "puchki.pdf")


# Рисунок: сечение сферы плоскостью через центр шара: изображение центра и евклидов центр эллипса
def fig_sravnenie():
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    disk(ax)
    c, rho = np.array([0.6, 0.0]), 1.0
    f = lambda x, y: (1 - x * c[0] - y * c[1]) / np.sqrt((1 - c @ c) * (1 - x * x - y * y))
    level(ax, f, [np.cosh(rho)], colors="tab:blue", linewidths=2)
    b = c @ c + np.cosh(rho) ** 2 * (1 - c @ c)
    ax.plot([-1, 1], [0, 0], color="0.75", lw=0.8, ls=":")
    ax.plot(*c, "ko", ms=4)
    ax.plot(c[0] / b, 0, "x", color="tab:blue", ms=8, mew=2)
    ax.plot(0, 0, "o", color="0.4", ms=3)
    ax.text(c[0] + 0.02, 0.06, "$C$", fontsize=13)
    ax.text(c[0] / b - 0.05, -0.13, "$C'$", fontsize=13, color="tab:blue")
    ax.text(-0.08, -0.13, "$O$", fontsize=12, color="0.4")
    fig.savefig(OUT + "sravnenie.pdf")


def ball(ax):
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="0.75", lw=0.3)
    ax.set_box_aspect((1, 1, 1), zoom=1.4)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.axis("off")


def ellipsoid(ax, center, direction, perp, par, color):
    import sys
    sys.path.insert(0, ".")
    from klein_surfaces import ellipsoid_mesh
    X, Y, Z = ellipsoid_mesh(center, direction, perp, par, 40)
    ax.plot_surface(X, Y, Z, color=color, alpha=0.45, lw=0)


def segment3(ax, p, d, **kw):
    d = d / np.linalg.norm(d)
    b = p @ d
    disc = b * b - (p @ p - 1)
    if disc <= 0:
        return
    t = np.array([-b - np.sqrt(disc), -b + np.sqrt(disc)])
    pts = p[None, :] + t[:, None] * d[None, :]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], **kw)


def fibo(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)


# Рисунок: три поверхности в шаре Клейна и их сечения
def fig_poverhnosti():
    import sys
    sys.path.insert(0, ".")
    from klein_surfaces import sphere_params, horosphere_params
    fig = plt.figure(figsize=(12, 8.4))
    # верхний ряд: поверхности
    ax = fig.add_subplot(231, projection="3d")
    ball(ax)
    c = np.array([0.45, 0.2, 0.25])
    ellipsoid(ax, *sphere_params(c, 0.9), "tab:green")
    for d in fibo(14):
        segment3(ax, c, d, color="tab:red", lw=0.6)
    ax.scatter(*c, color="k", s=14)
    ax.text(c[0] + 0.05, c[1], c[2] + 0.08, "$O$", fontsize=13)
    ax.set_title("а) сфера", fontsize=12)
    ax = fig.add_subplot(232, projection="3d")
    ball(ax)
    w = np.array([0.0, 0.6, 0.8])
    ellipsoid(ax, *horosphere_params(w, 0.6), "tab:green")
    for e in fibo(16):
        if np.linalg.norm(e - w) > 0.3:
            ax.plot(*zip(w, e), color="tab:red", lw=0.6)
    ax.scatter(*w, color="k", s=22)
    ax.text(w[0] + 0.05, w[1], w[2] + 0.1, "$\\omega$", fontsize=14)
    ax.set_title("б) орисфера", fontsize=12)
    ax = fig.add_subplot(233, projection="3d")
    ball(ax)
    sigma, h = 0.3, 0.45
    r, a_ = np.mgrid[0:1:30j, 0:2 * np.pi:60j]
    for sgn in (1, -1):
        s2 = np.sinh(h) ** 2 * (1 - sigma ** 2)
        A = 1 + s2
        rmax = np.sqrt(1 - sigma ** 2)
        x, y = rmax * r * np.cos(a_), rmax * r * np.sin(a_)
        C = sigma ** 2 - s2 * (1 - x ** 2 - y ** 2)
        disc = np.maximum(sigma ** 2 - A * C, 0)
        z = (sigma + sgn * np.sqrt(disc)) / A
        ax.plot_surface(x, y, z, color="tab:green", alpha=0.4, lw=0)
    rr = np.sqrt(1 - sigma ** 2)
    ax.plot_surface(rr * r * np.cos(a_), rr * r * np.sin(a_), sigma + 0 * r, color="0.5", alpha=0.25, lw=0)
    pole = np.array([0, 0, 1 / sigma])
    for q in [np.array([0.0, 0.0, sigma]), *[np.array([0.55 * np.cos(t), 0.55 * np.sin(t), sigma])
                                             for t in np.linspace(0, 2 * np.pi, 7)[:-1]]]:
        segment3(ax, q, q - pole, color="tab:red", lw=0.6)
    ax.set_title("в) эквидистантная поверхность", fontsize=12)
    # нижний ряд: сечения плоскостью, проходящей через центр шара и ось поверхности
    ax = fig.add_subplot(234)
    disk(ax)
    q = np.array([np.linalg.norm(c), 0.0])
    for th in np.linspace(0, np.pi, 12, endpoint=False):
        chord(ax, q, [np.cos(th), np.sin(th)], color="tab:red", lw=0.7)
    f = lambda x, y: (1 - x * q[0] - y * q[1]) / np.sqrt((1 - q @ q) * (1 - x * x - y * y))
    level(ax, f, [np.cosh(0.9)], colors="tab:green", linewidths=2.2)
    ax.plot(*q, "ko", ms=4)
    ax.text(q[0] + 0.04, q[1] + 0.05, "$O$", fontsize=13)
    ax.text(0, -1.25, "окружность лежит внутри круга,\nпрямые сходятся в точке $O$", ha="center", fontsize=11)
    ax.set_ylim(-1.45, 1.08)
    ax = fig.add_subplot(235)
    disk(ax)
    om = np.array([0.0, 1.0])
    rh = 0.6
    ctr = om * (1 - rh * rh)
    ax.plot(ctr[0] + rh * np.cos(T), ctr[1] + rh * rh * np.sin(T), color="tab:green", lw=2.2)
    for th in np.linspace(-np.pi / 2 - 1.35, -np.pi / 2 + 1.35, 11):
        e = np.array([np.cos(th), np.sin(th)])
        ax.plot([om[0], e[0]], [om[1], e[1]], color="tab:red", lw=0.7)
    ax.plot(*om, "ko", ms=5)
    ax.text(0.06, 1.02, "$\\omega$", fontsize=14)
    ax.text(0, -1.25, "орицикл касается абсолюта в точке $\\omega$,\nпрямые выходят из $\\omega$", ha="center", fontsize=11)
    ax.set_ylim(-1.45, 1.08)
    ax = fig.add_subplot(236)
    disk(ax)
    xs = np.linspace(-1, 1, 400)
    bw = np.sqrt(1 - sigma ** 2)
    ax.plot([-bw, bw], [sigma, sigma], color="k", lw=1.8)
    for sgn in (1, -1):
        s2 = np.sinh(h) ** 2 * (1 - sigma ** 2)
        A = 1 + s2
        xx = np.linspace(-bw, bw, 400)
        C = sigma ** 2 - s2 * (1 - xx ** 2)
        yy = (sigma + sgn * np.sqrt(np.maximum(sigma ** 2 - A * C, 0))) / A
        ax.plot(xx, yy, color="tab:green", lw=2.2)
    pl = np.array([0, 1 / sigma])
    for x0 in [-0.75, -0.45, -0.15, 0.15, 0.45, 0.75]:
        p0 = np.array([x0 * bw, sigma])
        chord(ax, p0, pl - p0, color="tab:red", lw=0.7)
    ax.text(0, -1.25, "эквидистанты подходят к абсолюту\nв концах базы", ha="center", fontsize=11)
    ax.set_ylim(-1.45, 1.08)
    fig.savefig(OUT + "poverhnosti.pdf")


# Рисунок: кубическая комната и сечение четырёх комнат у ребра
def refl2(n, s_, k):
    n = np.asarray(n, float)
    f = 2 * (n @ k - s_) / (1 - s_ * s_)
    return (k - f * n) / (1 - f * s_)


def fig_kub():
    fig = plt.figure(figsize=(14, 4.4))
    ax = fig.add_subplot(131)
    t_ = np.linspace(1e-3, 1 / np.sqrt(3), 300)
    d = np.arctanh(t_)
    alpha = np.degrees(np.arccos(t_ ** 2 / (1 - t_ ** 2)))
    ax.plot(d, alpha, color="tab:blue", lw=2)
    ax.axhline(90, color="0.6", ls="--", lw=0.8)
    ax.axhline(60, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("расстояние от центра комнаты до стены")
    ax.set_ylabel("двугранный угол, градусы")
    ax.set_ylim(55, 95)
    ax.grid(alpha=0.3)
    ax.set_title("а) двугранный угол", fontsize=12)
    ax = fig.add_subplot(132, projection="3d")
    ball(ax)
    s_ = 0.5
    for a1 in (-s_, s_):
        for b1 in (-s_, s_):
            ax.plot([-s_, s_], [a1, a1], [b1, b1], color="tab:brown", lw=1.2)
            ax.plot([a1, a1], [-s_, s_], [b1, b1], color="tab:brown", lw=1.2)
            ax.plot([a1, a1], [b1, b1], [-s_, s_], color="tab:brown", lw=1.2)
    ax.set_title("б) комната в модели", fontsize=12)
    ax = fig.add_subplot(133)
    disk(ax)
    t = 0.5
    sq = [np.array(p, float) for p in [(-t, -t), (t, -t), (t, t), (-t, t)]]
    R1 = lambda P: [refl2([1, 0], t, p) for p in P]
    R2 = lambda P: [refl2([0, 1], t, p) for p in P]
    for P, col in [(sq, "tab:blue"), (R1(sq), "tab:green"), (R2(sq), "tab:orange"), (R1(R2(sq)), "tab:red")]:
        xy = np.array(P + [P[0]])
        ax.fill(xy[:, 0], xy[:, 1], color=col, alpha=0.18)
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.6)
    ax.plot(t, t, "ko", ms=4)
    ax.text(t + 0.03, t - 0.1, "$V$", fontsize=13)
    al = np.degrees(np.arccos(t * t / (1 - t * t)))
    ax.annotate("щель", xy=(t + 0.06, t + 0.1), xytext=(0.15, 0.9), fontsize=12, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlim(-0.6, 1.02)
    ax.set_ylim(-0.6, 1.02)
    ax.set_title("в) сечение $z=0$: четыре комнаты у ребра", fontsize=12)
    fig.savefig(OUT + "kub.pdf")
    print("угол", round(al, 2), "щель", round(360 - 4 * al, 1))


# Рисунок: комната с полом-орисферой, идеальная точка (0, 0, 1)
def fig_komnata():
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ball(ax)
    a, h = 1.2, 0.9
    ell = np.array([1.0, 0, 0, 1.0])

    def klein(u1, u2, c):
        q = u1 ** 2 + u2 ** 2
        beta = (1 - c * c) / (2 * c)
        y0 = c * (1 + q / 2) + beta
        return np.array([c * u1, c * u2, c * q / 2 + beta]) / y0

    s = np.linspace(-a, a, 60)
    for c, col in [(1.0, "tab:brown"), (np.exp(-h), "tab:gray")]:
        for v in np.linspace(-a, a, 5):
            ax.plot(*klein(s, v + 0 * s, c), color=col, lw=0.9)
            ax.plot(*klein(v + 0 * s, s, c), color=col, lw=0.9)
    cs = np.exp(-np.linspace(0, h, 30))
    for u1 in (-a, a):
        for u2 in (-a, a):
            ax.plot(*np.array([klein(u1, u2, c) for c in cs]).T, color="k", lw=1.2)
    ax.scatter(0, 0, 1, color="k", s=10)
    ax.view_init(elev=22, azim=35)
    fig.savefig(OUT + "komnata.pdf")


if __name__ == "__main__":
    fig_klassifikaciya()
    fig_puchki()
    fig_sravnenie()
    fig_poverhnosti()
    fig_kub()
    fig_komnata()
    print("ok")
