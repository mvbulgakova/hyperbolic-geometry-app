import numpy as np

# Модель Бельтрами-Клейна: единичный шар, геодезические - хорды, абсолют - единичная сфера.
# Гиперболическая сфера и орисфера в ней - эллипсоиды вращения.


def rotation_to(n):
    # матрица поворота e_z -> n (n - единичный вектор)
    u_z = np.array([0., 0., 1.])
    v = np.cross(u_z, n)
    s = np.linalg.norm(v)
    c = np.dot(u_z, n)
    if s < 1e-9:
        return np.sign(c) * np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def unit_sphere_mesh(n=50):
    phi = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(0, np.pi, n)
    return (np.outer(np.cos(phi), np.sin(theta)),
            np.outer(np.sin(phi), np.sin(theta)),
            np.outer(np.ones_like(phi), np.cos(theta)))


def ellipsoid_mesh(center, direction, perp, par, n=50):
    # эллипсоид вращения: полуоси perp поперёк оси и par вдоль неё, ось направлена по direction
    ux, uy, uz = unit_sphere_mesh(n)
    pts = np.vstack([perp * ux.ravel(), perp * uy.ravel(), par * uz.ravel()])
    pts = rotation_to(direction) @ pts + np.asarray(center, float).reshape(3, 1)
    return [p.reshape(ux.shape) for p in pts]


def sphere_params(center, rho):
    # Гиперболическая сфера с гиперболическим центром center (|center| < 1) и радиусом rho:
    #   (1 - k.c)^2 = cosh(rho)^2 * (1 - |c|^2) * (1 - |k|^2)
    # Возвращает евклидов центр эллипсоида, направление оси и полуоси (поперёк, вдоль).
    # Евклидов центр эллипсоида не совпадает с гиперболическим центром.
    center = np.asarray(center, float)
    a = np.linalg.norm(center)
    if a >= 1:
        raise ValueError("центр должен лежать внутри абсолюта")
    c2 = np.cosh(rho) ** 2
    b = a * a + c2 * (1 - a * a)
    perp = np.sqrt((1 - a * a) * (c2 - 1) / b)
    par = (1 - a * a) * np.sqrt(c2 * (c2 - 1)) / b
    direction = center / a if a > 1e-9 else np.array([0., 0., 1.])
    return center / b, direction, perp, par


def hyperbolic_sphere(center, rho, n=50):
    ell_center, direction, perp, par = sphere_params(center, rho)
    return ellipsoid_mesh(ell_center, direction, perp, par, n)


def horosphere_params(omega, r):
    # Орисфера с идеальной точкой omega (единичный вектор); r (0 < r < 1) - полуось поперёк оси.
    # Эллипсоид с полуосями r, r, r^2, касается абсолюта в точке omega.
    omega = np.asarray(omega, float)
    return (1 - r * r) * omega, omega, r, r * r


def horosphere(omega, r, n=50):
    center, direction, perp, par = horosphere_params(omega, r)
    return ellipsoid_mesh(center, direction, perp, par, n)
