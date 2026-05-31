import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


EPS = 1e-6
PI = math.pi


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @staticmethod
    def from_list(a) -> "Vec3":
        return Vec3(a[0], a[1], a[2])

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k, self.z * k)

    def __rmul__(self, k: float) -> "Vec3":
        return self.__mul__(k)

    def mul(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)

    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: "Vec3") -> "Vec3":
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def norm(self) -> "Vec3":
        l = self.length()
        if l < EPS:
            return Vec3(0.0, 0.0, 0.0)
        return self * (1.0 / l)

    def max_comp(self) -> float:
        return max(self.x, self.y, self.z)

    def avg(self) -> float:
        return (self.x + self.y + self.z) / 3.0

    def clamp01(self) -> "Vec3":
        return Vec3(
            min(1.0, max(0.0, self.x)),
            min(1.0, max(0.0, self.y)),
            min(1.0, max(0.0, self.z)),
        )


def reflect(v: Vec3, n: Vec3) -> Vec3:
    # идеальное зеркало
    return v - n * (2.0 * v.dot(n))


@dataclass
class Ray:
    o: Vec3
    d: Vec3


@dataclass
class Material:
    diffuse: Vec3
    mirror: Vec3
    emission: Vec3


@dataclass
class Triangle:
    v0: Vec3
    v1: Vec3
    v2: Vec3
    mat: int
    n: Vec3
    area: float


@dataclass
class Hit:
    t: float
    p: Vec3
    n: Vec3
    tri_id: int
    mat_id: int


def triangle_make(v0: Vec3, v1: Vec3, v2: Vec3, mat: int) -> Triangle:
    e1 = v1 - v0
    e2 = v2 - v0
    nn = e1.cross(e2)
    area = 0.5 * nn.length()
    n = nn.norm()
    return Triangle(v0=v0, v1=v1, v2=v2, mat=mat, n=n, area=area)


def intersect_triangle(ray: Ray, tri: Triangle) -> Optional[Tuple[float, float, float]]:
    # Möller–Trumbore
    e1 = tri.v1 - tri.v0
    e2 = tri.v2 - tri.v0
    p = ray.d.cross(e2)
    det = e1.dot(p)
    if abs(det) < 1e-9:
        return None
    inv_det = 1.0 / det
    tvec = ray.o - tri.v0
    u = tvec.dot(p) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    q = tvec.cross(e1)
    v = ray.d.dot(q) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None
    t = e2.dot(q) * inv_det
    if t <= EPS:
        return None
    return (t, u, v)


def intersect_scene(ray: Ray, tris: List[Triangle]) -> Optional[Hit]:
    best_t = 1e30
    best_i = -1
    for i, tri in enumerate(tris):
        r = intersect_triangle(ray, tri)
        if r is None:
            continue
        t, _, _ = r
        if t < best_t:
            best_t = t
            best_i = i
    if best_i < 0:
        return None
    tri = tris[best_i]
    p = ray.o + ray.d * best_t
    n = tri.n
    # нормаль против луча
    if n.dot(ray.d) > 0.0:
        n = n * -1.0
    return Hit(t=best_t, p=p, n=n, tri_id=best_i, mat_id=tri.mat)


def orthonormal_basis(n: Vec3) -> Tuple[Vec3, Vec3]:
    # простой базис вокруг n
    if abs(n.x) > 0.9:
        a = Vec3(0.0, 1.0, 0.0)
    else:
        a = Vec3(1.0, 0.0, 0.0)
    t = n.cross(a).norm()
    b = t.cross(n)
    return t, b


def sample_cosine_hemisphere(n: Vec3, rng: random.Random) -> Tuple[Vec3, float]:
    r1 = rng.random()
    r2 = rng.random()
    phi = 2.0 * PI * r1
    r = math.sqrt(r2)
    x = r * math.cos(phi)
    y = r * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - r2))
    t, b = orthonormal_basis(n)
    d = (t * x + b * y + n * z).norm()
    pdf = max(0.0, n.dot(d)) / PI
    return d, pdf


def sample_point_on_triangle(tri: Triangle, rng: random.Random) -> Tuple[Vec3, Vec3]:
    # равномерно по площади (sqrt)
    r1 = rng.random()
    r2 = rng.random()
    su = math.sqrt(r1)
    a = 1.0 - su
    b = su * (1.0 - r2)
    c = su * r2
    p = tri.v0 * a + tri.v1 * b + tri.v2 * c
    return p, tri.n


@dataclass
class LightSampler:
    light_ids: List[int]
    cdf: List[float]
    total: float

    def sample(self, rng: random.Random) -> Tuple[int, float]:
        # возвращает tri_id и вероятность выбора этого источника
        if not self.light_ids or self.total <= 0.0:
            return (-1, 0.0)
        x = rng.random() * self.total
        lo = 0
        hi = len(self.cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if x <= self.cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        idx = lo
        tri_id = self.light_ids[idx]
        prev = self.cdf[idx - 1] if idx > 0 else 0.0
        weight = self.cdf[idx] - prev
        p_select = weight / self.total if self.total > 0.0 else 0.0
        return (tri_id, p_select)


def build_lights(tris: List[Triangle], mats: List[Material]) -> LightSampler:
    ids: List[int] = []
    weights: List[float] = []
    for i, tri in enumerate(tris):
        em = mats[tri.mat].emission
        if em.max_comp() <= 0.0:
            continue
        power = tri.area * (em.x + em.y + em.z)
        if power <= 0.0:
            continue
        ids.append(i)
        weights.append(power)
    cdf: List[float] = []
    s = 0.0
    for w in weights:
        s += w
        cdf.append(s)
    return LightSampler(light_ids=ids, cdf=cdf, total=s)


def trace_path(
    ray0: Ray,
    tris: List[Triangle],
    mats: List[Material],
    lights: LightSampler,
    max_depth: int,
    rng: random.Random,
) -> Vec3:
    L = Vec3(0.0, 0.0, 0.0)
    beta = Vec3(1.0, 1.0, 1.0)
    ray = ray0

    for depth in range(max_depth):
        hit = intersect_scene(ray, tris)
        if hit is None:
            break

        m = mats[hit.mat_id]

        # если попали в источник света
        if m.emission.max_comp() > 0.0:
            L = L + beta.mul(m.emission)
            break

        # прямой свет (1 выбор)
        if m.diffuse.max_comp() > 0.0 and lights.total > 0.0:
            light_tri_id, p_sel = lights.sample(rng)
            if light_tri_id >= 0 and p_sel > 0.0:
                ltri = tris[light_tri_id]
                lp, ln = sample_point_on_triangle(ltri, rng)
                wi = lp - hit.p
                dist2 = wi.dot(wi)
                if dist2 > EPS:
                    dist = math.sqrt(dist2)
                    wi = wi * (1.0 / dist)
                    cos_x = max(0.0, hit.n.dot(wi))
                    cos_l = max(0.0, ln.dot(wi * -1.0))
                    if cos_x > 0.0 and cos_l > 0.0 and ltri.area > EPS:
                        # теневой луч
                        shadow = Ray(hit.p + hit.n * (1e-4), wi)
                        occ = intersect_scene(shadow, tris)
                        visible = True
                        if occ is not None and occ.t < dist - 1e-4:
                            visible = False
                        if visible:
                            pdf_light = p_sel * (1.0 / ltri.area)
                            G = (cos_x * cos_l) / max(dist2, EPS)
                            f = m.diffuse * (1.0 / PI)  # Ламберт
                            Le = mats[ltri.mat].emission
                            contrib = beta.mul(Le).mul(f) * (G / max(pdf_light, EPS))
                            L = L + contrib

        # выбор события: диффузия или зеркало
        d_avg = m.diffuse.avg()
        s_avg = m.mirror.avg()
        sum_avg = d_avg + s_avg
        if sum_avg <= 0.0:
            break
        p_diff = d_avg / sum_avg
        if rng.random() < p_diff:
            # диффузный отскок
            ndir, pdf_dir = sample_cosine_hemisphere(hit.n, rng)
            if pdf_dir <= 0.0:
                break
            # при cosine-sampling множитель f*cos/pdf = diffuse
            beta = beta.mul(m.diffuse) * (1.0 / max(p_diff, EPS))
            ray = Ray(hit.p + hit.n * (1e-4), ndir)
        else:
            # зеркало
            ndir = reflect(ray.d, hit.n).norm()
            p_m = max(1.0 - p_diff, EPS)
            beta = beta.mul(m.mirror) * (1.0 / p_m)
            ray = Ray(hit.p + hit.n * (1e-4), ndir)

        # русская рулетка
        if depth >= 3:
            q = min(0.95, max(0.05, beta.max_comp()))
            if rng.random() > q:
                break
            beta = beta * (1.0 / q)

    return L


def make_camera_rays(
    cam_pos: Vec3,
    cam_look: Vec3,
    cam_up: Vec3,
    fov_deg: float,
    w: int,
    h: int,
):
    forward = (cam_look - cam_pos).norm()
    right = forward.cross(cam_up).norm()
    up = right.cross(forward).norm()
    scale = math.tan(0.5 * fov_deg * PI / 180.0)
    aspect = w / float(h)

    def gen_ray(x: float, y: float) -> Ray:
        # x,y в [0..1]
        px = (2.0 * x - 1.0) * aspect * scale
        py = (1.0 - 2.0 * y) * scale
        d = (right * px + up * py + forward).norm()
        return Ray(cam_pos, d)

    return gen_ray


def write_ppm_p3(path: str, img: List[Vec3], w: int, h: int):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"P3\n{w} {h}\n255\n")
        for y in range(h):
            for x in range(w):
                c = img[y * w + x]
                r = int(255.0 * c.x + 0.5)
                g = int(255.0 * c.y + 0.5)
                b = int(255.0 * c.z + 0.5)
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                f.write(f"{r} {g} {b} ")
            f.write("\n")


def load_scene(path: str) -> Tuple[dict, dict, List[Material], List[Triangle]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mats: List[Material] = []
    for m in data["materials"]:
        diff = Vec3.from_list(m.get("diffuse", [0, 0, 0]))
        mirr = Vec3.from_list(m.get("mirror", [0, 0, 0]))
        emis = Vec3.from_list(m.get("emission", [0, 0, 0]))

        # физичность: diff + mirror <= 1 по компонентам
        for ch in ("x", "y", "z"):
            d = getattr(diff, ch)
            s = getattr(mirr, ch)
            if d + s > 1.0 and d + s > EPS:
                k = 1.0 / (d + s)
                setattr(diff, ch, d * k)
                setattr(mirr, ch, s * k)

        mats.append(Material(diffuse=diff, mirror=mirr, emission=emis))

    tris: List[Triangle] = []
    for t in data["triangles"]:
        mat_id = int(t["mat"])
        v0 = Vec3.from_list(t["v0"])
        v1 = Vec3.from_list(t["v1"])
        v2 = Vec3.from_list(t["v2"])
        tris.append(triangle_make(v0, v1, v2, mat_id))

    return data["render"], data["camera"], mats, tris


def postprocess(img: List[Vec3], gamma: float, normalize: str) -> List[Vec3]:
    # нормировка яркости
    if normalize == "max":
        mx = 0.0
        for c in img:
            mx = max(mx, c.max_comp())
        if mx > 0.0:
            inv = 1.0 / mx
            img = [c * inv for c in img]

    inv_g = 1.0 / max(gamma, EPS)
    out: List[Vec3] = []
    for c in img:
        cc = c.clamp01()
        cc = Vec3(cc.x ** inv_g, cc.y ** inv_g, cc.z ** inv_g)
        out.append(cc)
    return out


def main():
    ap = argparse.ArgumentParser(description="Лаб 4: минимальный path tracing (треугольники, диффузия+зеркало, area lights)")
    ap.add_argument("--scene", default="lab_4/scene.json", help="Путь к scene.json")
    ap.add_argument("--out", default="lab_4/output.ppm", help="Куда сохранить PPM")
    ap.add_argument("--w", type=int, default=0, help="Ширина (0 = из сцены)")
    ap.add_argument("--h", type=int, default=0, help="Высота (0 = из сцены)")
    ap.add_argument("--spp", type=int, default=0, help="Сэмплы на пиксель (0 = из сцены)")
    ap.add_argument("--max-depth", type=int, default=0, help="Глубина (0 = из сцены)")
    ap.add_argument("--seed", type=int, default=1, help="Seed для случайных чисел")
    args = ap.parse_args()

    render, cam, mats, tris = load_scene(args.scene)
    lights = build_lights(tris, mats)

    w = int(args.w or render.get("width", 500))
    h = int(args.h or render.get("height", 500))
    spp = int(args.spp or render.get("spp", 4))
    max_depth = int(args.max_depth or render.get("max_depth", 8))
    gamma = float(render.get("gamma", 2.2))
    normalize = str(render.get("normalize", "max"))

    if w < 500 or h < 500:
        print("ВНИМАНИЕ: по заданию нужно >= 500x500 (сейчас так только для теста).")

    rng = random.Random(args.seed)

    cam_pos = Vec3.from_list(cam["pos"])
    cam_look = Vec3.from_list(cam["look_at"])
    cam_up = Vec3.from_list(cam["up"])
    fov_deg = float(cam["fov_deg"])
    gen_ray = make_camera_rays(cam_pos, cam_look, cam_up, fov_deg, w, h)

    img: List[Vec3] = [Vec3(0.0, 0.0, 0.0) for _ in range(w * h)]

    for y in range(h):
        if y % 25 == 0:
            print(f"Строка {y}/{h}")
        for x in range(w):
            acc = Vec3(0.0, 0.0, 0.0)
            for _ in range(spp):
                # антиалиасинг: случайная точка в пикселе
                u = (x + rng.random()) / float(w)
                v = (y + rng.random()) / float(h)
                ray = gen_ray(u, v)
                acc = acc + trace_path(ray, tris, mats, lights, max_depth, rng)
            img[y * w + x] = acc * (1.0 / float(spp))

    img2 = postprocess(img, gamma=gamma, normalize=normalize)
    write_ppm_p3(args.out, img2, w, h)
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()

