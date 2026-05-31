/* Лаб 4: минимальный path tracing на C (RGB, Lambert+зеркало, area lights, PPM) */
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "cJSON.h"

#define EPS 1e-6
#define PI 3.1415926

#define MAX_MATS 64
#define MAX_TRIS 512
#define MAX_LIGHTS 128
#define MAX_PATH 1024

typedef struct {
    double x, y, z;
} Vec3;

typedef struct {
    Vec3 o, d;
} Ray;

typedef struct {
    Vec3 diffuse, mirror, emission;
} Material;

typedef struct {
    Vec3 v0, v1, v2;
    int mat;
    Vec3 n;
    double area;
} Triangle;

typedef struct {
    int hit;
    double t;
    Vec3 p, n;
    int tri_id, mat_id;
} Hit;

typedef struct {
    int ids[MAX_LIGHTS];
    double cdf[MAX_LIGHTS];
    int count;
    double total;
} LightSampler;

typedef struct {
    int width, height, spp, max_depth;
    double gamma;
    char normalize[32];
} RenderCfg;

typedef struct {
    Vec3 pos, look_at, up;
    double fov_deg;
} CameraCfg;

typedef struct {
    Material mats[MAX_MATS];
    int mat_count;
    Triangle tris[MAX_TRIS];
    int tri_count;
    LightSampler lights;
    RenderCfg render;
    CameraCfg cam;
} Scene;

typedef struct {
    uint64_t state;
} Rng;

typedef struct {
    Scene scene;
    int w, h, spp, max_depth;
    uint64_t seed;
    Vec3 cam_pos, cam_fwd, cam_right, cam_upv;
    double cam_scale, aspect;
    Vec3 *img;
} RenderJob;

static Vec3 v3(double x, double y, double z) {
    Vec3 r = {x, y, z};
    return r;
}

static Vec3 v3_add(Vec3 a, Vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static Vec3 v3_sub(Vec3 a, Vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static Vec3 v3_mul(Vec3 a, double k) { return v3(a.x * k, a.y * k, a.z * k); }
static Vec3 v3_mulv(Vec3 a, Vec3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }

static double v3_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static Vec3 v3_cross(Vec3 a, Vec3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static double v3_len(Vec3 a) { return sqrt(v3_dot(a, a)); }

static Vec3 v3_norm(Vec3 a) {
    double l = v3_len(a);
    if (l < EPS) return v3(0, 0, 0);
    return v3_mul(a, 1.0 / l);
}

static double v3_maxc(Vec3 a) {
    double m = a.x;
    if (a.y > m) m = a.y;
    if (a.z > m) m = a.z;
    return m;
}

static double v3_avg(Vec3 a) { return (a.x + a.y + a.z) / 3.0; }

static Vec3 v3_clamp01(Vec3 a) {
    Vec3 r;
    r.x = a.x < 0 ? 0 : (a.x > 1 ? 1 : a.x);
    r.y = a.y < 0 ? 0 : (a.y > 1 ? 1 : a.y);
    r.z = a.z < 0 ? 0 : (a.z > 1 ? 1 : a.z);
    return r;
}

static Vec3 v3_reflect(Vec3 v, Vec3 n) {
    return v3_sub(v, v3_mul(n, 2.0 * v3_dot(v, n)));
}

static void rng_seed(Rng *r, uint64_t seed) {
    r->state = seed ? seed : 1;
}

static double rng_next(Rng *r) {
    uint64_t x = r->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->state = x;
    return (x * 2685821657736338717ULL) / (double)UINT64_C(18446744073709551615);
}

/* --- загрузка scene.json (cJSON) --- */
static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[sz] = '\0';
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

static Vec3 json_vec3(const cJSON *arr) {
    if (!cJSON_IsArray(arr) || cJSON_GetArraySize(arr) < 3) return v3(0, 0, 0);
    return v3(cJSON_GetArrayItem(arr, 0)->valuedouble, cJSON_GetArrayItem(arr, 1)->valuedouble,
              cJSON_GetArrayItem(arr, 2)->valuedouble);
}

static int json_obj_vec3(const cJSON *obj, const char *key, Vec3 *out) {
    const cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (!cJSON_IsArray(item) || cJSON_GetArraySize(item) < 3) return 0;
    *out = json_vec3(item);
    return 1;
}

static void clamp_material(Material *m) {
    double *d[3] = {&m->diffuse.x, &m->diffuse.y, &m->diffuse.z};
    double *s[3] = {&m->mirror.x, &m->mirror.y, &m->mirror.z};
    for (int i = 0; i < 3; i++) {
        if (*d[i] + *s[i] > 1.0 && *d[i] + *s[i] > EPS) {
            double k = 1.0 / (*d[i] + *s[i]);
            *d[i] *= k;
            *s[i] *= k;
        }
    }
}

static Triangle triangle_make(Vec3 v0, Vec3 v1, Vec3 v2, int mat) {
    Triangle t;
    t.v0 = v0;
    t.v1 = v1;
    t.v2 = v2;
    t.mat = mat;
    Vec3 e1 = v3_sub(v1, v0);
    Vec3 e2 = v3_sub(v2, v0);
    Vec3 nn = v3_cross(e1, e2);
    t.area = 0.5 * v3_len(nn);
    t.n = v3_norm(nn);
    return t;
}

static int load_scene(const char *path, Scene *sc) {
    size_t len;
    char *text = read_file(path, &len);
    if (!text) {
        fprintf(stderr, "Не удалось прочитать %s\n", path);
        return 0;
    }

    cJSON *root = cJSON_Parse(text);
    free(text);
    if (!root) {
        const char *err = cJSON_GetErrorPtr();
        fprintf(stderr, "Ошибка JSON в %s: %s\n", path, err ? err : "неизвестно");
        return 0;
    }

    memset(sc, 0, sizeof(*sc));

    const cJSON *render = cJSON_GetObjectItemCaseSensitive(root, "render");
    if (cJSON_IsObject(render)) {
        const cJSON *item;
        if ((item = cJSON_GetObjectItemCaseSensitive(render, "width")) && cJSON_IsNumber(item))
            sc->render.width = item->valueint;
        if ((item = cJSON_GetObjectItemCaseSensitive(render, "height")) && cJSON_IsNumber(item))
            sc->render.height = item->valueint;
        if ((item = cJSON_GetObjectItemCaseSensitive(render, "spp")) && cJSON_IsNumber(item))
            sc->render.spp = item->valueint;
        if ((item = cJSON_GetObjectItemCaseSensitive(render, "max_depth")) && cJSON_IsNumber(item))
            sc->render.max_depth = item->valueint;
        if ((item = cJSON_GetObjectItemCaseSensitive(render, "gamma")) && cJSON_IsNumber(item))
            sc->render.gamma = item->valuedouble;
        if ((item = cJSON_GetObjectItemCaseSensitive(render, "normalize")) && cJSON_IsString(item) &&
            item->valuestring)
            strncpy(sc->render.normalize, item->valuestring, sizeof(sc->render.normalize) - 1);
    }

    const cJSON *camera = cJSON_GetObjectItemCaseSensitive(root, "camera");
    if (cJSON_IsObject(camera)) {
        json_obj_vec3(camera, "pos", &sc->cam.pos);
        json_obj_vec3(camera, "look_at", &sc->cam.look_at);
        json_obj_vec3(camera, "up", &sc->cam.up);
        const cJSON *fov = cJSON_GetObjectItemCaseSensitive(camera, "fov_deg");
        if (cJSON_IsNumber(fov)) sc->cam.fov_deg = fov->valuedouble;
    }

    if (sc->render.width <= 0) sc->render.width = 500;
    if (sc->render.height <= 0) sc->render.height = 500;
    if (sc->render.spp <= 0) sc->render.spp = 4;
    if (sc->render.max_depth <= 0) sc->render.max_depth = 8;
    if (sc->render.gamma <= 0) sc->render.gamma = 2.2;
    if (sc->render.normalize[0] == '\0') strcpy(sc->render.normalize, "max");

    const cJSON *materials = cJSON_GetObjectItemCaseSensitive(root, "materials");
    if (cJSON_IsArray(materials)) {
        const cJSON *mat;
        cJSON_ArrayForEach(mat, materials) {
            if (!cJSON_IsObject(mat) || sc->mat_count >= MAX_MATS) break;
            Material m = {0};
            const cJSON *d = cJSON_GetObjectItemCaseSensitive(mat, "diffuse");
            const cJSON *mir = cJSON_GetObjectItemCaseSensitive(mat, "mirror");
            const cJSON *em = cJSON_GetObjectItemCaseSensitive(mat, "emission");
            if (cJSON_IsArray(d)) m.diffuse = json_vec3(d);
            if (cJSON_IsArray(mir)) m.mirror = json_vec3(mir);
            if (cJSON_IsArray(em)) m.emission = json_vec3(em);
            clamp_material(&m);
            sc->mats[sc->mat_count++] = m;
        }
    }

    const cJSON *triangles = cJSON_GetObjectItemCaseSensitive(root, "triangles");
    if (cJSON_IsArray(triangles)) {
        const cJSON *tri;
        cJSON_ArrayForEach(tri, triangles) {
            if (!cJSON_IsObject(tri) || sc->tri_count >= MAX_TRIS) break;
            int mat = 0;
            Vec3 v0, v1, v2;
            const cJSON *mp = cJSON_GetObjectItemCaseSensitive(tri, "mat");
            if (cJSON_IsNumber(mp)) mat = mp->valueint;
            if (!json_obj_vec3(tri, "v0", &v0) || !json_obj_vec3(tri, "v1", &v1) || !json_obj_vec3(tri, "v2", &v2))
                continue;
            sc->tris[sc->tri_count++] = triangle_make(v0, v1, v2, mat);
        }
    }

    cJSON_Delete(root);
    return sc->mat_count > 0 && sc->tri_count > 0;
}

static void build_lights(Scene *sc) {
    LightSampler *L = &sc->lights;
    L->count = 0;
    L->total = 0;
    for (int i = 0; i < sc->tri_count; i++) {
        Material *em = &sc->mats[sc->tris[i].mat];
        if (v3_maxc(em->emission) <= 0) continue;
        double power = sc->tris[i].area * (em->emission.x + em->emission.y + em->emission.z);
        if (power <= 0) continue;
        L->ids[L->count] = i;
        L->total += power;
        L->cdf[L->count] = L->total;
        L->count++;
        if (L->count >= MAX_LIGHTS) break;
    }
}

/* --- пересечение луч–треугольник (Möller–Trumbore) --- */
static int intersect_triangle(Ray ray, Triangle tri, double *out_t) {
    Vec3 e1 = v3_sub(tri.v1, tri.v0);
    Vec3 e2 = v3_sub(tri.v2, tri.v0);
    Vec3 p = v3_cross(ray.d, e2);
    double det = v3_dot(e1, p);
    if (fabs(det) < 1e-9) return 0;
    double inv = 1.0 / det;
    Vec3 tvec = v3_sub(ray.o, tri.v0);
    double u = v3_dot(tvec, p) * inv;
    if (u < 0.0 || u > 1.0) return 0;
    Vec3 q = v3_cross(tvec, e1);
    double v = v3_dot(ray.d, q) * inv;
    if (v < 0.0 || u + v > 1.0) return 0;
    double t = v3_dot(e2, q) * inv;
    if (t <= EPS) return 0;
    *out_t = t;
    return 1;
}

static Hit intersect_scene(Ray ray, Scene *sc) {
    Hit best = {0};
    best.t = 1e30;
    best.hit = 0;
    for (int i = 0; i < sc->tri_count; i++) {
        double t;
        if (!intersect_triangle(ray, sc->tris[i], &t)) continue;
        if (t < best.t) {
            best.hit = 1;
            best.t = t;
            best.tri_id = i;
            best.mat_id = sc->tris[i].mat;
            best.p = v3_add(ray.o, v3_mul(ray.d, t));
            best.n = sc->tris[i].n;
            if (v3_dot(best.n, ray.d) > 0) best.n = v3_mul(best.n, -1.0);
        }
    }
    return best;
}

static int is_occluded(Ray ray, Scene *sc, double max_dist) {
    for (int i = 0; i < sc->tri_count; i++) {
        double t;
        if (!intersect_triangle(ray, sc->tris[i], &t)) continue;
        if (t < max_dist - 1e-4) return 1;
    }
    return 0;
}

static void orthonormal_basis(Vec3 n, Vec3 *t, Vec3 *b) {
    Vec3 a = fabs(n.x) > 0.9 ? v3(0, 1, 0) : v3(1, 0, 0);
    *t = v3_norm(v3_cross(n, a));
    *b = v3_cross(*t, n);
}

static Vec3 sample_cosine_hemisphere(Vec3 n, Rng *rng) {
    double r1 = rng_next(rng);
    double r2 = rng_next(rng);
    double phi = 2.0 * PI * r1;
    double r = sqrt(r2);
    double x = r * cos(phi);
    double y = r * sin(phi);
    double z = sqrt(fmax(0.0, 1.0 - r2));
    Vec3 t, b;
    orthonormal_basis(n, &t, &b);
    return v3_norm(v3_add(v3_add(v3_mul(t, x), v3_mul(b, y)), v3_mul(n, z)));
}

static Vec3 sample_point_on_triangle(Triangle tri, Rng *rng) {
    double r1 = rng_next(rng);
    double r2 = rng_next(rng);
    double su = sqrt(r1);
    double a = 1.0 - su;
    double b = su * (1.0 - r2);
    double c = su * r2;
    return v3_add(v3_add(v3_mul(tri.v0, a), v3_mul(tri.v1, b)), v3_mul(tri.v2, c));
}

static void lights_sample(LightSampler *L, Rng *rng, int *tri_id, double *p_sel) {
    *tri_id = -1;
    *p_sel = 0;
    if (L->count <= 0 || L->total <= 0) return;
    double x = rng_next(rng) * L->total;
    int lo = 0, hi = L->count - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (x <= L->cdf[mid]) hi = mid;
        else lo = mid + 1;
    }
    *tri_id = L->ids[lo];
    double prev = lo > 0 ? L->cdf[lo - 1] : 0.0;
    *p_sel = (L->cdf[lo] - prev) / L->total;
}

static Vec3 trace_path(Ray ray0, Scene *sc, int max_depth, Rng *rng) {
    Vec3 L = v3(0, 0, 0);
    Vec3 beta = v3(1, 1, 1);
    Ray ray = ray0;

    for (int depth = 0; depth < max_depth; depth++) {
        Hit hit = intersect_scene(ray, sc);
        if (!hit.hit) break;

        Material m = sc->mats[hit.mat_id];

        if (v3_maxc(m.emission) > 0) {
            L = v3_add(L, v3_mulv(beta, m.emission));
            break;
        }

        /* прямой свет */
        if (v3_maxc(m.diffuse) > 0 && sc->lights.total > 0) {
            int ltri_id;
            double p_sel;
            lights_sample(&sc->lights, rng, &ltri_id, &p_sel);
            if (ltri_id >= 0 && p_sel > 0) {
                Triangle ltri = sc->tris[ltri_id];
                Vec3 lp = sample_point_on_triangle(ltri, rng);
                Vec3 wi = v3_sub(lp, hit.p);
                double dist2 = v3_dot(wi, wi);
                if (dist2 > EPS) {
                    double dist = sqrt(dist2);
                    wi = v3_mul(wi, 1.0 / dist);
                    double cos_x = fmax(0.0, v3_dot(hit.n, wi));
                    double cos_l = fmax(0.0, v3_dot(ltri.n, v3_mul(wi, -1.0)));
                    if (cos_x > 0 && cos_l > 0 && ltri.area > EPS) {
                        Ray shadow = {v3_add(hit.p, v3_mul(hit.n, 1e-4)), wi};
                        if (!is_occluded(shadow, sc, dist - 1e-4)) {
                            double pdf_light = p_sel * (1.0 / ltri.area);
                            double G = (cos_x * cos_l) / fmax(dist2, EPS);
                            Vec3 f = v3_mul(m.diffuse, 1.0 / PI);
                            Vec3 Le = sc->mats[ltri.mat].emission;
                            Vec3 contrib = v3_mul(v3_mulv(v3_mulv(beta, Le), f), G / fmax(pdf_light, EPS));
                            L = v3_add(L, contrib);
                        }
                    }
                }
            }
        }

        double d_avg = v3_avg(m.diffuse);
        double s_avg = v3_avg(m.mirror);
        double sum_avg = d_avg + s_avg;
        if (sum_avg <= 0) break;
        double p_diff = d_avg / sum_avg;

        if (rng_next(rng) < p_diff) {
            Vec3 ndir = sample_cosine_hemisphere(hit.n, rng);
            beta = v3_mul(v3_mulv(beta, m.diffuse), 1.0 / fmax(p_diff, EPS));
            ray.o = v3_add(hit.p, v3_mul(hit.n, 1e-4));
            ray.d = ndir;
        } else {
            Vec3 ndir = v3_norm(v3_reflect(ray.d, hit.n));
            double p_m = fmax(1.0 - p_diff, EPS);
            beta = v3_mul(v3_mulv(beta, m.mirror), 1.0 / p_m);
            ray.o = v3_add(hit.p, v3_mul(hit.n, 1e-4));
            ray.d = ndir;
        }

        if (depth >= 3) {
            double q = fmin(0.95, fmax(0.05, v3_maxc(beta)));
            if (rng_next(rng) > q) break;
            beta = v3_mul(beta, 1.0 / q);
        }
    }
    return L;
}

static Ray make_camera_ray(Vec3 pos, Vec3 fwd, Vec3 right, Vec3 upv, double scale, double aspect, double u,
                           double v) {
    double px = (2.0 * u - 1.0) * aspect * scale;
    double py = (1.0 - 2.0 * v) * scale;
    Vec3 d = v3_norm(v3_add(v3_add(v3_mul(right, px), v3_mul(upv, py)), fwd));
    Ray r = {pos, d};
    return r;
}

static void postprocess(Vec3 *img, int n, double gamma, const char *normalize) {
    if (strcmp(normalize, "max") == 0) {
        double mx = 0;
        for (int i = 0; i < n; i++) {
            double c = v3_maxc(img[i]);
            if (c > mx) mx = c;
        }
        if (mx > 0) {
            for (int i = 0; i < n; i++) img[i] = v3_mul(img[i], 1.0 / mx);
        }
    }
    double inv_g = 1.0 / fmax(gamma, EPS);
    for (int i = 0; i < n; i++) {
        Vec3 c = v3_clamp01(img[i]);
        img[i] = v3(pow(c.x, inv_g), pow(c.y, inv_g), pow(c.z, inv_g));
    }
}

static int mkdir_p(const char *path) {
    char tmp[MAX_PATH];
    snprintf(tmp, sizeof(tmp), "%s", path);
    size_t len = strlen(tmp);
    if (len == 0) return 0;
    if (tmp[len - 1] == '/') tmp[len - 1] = 0;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            if (mkdir(tmp, 0755) && errno != EEXIST) return 0;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) && errno != EEXIST) return 0;
    return 1;
}

static int write_ppm(const char *path, Vec3 *img, int w, int h) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "P3\n%d %d\n255\n", w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            Vec3 c = img[y * w + x];
            int r = (int)(255.0 * c.x + 0.5);
            int g = (int)(255.0 * c.y + 0.5);
            int b = (int)(255.0 * c.z + 0.5);
            if (r < 0) r = 0;
            if (g < 0) g = 0;
            if (b < 0) b = 0;
            if (r > 255) r = 255;
            if (g > 255) g = 255;
            if (b > 255) b = 255;
            fprintf(f, "%d %d %d ", r, g, b);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return 1;
}

static void render_image(RenderJob *job) {
    Rng rng;
    rng_seed(&rng, job->seed);
    for (int y = 0; y < job->h; y++) {
        if (y % 25 == 0) printf("Строка %d/%d\n", y, job->h);
        for (int x = 0; x < job->w; x++) {
            Vec3 acc = v3(0, 0, 0);
            for (int s = 0; s < job->spp; s++) {
                double u = (x + rng_next(&rng)) / (double)job->w;
                double v = (y + rng_next(&rng)) / (double)job->h;
                Ray ray = make_camera_ray(job->cam_pos, job->cam_fwd, job->cam_right, job->cam_upv, job->cam_scale,
                                          job->aspect, u, v);
                acc = v3_add(acc, trace_path(ray, &job->scene, job->max_depth, &rng));
            }
            job->img[y * job->w + x] = v3_mul(acc, 1.0 / job->spp);
        }
    }
}

static void setup_camera(CameraCfg *cam, int w, int h, Vec3 *pos, Vec3 *fwd, Vec3 *right, Vec3 *upv, double *scale,
                           double *aspect) {
    *pos = cam->pos;
    *fwd = v3_norm(v3_sub(cam->look_at, cam->pos));
    *right = v3_norm(v3_cross(*fwd, cam->up));
    *upv = v3_norm(v3_cross(*right, *fwd));
    *scale = tan(0.5 * cam->fov_deg * PI / 180.0);
    *aspect = w / (double)h;
}

static void usage(const char *prog) {
    fprintf(stderr,
            "Использование: %s [опции]\n"
            "  --scene PATH     JSON-сцена (по умолчанию assets/scenes/scene.json)\n"
            "  --out PATH       выходной PPM\n"
            "  --w N --h N      размер (0 = из сцены)\n"
            "  --spp N          сэмплы на пиксель\n"
            "  --max-depth N    глубина трассировки\n"
            "  --seed N         seed RNG\n",
            prog);
}

int main(int argc, char **argv) {
    char scene_path[MAX_PATH] = "assets/scenes/scene.json";
    char out_path[MAX_PATH] = "renders/output.ppm";
    int ow = 0, oh = 0, ospp = 0, odepth = 0;
    uint64_t seed = 1;

    static struct option opts[] = {
        {"scene", required_argument, 0, 's'},
        {"out", required_argument, 0, 'o'},
        {"w", required_argument, 0, 'W'},
        {"h", required_argument, 0, 'H'},
        {"spp", required_argument, 0, 'p'},
        {"max-depth", required_argument, 0, 'd'},
        {"seed", required_argument, 0, 'S'},
        {0, 0, 0, 0}};

    int c;
    while ((c = getopt_long(argc, argv, "", opts, NULL)) != -1) {
        switch (c) {
        case 's':
            strncpy(scene_path, optarg, sizeof(scene_path) - 1);
            break;
        case 'o':
            strncpy(out_path, optarg, sizeof(out_path) - 1);
            break;
        case 'W':
            ow = atoi(optarg);
            break;
        case 'H':
            oh = atoi(optarg);
            break;
        case 'p':
            ospp = atoi(optarg);
            break;
        case 'd':
            odepth = atoi(optarg);
            break;
        case 'S':
            seed = strtoull(optarg, NULL, 10);
            break;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    Scene sc;
    if (!load_scene(scene_path, &sc)) return 1;
    build_lights(&sc);

    int w = ow > 0 ? ow : sc.render.width;
    int h = oh > 0 ? oh : sc.render.height;
    int spp = ospp > 0 ? ospp : sc.render.spp;
    int max_depth = odepth > 0 ? odepth : sc.render.max_depth;

    if (w < 500 || h < 500) {
        printf("ВНИМАНИЕ: по заданию нужно >= 500x500 (сейчас так только для теста).\n");
    }

    Vec3 cam_pos, cam_fwd, cam_right, cam_upv;
    double cam_scale, aspect;
    setup_camera(&sc.cam, w, h, &cam_pos, &cam_fwd, &cam_right, &cam_upv, &cam_scale, &aspect);

    size_t npix = (size_t)w * (size_t)h;
    Vec3 *img = (Vec3 *)calloc(npix, sizeof(Vec3));
    if (!img) {
        fprintf(stderr, "Нет памяти\n");
        return 1;
    }

    clock_t t0 = clock();
    RenderJob job = {0};
    job.scene = sc;
    job.w = w;
    job.h = h;
    job.spp = spp;
    job.max_depth = max_depth;
    job.seed = seed;
    job.cam_pos = cam_pos;
    job.cam_fwd = cam_fwd;
    job.cam_right = cam_right;
    job.cam_upv = cam_upv;
    job.cam_scale = cam_scale;
    job.aspect = aspect;
    job.img = img;
    render_image(&job);
    printf("Рендер: %.3f c\n", (double)(clock() - t0) / CLOCKS_PER_SEC);

    postprocess(img, (int)npix, sc.render.gamma, sc.render.normalize);

    char dir[MAX_PATH];
    strncpy(dir, out_path, sizeof(dir) - 1);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
        mkdir_p(dir);
    }
    if (!write_ppm(out_path, img, w, h)) {
        fprintf(stderr, "Не удалось записать %s\n", out_path);
        free(img);
        return 1;
    }
    printf("Готово: %s\n", out_path);
    free(img);
    return 0;
}
