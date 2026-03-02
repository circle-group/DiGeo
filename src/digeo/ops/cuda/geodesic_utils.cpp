#define _USE_MATH_DEFINES
#include <cmath>
#include <string>
#include <tuple>

#include <torch/extension.h>

#define CT_CUDA_DEBUG 0

extern "C" PyObject* PyInit__C(void) {
    static PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C", /* name of module */
        NULL,  /* module documentation, may be NULL */
        -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        NULL, /* methods */
    };
    return PyModule_Create(&module_def);
}

template <typename T>
constexpr T epsilon();

template <>
constexpr float epsilon<float>() { return 1e-4f; }

template <>
constexpr double epsilon<double>() { return 1e-7; }

namespace geodesic_utils {

template <typename T>
struct vec3 {
    T values[3];

    inline vec3() : values{T(0), T(0), T(0)} {}

    inline vec3(T x, T y, T z) : values{x, y, z} {}

    inline void zero() {
        values[0] = T(0);
        values[1] = T(0);
        values[2] = T(0);
    }

    inline vec3<T> copy() const {
        return vec3<T>(values[0], values[1], values[2]);
    }

    inline T& operator[](int index) {
        return values[index];
    }

    inline const T& operator[](int index) const {
        return values[index];
    }

    inline vec3<T> operator+(const vec3<T>& other) const {
        return vec3<T>(values[0] + other[0], values[1] + other[1], values[2] + other[2]);
    }

    inline vec3<T> operator-(const vec3<T>& other) const {
        return vec3<T>(values[0] - other[0], values[1] - other[1], values[2] - other[2]);
    }

    inline vec3<T> operator-() const {
        return vec3<T>(-values[0], -values[1], -values[2]);
    }

    inline vec3<T> operator*(T scalar) const {
        return vec3<T>(values[0] * scalar, values[1] * scalar, values[2] * scalar);
    }

    inline vec3<T> operator/(T scalar) const {
        return vec3<T>(values[0] / scalar, values[1] / scalar, values[2] / scalar);
    }
};

template <typename T>
inline vec3<T> operator*(T scalar, const vec3<T>& v) {
    return vec3<T>(v[0] * scalar, v[1] * scalar, v[2] * scalar);
}

template <typename T>
inline T length(const vec3<T>& v) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

template <typename T>
inline vec3<T> normalize(const vec3<T>& v) {
    T l = length(v);
    if (l == 0) { // Avoid division by zero
        return v;
    }
    return v / l;
}

template <typename T>
inline vec3<T> cross(const vec3<T>& a, const vec3<T>& b) {
    return vec3<T>(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    );
}

template <typename T>
inline T dot(const vec3<T>& a, const vec3<T>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
inline vec3<T> project_vec(const vec3<T>& vec, const vec3<T>& normal) {
    vec3<T> n = normalize(normal);
    return vec - geodesic_utils::dot(vec, n) * n;
}

template <typename T>
inline vec3<T> triangle_normal(const vec3<T>& p0, const vec3<T>& p1, const vec3<T>& p2) {
    return normalize(geodesic_utils::cross(p1 - p0, p2 - p0));
}

template <typename T>
struct Mesh {
    const torch::TensorAccessor<T, 2> positions;
    const torch::TensorAccessor<int, 2> triangles;
    const torch::TensorAccessor<int, 2> adjacencies;
    const torch::TensorAccessor<T, 2> triangle_normals;
    const torch::TensorAccessor<int, 2> v2t;
    const torch::TensorAccessor<T, 2> vertex_normals;

    Mesh(
        const torch::TensorAccessor<T, 2>& positions,
        const torch::TensorAccessor<int, 2>& triangles,
        const torch::TensorAccessor<int, 2>& adjacencies,
        const torch::TensorAccessor<T, 2>& triangle_normals,
        const torch::TensorAccessor<int, 2>& v2t,
        const torch::TensorAccessor<T, 2>& vertex_normals
    ) :
        positions(positions),
        triangles(triangles),
        adjacencies(adjacencies),
        triangle_normals(triangle_normals),
        v2t(v2t),
        vertex_normals(vertex_normals) {}

    inline vec3<T> get_vertex(int vertex_id) const {
        return vec3<T>(
            positions[vertex_id][0],
            positions[vertex_id][1],
            positions[vertex_id][2]
        );
    }

    inline vec3<T> get_triangle_normal(int triangle_id) const {
        return vec3<T>(
            triangle_normals[triangle_id][0],
            triangle_normals[triangle_id][1],
            triangle_normals[triangle_id][2]
        );
    }

    inline vec3<T> get_vertex_normal(int vertex_id) const {
        return vec3<T>(
            vertex_normals[vertex_id][0],
            vertex_normals[vertex_id][1],
            vertex_normals[vertex_id][2]
        );
    }
};


template <typename T>
struct MeshPoint {
    int face;
    T u;
    T v;

    MeshPoint(int face_id, T u, T v) : face(face_id), u(u), v(v) {}

    inline vec3<T> interpolate(const Mesh<T>& mesh) const {
        vec3<T> p0 = mesh.get_vertex(mesh.triangles[face][0]);
        vec3<T> p1 = mesh.get_vertex(mesh.triangles[face][1]);
        vec3<T> p2 = mesh.get_vertex(mesh.triangles[face][2]);
        return p0 * (T(1)-u-v) + p1 * u + p2 * v;
    }

    inline vec3<T> get_barycentric_coords() const {
        return vec3<T>(
            T(1) - u - v,
            u,
            v
        );
    }
};


template <typename T>
inline vec3<T> rotate_vector(const vec3<T>& v, const T angle, const vec3<T>& n) {
    return v * cos(angle) + cross(n, v) * sin(angle) + n * dot(n, v) * (T(1.0f) - cos(angle));
}


template <typename T>
inline vec3<T> tri_bary_coords(
    const vec3<T>& p0, const vec3<T>& p1, const vec3<T>& p2, const vec3<T>& p
) {
    vec3<T> v0 = p1-p0;
    vec3<T> v1 = p2-p0;
    vec3<T> v2 = p-p0;

    T d00 = geodesic_utils::dot(v0, v0);
    T d01 = geodesic_utils::dot(v0, v1);
    T d11 = geodesic_utils::dot(v1, v1);
    T d20 = geodesic_utils::dot(v2, v0);
    T d21 = geodesic_utils::dot(v2, v1);

    T denom = d00 * d11 - d01 * d01;
    if (denom == T(0)) {
        return vec3<T>(T(1),T(0),T(0));
    }

    T v = (d11 * d20 - d01 * d21) / denom;
    T w = (d00 * d21 - d01 * d20) / denom;
    T u = T(1) - v - w;

    return vec3<T>(u, v, w);
}

template <typename T>
inline vec3<T> tri_edge_coords(
    const vec3<T>& p0, const vec3<T>& p1, const vec3<T>& p2, const vec3<T>& d
) {
    vec3<T> e1 = p1 - p0;
    vec3<T> e2 = p2 - p0;

    T d00 = dot(e1, e1);
    T d01 = dot(e1, e2);
    T d11 = dot(e2, e2);
    T d20 = dot(d, e1);
    T d21 = dot(d, e2);

    T denom = d00 * d11 - d01 * d01;

    if (denom == 0)
        return vec3<T>();

    T a = (d11 * d20 - d01 * d21) / denom;
    T b = (d00 * d21 - d01 * d20) / denom;

    return vec3<T>(-a-b, a, b);
}

template <typename T>
inline void bary_is_edge(const vec3<T>& bary, const T eps, bool* is_edge, int* edge_idx_out) {
    if (bary[0] < eps) {
        *edge_idx_out = 1;
        *is_edge = true;
        return;
    }
    if (bary[1] < eps) {
        *edge_idx_out = 2;
        *is_edge = true;
        return;
    }
    if (bary[2] < eps) {
        *edge_idx_out = 0;
        *is_edge = true;
        return;
    }
    *edge_idx_out = -1;
    *is_edge = false;
}

template <typename T>
inline void bary_is_vert(const vec3<T>& bary, const T eps, bool* is_vert, int* vert_idx_out) {
    if (bary[0] > T(1) - eps) {
        *vert_idx_out = 0;
        *is_vert = true;
        return;
    }
    if (bary[1] > T(1) - eps) {
        *vert_idx_out = 1;
        *is_vert = true;
        return;
    }
    if (bary[2] > T(1) - eps) {
        *vert_idx_out = 2;
        *is_vert = true;
        return;
    }
    *vert_idx_out = -1;
    *is_vert = false;
}


template <typename T>
inline void trace_around_hole(
    const Mesh<T>& mesh, const vec3<T>& dir_3d,
    const vec3<T>& curr_bary, const vec3<T>& curr_pos,
    int curr_tri, float max_len, const T eps,
    vec3<T>& next_pos, vec3<T>& next_bary
) {
    int vertex_idx = -1;
    int edge_idx = -1;
    for (int i=0; i < 3; i++) {
        if (curr_bary[i] > T(1)-eps) {
            vertex_idx = i;
            break;
        }
        else if (curr_bary[i] < eps) {
            edge_idx = i;
        }
    }

    int next_vertex = -1;
    if (vertex_idx != -1) {
        int hole_edge = -1;
        for (int k=0; k<3; k++) {
            if (mesh.adjacencies[curr_tri][k] == -1) {
                hole_edge = k;
            }
        }

        if (hole_edge == -1) {
            // No hole edge found, just return the current position and barycentric coordinates
            next_pos = curr_pos;
            next_bary = curr_bary;
            return;
        }

        int edge_map[3][2] =  {{0,1},{1,2},{2,0}};
        next_vertex = (edge_map[hole_edge][0] == vertex_idx) ? edge_map[hole_edge][1] : edge_map[hole_edge][0];
    }
    else if (edge_idx != -1) { // Select arbitrary next vertex on the edge
        int next_vertex_map[3] = {1,2,0};
        next_vertex = next_vertex_map[edge_idx];
    }
    else {
        next_vertex = vertex_idx;
    }

    vec3<T> dir = mesh.get_vertex(mesh.triangles[curr_tri][next_vertex]) - curr_pos;
    T length_dir = length(dir);
    if (length_dir > max_len) {
        dir = dir/length_dir * max_len;
    }
    next_pos = curr_pos + dir;
    next_bary.zero();
    next_bary[next_vertex] = 1.0;
}


template <typename T>
inline void trace_in_triangles(
    const Mesh<T>& mesh, const vec3<T>& dir_3d,
    const vec3<T>& curr_bary, const vec3<T>& curr_pos,
    int curr_tri, T max_len, const T eps,
    vec3<T>& next_pos, vec3<T>& next_bary
) {
    vec3<T> p0 = mesh.get_vertex(mesh.triangles[curr_tri][0]);
    vec3<T> p1 = mesh.get_vertex(mesh.triangles[curr_tri][1]);
    vec3<T> p2 = mesh.get_vertex(mesh.triangles[curr_tri][2]);
    vec3<T> dir_bary = tri_edge_coords(p0, p1, p2, dir_3d);

    T best_t = T(10000);

    for (int i = 0; i < 3; i++) {
        if (dir_bary[i] < -eps) {
            T t = -curr_bary[i] / dir_bary[i];
            if (t > 0 && t < best_t) {
                best_t = t;
            }
        }
    }

    if (best_t > max_len) {
        best_t = max_len;
    }

    next_pos = curr_pos + best_t * dir_3d;
    next_bary = curr_bary + best_t * dir_bary;

    // check if point is valid
    for (int i=0; i < 3; i++) {
        if (next_bary[i] < -eps) {
            next_pos = curr_pos.copy();
            next_bary = curr_bary.copy();
            return;
        }
    }

    bool is_vertex = int(next_bary[0]<eps) + int(next_bary[1]<eps) + int(next_bary[2]<eps) == 2;

    if (is_vertex) {
        for (int i=0; i < 3; i++) {
            if (next_bary[i] > 0.5) {
                next_bary[i] = T(1);
                next_pos = mesh.get_vertex(mesh.triangles[curr_tri][i]);
            }
            else {
                next_bary[i] = T(0);
            }
        }
    }
}


template <typename T>
inline void common_edge(
    const Mesh<T>& mesh, int tri1, int tri2,
    int common_verts[2], int& diff_vert_1, int& diff_vert_2, bool& valid
) {
    int verts1[3] = {
        mesh.triangles[tri1][0],
        mesh.triangles[tri1][1],
        mesh.triangles[tri1][2]
    };
    int verts2[3] = {
        mesh.triangles[tri2][0],
        mesh.triangles[tri2][1],
        mesh.triangles[tri2][2]
    };

    // Find common vertices
    int common_count = 0;
    bool common_flags1[3] = {false, false, false};
    bool common_flags2[3] = {false, false, false};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (verts1[i] == verts2[j]) {
                if (common_count < 2) {
                    common_verts[common_count] = verts1[i];
                }
                common_flags1[i] = true;
                common_flags2[j] = true;
                common_count++;
            }
        }
    }

    if (common_count != 2) {
        valid = false;
        diff_vert_1 = -1;
        diff_vert_2 = -1;
        return;
    }

    // Find unique verts (not common)
    diff_vert_1 = -1;
    diff_vert_2 = -1;

    for (int i = 0; i < 3; i++) {
        if (!common_flags1[i]) diff_vert_1 = verts1[i];
        if (!common_flags2[i]) diff_vert_2 = verts2[i];
    }

    valid = true;
}

template <typename T>
inline T signed_angle(const vec3<T>& A, const vec3<T>& B, const vec3<T>& N) {
    vec3<T> a = A - geodesic_utils::dot(A, N) * N;
    vec3<T> b = B - geodesic_utils::dot(B, N) * N;

    T lenA = length(a);
    T lenB = length(b);

    if (lenA == T(0) || lenB == T(0)) return T(0);

    a = a / lenA;
    b = b / lenB;

    vec3<T> cross_prod = geodesic_utils::cross(A, B);
    T dot_prod = geodesic_utils::dot(A, B);
    T sign = geodesic_utils::dot(N, cross_prod);

    T angle = atan2(sign, dot_prod); // result in radians
    return angle;
}

template <typename T>
inline vec3<T> get_next_bary(const Mesh<T>& mesh, int curr_tri, int next_tri, const vec3<T>& curr_bary, int* common) {
    if (curr_tri == next_tri) {
        return curr_bary.copy();
    }

    vec3<T> next_bary = vec3<T>();

    for (int i = 0; i < 3; i++) {
        int v = mesh.triangles[curr_tri][i];
        if (v == common[0] || v == common[1]) {
            // find index of v in next_triangle
            for (int j = 0; j < 3; j++) {
                if (v == mesh.triangles[next_tri][j]) {
                    next_bary[j] = curr_bary[i];
                    break;
                }
            }
        }
    }

    return next_bary;
}

template <typename T>
inline void compute_parallel_transport_edge(
    const Mesh<T>& mesh,
    int curr_tri,
    int next_tri,
    const vec3<T>& dir_3d,
    const vec3<T>& curr_normal,
    vec3<T>& out_dir,
    vec3<T>& out_normal
) {
    if (curr_tri == next_tri) {
        out_dir = dir_3d;
        out_normal = curr_normal;
        return;
    }

    int common_e[2];
    int other_v1, other_v2;
    bool valid_edge;
    common_edge(mesh, curr_tri, next_tri, common_e, other_v1, other_v2, valid_edge);

    if (!valid_edge) {
        out_dir = dir_3d;
        out_normal = curr_normal;
        return;
    }

    vec3<T> p0 = mesh.get_vertex(common_e[0]);
    vec3<T> p1 = mesh.get_vertex(common_e[1]);
    vec3<T> p2_curr = mesh.get_vertex(other_v1);
    vec3<T> p2_next = mesh.get_vertex(other_v2);

    vec3<T> n1 = triangle_normal(p0, p1, p2_curr);
    vec3<T> n2 = triangle_normal(p1, p0, p2_next);

    vec3<T> edge_dir = p1 - p0;
    vec3<T> axis = normalize(edge_dir);

    T angle = signed_angle(n1, n2, axis);

    out_dir = rotate_vector(dir_3d, angle, axis);

    T normal_sign = geodesic_utils::dot(n1, curr_normal);
    out_normal = normal_sign * n2;
}


template <typename T>
inline void compute_parallel_transport_vertex_hole(
    const Mesh<T>& mesh,
    int curr_tri,
    int vertex_id,
    const vec3<T>& dir_3d,
    const vec3<T>& curr_normal,
    const T eps,
    vec3<T>& out_dir,
    int& out_tri,
    vec3<T>& out_normal,
    bool& is_near_hole
) {
    int best_tri_final = -1;
    T best_cos_final = T(-1);
    int best_tri = -1;
    T best_cos = T(-1);
    int len_connected_triangles = mesh.v2t[vertex_id][0];
    vec3<T> n, dir_proj;

    for (int tri_idx = 1; tri_idx <= len_connected_triangles; tri_idx++) {
        int tri = mesh.v2t[vertex_id][tri_idx];
        if (tri == curr_tri) {
            continue;
        }

        // Get the vertices of the triangle
        int vertices[2] = {0,0};
        int idx = 0;
        for (int v_idx = 0; v_idx < 3; v_idx++) {
            int v = mesh.triangles[tri][v_idx];
            if (v != vertex_id) {
                vertices[idx] = v;
                idx++;
            }
        }

        vec3<T> p0 = mesh.get_vertex(vertex_id);
        vec3<T> p1 = mesh.get_vertex(vertices[0]);
        vec3<T> p2 = mesh.get_vertex(vertices[1]);
        vec3<T> e1 = normalize(p1 - p0);
        vec3<T> e2 = normalize(p2 - p0);

        n = mesh.get_triangle_normal(tri);
        dir_proj = normalize(project_vec(dir_3d, n));
        T cos = dot(dir_proj, dir_3d);

        if (
            dot(cross(dir_proj, e1), cross(dir_proj, e2)) <= 0 &&
            acos(dot(dir_proj, e1)) + acos(dot(dir_proj, e2)) < T(M_PI) - eps
        ) {
            if (cos > best_cos_final) {
                best_cos_final = cos;
                best_tri_final = tri;
            }
        }
        else if (cos > best_cos) {
            best_tri = tri;
            best_cos = cos;
        }
    }

    bool is_near_hole_flag = false;
    if (best_tri_final == -1) {
        // No valid triangle found, use next edge close to hole
        int edges[3][2] = {{0,1},{1,2},{2,0}};
        for (int tri_idx = 1; tri_idx <= len_connected_triangles; tri_idx++) {
            int tri = mesh.v2t[vertex_id][tri_idx];
            if (tri == curr_tri) {
                continue;
            }

            for (int k=0; k<3; k++) {
                int v1 = mesh.triangles[tri][edges[k][0]];
                int v2 = mesh.triangles[tri][edges[k][1]];
                if (v1 != vertex_id && v2 != vertex_id) {
                    continue;
                }

                if (mesh.adjacencies[tri][k] == -1) {
                    best_tri_final = tri;
                    is_near_hole_flag = true;
                    break;
                }
            }

            if (is_near_hole_flag) {
                break;
            }
        }
        // No valid triangle found, select best triangle instead
        if (best_tri_final == -1) {
            best_tri_final = best_tri;
        }
    }

    n = mesh.get_triangle_normal(best_tri_final);
    dir_proj = normalize(project_vec(dir_3d, n));

    out_dir = dir_proj;
    out_tri = best_tri_final;
    out_normal = n;
    is_near_hole = is_near_hole_flag;
}


template <typename T>
inline void compute_parallel_transport_vertex(
    const Mesh<T>& mesh,
    int curr_tri,
    int vertex_id,
    const vec3<T>& dir_3d,
    const vec3<T>& curr_normal,
    const T eps,
    vec3<T>& out_dir,
    int& out_tri,
    vec3<T>& out_normal
) {
    int len_connected_triangles = mesh.v2t[vertex_id][0];
    // v2t layout: [count, tri_id1, tri_id2, ...]

    T total_angle = T(0);

    // Sum absolute angles around the vertex
    for (int idx = 1; idx <= len_connected_triangles; idx++) {
        int tri_id = mesh.v2t[vertex_id][idx];

        // Find the two other vertices in this triangle that are not vertex_id
        int verts[2];
        int vcount = 0;
        for (int i = 0; i < 3; i++) {
            int v = mesh.triangles[tri_id][i];
            if (v != vertex_id) {
                verts[vcount++] = v;
            }
        }

        vec3<T> vec0 = mesh.get_vertex(verts[0]) - mesh.get_vertex(vertex_id);
        vec3<T> vec1 = mesh.get_vertex(verts[1]) - mesh.get_vertex(vertex_id);
        vec3<T> n = mesh.get_triangle_normal(tri_id);

        T angle = abs(signed_angle(vec0, vec1, n));
        total_angle += angle;
    }

    T half_angle = total_angle * T(0.5);

    vec3<T> dir = -dir_3d;

    // Find first v1 vertex in current triangle different from vertex_id
    int v1 = -1;
    for (int i = 0; i < 3; i++) {
        int v = mesh.triangles[curr_tri][i];
        if (v != vertex_id) {
            v1 = v;
            break;
        }
    }

    // Find v2 vertex different from vertex_id and v1
    int v2 = -1;
    for (int i = 0; i < 3; i++) {
        int v = mesh.triangles[curr_tri][i];
        if (v != vertex_id && v != v1) {
            v2 = v;
            break;
        }
    }

    vec3<T> p0 = mesh.get_vertex(vertex_id);
    vec3<T> p1 = mesh.get_vertex(v1);
    vec3<T> p2 = mesh.get_vertex(v2);

    vec3<T> n = triangle_normal(p0, p1, p2);

    T normal_sign = (geodesic_utils::dot(n, curr_normal) > 0) ? T(1) : T(-1);

    T angle = signed_angle(dir, p1 - p0, n);

    if (angle > half_angle) {
        angle = signed_angle(
            p2-p0,
            dir_3d,
            n
        );
        int temp = v1;
        v1 = v2;
        v2 = temp;
    }

    while (angle < half_angle) {
        // Find v2 in next_tri different from v1 and vertex_id
        v2 = -1;
        for (int i = 0; i < 3; i++) {
            int v = mesh.triangles[curr_tri][i];
            if (v != v1 && v != vertex_id) {
                v2 = v;
                break;
            }
        }

        p0 = mesh.get_vertex(vertex_id);
        p1 = mesh.get_vertex(v1);
        p2 = mesh.get_vertex(v2);

        n = triangle_normal(p0, p1, p2);

        T tri_angle = abs(signed_angle(p1 - p0, p2 - p0, n));

        if (angle + tri_angle >= half_angle - eps) {
            T angle_diff = half_angle - angle;
            vec3<T> axis = n;
            vec3<T> edge = normalize(p1 - p0);
            vec3<T> new_dir = rotate_vector(edge, angle_diff, axis);

            out_dir = new_dir;
            out_tri = curr_tri;
            out_normal = normal_sign * n;
            return;
        }

        // Find local_edge_idx: the index in curr_tri of the vertex != v1 and != vertex_id, then +1 % 3
        int local_edge_idx = -1;
        for (int i = 0; i < 3; i++) {
            int v = mesh.triangles[curr_tri][i];
            if (v != v2 && v != vertex_id) {
                local_edge_idx = (i + 1) % 3;
                break;
            }
        }

        int next_tri = mesh.adjacencies[curr_tri][local_edge_idx];
        if (next_tri == -1) {
            // No adjacent triangle, rotate and return
            T angle_diff = half_angle - angle;
            vec3<T> axis = n;
            vec3<T> edge = normalize(p1 - p0);
            vec3<T> new_dir = rotate_vector(edge, angle_diff, axis);

            out_dir = new_dir;
            out_tri = curr_tri;
            out_normal = normal_sign * n;
            return;
        }

        curr_tri = next_tri;
        v1 = v2;
        angle += tri_angle;
    }

    // Fallback return zero vector, 0, and curr_normal
    out_dir = vec3<T>();
    out_tri = 0;
    out_normal = curr_normal;
}


template <typename T>
inline void project_vertex_dir(
    const Mesh<T>& mesh, int vertex_id, vec3<T>& dir,
    int& new_tri, vec3<T>& next_dir
) {
    vec3<T> v_normal = mesh.get_vertex_normal(vertex_id);

    int len_connected_triangles = mesh.v2t[vertex_id][0];
    T total_angle = T(0);
    T best_proj = T(1000000);
    int best_tri = -1;
    int best_v = -1;
    vec3<T> v_pos = mesh.get_vertex(vertex_id);

    for (int tri_k = 1; tri_k <= len_connected_triangles; tri_k++) {
        int tri_id = mesh.v2t[vertex_id][tri_k];
        int v1=0, v2=0;
        for (int k=0; k<3; k++) {
            if (mesh.triangles[tri_id][k] == vertex_id) {
                v1 = mesh.triangles[tri_id][(k + 1) % 3];
                v2 = mesh.triangles[tri_id][(k + 2) % 3];
                break;
            }
        }
        vec3<T> e1 = mesh.get_vertex(v1) - v_pos;
        vec3<T> e2 = mesh.get_vertex(v2) - v_pos;
        e1 = normalize(e1);
        e2 = normalize(e2);

        total_angle += acos(dot(e1, e2));

        T proj_loss = abs(dot(e1, v_normal));
        if (proj_loss < best_proj) {
            best_proj = proj_loss;
            best_tri = tri_id;
            best_v = v1;
        }

        proj_loss = abs(dot(e2, v_normal));
        if (proj_loss < best_proj) {
            best_proj = proj_loss;
            best_tri = tri_id;
            best_v = v2;
        }
    }

    vec3<T> mesh_edge = mesh.get_vertex(best_v) - v_pos;
    vec3<T> tangent_edge = mesh_edge - dot(mesh_edge, v_normal) * v_normal;

    T tangent_angle = fmod(signed_angle(tangent_edge, dir, v_normal), 2 * T(M_PI));
    if (tangent_angle < 0) tangent_angle += 2 * T(M_PI);
    T mesh_angle = (tangent_angle / (2*T(M_PI))) * total_angle;

    T angle = T(0.0);
    int tri = best_tri;
    int v1 = best_v;
    int v2 = 0;
    for (int k=0; k<3; k++) {
        int v_k = mesh.triangles[tri][k];
        if (v_k != vertex_id && v_k != v1) {
            v2 = v_k;
            break;
        }
    }
    vec3<T> e1 = mesh.get_vertex(v1) - v_pos;
    vec3<T> e2 = mesh.get_vertex(v2) - v_pos;
    e1 = normalize(e1);
    e2 = normalize(e2);

    while (angle < mesh_angle) {
        T tri_angle = acos(dot(e1, e2));

        if (angle + tri_angle >= mesh_angle) {
            vec3<T> axis = normalize(cross(e1, e2));
            new_tri = tri;
            next_dir = rotate_vector(e1, mesh_angle - angle, axis);
            return;
        }

        angle += tri_angle;
        int next_tri=0;
        for (int k=0; k<3; k++) {
            if (mesh.triangles[tri][k] == v1) {
                next_tri = mesh.adjacencies[tri][(k+1)%3];
            }
        }

        if (next_tri == -1) {
            new_tri = tri;
            next_dir = normalize(e2);
            return;
        }

        tri = next_tri;
        v1 = v2;
        for (int k=0; k<3; k++) {
            int v_k = mesh.triangles[tri][k];
            if (v_k != vertex_id && v_k != v1) {
                v2 = v_k;
                break;
            }
        }
        e1 = e2;
        e2 = normalize(mesh.get_vertex(v2) - v_pos);
    }
    new_tri = tri;
    next_dir = normalize(mesh.get_vertex(v2)-v_pos);
}


template <typename T>
void straightest_geodesic_single(
    int i,
    const Mesh<T>& mesh,
    const torch::TensorAccessor<int, 1> start_face,
    const torch::TensorAccessor<T, 2> start_uvs,
    const torch::TensorAccessor<T, 2> start_dirs,
    const int64_t max_steps,
    const bool debug,
    const bool print_warnings,
    T eps,
    bool avoid_holes,
    torch::TensorAccessor<T, 3> final_results,
    torch::TensorAccessor<T, 4> debug_tensor
) {
    if (eps < 0) {
        eps = epsilon<T>();
    }

    MeshPoint<T> curr_point(
        start_face[i],
        start_uvs[i][0],
        start_uvs[i][1]
    );
    int curr_tri = curr_point.face;

    vec3<T> dir = vec3<T>(
        start_dirs[i][0],
        start_dirs[i][1],
        start_dirs[i][2]
    );

    bool is_vert_bary;
    int vert_idx;
    vec3<T> curr_bary = curr_point.get_barycentric_coords();
    bary_is_vert(curr_bary, eps, &is_vert_bary, &vert_idx);
    vec3<T> current_normal;
    T max_len_path;
    int v_id = mesh.triangles[curr_tri][vert_idx];
    if (is_vert_bary) {
        current_normal = mesh.get_vertex_normal(v_id);
        dir = project_vec(dir, current_normal);
        max_len_path = length(dir);
        vec3<T> new_dir;
        project_vertex_dir(mesh, v_id, dir, curr_tri, new_dir);
        dir = new_dir;
        T uv[2];
        for (int k=1;k<3;k++) {
            if (mesh.triangles[curr_tri][k] == v_id) {
                uv[k-1] = T(1.0f);
            }
            else {
                uv[k-1] = T(0.0f);
            }
        }
        curr_point = MeshPoint<T>(
            curr_tri,
            uv[0],
            uv[1]
        );
        curr_bary = curr_point.get_barycentric_coords();
        current_normal = mesh.get_triangle_normal(curr_tri);
    } else {
        current_normal = mesh.get_triangle_normal(curr_point.face);
        dir = project_vec(dir, current_normal);
        max_len_path = length(dir);
    }

    T len_path = T(0.0);
    dir = normalize(dir);

    vec3<T> curr_pos = curr_point.interpolate(mesh);
    vec3<T> curr_normal = current_normal;
    vec3<T> next_pos = vec3<T>();
    vec3<T> next_bary = vec3<T>();

    int next_tri = -1;
    bool is_near_hole = false;
    int vertex_crossings = 0;

    if (avoid_holes) {
        bool is_edge_bary;
        int edge_idx;
        bary_is_edge(curr_bary, eps, &is_edge_bary, &edge_idx);

        bary_is_vert(curr_bary, eps, &is_vert_bary, &vert_idx);
        if (is_vert_bary) {
            for (int k = 0; k < 3; k++) {
                if (k != vert_idx+1) {
                    if (mesh.adjacencies[curr_tri][k] == -1) {
                        is_near_hole = true;
                        break;
                    }
                }
            }
        }
        else if (is_edge_bary) {
            // Check if the edge is adjacent to a hole
            int edge_local_idx = edge_idx - 1;
            if (mesh.adjacencies[curr_tri][edge_local_idx] == -1) {
                is_near_hole = true;
            }
        }
    }

    if (debug) {
        debug_tensor[i][0][1][0] = curr_pos[0];
        debug_tensor[i][0][1][1] = curr_pos[1];
        debug_tensor[i][0][1][2] = curr_pos[2];
        debug_tensor[i][1][1][0] = dir[0];
        debug_tensor[i][1][1][1] = dir[1];
        debug_tensor[i][1][1][2] = dir[2];
        debug_tensor[i][2][1][0] = current_normal[0];
        debug_tensor[i][2][1][1] = current_normal[1];
        debug_tensor[i][2][1][2] = current_normal[2];
    }

    int step = 0;

    while (len_path < max_len_path - eps) {
        step++;
        if (step > max_steps) {
            if (print_warnings) {
                printf("Warning: Exceeded maximum steps in straightest_geodesic.\n");
                printf("UV: [%.9g, %.9g], face: %d, direction: [%.9g, %.9g, %.9g]\n",
                    curr_point.u, curr_point.v, curr_point.face,
                    start_dirs[i][0], start_dirs[i][1], start_dirs[i][2]);
                printf("Start UV: [%.9g, %.9g], face: %d\n", start_uvs[i][0], start_uvs[i][1], start_face[i]);
            }
            break;
        }

        if (length(dir) < eps)
            break;

        // Trace the ray in the current triangle
        if (is_near_hole) {
            trace_around_hole(mesh, dir, curr_bary, curr_pos, curr_tri, max_len_path - len_path, eps, next_pos, next_bary);
        }
        else {
            trace_in_triangles(mesh, dir, curr_bary, curr_pos, curr_tri, max_len_path - len_path, eps, next_pos, next_bary);
        }

        bool is_edge_bary;
        int edge_idx;
        bary_is_edge(next_bary, eps, &is_edge_bary, &edge_idx);

        bool is_vert_bary;
        int vert_idx;
        bary_is_vert(next_bary, eps, &is_vert_bary, &vert_idx);

        len_path += length(next_pos - curr_pos);

        # if CT_CUDA_DEBUG
            printf("Current triangle: %d, next position: [%f, %f, %f], next bary: [%f, %f, %f], dir: [%f, %f, %f]\n",
                curr_tri,
                next_pos[0], next_pos[1], next_pos[2],
                next_bary[0], next_bary[1], next_bary[2],
                dir[0], dir[1], dir[2]
            );
        # endif

        if (is_vert_bary) {
            vertex_crossings++;
            int v_idx = mesh.triangles[curr_tri][vert_idx];

            bool hole_parallel_transport = false;
            if (is_near_hole) {
                hole_parallel_transport = true;
                compute_parallel_transport_vertex_hole(
                    mesh, curr_tri, v_idx, dir, curr_normal, eps,
                    dir, next_tri, current_normal, is_near_hole
                );
            }
            else {
                compute_parallel_transport_vertex(
                    mesh, curr_tri, v_idx, dir, curr_normal, eps,
                    dir, next_tri, current_normal
                );
            }

            // Determine closest vertex in next_tri
            vec3<T> p0 = mesh.get_vertex(mesh.triangles[next_tri][0]);
            vec3<T> p1 = mesh.get_vertex(mesh.triangles[next_tri][1]);
            vec3<T> p2 = mesh.get_vertex(mesh.triangles[next_tri][2]);
            vec3<T> d0 = next_pos - p0;
            vec3<T> d1 = next_pos - p1;
            vec3<T> d2 = next_pos - p2;
            T dists[3] = {
                geodesic_utils::dot(d0, d0),
                geodesic_utils::dot(d1, d1),
                geodesic_utils::dot(d2, d2)
            };

            // Take argmin of distances
            int vert_idx = -1;
            next_bary[0] = next_bary[1] = next_bary[2] = T(0);
            if (dists[0] < dists[1] && dists[0] < dists[2]) {
                vert_idx = 0;
            }
            else if (dists[1] < dists[0] && dists[1] < dists[2]) {
                vert_idx = 1;
            }
            else {
                vert_idx = 2;
            }
            next_bary[vert_idx] = T(1);

            // Check if adjacent to a hole
            if (avoid_holes && !hole_parallel_transport) {
                for (int k = 0; k < 3; k++) {
                    if (k != vert_idx+1 && mesh.adjacencies[next_tri][k] == -1) {
                        is_near_hole = true;
                        break;
                    }
                }
            }

            curr_tri = next_tri;
            curr_bary = next_bary;
            curr_pos = next_pos;
            curr_point = MeshPoint<T>(curr_tri, curr_bary[1], curr_bary[2]);
            curr_normal = current_normal;
        }
        else if (is_edge_bary) {
            int adj_tri = mesh.adjacencies[curr_tri][edge_idx];

            if (adj_tri == -1) {
                is_near_hole = true;
                if (!avoid_holes) {
                    adj_tri = curr_tri;
                    curr_point = MeshPoint<T>(curr_tri, next_bary[1], next_bary[2]);
                    break;
                }
                else {
                    adj_tri = curr_tri;
                }
            }
            else {
                int e[2];
                int diff_vert_1, diff_vert_2;
                bool found;
                common_edge(mesh, curr_tri, adj_tri, e, diff_vert_1, diff_vert_2, found);
                if (!found) {
                    curr_point = MeshPoint<T>(adj_tri, next_bary[1], next_bary[2]);
                    break;
                }

                next_bary = get_next_bary(mesh, curr_tri, adj_tri, next_bary, e);

                compute_parallel_transport_edge(
                    mesh, curr_tri, adj_tri, dir, curr_normal,
                    dir, current_normal
                );
            }

            curr_tri = adj_tri;
            curr_bary = next_bary;
            curr_pos = next_pos;
            curr_point = MeshPoint<T>(curr_tri, curr_bary[1], curr_bary[2]);
            curr_normal = current_normal;
        }
        else {
            curr_pos = next_pos;
            curr_bary = next_bary;
            curr_point = MeshPoint<T>(curr_tri, curr_bary[1], curr_bary[2]);
            curr_normal = current_normal;
        }
        if (debug) {
            debug_tensor[i][0][step+1][0] = curr_pos[0];
            debug_tensor[i][0][step+1][1] = curr_pos[1];
            debug_tensor[i][0][step+1][2] = curr_pos[2];
            debug_tensor[i][1][step+1][0] = dir[0];
            debug_tensor[i][1][step+1][1] = dir[1];
            debug_tensor[i][1][step+1][2] = dir[2];
            debug_tensor[i][2][step+1][0] = current_normal[0];
            debug_tensor[i][2][step+1][1] = current_normal[1];
            debug_tensor[i][2][step+1][2] = current_normal[2];
        }
    }

    # if CT_CUDA_DEBUG
        printf("Current triangle: %d, next position: [%f, %f, %f], next bary: [%f, %f, %f], dir: [%f, %f, %f]\n",
            curr_tri,
            next_pos[0], next_pos[1], next_pos[2],
            next_bary[0], next_bary[1], next_bary[2],
            dir[0], dir[1], dir[2]
        );
    # endif

    if (debug) {
        debug_tensor[i][0][0][0] = step;
        debug_tensor[i][0][0][1] = vertex_crossings;
    }

    final_results[i][0][0] = T(curr_point.face);
    final_results[i][0][1] = curr_point.u;
    final_results[i][0][2] = curr_point.v;
    final_results[i][1][0] = dir[0];
    final_results[i][1][1] = dir[1];
    final_results[i][1][2] = dir[2];
    final_results[i][2][0] = curr_normal[0];
    final_results[i][2][1] = curr_normal[1];
    final_results[i][2][2] = curr_normal[2];
}

template <typename scalar_t>
std::tuple<torch::Tensor, torch::Tensor> straightest_geodesics_impl(
    const torch::Tensor& mesh_positions,
    const torch::Tensor& mesh_triangles,
    const torch::Tensor& mesh_adjacencies,
    const torch::Tensor& mesh_triangle_normals,
    const torch::Tensor& mesh_v2t,
    const torch::Tensor& mesh_vertex_normals,
    const torch::Tensor& start_uv,
    const torch::Tensor& start_dir,
    const torch::Tensor& start_face,
    const int64_t max_steps,
    const bool debug,
    const bool print_warnings,
    scalar_t eps,
    bool avoid_holes
) {
    geodesic_utils::Mesh<scalar_t> mesh(
        mesh_positions.accessor<scalar_t, 2>(),
        mesh_triangles.accessor<int, 2>(),
        mesh_adjacencies.accessor<int, 2>(),
        mesh_triangle_normals.accessor<scalar_t, 2>(),
        mesh_v2t.accessor<int, 2>(),
        mesh_vertex_normals.accessor<scalar_t, 2>()
    );

    int num_starts = start_uv.size(0);

    torch::Tensor final_results = torch::empty({num_starts, 3, 3}, start_uv.options());
    torch::Tensor debug_tensor;
    if (debug) {
        debug_tensor = torch::empty({num_starts, 3, max_steps+1, 3}, start_uv.options());
    }
    else {
        debug_tensor = torch::empty({1,1,1,1}, start_uv.options());
    }

    for (int i=0; i < num_starts; i++) {
        straightest_geodesic_single<scalar_t>(
            i,
            mesh,
            start_face.accessor<int, 1>(),
            start_uv.accessor<scalar_t, 2>(),
            start_dir.accessor<scalar_t, 2>(),
            max_steps,
            debug,
            print_warnings,
            eps,
            avoid_holes,
            final_results.accessor<scalar_t, 3>(),
            debug_tensor.accessor<scalar_t, 4>()
        );
    }
    return std::make_tuple(final_results, debug_tensor);
}

std::tuple<torch::Tensor, torch::Tensor> straightest_geodesics(
    const torch::Tensor& mesh_positions,
    const torch::Tensor& mesh_triangles,
    const torch::Tensor& mesh_adjacencies,
    const torch::Tensor& mesh_triangle_normals,
    const torch::Tensor& mesh_v2t,
    const torch::Tensor& mesh_vertex_normals,
    const torch::Tensor& start_uv,
    const torch::Tensor& start_dir,
    const torch::Tensor& start_face,
    const int64_t max_steps,
    const bool debug,
    const bool print_warnings,
    double eps,
    bool avoid_holes
) {
    TORCH_CHECK(mesh_positions.scalar_type() == start_uv.scalar_type(), "Inconsistent types");

    if (mesh_positions.scalar_type() == torch::kFloat) {
        return straightest_geodesics_impl<float>(
            mesh_positions, mesh_triangles, mesh_adjacencies,
            mesh_triangle_normals, mesh_v2t, mesh_vertex_normals,
            start_uv, start_dir, start_face,
            max_steps, debug, print_warnings, float(eps), avoid_holes
        );
    } else if (mesh_positions.scalar_type() == torch::kDouble) {
        return straightest_geodesics_impl<double>(
            mesh_positions, mesh_triangles, mesh_adjacencies,
            mesh_triangle_normals, mesh_v2t, mesh_vertex_normals,
            start_uv, start_dir, start_face,
            max_steps, debug, print_warnings, eps, avoid_holes
        );
    } else {
        TORCH_CHECK(false, "Unsupported tensor dtype (expected float or double)");
    }
}

TORCH_LIBRARY(geodesic_utils, m) {
    m.def(
        R"(straightest_geodesics(
            Tensor mesh_positions,
            Tensor mesh_triangles,
            Tensor mesh_adjacencies,
            Tensor mesh_triangle_normals,
            Tensor mesh_v2t,
            Tensor mesh_vertex_normals,
            Tensor start_uv,
            Tensor start_dir,
            Tensor start_face,
            int max_steps,
            bool debug,
            bool print_warnings,
            float eps,
            bool avoid_holes
        )->(Tensor,Tensor))"
    );
}

TORCH_LIBRARY_IMPL(geodesic_utils, CPU, m) {
    m.impl("straightest_geodesics", &straightest_geodesics);
}

TORCH_LIBRARY_IMPL(geodesic_utils, Autograd, m) {
    m.impl("straightest_geodesics", torch::CppFunction::makeFallthrough());
}

}
