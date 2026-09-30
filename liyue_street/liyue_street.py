"""
原神·璃月港 街道场景 —— Blender 程序化生成脚本
=================================================

参考构图：从木桥上望向石阶、红色楼阁、大红拱门与悬挂的云纹灯笼，
远处是红色天桥、月洞门和璃月特有的石林山峰。

用法
----
1. Blender 图形界面：打开 Scripting（脚本）工作区 → 打开本文件 → 运行脚本（Alt+P）。
   会清空当前场景并生成整个街道、相机、灯光和卡通材质。
2. 命令行后台渲染（需要 Blender 4.2+，已在 5.0 测试）：
     blender -b -P liyue_street.py -- --render preview.png --save liyue_street.blend
   可选参数：
     --engine eevee|cycles   渲染器（默认 eevee，卡通着色效果最好）
     --res 1600x1000         分辨率
     --samples 32            采样数
     --outline               开启 Freestyle 描边（更像游戏截图，但渲染更慢）
     --view bridge|top       相机机位：bridge=桥上仰视（参考图1）/ top=桥头平视（参考图2）

所有尺寸单位为米。每个部件都是独立对象，并按集合（Collection）分组，方便之后在
Blender 里手动调整、替换成更精细的模型，或者导入 MMD 模型拍摄。
"""

import argparse
import math
import os
import random
import sys

import bpy  # 必须先导入 bpy（作为 Python 模块使用时 bmesh 依赖它）
import bmesh
from mathutils import Matrix, Vector

RNG = random.Random(7)


# ---------------------------------------------------------------------------
# 颜色 / 材质
# ---------------------------------------------------------------------------

def srgb(hex_str):
    """'#RRGGBB' -> 线性 RGBA（着色器节点里的颜色是线性空间）"""
    h = hex_str.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return (*lin, 1.0)


SHADOW_NEAR = (0.62, 0.58, 0.74)   # 近景阴影：偏蓝紫，原神风格
SHADOW_FAR = (0.80, 0.83, 0.93)    # 远景阴影：更浅，模拟空气透视


def _pattern_factor(nt, kind, scale, plane):
    """返回一个 0..1 的浮点输出，用于给底色压暗形成纹理（木板缝、瓦片、砖缝等）"""
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord")
    sep = N("ShaderNodeSeparateXYZ")
    L(tc.outputs["Object"], sep.inputs[0])
    comb = N("ShaderNodeCombineXYZ")
    # plane 决定纹理贴在哪两个轴上
    order = {"XY": ("X", "Y", "Z"), "XZ": ("X", "Z", "Y"), "YZ": ("Y", "Z", "X")}[plane]
    for dst, src in zip(("X", "Y", "Z"), order):
        L(sep.outputs[src], comb.inputs[dst])
    vec = comb.outputs[0]

    if kind == "bands":            # 平行线：木板、竖向板墙
        t = N("ShaderNodeTexWave")
        t.wave_type = "BANDS"
        t.bands_direction = "X"
        t.inputs["Scale"].default_value = scale
        t.inputs["Distortion"].default_value = 0.0
        L(vec, t.inputs["Vector"])
        fac = t.outputs["Fac"]
    elif kind == "grid":           # 网格：琉璃瓦
        a = N("ShaderNodeTexWave"); a.bands_direction = "X"
        b = N("ShaderNodeTexWave"); b.bands_direction = "Y"
        for t in (a, b):
            t.wave_type = "BANDS"
            t.inputs["Scale"].default_value = scale
            t.inputs["Distortion"].default_value = 0.0
            L(vec, t.inputs["Vector"])
        mx = N("ShaderNodeMath"); mx.operation = "MAXIMUM"
        L(a.outputs["Fac"], mx.inputs[0]); L(b.outputs["Fac"], mx.inputs[1])
        fac = mx.outputs[0]
    elif kind == "diamond":        # 斜格石板：台阶中间的坡道
        mp = N("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, 0, math.radians(45))
        L(vec, mp.inputs["Vector"])
        t = N("ShaderNodeTexChecker")
        t.inputs["Scale"].default_value = scale
        t.inputs["Color1"].default_value = (1, 1, 1, 1)
        t.inputs["Color2"].default_value = (0, 0, 0, 1)
        L(mp.outputs[0], t.inputs["Vector"])
        fac = t.outputs["Fac"]
        return fac
    elif kind == "bricks":         # 砖墙：河岸、台基
        t = N("ShaderNodeTexBrick")
        t.inputs["Scale"].default_value = scale
        t.inputs["Mortar Size"].default_value = 0.025
        t.inputs["Color1"].default_value = (0, 0, 0, 1)
        t.inputs["Color2"].default_value = (0.25, 0.25, 0.25, 1)
        t.inputs["Mortar"].default_value = (1, 1, 1, 1)
        L(vec, t.inputs["Vector"])
        fac = t.outputs["Color"]
        bw = N("ShaderNodeRGBToBW")
        L(fac, bw.inputs[0])
        return bw.outputs[0]
    elif kind == "swirl":          # 云纹：灯笼
        t = N("ShaderNodeTexWave")
        t.wave_type = "RINGS"
        t.inputs["Scale"].default_value = scale
        t.inputs["Distortion"].default_value = 7.0
        t.inputs["Detail"].default_value = 1.0
        L(vec, t.inputs["Vector"])
        fac = t.outputs["Fac"]
    else:
        raise ValueError(kind)

    # 把正弦波变成细线：只有波峰附近才压暗
    ramp = N("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    ramp.color_ramp.elements[1].position = 0.88
    ramp.color_ramp.elements[1].color = (1, 1, 1, 1)
    L(fac, ramp.inputs["Fac"])
    return ramp.outputs["Color"]


def toon(name, color, pattern=None, scale=5.0, plane="XY", dark=0.8,
         shadow=SHADOW_NEAR, emit=0.0):
    """
    卡通材质：
      - EEVEE：Diffuse -> Shader to RGB -> 两段色阶，得到明暗分界清晰的赛璐璐着色
      - Cycles：退化为普通 Principled BSDF（Cycles 不支持 Shader to RGB）
    """
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    nt.nodes.clear()
    N, L = nt.nodes.new, nt.links.new

    rgb = N("ShaderNodeRGB")
    rgb.outputs[0].default_value = srgb(color)
    base = rgb.outputs[0]

    if pattern:
        f = _pattern_factor(nt, pattern, scale, plane)
        mr = N("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 1.0
        mr.inputs["To Max"].default_value = dark
        L(f, mr.inputs["Value"])
        sc = N("ShaderNodeVectorMath"); sc.operation = "SCALE"
        L(base, sc.inputs[0]); L(mr.outputs[0], sc.inputs["Scale"])
        base = sc.outputs[0]

    em = N("ShaderNodeEmission")
    if emit > 0:
        L(base, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0 + emit
    else:
        diff = N("ShaderNodeBsdfDiffuse")
        diff.inputs["Color"].default_value = (1, 1, 1, 1)
        s2r = N("ShaderNodeShaderToRGB")
        L(diff.outputs[0], s2r.inputs[0])
        ramp = N("ShaderNodeValToRGB")
        cr = ramp.color_ramp
        cr.elements[0].position = 0.40
        cr.elements[0].color = (*shadow, 1)
        cr.elements[1].position = 0.52
        cr.elements[1].color = (1, 1, 1, 1)
        L(s2r.outputs["Color"], ramp.inputs["Fac"])
        mul = N("ShaderNodeVectorMath"); mul.operation = "MULTIPLY"
        L(base, mul.inputs[0]); L(ramp.outputs["Color"], mul.inputs[1])
        L(mul.outputs[0], em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0

    out_e = N("ShaderNodeOutputMaterial"); out_e.target = "EEVEE"
    L(em.outputs[0], out_e.inputs["Surface"])

    pb = N("ShaderNodeBsdfPrincipled")
    L(base, pb.inputs["Base Color"])
    pb.inputs["Roughness"].default_value = 0.85
    if emit > 0:
        L(base, pb.inputs["Emission Color"])
        pb.inputs["Emission Strength"].default_value = 1.0 + emit
    out_c = N("ShaderNodeOutputMaterial"); out_c.target = "CYCLES"
    L(pb.outputs[0], out_c.inputs["Surface"])
    return m


class Mats:
    """整个场景的调色板（取自参考截图）"""

    def __init__(self):
        self.red = toon("朱红墙", "#C8483C", "bands", 1.2, "XZ", 0.95)
        self.red_dark = toon("深红柱", "#A8302A")
        self.stone = toon("台基石", "#B8BCB6", "bricks", 0.9, "XZ", 0.86)
        self.stone_side = toon("台基石_侧", "#B8BCB6", "bricks", 0.9, "YZ", 0.86)
        self.stone_light = toon("浅石", "#D6D9D2")
        self.stone_dark = toon("深石", "#9EA39E")
        self.step_riser = toon("台阶踢面", "#A9ADA6")
        self.step_tread = toon("台阶踏面", "#E3E5DD")
        self.paving = toon("铺地石", "#C9CCC4", "grid", 0.35, "XY", 0.9)
        self.ramp = toon("斜格石坡", "#DEE0D8", "diamond", 0.9, "XY", 0.88)
        self.cream = toon("米白墙", "#F1E4C6")
        self.gold = toon("金饰", "#CDA955")
        self.wood = toon("原木", "#C68D55", "bands", 3.0, "YZ", 0.82)
        self.wood_x = toon("原木_横", "#C68D55", "bands", 3.0, "XZ", 0.82)
        self.wood_brown = toon("棕木", "#8C6A48", "bands", 2.0, "XZ", 0.88)
        self.wood_dark = toon("深木", "#5C4430")
        self.roof = toon("琉璃瓦", "#3F9C7C", "grid", 3.0, "XY", 0.78)
        self.roof_ridge = toon("屋脊", "#2C6E5A")
        self.window = toon("窗格", "#2F4E57")
        self.lantern = toon("灯笼", "#F4B43C", "swirl", 1.6, "XZ", 0.72, emit=0.25)
        self.lantern_small = toon("小灯笼", "#FFD45A", emit=0.6)
        self.warm = toon("暖光室内", "#F8D690", emit=0.3)
        self.water = toon("水面", "#3F92B4", "bands", 0.6, "YZ", 0.92)
        self.maple = toon("枫叶", "#EE6A3E")
        self.maple2 = toon("枫叶_亮", "#F59A4E")
        self.bark = toon("树干", "#6B4A36")
        self.pine = toon("松针", "#3E8A55")
        self.grass = toon("草", "#C8E05A")
        self.rock_far = toon("远山岩", "#8FA2BE", shadow=SHADOW_FAR)
        self.grass_far = toon("远山草", "#7FC37C", shadow=SHADOW_FAR)
        self.rock_far2 = toon("更远山岩", "#A9BAD2", shadow=SHADOW_FAR)
        self.cloud = toon("云", "#FFFFFF", shadow=(0.80, 0.87, 0.98))
        self.rope = toon("绳", "#3A2A20")


# ---------------------------------------------------------------------------
# 几何工具：一个 Part = 一个对象，内部可以有多种材质
# ---------------------------------------------------------------------------

def collection(name):
    c = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c


class Part:
    def __init__(self, name, coll):
        self.name, self.coll = name, coll
        self.bm = bmesh.new()
        self.mats = []

    # -- 内部 -------------------------------------------------------------
    def _mi(self, mat):
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def _tag(self, verts, mat, smooth=False):
        mi = self._mi(mat)
        for f in {f for v in verts for f in v.link_faces}:
            f.material_index = mi
            f.smooth = smooth

    @staticmethod
    def _mat(loc, rot=(0, 0, 0), size=(1, 1, 1)):
        from mathutils import Euler
        return (Matrix.Translation(loc) @ Euler(rot).to_matrix().to_4x4()
                @ Matrix.Diagonal((*size, 1.0)))

    # -- 基本体 -------------------------------------------------------------
    def box(self, c, s, mat, rot=(0, 0, 0)):
        r = bmesh.ops.create_cube(self.bm, size=1.0, matrix=self._mat(c, rot, s))
        self._tag(r["verts"], mat)

    def box_mm(self, x0, x1, y0, y1, z0, z1, mat):
        """用最小/最大坐标定义长方体，更直观"""
        self.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                 (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)), mat)

    def cyl(self, c, r, h, mat, seg=12, r2=None, rot=(0, 0, 0), smooth=False):
        r = bmesh.ops.create_cone(self.bm, cap_ends=True, cap_tris=False, segments=seg,
                                  radius1=r, radius2=r if r2 is None else r2, depth=h,
                                  matrix=self._mat(c, rot))
        self._tag(r["verts"], mat, smooth)

    def sphere(self, c, r, mat, scale=(1, 1, 1), seg=16, smooth=True):
        res = bmesh.ops.create_uvsphere(self.bm, u_segments=seg, v_segments=max(6, seg // 2),
                                        radius=r, matrix=self._mat(c, (0, 0, 0), scale))
        self._tag(res["verts"], mat, smooth)

    def ico(self, c, r, mat, scale=(1, 1, 1), sub=1, jitter=0.0, smooth=False):
        res = bmesh.ops.create_icosphere(self.bm, subdivisions=sub, radius=r,
                                         matrix=self._mat(c, (0, 0, 0), scale))
        if jitter:
            for v in res["verts"]:
                v.co += Vector((RNG.uniform(-1, 1) for _ in range(3))) * jitter
        self._tag(res["verts"], mat, smooth)

    def beam(self, p0, p1, w, h, mat):
        """两点之间的方梁（栏杆扶手、斜梁、楼梯斜板）"""
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        q = d.to_track_quat("X", "Z")
        m = (Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4()
             @ Matrix.Diagonal((d.length, w, h, 1.0)))
        r = bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)
        self._tag(r["verts"], mat)

    def prism(self, profile, mat, to3d):
        """
        把二维轮廓挤出成柱体。to3d(u, v, t) 把轮廓点 (u, v) 和挤出参数 t∈{0,1}
        映射到世界坐标。
        """
        bm = self.bm
        a = [bm.verts.new(to3d(u, v, 0)) for u, v in profile]
        b = [bm.verts.new(to3d(u, v, 1)) for u, v in profile]
        n = len(profile)
        bm.faces.new(a)
        bm.faces.new(list(reversed(b)))
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
        self._tag(a + b, mat)

    def prism_xz(self, profile, y0, y1, mat):
        self.prism(profile, mat, lambda u, v, t: (u, y0 + (y1 - y0) * t, v))

    def prism_yz(self, profile, x0, x1, mat):
        self.prism(profile, mat, lambda u, v, t: (x0 + (x1 - x0) * t, u, v))

    def roof(self, c, W, D, H, top, under, rot90=False, thick=0.35, ridge_mat=None):
        """
        中式歇山/庑殿顶的简化版：四角起翘、屋面下凹、带屋脊和翘角。
        c = 檐口中心 (x, y, z)；W/D = 檐口半宽/半深；H = 屋顶高度。
        """
        if D > W:
            W, D = D, W
            rot90 = not rot90
        cx, cy, cz = c
        bm = self.bm
        curl = min(W, D) * 0.2
        k = min(W, D) * 0.45

        def V(x, y, z):
            if rot90:
                x, y = -y, x
            return bm.verts.new((cx + x, cy + y, cz + z))

        E_pts = [(W, D, curl), (W - k, D, 0), (-W + k, D, 0), (-W, D, curl),
                 (-W, D - k, 0), (-W, -D + k, 0), (-W, -D, curl), (-W + k, -D, 0),
                 (W - k, -D, 0), (W, -D, curl), (W, -D + k, 0), (W, D - k, 0)]
        E = [V(*p) for p in E_pts]
        Lw = [V(x, y, z - thick) for x, y, z in E_pts]
        s = D * 0.5
        zm = H * 0.42
        M = [V(W - s, D - s, zm), V(-W + s, D - s, zm), V(-W + s, -D + s, zm), V(W - s, -D + s, zm)]
        rx = max(W - D * 0.95, 0.15)
        R = [V(rx, 0, H), V(-rx, 0, H)]
        F = bm.faces.new
        tops = [
            F((E[0], E[1], M[0])), F((E[1], E[2], M[1], M[0])), F((E[2], E[3], M[1])),
            F((E[3], E[4], M[1])), F((E[4], E[5], M[2], M[1])), F((E[5], E[6], M[2])),
            F((E[6], E[7], M[2])), F((E[7], E[8], M[3], M[2])), F((E[8], E[9], M[3])),
            F((E[9], E[10], M[3])), F((E[10], E[11], M[0], M[3])), F((E[11], E[0], M[0])),
            F((M[0], M[1], R[1], R[0])), F((M[1], M[2], R[1])),
            F((M[2], M[3], R[0], R[1])), F((M[3], M[0], R[0])),
        ]
        unders = [F((E[i], E[(i + 1) % 12], Lw[(i + 1) % 12], Lw[i])) for i in range(12)]
        unders.append(F(list(reversed(Lw))))
        ti, ui = self._mi(top), self._mi(under)
        for f in tops:
            f.material_index = ti
        for f in unders:
            f.material_index = ui

        # 屋脊 + 两端鸱吻 + 四角翘
        rm = ridge_mat or top
        o = max(0.12, min(0.5, D * 0.15))      # 装饰件尺寸随屋顶大小缩放
        r0, r1 = R[0].co.copy(), R[1].co.copy()
        self.beam(r1 + Vector((0, 0, o * 0.3)), r0 + Vector((0, 0, o * 0.3)), o, o, rm)
        for p, sgn in ((r0, 1), (r1, -1)):
            dirv = (r0 - r1).normalized() * sgn if (r0 - r1).length > 1e-4 else Vector((1, 0, 0))
            self.beam(p + Vector((0, 0, o * 0.3)), p + dirv * o + Vector((0, 0, o * 2.4)), o * 0.8, o * 0.8, rm)
        for i in (0, 3, 6, 9):
            p = E[i].co.copy()
            out = Vector((p.x - cx, p.y - cy, 0)).normalized()
            self.beam(p, p + out * o * 2.0 + Vector((0, 0, o * 1.5)), o * 0.6, o * 0.6, rm)
            # 屋面斜脊
            mi = {0: 0, 3: 1, 6: 2, 9: 3}[i]
            self.beam(p + Vector((0, 0, o * 0.15)), M[mi].co + Vector((0, 0, o * 0.15)), o * 0.6, o * 0.5, rm)

    # -- 生成对象 -----------------------------------------------------------
    def finish(self):
        bm = self.bm
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(self.name)
        bm.to_mesh(me)
        bm.free()
        for m in self.mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(self.name, me)
        self.coll.objects.link(ob)
        return ob


# ---------------------------------------------------------------------------
# 可复用的建筑构件
# ---------------------------------------------------------------------------

def window(P, M, x, y, z, w, h, facing):
    """在墙面上放一扇格子窗。facing: '-y' '+y' '-x' '+x' 表示窗朝向"""
    axis, sgn = facing[1], (1 if facing[0] == "+" else -1)
    t = 0.12

    def B(du, dz, su, sz, dn, sn, mat):
        if axis == "y":
            P.box((x + du, y + sgn * dn, z + dz), (su, sn, sz), mat)
        else:
            P.box((x + sgn * dn, y + du, z + dz), (sn, su, sz), mat)

    B(0, 0, w, h, 0.02, 0.06, M.window)                 # 深色窗芯
    for dz in (-h / 2, h / 2):                           # 上下框
        B(0, dz, w + t, t, 0.08, 0.16, M.wood_dark)
    for du in (-w / 2, w / 2):                           # 左右框
        B(du, 0, t, h + t, 0.08, 0.16, M.wood_dark)
    for i in (1, 2):                                     # 窗格
        B(-w / 2 + w * i / 3, 0, 0.05, h, 0.07, 0.08, M.wood_brown)
        B(0, -h / 2 + h * i / 3, w, 0.05, 0.07, 0.08, M.wood_brown)


def railing_x(P, x0, x1, y, z, mat, h=1.0, spacing=1.2, post=0.12):
    """沿 X 方向的栏杆"""
    n = max(1, int(abs(x1 - x0) / spacing))
    for i in range(n + 1):
        P.box((x0 + (x1 - x0) * i / n, y, z + h / 2), (post, post, h), mat)
    P.box(((x0 + x1) / 2, y, z + h), (abs(x1 - x0), post * 1.2, post * 1.2), mat)
    P.box(((x0 + x1) / 2, y, z + h * 0.45), (abs(x1 - x0), post * 0.7, post * 0.7), mat)


def railing_y(P, x, y0, y1, z, mat, h=1.0, spacing=1.2, post=0.12):
    n = max(1, int(abs(y1 - y0) / spacing))
    for i in range(n + 1):
        P.box((x, y0 + (y1 - y0) * i / n, z + h / 2), (post, post, h), mat)
    P.box((x, (y0 + y1) / 2, z + h), (post * 1.2, abs(y1 - y0), post * 1.2), mat)
    P.box((x, (y0 + y1) / 2, z + h * 0.45), (post * 0.7, abs(y1 - y0), post * 0.7), mat)


def lattice_fence_x(P, x0, x1, y, z, mat, h=0.9):
    """河边的几何纹木栏（参考图2远处那种带斜撑的矮栏）"""
    n = max(1, int(abs(x1 - x0) / 2.0))
    step = (x1 - x0) / n
    for i in range(n + 1):
        P.box((x0 + step * i, y, z + h / 2), (0.14, 0.14, h), mat)
    P.box(((x0 + x1) / 2, y, z + h), (abs(x1 - x0), 0.12, 0.1), mat)
    P.box(((x0 + x1) / 2, y, z + 0.12), (abs(x1 - x0), 0.1, 0.08), mat)
    for i in range(n):
        a = x0 + step * i
        P.beam((a + step * 0.15, y, z + 0.15), (a + step * 0.5, y, z + h - 0.1), 0.06, 0.06, mat)
        P.beam((a + step * 0.5, y, z + h - 0.1), (a + step * 0.85, y, z + 0.15), 0.06, 0.06, mat)


def stone_lantern(P, M, x, y, z):
    """石灯笼"""
    P.box((x, y, z + 0.3), (1.0, 1.0, 0.6), M.stone_light)
    P.box((x, y, z + 0.7), (0.75, 0.75, 0.2), M.stone_dark)
    P.cyl((x, y, z + 1.25), 0.2, 0.9, M.stone_light, seg=8)
    P.box((x, y, z + 1.75), (0.95, 0.95, 0.15), M.stone_light)
    P.box((x, y, z + 2.15), (0.7, 0.7, 0.65), M.stone_light)
    for dx, dy, sx, sy in ((0, -0.33, 0.4, 0.06), (0, 0.33, 0.4, 0.06),
                           (-0.33, 0, 0.06, 0.4), (0.33, 0, 0.06, 0.4)):
        P.box((x + dx, y + dy, z + 2.18), (sx, sy, 0.4), M.stone_dark)
    P.roof((x, y, z + 2.55), 0.72, 0.72, 0.55, M.stone_light, M.stone_dark, thick=0.12)
    P.sphere((x, y, z + 3.25), 0.13, M.stone_light, seg=8)


def hanging_lantern(P, M, x, y, z, r=0.25, n=3, gap=0.75):
    """一串黄色小灯笼"""
    P.cyl((x, y, z + 0.25), 0.02, 0.5, M.rope, seg=4)
    for i in range(n):
        zz = z - i * gap
        P.cyl((x, y, zz - 0.25), r, r * 1.6, M.lantern_small, seg=10, smooth=True)
        P.cyl((x, y, zz - 0.25 + r * 0.85), r * 0.6, 0.08, M.red_dark, seg=8)
        P.cyl((x, y, zz - 0.25 - r * 0.85), r * 0.6, 0.08, M.red_dark, seg=8)
        if i < n - 1:
            P.cyl((x, y, zz - gap / 2 - 0.25), 0.015, gap - r * 1.6, M.rope, seg=4)


def stair_flight(P, x0, x1, y0, y1, z0, z1, n, riser, tread=None):
    """沿 +Y 上升的实心台阶：踢面用 riser 材质，踏面盖一层略外挑的 tread 石板"""
    dy, dz = (y1 - y0) / n, (z1 - z0) / n
    for i in range(n):
        top = z0 + (i + 1) * dz
        P.box_mm(x0, x1, y0 + i * dy, y1, z0, top - 0.06, riser)
        if tread:
            P.box_mm(x0, x1, y0 + i * dy - 0.05, y0 + (i + 1) * dy, top - 0.06, top, tread)


# ---------------------------------------------------------------------------
# 场景各部分
# ---------------------------------------------------------------------------

BRIDGE_Y0, BRIDGE_Y1, BRIDGE_ARCH = -24.0, -0.5, 0.9


def bridge_z(y):
    t = (y - BRIDGE_Y0) / (BRIDGE_Y1 - BRIDGE_Y0)
    return BRIDGE_ARCH * math.sin(math.pi * t) - 0.05


def build_ground(M):
    C = collection("地面与河道")
    P = Part("河面", C)
    P.box_mm(-120, 120, -80, -0.5, -3.6, -3.3, M.water)
    P.finish()

    P = Part("广场与河岸", C)
    P.box_mm(-60, 60, -0.5, 3.0, -3.5, 0.0, M.paving)        # 桥头广场
    P.box_mm(-60, 60, -0.5, -0.3, -3.5, -0.05, M.stone)      # 河岸砖墙（朝向镜头）
    # 桥两侧的石砌堤岸（参考图2中桥左右的石墙）
    for s in (-1, 1):
        x0, x1 = s * 4.3, s * 40
        P.box_mm(min(x0, x1), max(x0, x1), -9.0, -0.5, -3.5, -0.4, M.stone)
        P.box_mm(min(x0, x1), max(x0, x1), -9.2, -0.5, -0.4, 0.0, M.stone_light)
        xs = min(x0, x1) if s > 0 else max(x0, x1)
        P.box_mm(xs - 0.05, xs + 0.05, -9.0, -0.5, -3.5, -0.4, M.stone_side)
        # 石块墩子
        for yy in (-8.2, -4.5):
            P.box((s * 5.6, yy, 0.45), (2.2, 1.6, 0.9), M.stone_light)
    P.box_mm(-60, 60, 8.5, 90, -3.5, 3.0, M.paving)           # 上层街面（台阶上方）
    P.finish()

    P = Part("河边木栏", C)
    for s in (-1, 1):
        lattice_fence_x(P, s * 4.4, s * 30, -8.9, 0.0, M.wood_brown)
        lattice_fence_x(P, s * 9.8, s * 30, 0.1, 0.0, M.wood_brown)
    P.finish()


def build_bridge(M):
    C = collection("木桥")
    P = Part("拱形木桥", C)
    n = 24
    hw = 3.0
    ys = [BRIDGE_Y0 + (BRIDGE_Y1 - BRIDGE_Y0) * i / n for i in range(n + 1)]
    for a, b in zip(ys, ys[1:]):
        za, zb = bridge_z(a), bridge_z(b)
        P.beam((0, a, za - 0.15), (0, b, zb - 0.15), hw * 2, 0.3, M.wood)       # 桥面
        for s in (-1, 1):
            P.beam((s * (hw - 0.05), a, za - 0.55), (s * (hw - 0.05), b, zb - 0.55), 0.3, 0.6, M.wood_x)
            # 扶手 & 腰杆
            P.beam((s * (hw - 0.1), a, za + 1.05), (s * (hw - 0.1), b, zb + 1.05), 0.2, 0.18, M.wood_x)
            P.beam((s * (hw - 0.1), a, za + 0.55), (s * (hw - 0.1), b, zb + 0.55), 0.12, 0.12, M.wood_x)
    # 望柱（带尖帽）
    for i in range(0, n + 1, 3):
        y = ys[i]
        z = bridge_z(y)
        for s in (-1, 1):
            x = s * (hw - 0.1)
            P.box((x, y, z + 0.6), (0.34, 0.34, 1.2), M.wood_x)
            P.box((x, y, z + 1.27), (0.42, 0.42, 0.14), M.wood_x)
            P.cyl((x, y, z + 1.45), 0.3, 0.3, M.wood_x, seg=4, r2=0.02, rot=(0, 0, math.pi / 4))
    # 桥墩
    for y in (-18, -12, -6):
        for s in (-1, 1):
            P.box((s * 2.4, y, -2.2), (0.45, 0.45, 2.6), M.wood_dark)
    P.finish()


def build_stairs(M):
    C = collection("台阶与石灯")
    P = Part("大台阶", C)
    y0, y1, z1, n = 3.0, 8.5, 3.0, 14
    for s in (-1, 1):
        stair_flight(P, min(s * 3.9, s * 8.2), max(s * 3.9, s * 8.2), y0, y1, 0.0, z1, n,
                     M.step_riser, M.step_tread)
        # 坡道两侧的石栏（斜）
        xi = s * 3.55
        P.prism_yz([(y0 - 0.4, 0), (y0 - 0.4, 0.35), (y1, z1 + 0.35), (y1, 0)],
                   xi - 0.35, xi + 0.35, M.stone_light)
        # 台阶外侧的石墙
        xo = s * 8.6
        P.prism_yz([(y0 - 0.4, 0), (y0 - 0.4, 0.5), (y1, z1 + 0.5), (y1, 0)],
                   xo - 0.4, xo + 0.4, M.stone_light)
    # 中间斜格石板坡道
    P.prism_yz([(y0, 0), (y1, z1), (y1, 0)], -3.2, 3.2, M.ramp)
    P.finish()

    P = Part("石灯笼", C)
    for s in (-1, 1):
        stone_lantern(P, M, s * 4.6, 3.1, 0.0)
    P.finish()


def big_building(P, M, x0, x1, y0, y1, base_h, red_h, cream_h, street_x, roof_h=4.0):
    """近景两侧的大楼：灰色石台基 + 朱红墙身 + 米白上层 + 绿琉璃顶"""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    sx = 1 if street_x > cx else -1
    P.box((cx, cy, base_h / 2), (dx, dy, base_h), M.stone)
    P.box((cx, cy, base_h + 0.2), (dx + 0.5, dy + 0.5, 0.4), M.stone_light)
    zr = base_h + 0.4
    P.box((cx, cy, zr + red_h / 2), (dx - 0.6, dy - 0.6, red_h), M.red)
    # 红墙顶部的金色/木色腰线
    P.box((cx, cy, zr + red_h + 0.15), (dx - 0.2, dy - 0.2, 0.3), M.gold)
    P.box((cx, cy, zr + red_h + 0.55), (dx + 0.6, dy + 0.6, 0.5), M.wood_brown)
    zc = zr + red_h + 0.8
    P.box((cx, cy, zc + cream_h / 2), (dx - 1.0, dy - 1.0, cream_h), M.cream)
    # 米白层的木框架（竖柱 + 横梁）
    for yy in (y0 + 0.45, y1 - 0.45):
        n = max(2, int(dx / 3.2))
        for i in range(n + 1):
            P.box((x0 + 0.5 + (dx - 1.0) * i / n, yy, zc + cream_h / 2), (0.3, 0.2, cream_h), M.wood_brown)
    for xx in (x0 + 0.45, x1 - 0.45):
        n = max(2, int(dy / 3.2))
        for i in range(n + 1):
            P.box((xx, y0 + 0.5 + (dy - 1.0) * i / n, zc + cream_h / 2), (0.2, 0.3, cream_h), M.wood_brown)
    P.box((cx, cy, zc + cream_h * 0.45), (dx - 0.9, dy - 0.9, 0.2), M.wood_brown)
    P.box((cx, cy, zc + cream_h + 0.2), (dx - 0.2, dy - 0.2, 0.4), M.wood_brown)
    # 窗：正面（朝 -Y 即朝向镜头）与临街侧
    n = max(2, int(dx / 3.2))
    for i in range(n):
        wx = x0 + 0.5 + (dx - 1.0) * (i + 0.5) / n
        window(P, M, wx, y0 + 0.5, zc + cream_h * 0.72, 1.5, 1.6, "-y")
        window(P, M, wx, y0 + 0.5, zc + cream_h * 0.22, 1.5, 1.0, "-y")
    n = max(2, int(dy / 3.2))
    for i in range(n):
        wy = y0 + 0.5 + (dy - 1.0) * (i + 0.5) / n
        window(P, M, street_x - sx * 0.5, wy, zc + cream_h * 0.72, 1.5, 1.6, "+x" if sx > 0 else "-x")
    # 正面红墙上的小石牌 & 竖向深红壁柱
    n = max(2, int(dx / 4.5))
    for i in range(n + 1):
        P.box((x0 + 0.5 + (dx - 1.0) * i / n, y0 + 0.2, zr + red_h / 2), (0.55, 0.3, red_h), M.red_dark)
    n = max(2, int(dy / 4.5))
    for i in range(n + 1):
        P.box((street_x - sx * 0.2, y0 + 0.5 + (dy - 1.0) * i / n, zr + red_h / 2), (0.3, 0.55, red_h), M.red_dark)
    # 屋顶
    P.roof((cx, cy, zc + cream_h + 0.5), dx / 2 + 1.6, dy / 2 + 1.6, roof_h, M.roof, M.wood_brown,
           ridge_mat=M.roof_ridge)


def build_front_buildings(M):
    C = collection("近景楼阁")
    P = Part("左侧大楼", C)
    big_building(P, M, -32, -8.8, 5.0, 24, base_h=5.5, red_h=6.5, cream_h=5.0, street_x=-8.8)
    # 临街一侧的木挑台
    P.box_mm(-10.2, -8.8, 8, 20, 12.6, 13.0, M.wood_brown)
    railing_y(P, -10.1, 8, 20, 13.0, M.wood_brown, h=0.9)
    P.finish()

    P = Part("右侧大楼", C)
    big_building(P, M, 8.8, 32, 6.0, 22, base_h=6.0, red_h=7.5, cream_h=4.5, street_x=8.8)
    # 台基上的一丛草
    for i in range(14):
        x = RNG.uniform(10, 17)
        P.cyl((x, 5.9, 6.4), 0.12, RNG.uniform(0.6, 1.2), M.grass, seg=3, r2=0.0,
              rot=(RNG.uniform(-0.3, 0.3), RNG.uniform(-0.3, 0.3), 0))
    P.finish()


def build_arch(M):
    """横跨街道的大红拱楼 + 悬挂的云纹大灯笼"""
    C = collection("红色拱楼")
    P = Part("大红拱", C)
    zb, zt = 7.5, 22.0          # 拱体底 / 顶
    # 拱洞轮廓（八边形的上半部分）
    def hole(e):
        return [(-8.4 - e, zb), (-8.4 - e, 13.0 + e * 0.3), (-6.0 - e * 0.8, 16.8 + e * 0.8),
                (-3.0 - e * 0.5, 18.2 + e), (3.0 + e * 0.5, 18.2 + e), (6.0 + e * 0.8, 16.8 + e * 0.8),
                (8.4 + e, 13.0 + e * 0.3), (8.4 + e, zb)]

    def arch_layer(e, y0, y1, xo, ztop, mat):
        h = hole(e)
        P.box_mm(-xo, h[0][0], y0, y1, zb, ztop, mat)            # 左腿
        P.box_mm(h[-1][0], xo, y0, y1, zb, ztop, mat)            # 右腿
        P.box_mm(h[3][0], h[4][0], y0, y1, h[3][1], ztop, mat)   # 顶梁
        for side in (h[:4], [(-u, v) for u, v in reversed(h[4:])]):
            (ax, az), (bx, bz), (ccx, cz), (dx, dz) = side[0], side[1], side[2], side[3]
            # 用三角/四边形把腿和顶梁之间的斜角补齐
            P.prism_xz([(bx, bz), (ccx, cz), (bx, ztop)], y0, y1, mat)
            P.prism_xz([(ccx, cz), (dx, dz), (dx, ztop), (bx, ztop)], y0, y1, mat)

    # 三层退台，形成参考图中层层叠叠的厚重拱口
    arch_layer(0.0, 19.0, 23.5, 16, zt, M.red)
    arch_layer(0.6, 18.3, 19.0, 15.5, zt - 0.4, M.red)
    arch_layer(1.2, 17.6, 18.3, 15.0, zt - 0.8, M.red)
    # 拱口下沿的阴影线（深红）
    P.box_mm(-3.0, 3.0, 17.55, 23.55, 18.05, 18.22, M.red_dark)
    # 拱体上方的深色木梁托（斗拱感）
    for x in range(-14, 15, 2):
        P.box((x, 17.4, zt - 0.3), (0.5, 0.8, 0.6), M.wood_brown)
    P.box_mm(-16.5, 16.5, 17.0, 24.0, zt, zt + 0.5, M.wood_brown)
    P.finish()

    P = Part("拱上木楼", C)
    z0 = zt + 0.5
    P.box_mm(-15, 15, 18, 23.5, z0, z0 + 3.5, M.wood_brown)
    railing_x(P, -15.5, 15.5, 17.4, z0, M.wood_dark, h=1.1, spacing=0.9)
    P.box_mm(-15.8, 15.8, 16.8, 18.0, z0 - 0.05, z0 + 0.05, M.wood_brown)
    for x in (-14, -7, 0, 7, 14):
        P.box((x, 17.5, z0 + 3.0), (0.4, 0.4, 6.0), M.wood_dark)
    P.box_mm(-16, 16, 16.5, 24.5, z0 + 5.8, z0 + 6.4, M.wood_brown)
    P.roof((0, 20.5, z0 + 6.4), 17.5, 6.5, 4.5, M.roof, M.wood_brown, ridge_mat=M.roof_ridge)
    P.finish()

    # 悬挂的大灯笼（云纹金色）
    P = Part("云纹大灯笼", C)
    lx, ly, lz = 0.0, 20.5, 13.6
    P.cyl((lx, ly, (lz + 2.6 + 18.2) / 2), 0.06, 18.2 - (lz + 2.6), M.rope, seg=6)
    P.sphere((lx, ly, lz), 1.9, M.lantern, scale=(1, 1, 1.1), seg=24)
    P.cyl((lx, ly, lz + 2.0), 0.95, 0.45, M.roof_ridge, seg=12, r2=0.7)
    for i in range(6):                                   # 顶部叶片装饰
        a = i / 6 * math.tau
        P.beam((lx, ly, lz + 2.1), (lx + 1.1 * math.cos(a), ly + 1.1 * math.sin(a), lz + 2.6),
               0.3, 0.12, M.roof_ridge)
    P.cyl((lx, ly, lz + 2.4), 0.35, 0.5, M.roof_ridge, seg=8)
    P.cyl((lx, ly, lz - 2.05), 0.8, 0.3, M.gold, seg=12, r2=1.0)
    for i in range(14):
        a = i / 14 * math.tau
        P.cyl((lx + 0.75 * math.cos(a), ly + 0.75 * math.sin(a), lz - 2.9), 0.05, 1.4,
              M.lantern_small, seg=4)
    P.finish()


def shop(P, M, away, xe, y0, y1, z0, floors, balcony=True):
    """
    沿街商铺。away=-1 表示街道左侧（建筑向 -X 延伸），+1 表示右侧。
    xe 是临街立面的 X 坐标。
    """
    depth = 9.0
    toward = -away
    xf = xe + away * depth
    L = y1 - y0
    cy = (y0 + y1) / 2

    def X(a, b):
        return min(a, b), max(a, b)

    h1 = 4.2
    rec = 1.8
    # 一层：后退的铺面 + 暖光门窗 + 红柱廊
    P.box_mm(*X(xe + away * rec, xf), y0, y1, z0, z0 + h1, M.cream)
    n = max(2, int(L / 3.0))
    for i in range(n):
        yy = y0 + L * (i + 0.5) / n
        P.box((xe + away * (rec - 0.04), yy, z0 + 1.6), (0.1, L / n * 0.62, 2.6), M.warm)
        P.box((xe + away * (rec - 0.08), yy, z0 + 3.3), (0.12, L / n * 0.7, 0.35), M.wood_brown)
    for i in range(n + 1):
        P.cyl((xe + away * 0.3, y0 + L * i / n, z0 + h1 / 2), 0.2, h1, M.red_dark, seg=10)
    P.box_mm(*X(xe - away * 0.1, xf), y0, y1, z0 + h1 - 0.35, z0 + h1, M.wood_brown)
    # 门廊小檐
    P.beam((xe - away * 1.1, cy, z0 + h1 - 0.05), (xe + away * 1.2, cy, z0 + h1 + 0.8), L + 0.4, 0.22, M.roof)
    # 招牌
    P.box((xe - away * 0.05, cy, z0 + h1 - 0.9), (0.15, min(3.0, L * 0.4), 0.7), M.gold)
    P.box((xe - away * 0.1, cy, z0 + h1 - 0.9), (0.12, min(2.6, L * 0.34), 0.45), M.red_dark)

    z = z0 + h1
    fh = 3.4
    for f in range(floors):
        ov = 0.6 + 0.3 * f            # 上层逐层外挑
        xs = xe - away * ov
        P.box_mm(*X(xs, xf), y0, y1, z, z + fh, M.cream)
        P.box_mm(*X(xs - away * 0.1, xf), y0 - 0.1, y1 + 0.1, z, z + 0.35, M.wood_brown)
        P.box_mm(*X(xs - away * 0.1, xf), y0 - 0.1, y1 + 0.1, z + fh - 0.3, z + fh, M.wood_brown)
        for yy in (y0 + 0.15, y1 - 0.15):
            P.box((xs - away * 0.05, yy, z + fh / 2), (0.35, 0.35, fh), M.red_dark)
        nw = max(1, int(L / 3.2))
        for i in range(nw):
            window(P, M, xs, y0 + L * (i + 0.5) / nw, z + fh * 0.55, 1.3, 1.5,
                   "+x" if toward > 0 else "-x")
        if balcony and f == 0:
            bx = xs - away * 0.9
            P.box_mm(*X(bx, xs), y0, y1, z, z + 0.2, M.wood_brown)
            railing_y(P, bx, y0, y1, z + 0.2, M.red_dark, h=0.95, spacing=1.0)
        z += fh
    P.roof((xe + away * (depth / 2 - 0.4), cy, z + 0.2), L / 2 + 1.1, depth / 2 + 1.6, 3.2,
           M.roof, M.wood_brown, ridge_mat=M.roof_ridge)
    return z


def build_street(M):
    C = collection("街道商铺")
    Z0 = 3.0
    # 左侧
    P = Part("左侧商铺", C)
    shop(P, M, -1, -6.0, 23.5, 31.0, Z0, 2)
    shop(P, M, -1, -5.8, 31.5, 38.0, Z0, 2, balcony=False)
    shop(P, M, -1, -5.6, 42.0, 49.0, Z0, 1)
    shop(P, M, -1, -5.4, 49.5, 56.0, Z0, 2, balcony=False)
    shop(P, M, -1, -5.2, 60.0, 66.0, Z0, 1)
    P.finish()
    # 右侧
    P = Part("右侧商铺", C)
    shop(P, M, 1, 6.2, 23.5, 30.0, Z0, 2, balcony=False)
    shop(P, M, 1, 6.0, 34.0, 41.0, Z0, 2)
    shop(P, M, 1, 5.7, 41.5, 48.0, Z0, 1, balcony=False)
    shop(P, M, 1, 5.5, 52.0, 58.5, Z0, 2)
    shop(P, M, 1, 5.3, 60.0, 66.0, Z0, 1)
    # 右侧外挂红色楼梯（参考图右侧那段斜上的红梯）
    x0, x1 = 4.4, 5.9
    ya, yb, za, zb = 30.5, 38.0, Z0, Z0 + 4.4
    nstep = 16
    for i in range(nstep):
        t = i / nstep
        yy = ya + (yb - ya) * t
        zz = za + (zb - za) * (i + 1) / nstep
        P.box_mm(x0, x1, yy, yy + (yb - ya) / nstep, zz - 0.15, zz, M.red)
    for xx in (x0, x1):
        P.beam((xx, ya, za), (xx, yb, zb), 0.15, 0.5, M.red_dark)
        P.beam((xx, ya, za + 1.0), (xx, yb, zb + 1.0), 0.1, 0.1, M.red_dark)
        for i in range(0, nstep + 1, 4):
            t = i / nstep
            P.box((xx, ya + (yb - ya) * t, za + (zb - za) * t + 0.5), (0.1, 0.1, 1.0), M.red_dark)
    P.box_mm(x0, 6.2, 38.0, 41.0, zb - 0.3, zb, M.red)
    P.finish()

    # 小黄灯笼串
    P = Part("小灯笼串", C)
    for y in (25.0, 27.0, 29.0, 33.0, 35.0):
        hanging_lantern(P, M, -5.0, y, Z0 + 3.4, n=3)
    for y in (36.0, 38.5, 44.0):
        hanging_lantern(P, M, 5.4, y, Z0 + 3.4, n=2)
    P.finish()


def arch_bridge(P, M, y, x0, x1, deck, sag, width=2.4, rail=True):
    """横跨街道的红色天桥，底部是弧形拱"""
    n = 16
    top = deck - 0.3
    for i in range(n):
        a = x0 + (x1 - x0) * i / n
        b = x0 + (x1 - x0) * (i + 1) / n

        def bot(x):
            t = (x - x0) / (x1 - x0)
            return deck - sag + (sag - 0.9) * math.sin(math.pi * t)

        P.prism_xz([(a, bot(a)), (b, bot(b)), (b, top), (a, top)], y - width / 2 + 0.2, y + width / 2 - 0.2, M.red)
    P.box_mm(x0 - 0.3, x1 + 0.3, y - width / 2, y + width / 2, deck - 0.3, deck, M.wood_brown)
    if rail:
        for s in (-1, 1):
            railing_x(P, x0, x1, y + s * (width / 2 - 0.1), deck, M.red_dark, h=1.0, spacing=1.0)


def build_bridges_and_gate(M):
    C = collection("天桥与月洞门")
    P = Part("中段天桥", C)
    arch_bridge(P, M, 37.5, -9.5, 9.5, 13.0, 5.0)
    P.finish()

    P = Part("远处红桥", C)
    arch_bridge(P, M, 58.5, -8.0, 8.0, 8.2, 3.2, width=2.0)
    P.finish()

    # 月洞门：带圆洞的白墙 + 绿瓦顶
    P = Part("月洞门", C)
    y0, y1 = 67.5, 68.5
    cx, cz, r = 0.0, 6.2, 2.5
    X0, X1, Z0, Z1 = -6.5, 6.5, 3.0, 10.0
    n = 32
    for i in range(n):
        a0 = i / n * math.tau
        a1 = (i + 1) / n * math.tau

        def edge(a):
            dx, dz = math.cos(a), math.sin(a)
            t = min((X1 - cx) / dx if dx > 1e-6 else (X0 - cx) / dx if dx < -1e-6 else 1e9,
                    (Z1 - cz) / dz if dz > 1e-6 else (Z0 - cz) / dz if dz < -1e-6 else 1e9)
            return cx + dx * t, cz + dz * t

        p0 = (cx + r * math.cos(a0), cz + r * math.sin(a0))
        p1 = (cx + r * math.cos(a1), cz + r * math.sin(a1))
        q0, q1 = edge(a0), edge(a1)
        # 如果这一段跨过了墙角，把角点加进去
        corners = [(X1, Z1), (X0, Z1), (X0, Z0), (X1, Z0)]
        poly = [p0, p1, q1]
        for c in corners:
            ang = math.atan2(c[1] - cz, c[0] - cx) % math.tau
            if a0 < ang < a1:
                poly.append(c)
        poly.append(q0)
        # 排序保证是凸多边形顺序：p0,p1,q1,(corner),q0
        if len(poly) == 5:
            poly = [p0, p1, q1, poly[3], q0]
        P.prism_xz(poly, y0, y1, M.cream)
    # 圆洞描边
    for i in range(n):
        a0 = i / n * math.tau
        a1 = (i + 1) / n * math.tau
        P.prism_xz([(cx + r * math.cos(a0), cz + r * math.sin(a0)),
                    (cx + r * math.cos(a1), cz + r * math.sin(a1)),
                    (cx + (r + 0.3) * math.cos(a1), cz + (r + 0.3) * math.sin(a1)),
                    (cx + (r + 0.3) * math.cos(a0), cz + (r + 0.3) * math.sin(a0))],
                   y0 - 0.1, y1 + 0.1, M.stone_light)
    P.box_mm(X0 - 0.3, X1 + 0.3, y0 - 0.3, y1 + 0.3, Z1, Z1 + 0.4, M.wood_brown)
    P.roof((0, (y0 + y1) / 2, Z1 + 0.4), 7.8, 1.9, 1.8, M.roof, M.wood_brown, ridge_mat=M.roof_ridge)
    P.box_mm(X0 - 1.0, X0, y0 - 0.3, y1 + 0.3, Z0, Z1, M.stone_light)
    P.box_mm(X1, X1 + 1.0, y0 - 0.3, y1 + 0.3, Z0, Z1, M.stone_light)
    P.finish()


def maple(P, M, x, y, z, h=6.0, spread=3.2, n=9, leaf=(1.3, 2.0)):
    P.cyl((x, y, z + h / 2), 0.3, h, M.bark, seg=6, r2=0.18)
    for i in range(n):
        a = RNG.uniform(0, math.tau)
        rr = RNG.uniform(0.2, 1.0) * spread
        c = (x + math.cos(a) * rr, y + math.sin(a) * rr * 0.8, z + h + RNG.uniform(-1.2, 1.4))
        P.beam((x, y, z + h * 0.7), c, 0.16, 0.16, M.bark)
        P.ico(c, RNG.uniform(*leaf), M.maple if i % 3 else M.maple2,
              scale=(1, 1, 0.75), sub=1, jitter=0.18)


def pine(P, M, x, y, z, h=7.0):
    P.cyl((x, y, z + h / 2), 0.35, h, M.bark, seg=6, r2=0.2)
    for i, (dz, r) in enumerate(((0.55, 3.2), (0.72, 2.6), (0.88, 1.8), (1.0, 1.1))):
        off = Vector((RNG.uniform(-0.6, 0.6), RNG.uniform(-0.6, 0.6), 0))
        P.ico((x + off.x, y + off.y, z + h * dz), r, M.pine, scale=(1, 1, 0.32), sub=1, jitter=0.12)


def karst(P, M, x, y, z, r, h, rock, grass, seed):
    """璃月石林：带草顶、层层草台的岩柱"""
    rng = random.Random(seed)
    segs = 7
    rings = 5
    bm = P.bm
    prev = None
    rows = []
    for j in range(rings + 1):
        t = j / rings
        rr = r * (1.0 - 0.35 * t) * rng.uniform(0.92, 1.08)
        zz = z + h * t
        row = []
        for i in range(segs):
            a = i / segs * math.tau + rng.uniform(-0.15, 0.15)
            row.append(bm.verts.new((x + math.cos(a) * rr * rng.uniform(0.85, 1.1),
                                     y + math.sin(a) * rr * rng.uniform(0.85, 1.1), zz)))
        rows.append(row)
    faces = []
    for j in range(rings):
        for i in range(segs):
            k = (i + 1) % segs
            faces.append(bm.faces.new((rows[j][i], rows[j][k], rows[j + 1][k], rows[j + 1][i])))
    faces.append(bm.faces.new(list(reversed(rows[-1]))))
    mi = P._mi(rock)
    for f in faces:
        f.material_index = mi
    # 草顶
    P.ico((x, y, z + h + r * 0.1), r * 0.72, grass, scale=(1, 1, 0.35), sub=2, jitter=r * 0.04)
    # 侧面的草台
    for _ in range(rng.randint(1, 3)):
        t = rng.uniform(0.35, 0.85)
        a = rng.uniform(0, math.tau)
        rr = r * (1.0 - 0.35 * t) * 0.75
        P.ico((x + math.cos(a) * rr, y + math.sin(a) * rr, z + h * t), r * 0.45, grass,
              scale=(1, 1, 0.25), sub=1, jitter=r * 0.03)


def build_nature(M):
    C = collection("自然景观")
    P = Part("枫树", C)
    maple(P, M, -11.5, -2.0, 0.0, h=6.5, spread=2.6, n=10, leaf=(0.9, 1.4))     # 左上角近景枫叶
    maple(P, M, -4.0, 51.5, 3.0, h=4.5, spread=2.2, n=7)
    maple(P, M, 3.2, 55.0, 3.0, h=4.0, spread=2.0, n=6)
    maple(P, M, -7.5, 40.0, 3.0, h=6.0, spread=2.4, n=6)
    P.finish()

    P = Part("松树", C)
    pine(P, M, 16.0, -2.5, 0.0, h=6.5)
    pine(P, M, -20.0, 1.5, 0.0, h=7.5)
    P.finish()

    P = Part("远山石林", C)
    # 正后方的大岩山（参考图中拱洞里看到的那座）
    karst(P, M, 6, 110, -10, 20, 62, M.rock_far, M.grass_far, 1)
    karst(P, M, -14, 125, -10, 16, 50, M.rock_far, M.grass_far, 2)
    karst(P, M, 26, 135, -10, 14, 70, M.rock_far, M.grass_far, 3)
    karst(P, M, -34, 150, -10, 13, 58, M.rock_far2, M.grass_far, 4)
    karst(P, M, 46, 170, -10, 15, 80, M.rock_far2, M.grass_far, 5)
    karst(P, M, -60, 190, -10, 18, 72, M.rock_far2, M.grass_far, 6)
    karst(P, M, 80, 210, -10, 20, 90, M.rock_far2, M.grass_far, 7)
    for i in range(10):
        karst(P, M, RNG.uniform(-160, 160), RNG.uniform(230, 320), -10, RNG.uniform(10, 22),
              RNG.uniform(50, 110), M.rock_far2, M.grass_far, 10 + i)
    # 山脚绿地
    P.box_mm(-300, 300, 70, 400, -12, -1.0, M.grass_far)
    P.finish()


def build_sky(M):
    C = collection("天空")
    P = Part("云朵", C)
    clouds = [(-8, 150, 34, 9), (18, 190, 52, 14), (-50, 230, 70, 16), (60, 260, 45, 18),
              (-90, 280, 95, 20), (5, 300, 110, 22), (110, 300, 120, 24), (-20, 120, 26, 5)]
    for cx, cy, cz, s in clouds:
        for i in range(7):
            P.ico((cx + RNG.uniform(-1.6, 1.6) * s, cy + RNG.uniform(-0.3, 0.3) * s,
                   cz + RNG.uniform(-0.2, 0.35) * s),
                  s * RNG.uniform(0.5, 0.85), M.cloud, scale=(1.2, 0.8, 0.7), sub=2, smooth=True)
    P.finish()


# ---------------------------------------------------------------------------
# 世界、灯光、相机、渲染设置
# ---------------------------------------------------------------------------

def setup_world():
    w = bpy.data.worlds.new("璃月天空")
    bpy.context.scene.world = w
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    nt.nodes.clear()
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord")
    sep = N("ShaderNodeSeparateXYZ")
    L(tc.outputs["Generated"], sep.inputs[0])
    mr = N("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -0.05
    mr.inputs["From Max"].default_value = 0.7
    L(sep.outputs["Z"], mr.inputs["Value"])
    ramp = N("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = srgb("#D8ECFA")
    cr.elements[1].position = 1.0
    cr.elements[1].color = srgb("#2F7BD8")
    e = cr.elements.new(0.3)
    e.color = srgb("#72B5EE")
    L(mr.outputs[0], ramp.inputs["Fac"])
    sky = N("ShaderNodeBackground")
    L(ramp.outputs["Color"], sky.inputs["Color"])
    amb = N("ShaderNodeBackground")
    amb.inputs["Color"].default_value = srgb("#A8BEDC")
    amb.inputs["Strength"].default_value = 0.35
    lp = N("ShaderNodeLightPath")
    mix = N("ShaderNodeMixShader")
    L(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    L(amb.outputs[0], mix.inputs[1])
    L(sky.outputs[0], mix.inputs[2])
    out = N("ShaderNodeOutputWorld")
    L(mix.outputs[0], out.inputs["Surface"])


def setup_sun():
    C = collection("灯光相机")
    d = bpy.data.lights.new("太阳", "SUN")
    d.energy = 4.0
    d.color = (1.0, 0.95, 0.86)
    d.angle = math.radians(1.0)
    ob = bpy.data.objects.new("太阳", d)
    C.objects.link(ob)
    direction = Vector((-0.55, 0.55, -0.62)).normalized()
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    ob.location = (20, -30, 40)
    return C


VIEWS = {
    # 参考图1：站在桥中央，微微仰视拱门
    "bridge": dict(loc=(0.0, -11.5, bridge_z(-11.5) + 1.55), target=(0.0, 30.0, 7.5), lens=19),
    # 参考图2：退后一点，从桥的一端看过去，能看到整座拱桥
    "top": dict(loc=(0.0, -30.0, 3.2), target=(0.0, 30.0, 6.0), lens=24),
}


def setup_camera(C, view):
    v = VIEWS[view]
    cam = bpy.data.cameras.new("相机")
    cam.lens = v["lens"]
    cam.clip_end = 2000
    ob = bpy.data.objects.new("相机", cam)
    C.objects.link(ob)
    ob.location = v["loc"]
    d = Vector(v["target"]) - Vector(v["loc"])
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = ob
    return ob


def setup_render(engine, res, samples, outline):
    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = res
    s.render.resolution_percentage = 100
    if engine == "cycles":
        s.render.engine = "CYCLES"
        s.cycles.samples = samples
        s.cycles.device = "CPU"
        try:
            s.cycles.use_denoising = True
        except Exception:
            pass
    else:
        s.render.engine = "BLENDER_EEVEE"
        try:
            s.eevee.taa_render_samples = samples
        except Exception:
            pass
    s.view_settings.view_transform = "Standard"
    s.view_settings.look = "None"
    s.render.use_freestyle = outline
    if outline:
        s.render.line_thickness_mode = "ABSOLUTE"
        s.render.line_thickness = 1.0
        vl = s.view_layers[0]
        vl.use_freestyle = True
        for ls in vl.freestyle_settings.linesets:
            ls.linestyle.color = srgb("#4A3328")[:3]
            ls.linestyle.thickness = 1.2


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                  bpy.data.worlds):
        for item in list(block):
            block.remove(item)


def build(view="bridge"):
    clear_scene()
    M = Mats()
    build_ground(M)
    build_bridge(M)
    build_stairs(M)
    build_front_buildings(M)
    build_arch(M)
    build_street(M)
    build_bridges_and_gate(M)
    build_nature(M)
    build_sky(M)
    setup_world()
    C = setup_sun()
    setup_camera(C, view)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="生成原神璃月街道场景")
    p.add_argument("--render", help="渲染输出 PNG 路径")
    p.add_argument("--save", help="保存 .blend 路径")
    p.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    p.add_argument("--res", default="1600x1000")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--outline", action="store_true")
    p.add_argument("--view", choices=tuple(VIEWS), default="bridge")
    args, _ = p.parse_known_args(argv)
    return args


def main():
    args = parse_args()
    build(args.view)
    w, h = (int(v) for v in args.res.lower().split("x"))
    setup_render(args.engine, (w, h), args.samples, args.outline)
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
    if args.render:
        bpy.context.scene.render.filepath = os.path.abspath(args.render)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
