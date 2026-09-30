"""
原神·枫丹廷 芙宁娜家门口 —— Blender 程序化生成脚本（精细版 v1）
====================================================================

参考：枫丹廷的一栋两层小楼（芒萨尔式石板屋顶 + 玻璃天窗、米黄砂岩墙、
白石壁柱与金色书本形柱头、青绿色装饰艺术风格彩窗、拱形雨棚大门、
粉色遮阳篷书报摊），左侧是柏树和花丛的高台花园，前方是大台阶、
台阶中间的流水槽和八角喷泉，右侧是巨大的圆形石塔和城墙。

用法
----
Blender 图形界面：Scripting 工作区 → 打开本文件 → 运行（会清空当前场景）。

命令行：
  blender -b -P fontaine_furina.py -- --render out.png --save scene.blend
参数：
  --view front|stairs|plaza  机位（正面 / 台阶顶俯视 / 广场斜俯视）
  --part all|building        只生成主楼（调细节时更快）
  --engine eevee|cycles      渲染器（卡通着色只在 EEVEE 下生效）
  --res 1600x1000  --samples 32  --outline

设计思路
--------
* 每个构件都是一个函数、参数写在函数开头，方便之后一点一点改。
* 墙面装饰全部写在“墙面局部坐标系” Frame 里：u = 沿墙水平方向，
  z = 高度，d = 离墙面向外的距离。这样正面和侧面可以共用同一套立面函数。
* 线脚（腰线、檐口、屋顶）用“截面沿路径扫掠” sweep 生成；
  灯柱、花盆、塔身等用“车削” lathe 生成；所有硬边加倒角修改器。
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import bpy  # 必须先导入 bpy（作为 Python 模块使用时 bmesh 依赖它）
import bmesh
from mathutils import Euler, Matrix, Vector

RNG = random.Random(11)


# ===========================================================================
# 颜色与材质
# ===========================================================================

def srgb(hex_str):
    """'#RRGGBB' -> 线性 RGBA"""
    h = hex_str.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return (*lin, 1.0)


SHADOW = (0.70, 0.72, 0.88)       # 枫丹的阴影偏蓝紫、比较柔和
SHADOW_FAR = (0.82, 0.85, 0.95)


def _coords(nt, plane, radius=1.0):
    """
    纹理坐标：
      XY / XZ / YZ   直接取两个轴
      W              “包裹”：u = x + y，v = z。建筑四个立面都能连续贴砖
      CYL            圆柱：u = 角度 × 半径，v = z（塔身）
    """
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord")
    sep = N("ShaderNodeSeparateXYZ")
    L(tc.outputs["Object"], sep.inputs[0])
    comb = N("ShaderNodeCombineXYZ")
    if plane in ("XY", "XZ", "YZ"):
        a, b = plane[0], plane[1]
        L(sep.outputs[a], comb.inputs["X"])
        L(sep.outputs[b], comb.inputs["Y"])
    elif plane == "W":
        add = N("ShaderNodeMath"); add.operation = "ADD"
        L(sep.outputs["X"], add.inputs[0]); L(sep.outputs["Y"], add.inputs[1])
        L(add.outputs[0], comb.inputs["X"])
        L(sep.outputs["Z"], comb.inputs["Y"])
    elif plane == "CYL":
        at = N("ShaderNodeMath"); at.operation = "ARCTAN2"
        L(sep.outputs["Y"], at.inputs[0]); L(sep.outputs["X"], at.inputs[1])
        mul = N("ShaderNodeMath"); mul.operation = "MULTIPLY"
        mul.inputs[1].default_value = radius
        L(at.outputs[0], mul.inputs[0])
        L(mul.outputs[0], comb.inputs["X"])
        L(sep.outputs["Z"], comb.inputs["Y"])
    return comb.outputs[0]


def _math(nt, op, a, b=None, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def _lines(nt, vec, spacing, width, angle=0.0):
    """平行细线：返回线上为 1、其余为 0"""
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(vec, sep.inputs[0])
    ca, sa = math.cos(angle), math.sin(angle)
    u = _math(nt, "ADD", _math(nt, "MULTIPLY", sep.outputs["X"], ca),
              _math(nt, "MULTIPLY", sep.outputs["Y"], sa))
    f = _math(nt, "FRACT", _math(nt, "DIVIDE", u, spacing))
    d = _math(nt, "MINIMUM", f, _math(nt, "SUBTRACT", 1.0, f))
    return _math(nt, "LESS_THAN", d, width / spacing / 2)


def _pattern(nt, kind, scale, plane, radius=1.0):
    """返回 0..1 的遮罩；1 的地方颜色会被乘上 dark（<1 变暗，>1 变亮）"""
    N, L = nt.nodes.new, nt.links.new
    vec = _coords(nt, plane, radius)
    if kind == "bricks":                      # 石块/砖：灰缝 + 每块轻微色差
        t = N("ShaderNodeTexBrick")
        t.inputs["Scale"].default_value = scale
        t.inputs["Mortar Size"].default_value = 0.012
        t.inputs["Mortar Smooth"].default_value = 0.2
        t.inputs["Color1"].default_value = (0, 0, 0, 1)
        t.inputs["Color2"].default_value = (0.35, 0.35, 0.35, 1)
        t.inputs["Mortar"].default_value = (1, 1, 1, 1)
        L(vec, t.inputs["Vector"])
        bw = N("ShaderNodeRGBToBW")
        L(t.outputs["Color"], bw.inputs[0])
        return bw.outputs[0]
    if kind == "slate":                       # 石板瓦：细灰缝，块更扁
        t = N("ShaderNodeTexBrick")
        t.inputs["Scale"].default_value = scale
        t.inputs["Mortar Size"].default_value = 0.03
        t.inputs["Row Height"].default_value = 0.18
        t.inputs["Brick Width"].default_value = 0.45
        t.inputs["Color1"].default_value = (0, 0, 0, 1)
        t.inputs["Color2"].default_value = (0.4, 0.4, 0.4, 1)
        t.inputs["Mortar"].default_value = (1, 1, 1, 1)
        L(vec, t.inputs["Vector"])
        bw = N("ShaderNodeRGBToBW")
        L(t.outputs["Color"], bw.inputs[0])
        return bw.outputs[0]
    if kind == "tiles":                       # 广场地砖：方格 + 斜线组成星形图案
        s = scale
        m = _lines(nt, vec, s, 0.035)
        for ang, sp in ((math.pi / 2, s), (math.pi / 4, s * math.sqrt(0.5)),
                        (-math.pi / 4, s * math.sqrt(0.5))):
            m = _math(nt, "MAXIMUM", m, _lines(nt, vec, sp, 0.02 if abs(ang) < 1 else 0.035, ang))
        return m
    if kind == "scallop":                     # 枫丹的同心拱纹（花园挡土墙浮雕）
        sep = N("ShaderNodeSeparateXYZ")
        L(vec, sep.inputs[0])
        fu = _math(nt, "SUBTRACT", _math(nt, "FRACT", _math(nt, "DIVIDE", sep.outputs["X"], scale)), 0.5)
        fz = _math(nt, "FRACT", _math(nt, "DIVIDE", sep.outputs["Y"], scale))
        dist = _math(nt, "SQRT", _math(nt, "ADD", _math(nt, "MULTIPLY", fu, fu), _math(nt, "MULTIPLY", fz, fz)))
        ring = _math(nt, "FRACT", _math(nt, "MULTIPLY", dist, 7.0))
        return _math(nt, "LESS_THAN", ring, 0.22)
    if kind == "hlines":                      # 水平线：金属、木纹
        return _lines(nt, vec, scale, scale * 0.12, math.pi / 2)
    raise ValueError(kind)


def toon(name, color, pattern=None, scale=1.0, plane="W", dark=0.85, shadow=SHADOW,
         emit=0.0, sheen=None, radius=1.0, alpha=1.0, image=None, stains=0.0):
    """
    卡通材质（EEVEE）：Diffuse → Shader to RGB → 柔和两段色阶 → 乘底色 → Emission
    Cycles 下自动退化为 Principled BSDF。
      pattern/scale/plane/dark  程序纹理（石块、瓦片、地砖……）
      sheen=(hex, amount)       视角掠射时的亮边，用于金饰、玻璃
      emit                      自发光（灯、水）
      image=dict(color=, height=, size=(宽米, 高米), rot=度, bump=强度)
                                平铺贴图（地砖）：按世界尺寸平铺，高度图做凹凸
                                也可以用 basis=((ux,uy),(vx,vy)) 按任意平行四边形周期平铺
      stains                    地面水渍强度（0 = 没有）
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
    normal = None

    if image:
        tc = N("ShaderNodeTexCoord")
        if "basis" in image:
            # 世界坐标 → 平行四边形周期坐标 (u, v)：乘以 [U V] 矩阵的逆
            (ux, uy), (vx, vy) = image["basis"]
            det = ux * vy - vx * uy
            rows = ((vy / det, -vx / det), (-uy / det, ux / det))
            comb = N("ShaderNodeCombineXYZ")
            for axis, (r0, r1) in zip(("X", "Y"), rows):
                dp = N("ShaderNodeVectorMath"); dp.operation = "DOT_PRODUCT"
                L(tc.outputs["Object"], dp.inputs[0])
                dp.inputs[1].default_value = (r0, r1, 0)
                L(dp.outputs["Value"], comb.inputs[axis])
            uv = comb.outputs[0]
        else:
            mp = N("ShaderNodeMapping")
            sx, sy = image["size"]
            mp.inputs["Scale"].default_value = (1 / sx, 1 / sy, 1)
            mp.inputs["Rotation"].default_value = (0, 0, math.radians(image.get("rot", 0)))
            L(tc.outputs["Object"], mp.inputs["Vector"])
            uv = mp.outputs[0]
        it = N("ShaderNodeTexImage")
        it.image = image["color"]
        it.interpolation = "Cubic"
        L(uv, it.inputs["Vector"])
        base = it.outputs["Color"]
        if image.get("height"):
            ht = N("ShaderNodeTexImage")
            ht.image = image["height"]
            L(uv, ht.inputs["Vector"])
            bp = N("ShaderNodeBump")
            bp.inputs["Strength"].default_value = image.get("bump", 0.6)
            bp.inputs["Distance"].default_value = 0.01
            L(ht.outputs["Color"], bp.inputs["Height"])
            normal = bp.outputs["Normal"]

    if stains > 0:
        # 水渍：大块低频噪声 → 阈值 → 局部压暗（“意思意思”，不追求还原）
        tcs = N("ShaderNodeTexCoord")
        nz = N("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 0.35
        nz.inputs["Detail"].default_value = 8.0
        nz.inputs["Roughness"].default_value = 0.65
        L(tcs.outputs["Object"], nz.inputs["Vector"])
        rp = N("ShaderNodeValToRGB")
        rp.color_ramp.elements[0].position = 0.56
        rp.color_ramp.elements[1].position = 0.68
        L(nz.outputs["Fac"], rp.inputs["Fac"])
        k = _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", rp.outputs["Color"], stains))
        sc = N("ShaderNodeVectorMath"); sc.operation = "SCALE"
        L(base, sc.inputs[0]); L(k, sc.inputs["Scale"])
        base = sc.outputs[0]

    if pattern:
        f = _pattern(nt, pattern, scale, plane, radius)
        mr = N("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 1.0
        mr.inputs["To Max"].default_value = dark
        L(f, mr.inputs["Value"])
        sc = N("ShaderNodeVectorMath"); sc.operation = "SCALE"
        L(base, sc.inputs[0]); L(mr.outputs[0], sc.inputs["Scale"])
        base = sc.outputs[0]

    if sheen:
        lw = N("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.35
        fac = _math(nt, "MULTIPLY", lw.outputs["Facing"], sheen[1])
        mix = N("ShaderNodeMix"); mix.data_type = "RGBA"
        L(fac, mix.inputs[0])
        L(base, mix.inputs[6])
        mix.inputs[7].default_value = srgb(sheen[0])
        base = mix.outputs[2]

    em = N("ShaderNodeEmission")
    if emit > 0:
        L(base, em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0 + emit
    else:
        diff = N("ShaderNodeBsdfDiffuse")
        diff.inputs["Color"].default_value = (1, 1, 1, 1)
        if normal:
            L(normal, diff.inputs["Normal"])
        s2r = N("ShaderNodeShaderToRGB")
        L(diff.outputs[0], s2r.inputs[0])
        ramp = N("ShaderNodeValToRGB")
        cr = ramp.color_ramp
        cr.elements[0].position = 0.30
        cr.elements[0].color = (*shadow, 1)
        cr.elements[1].position = 0.55
        cr.elements[1].color = (1, 1, 1, 1)
        L(s2r.outputs["Color"], ramp.inputs["Fac"])
        # 环境光遮蔽：让墙角、窗框内侧有一点体积感
        ao = N("ShaderNodeAmbientOcclusion")
        ao.inputs["Distance"].default_value = 0.6
        aof = _math(nt, "ADD", _math(nt, "MULTIPLY", ao.outputs["AO"], 0.25), 0.75)
        sc2 = N("ShaderNodeVectorMath"); sc2.operation = "SCALE"
        L(ramp.outputs["Color"], sc2.inputs[0]); L(aof, sc2.inputs["Scale"])
        mul = N("ShaderNodeVectorMath"); mul.operation = "MULTIPLY"
        L(base, mul.inputs[0]); L(sc2.outputs[0], mul.inputs[1])
        L(mul.outputs[0], em.inputs["Color"])
        em.inputs["Strength"].default_value = 1.0

    surf = em.outputs[0]
    if alpha < 1.0:
        tr = N("ShaderNodeBsdfTransparent")
        mx = N("ShaderNodeMixShader")
        mx.inputs[0].default_value = alpha
        L(tr.outputs[0], mx.inputs[1]); L(surf, mx.inputs[2])
        surf = mx.outputs[0]
        try:
            m.surface_render_method = "BLENDED"
        except Exception:
            pass

    out_e = N("ShaderNodeOutputMaterial"); out_e.target = "EEVEE"
    L(surf, out_e.inputs["Surface"])

    pb = N("ShaderNodeBsdfPrincipled")
    L(base, pb.inputs["Base Color"])
    pb.inputs["Roughness"].default_value = 0.75
    pb.inputs["Alpha"].default_value = alpha
    if normal:
        L(normal, pb.inputs["Normal"])
    if emit > 0:
        L(base, pb.inputs["Emission Color"])
        pb.inputs["Emission Strength"].default_value = 1.0 + emit
    out_c = N("ShaderNodeOutputMaterial"); out_c.target = "CYCLES"
    L(pb.outputs[0], out_c.inputs["Surface"])
    return m


class Mats:
    """调色板：取自参考截图"""

    def __init__(self):
        # 主楼
        self.sand = toon("砂岩墙", "#EBC896", "bricks", 0.55, "W", 0.9)
        self.stone = toon("白石", "#EEE7DA")
        self.stone_block = toon("白石_分块", "#EEE7DA", "bricks", 0.45, "W", 0.93)
        self.gold = toon("金饰", "#D6A343", sheen=("#FFE7A0", 0.7))
        self.gold_dark = toon("暗金", "#A87A2A")
        self.slate = toon("石板瓦", "#8EA3AE", "slate", 1.6, "W", 0.86)
        self.metal = toon("青灰金属", "#5E7A82", sheen=("#A9C4CC", 0.5))
        self.glass = toon("青绿彩窗", "#2C9C98", sheen=("#8FE6DA", 0.8))
        self.glass_dark = toon("深青彩窗", "#1F6F6E", sheen=("#5CC5BC", 0.6))
        self.skylight = toon("天窗玻璃", "#6FB9B8", "hlines", 0.6, "XY", 0.8, sheen=("#D2FAF4", 0.8))
        self.door = toon("青铜门", "#3F6C68", sheen=("#7FB0A8", 0.4))
        # 书报摊
        self.awning = toon("粉色篷布", "#F39C98")
        self.awning_light = toon("浅粉篷布", "#F9CFC8")
        self.teal_frame = toon("青色铁架", "#3E7C74")
        self.books = [toon("书_%d" % i, c) for i, c in
                      enumerate(("#E8D7B0", "#9BC4A6", "#D98C6A", "#6E9BB8", "#F0E6CC"))]
        # 地面
        # 地砖贴图（程序生成，见 gen_plaza_tiles / gen_hex_pavers）
        pc, ph = gen_plaza_tiles()
        hc, hh, hu, hv = gen_hex_pavers()
        ca, sa = math.cos(math.radians(HEX_ROT)), math.sin(math.radians(HEX_ROT))
        hex_basis = tuple((ca * x - sa * y, sa * x + ca * y) for x, y in (hu, hv))
        self.tex = [make_image("广场地砖_颜色", pc), make_image("广场地砖_高度", ph, True),
                    make_image("风车砖_颜色", hc), make_image("风车砖_高度", hh, True)]
        self.plaza = toon("广场地砖", "#E7CDBF", stains=0.06, image=dict(
            color=self.tex[0], height=self.tex[1], size=(2 * PLAZA_TILE, 2 * PLAZA_TILE), rot=45, bump=0.5))
        hex_img = dict(color=self.tex[2], height=self.tex[3], basis=hex_basis, bump=0.6)
        self.sidewalk = toon("风车砖_人行道", "#B98A86", stains=0.1, image=hex_img)
        self.road = toon("风车砖_砖路", "#B98A86", stains=0.35, image=hex_img)
        self.manhole = toon("井盖铸铁", "#45403B", sheen=("#7A6E60", 0.4))
        self.curb = toon("路缘石", "#E4D6CC")
        self.road_curb = toon("灰色路缘石", "#B8B6B6")
        self.step = toon("台阶", "#DEDAD4")
        self.step_riser = toon("台阶踢面", "#B8B4B0")
        self.wall_relief = toon("拱纹挡土墙", "#E3DDD4", "scallop", 0.9, "W", 0.9)
        self.water = toon("水", "#8FE0EC", emit=0.15, sheen=("#E6FFFF", 0.6))
        self.water_jet = toon("喷泉水柱", "#E8FBFF", emit=0.4, alpha=0.8)
        # 远景建筑
        self.tower = toon("石塔", "#DEDAD3", "bricks", 0.32, "CYL", 0.93, radius=4.4)
        self.bigwall = toon("城墙", "#D8D5CF", "bricks", 0.3, "W", 0.93)
        self.bigwall_dark = toon("城墙_背光", "#B9BCC9")
        self.banner = toon("蓝色旗幡", "#63B7EE", sheen=("#C8ECFF", 0.8), alpha=0.9)
        self.copper = toon("铜顶", "#B98A5A", sheen=("#E6C39A", 0.5))
        # 植物
        self.grass = toon("草地", "#78C03E")
        self.hedge = toon("树篱", "#4FA23A")
        self.hedge_dark = toon("树篱_暗", "#3C8A33")
        self.flower = toon("黄花", "#F7C624")
        self.flower_white = toon("白花", "#F5F2E6")
        self.cypress = toon("柏树", "#1E9B7F")
        self.cypress_dark = toon("柏树_暗", "#12705F")
        self.bark = toon("树干", "#8A5A3A")
        self.pot = toon("花盆", "#E5DDD0")
        self.lamp_glass = toon("灯罩", "#EAF6F0", emit=0.3)


# ===========================================================================
# 地砖贴图：用 numpy 逐像素画出可无缝平铺的图案（颜色图 + 高度图）
# ===========================================================================


PLAZA_TILE = 1.7            # 广场大方砖边长（米），贴图里放 2×2 块
PLAZA_PX = 2048
HEX_G, HEX_SQ, HEX_CUT = 0.30, 0.48, 0.44   # 风车砖：小方砖边长 / 大方砖边长 / 六边形切角
HEX_CELLS = 4               # 贴图里放 4×4 个周期（色差不容易看出重复）
HEX_ROT = -63              # 整体旋转：按截图量出六边形长边与立面约成 -18°，拼法不与楼对齐
HEX_PX = 2048


def _hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _hash(*xs):
    """整数 → 0..1 的伪随机数（每块砖一个固定色差）"""
    s = np.zeros_like(np.asarray(xs[0], dtype=np.float64))
    for k, x in enumerate(xs):
        s = s + np.asarray(x, dtype=np.float64) * (12.9898 + 41.23 * k)
    return np.modf(np.abs(np.sin(s) * 43758.5453))[0]


def _cover(d, w, px):
    """到线中心的距离 d → 线宽 w 的覆盖率（抗锯齿 1 像素）"""
    return np.clip(0.5 - (d - w / 2) / px, 0.0, 1.0)


def _lowfreq_noise(x, y, seed):
    """几层正弦叠加的低频斑驳（石材的不均匀感），周期与贴图对齐保证无缝"""
    rng = np.random.RandomState(seed)
    n = np.zeros_like(x)
    for f in (1, 2, 3, 5, 8):
        for _ in range(2):
            ax, ay = rng.randint(-f, f + 1, size=2)
            ph = rng.uniform(0, 2 * np.pi)
            n += np.sin(2 * np.pi * (ax * x + ay * y) + ph) / f
    return n / 4.0


def gen_plaza_tiles(n=PLAZA_PX):
    """
    广场“画框”砖（参考截图 6）：
      大方砖（贴图里轴对齐，材质里再转 45°）
        ├ 中心小方块
        ├ 两圈砖条，沿对角线斜接（形成从中心放射的 X 线）
        └ 每圈按错缝切成小砖
      白色细缝，大砖之间是深一点的粉褐色缝
    返回 (rgb[H,W,3], height[H,W])，覆盖 2×2 块大砖
    """
    t = (np.arange(n) + 0.5) / n                       # 0..1
    X, Y = np.meshgrid(t, t)                           # Y 向上（第 0 行 = 贴图底部）
    gx, gy = X * 2, Y * 2
    ix, iy = np.floor(gx), np.floor(gy)
    u, v = (gx - ix) * 2 - 1, (gy - iy) * 2 - 1        # 每块砖内 -1..1
    px = 4.0 / n                                       # 一个像素在 u 单位里的大小
    au, av = np.abs(u), np.abs(v)
    r = np.maximum(au, av)

    R0, R1 = 0.30, 0.64                                # 中心块 / 内圈 / 外圈 分界
    ring = np.where(r < R0, 0, np.where(r < R1, 1, 2))
    horiz = au >= av
    side = np.where(horiz, np.where(u > 0, 0, 1), np.where(v > 0, 2, 3))
    along = np.where(horiz, v, u)                      # 沿砖条方向的坐标

    # 错缝：两圈砖条长度、起点不同
    # 内圈：接缝在 ±0.22（中间一块长砖 + 两端梯形砖）；外圈：接缝在 0、±0.5（与内圈错开）
    L = np.where(ring == 1, 0.44, 0.50)
    off = np.where(ring == 1, 0.22, 0.0)
    q = (along + off) / L
    brick = np.floor(q)
    d_joint = np.abs(q - np.round(q)) * L

    d_miter = np.abs(au - av) / np.sqrt(2)
    d_ring = np.minimum(np.abs(r - R0), np.abs(r - R1))
    inf = np.full_like(r, 9.0)
    d_joint = np.where(ring > 0, d_joint, inf)
    d_miter = np.where(r > R0 - 0.005, d_miter, inf)
    d_white = np.minimum(np.minimum(d_ring, d_miter), d_joint)
    d_white = np.where(r < 1 - 0.02, d_white, inf)

    white = _cover(d_white, 0.014, px)
    border = _cover(1 - r, 0.03, px)                   # 大砖之间的缝（离边 0 处）
    bevel = np.clip((1 - r) / 0.05, 0, 1)              # 大砖边缘的小倒角

    # 每小块颜色差异
    pid = np.where(ring == 0, 0, ring * 100 + side * 20 + brick + 50)
    h = _hash(ix * 7 + iy * 13, pid)
    base = _hex2rgb("#E8CFC0")
    warm = _hex2rgb("#F1D2BA")
    col = base[None, None, :] * (0.965 + 0.07 * h[..., None])
    col = col * (1 - 0.35 * h[..., None] * (h[..., None] > 0.8)) + warm * 0.35 * h[..., None] * (h[..., None] > 0.8)
    col *= (1 + 0.025 * _lowfreq_noise(X, Y, 3))[..., None]
    col *= (0.93 + 0.07 * bevel)[..., None]
    col = col * (1 - white[..., None]) + _hex2rgb("#F7EEE6") * white[..., None]
    col = col * (1 - border[..., None]) + _hex2rgb("#B58A80") * border[..., None]

    height = 0.35 + 0.65 * bevel ** 0.5
    height = height - 0.25 * white
    height = height * (1 - border)
    return np.clip(col, 0, 1), np.clip(height, 0, 1)


def gen_hex_pavers(n=HEX_PX, g=HEX_G, G=HEX_SQ, c=HEX_CUT, K=HEX_CELLS):
    """
    玫瑰色风车拼砖（参考截图 8、9）——四种砖：
      暗红小方砖（边长 g）、浅色大方砖（边长 G）、
      横向 / 竖向六边形 = 长方形切掉左上、右下两个角（切角直角边 c），带一圈内倒角线
    横竖六边形共用长斜边，小方砖和大方砖沿对角线角碰角，整体是旋转对称的“风车”排布。
    周期格子 a = (G+g+c, c)，b = (c, G+g+c)，每格正好 2 块六边形 + 1 大方 + 1 小方（面积严格相等）。

    贴图覆盖 K×K 个周期的平行四边形（u, v ∈ [0,1)），材质里用 basis 把世界坐标换算成 (u, v)。
    返回 (rgb, height, 贴图 u 方向世界向量, v 方向世界向量)
    """
    A1 = G + g + c
    a = np.array([A1, c]); b = np.array([c, A1])
    tiles = {   # 逆时针顶点
        0: [(-g, 0), (0, 0), (0, g), (-g, g)],                                               # 小方砖
        1: [(0, 0), (G, 0), (G + c, c), (G + c, g + c), (c, g + c), (0, g)],                  # 横向六边形
        2: [(-g, g), (0, g), (c, g + c), (c, g + G + c), (-g + c, g + G + c), (-g, g + G)],   # 竖向六边形
        3: [(c, g + c), (G + c, g + c), (G + c, g + c + G), (c, g + c + G)],                  # 大方砖
    }
    t = (np.arange(n) + 0.5) / n * K
    U, V = np.meshgrid(t, t)
    PX = U * a[0] + V * b[0]
    PY = U * a[1] + V * b[1]
    px = A1 * K / n
    i0, j0 = np.floor(U), np.floor(V)
    best = np.full(U.shape, 1e9)
    btype = np.zeros(U.shape, int)
    bi = np.zeros(U.shape); bj = np.zeros(U.shape)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            I, J = i0 + di, j0 + dj
            qx = PX - I * a[0] - J * b[0]
            qy = PY - I * a[1] - J * b[1]
            for k, poly in tiles.items():
                sd = np.full(U.shape, -1e9)          # 凸多边形有向距离（负 = 在砖内）
                for m in range(len(poly)):
                    (x0, y0), (x1, y1) = poly[m], poly[(m + 1) % len(poly)]
                    ex, ey = x1 - x0, y1 - y0
                    sd = np.maximum(sd, ((qx - x0) * ey - (qy - y0) * ex) / np.hypot(ex, ey))
                cl = sd < best
                best = np.where(cl, sd, best)
                btype = np.where(cl, k, btype)
                bi = np.where(cl, I, bi); bj = np.where(cl, J, bj)
    d = np.maximum(-best, 0)
    is_hex = (btype == 1) | (btype == 2)

    grout = _cover(d, 0.012, px)
    inset = _cover(np.abs(d - 0.055), 0.01, px) * is_hex
    bevel = np.clip(d / 0.05, 0, 1)
    h = _hash(np.mod(bi, K), np.mod(bj, K) + 7, btype)
    cols = np.stack([_hex2rgb("#9C5E63"), _hex2rgb("#B98A86"), _hex2rgb("#B98A86"), _hex2rgb("#C0928D")])
    col = cols[btype] * (0.93 + 0.12 * h[..., None])
    col *= (1 + 0.035 * _lowfreq_noise(U / K, V / K, 5))[..., None]
    col *= (0.88 + 0.12 * bevel)[..., None]
    col *= (1 - 0.12 * inset)[..., None]
    col = col * (1 - grout[..., None]) + _hex2rgb("#6E4A48") * grout[..., None]
    height = (0.3 + 0.7 * bevel ** 0.6 - 0.2 * inset) * (1 - grout)
    return np.clip(col, 0, 1), np.clip(height, 0, 1), a * K, b * K


def make_image(name, arr, non_color=False):
    """numpy 数组 → Blender 图像（打包进 .blend，不依赖外部文件）"""
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    rgba = np.concatenate([arr, np.ones((h, w, 1))], axis=2).astype(np.float32)
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.pixels.foreach_set(rgba.ravel())
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    img.pack()
    return img


def export_images(folder, *imgs):
    os.makedirs(folder, exist_ok=True)
    for img in imgs:
        img.filepath_raw = os.path.join(folder, img.name + ".png")
        img.file_format = "PNG"
        img.save()


# ===========================================================================
# 几何工具
# ===========================================================================

def collection(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    return c


class Frame:
    """
    墙面局部坐标系：u 沿墙水平、z 向上、d 沿外法线。
    origin 是墙面左下角（面向墙时的左边）。
    """

    def __init__(self, origin, u_dir, n_dir):
        self.o = Vector(origin)
        self.u = Vector(u_dir).normalized()
        self.n = Vector(n_dir).normalized()
        self.R = Matrix((self.u, self.n, (0, 0, 1))).transposed().to_4x4()

    def p(self, u, z, d=0.0):
        return self.o + self.u * u + Vector((0, 0, z)) + self.n * d


class Part:
    """一个 Part = 一个 Blender 对象，可含多种材质"""

    def __init__(self, name, coll, bevel=0.0, bevel_segments=2):
        self.name, self.coll = name, coll
        self.bm = bmesh.new()
        self.mats = []
        self.bevel = bevel
        self.bevel_segments = bevel_segments
        self.modifiers = []

    # -- 内部 --------------------------------------------------------------
    def _mi(self, mat):
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def _tag_faces(self, faces, mat, smooth=False):
        mi = self._mi(mat)
        for f in faces:
            f.material_index = mi
            f.smooth = smooth

    def _tag(self, verts, mat, smooth=False):
        self._tag_faces({f for v in verts for f in v.link_faces}, mat, smooth)

    @staticmethod
    def M(loc, rot=(0, 0, 0), size=(1, 1, 1)):
        R = rot.to_matrix().to_4x4() if hasattr(rot, "to_matrix") else Euler(rot).to_matrix().to_4x4()
        return Matrix.Translation(loc) @ R @ Matrix.Diagonal((*size, 1.0))

    # -- 基本体 --------------------------------------------------------------
    def box(self, c, s, mat, rot=(0, 0, 0)):
        r = bmesh.ops.create_cube(self.bm, size=1.0, matrix=self.M(c, rot, s))
        self._tag(r["verts"], mat)

    def box_mm(self, x0, x1, y0, y1, z0, z1, mat):
        self.box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                 (abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)), mat)

    def cyl(self, c, r, h, mat, seg=16, r2=None, rot=(0, 0, 0), smooth=True):
        res = bmesh.ops.create_cone(self.bm, cap_ends=True, cap_tris=False, segments=seg,
                                    radius1=r, radius2=r if r2 is None else r2, depth=h,
                                    matrix=self.M(c, rot))
        self._tag(res["verts"], mat, smooth)

    def ico(self, c, r, mat, scale=(1, 1, 1), sub=1, jitter=0.0, smooth=False):
        res = bmesh.ops.create_icosphere(self.bm, subdivisions=sub, radius=r,
                                         matrix=self.M(c, (0, 0, 0), scale))
        if jitter:
            for v in res["verts"]:
                v.co += Vector([RNG.uniform(-1, 1) for _ in range(3)]) * jitter
        self._tag(res["verts"], mat, smooth)

    def beam(self, p0, p1, w, h, mat, up="Z"):
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        q = d.to_track_quat("X", up)
        m = Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((d.length, w, h, 1))
        r = bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)
        self._tag(r["verts"], mat)

    def cone_dir(self, base, direction, r, length, mat, seg=5):
        """从 base 出发沿 direction 的尖锥（柏树叶簇）"""
        d = Vector(direction).normalized()
        q = d.to_track_quat("Z", "Y")
        c = Vector(base) + d * (length / 2)
        res = bmesh.ops.create_cone(self.bm, cap_ends=True, cap_tris=False, segments=seg,
                                    radius1=r, radius2=0.0, depth=length,
                                    matrix=Matrix.Translation(c) @ q.to_matrix().to_4x4())
        self._tag(res["verts"], mat)

    def prism(self, profile, mat, to3d, smooth=False):
        """二维轮廓挤出：to3d(u, v, t)，t∈{0,1}"""
        bm = self.bm
        a = [bm.verts.new(to3d(u, v, 0)) for u, v in profile]
        b = [bm.verts.new(to3d(u, v, 1)) for u, v in profile]
        n = len(profile)
        faces = [bm.faces.new(a), bm.faces.new(list(reversed(b)))]
        for i in range(n):
            j = (i + 1) % n
            faces.append(bm.faces.new((a[i], a[j], b[j], b[i])))
        self._tag_faces(faces, mat, smooth)

    def lathe(self, c, profile, mat, seg=24, smooth=True, caps=True):
        """车削：profile = [(半径, 高度), ...] 自下而上"""
        bm = self.bm
        cx, cy, cz = c
        rings = []
        for r, z in profile:
            if r < 1e-5:
                rings.append([bm.verts.new((cx, cy, cz + z))])
            else:
                rings.append([bm.verts.new((cx + r * math.cos(a), cy + r * math.sin(a), cz + z))
                              for a in (i / seg * math.tau for i in range(seg))])
        faces = []
        for A, B in zip(rings, rings[1:]):
            for i in range(seg):
                j = (i + 1) % seg
                if len(A) == 1 and len(B) == 1:
                    continue
                if len(A) == 1:
                    faces.append(bm.faces.new((A[0], B[i], B[j])))
                elif len(B) == 1:
                    faces.append(bm.faces.new((A[i], A[j], B[0])))
                else:
                    faces.append(bm.faces.new((A[i], A[j], B[j], B[i])))
        if caps and len(rings[0]) > 1:
            faces.append(bm.faces.new(list(reversed(rings[0]))))
        if caps and len(rings[-1]) > 1:
            faces.append(bm.faces.new(rings[-1]))
        self._tag_faces(faces, mat, smooth)

    def sweep(self, path, profile, mat, closed=True, cap=False):
        """
        截面沿水平路径扫掠（线脚、檐口、屋顶）。
        path    = [(x, y), ...] 逆时针（从上往下看），截面自动朝路径外侧
        profile = [(d, z), ...] d = 向外偏移，z = 高度
        """
        bm = self.bm
        pts = [Vector((x, y, 0)) for x, y in path]
        n = len(pts)
        rings = []
        for i in range(n):
            if closed or 0 < i < n - 1:
                e0 = (pts[i] - pts[i - 1]).normalized()
                e1 = (pts[(i + 1) % n] - pts[i]).normalized()
                n0 = Vector((e0.y, -e0.x, 0))
                n1 = Vector((e1.y, -e1.x, 0))
                m = (n0 + n1).normalized()
                s = 1.0 / max(0.3, m.dot(n0))
            else:
                e = (pts[1] - pts[0]) if i == 0 else (pts[-1] - pts[-2])
                e.normalize()
                m, s = Vector((e.y, -e.x, 0)), 1.0
            rings.append([bm.verts.new(pts[i] + m * d * s + Vector((0, 0, z))) for d, z in profile])
        faces = []
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            for k in range(len(profile) - 1):
                faces.append(bm.faces.new((rings[i][k], rings[j][k], rings[j][k + 1], rings[i][k + 1])))
        if cap and not closed:
            for r in (rings[0], rings[-1]):
                if len(r) >= 3:
                    faces.append(bm.faces.new(r))
        self._tag_faces(faces, mat)
        return rings

    # -- 墙面坐标系下的构件 --------------------------------------------------
    def fbox(self, F, u0, u1, z0, z1, d0, d1, mat):
        c = F.p((u0 + u1) / 2, (z0 + z1) / 2, (d0 + d1) / 2)
        m = Matrix.Translation(c) @ F.R @ Matrix.Diagonal((abs(u1 - u0), abs(d1 - d0), abs(z1 - z0), 1))
        r = bmesh.ops.create_cube(self.bm, size=1.0, matrix=m)
        self._tag(r["verts"], mat)

    def fpoly(self, F, poly, d0, d1, mat, du=0.0, dz=0.0):
        """墙面上的多边形浮雕/面板"""
        self.prism(poly, mat, lambda u, z, t: F.p(u + du, z + dz, d0 + (d1 - d0) * t))

    def fring(self, F, outer, inner, d0, d1, mat, du=0.0, dz=0.0):
        """两个同顶点数多边形之间的“框”（窗框、门框）"""
        bm = self.bm
        n = len(outer)
        P = lambda p, d: bm.verts.new(F.p(p[0] + du, p[1] + dz, d))
        Of = [P(p, d1) for p in outer]
        If = [P(p, d1) for p in inner]
        Ob = [P(p, d0) for p in outer]
        Ib = [P(p, d0) for p in inner]
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces += [bm.faces.new((Of[i], Of[j], If[j], If[i])),
                      bm.faces.new((Ob[i], Ib[i], Ib[j], Ob[j])),
                      bm.faces.new((Ob[i], Ob[j], Of[j], Of[i])),
                      bm.faces.new((Ib[i], If[i], If[j], Ib[j]))]
        self._tag_faces(faces, mat)

    def fband(self, F, pts, t, d0, d1, mat):
        """沿墙面上折线的窄带（装饰线条）"""
        for (u0, z0), (u1, z1) in zip(pts, pts[1:]):
            dx, dz = u1 - u0, z1 - z0
            L = math.hypot(dx, dz)
            nx, nz = -dz / L * t / 2, dx / L * t / 2
            ex, ez = dx / L * t / 2, dz / L * t / 2       # 两端稍微延长，接缝不露缝
            self.fpoly(F, [(u0 - nx - ex, z0 - nz - ez), (u1 - nx + ex, z1 - nz + ez),
                           (u1 + nx + ex, z1 + nz + ez), (u0 + nx - ex, z0 + nz - ez)], d0, d1, mat)

    def farc(self, F, cu, cz, r0, r1, a0, a1, d0, d1, mat, n=12):
        """墙面上的圆弧带"""
        for i in range(n):
            t0 = a0 + (a1 - a0) * i / n
            t1 = a0 + (a1 - a0) * (i + 1) / n
            self.fpoly(F, [(cu + r0 * math.cos(t0), cz + r0 * math.sin(t0)),
                           (cu + r0 * math.cos(t1), cz + r0 * math.sin(t1)),
                           (cu + r1 * math.cos(t1), cz + r1 * math.sin(t1)),
                           (cu + r1 * math.cos(t0), cz + r1 * math.sin(t0))], d0, d1, mat)

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
        if self.bevel > 0:
            b = ob.modifiers.new("倒角", "BEVEL")
            b.width = self.bevel
            b.segments = self.bevel_segments
            b.limit_method = "ANGLE"
            b.angle_limit = math.radians(40)
            b.use_clamp_overlap = True
        for fn in self.modifiers:
            fn(ob)
        return ob


# ===========================================================================
# 形状函数（墙面二维轮廓）
# ===========================================================================

def coffin(w, h, c, t=0.0):
    """切角长方形（上下四角斜切）——装饰艺术风格窗。t = 向外扩的厚度"""
    w, h, c = w + 2 * t, h + 2 * t, c + t * 0.414
    return [(-w / 2, c - t), (-w / 2 + c, -t), (w / 2 - c, -t), (w / 2, c - t),
            (w / 2, h - c - t), (w / 2 - c, h - t), (-w / 2 + c, h - t), (-w / 2, h - c - t)]


def arched(w, h, rise, t=0.0, n=12):
    """直边 + 椭圆拱顶"""
    w, h, rise = w + 2 * t, h + 2 * t, rise + t
    pts = [(-w / 2, -t), (w / 2, -t)]
    for i in range(n + 1):
        a = math.pi * i / n
        pts.append((w / 2 * math.cos(a), h - t - rise + rise * math.sin(a)))
    return pts


def shift(poly, du, dz):
    return [(u + du, z + dz) for u, z in poly]


# ===========================================================================
# 主楼
# ===========================================================================

# 主楼尺寸（米）
BW, BD = 15.0, 10.0          # 宽、深
BX0, BY0 = -7.5, 0.0         # 左前角
Z_PLINTH = 0.55              # 勒脚
Z_BELT0, Z_BELT1 = 4.7, 5.2  # 腰线
Z_CORNICE = 8.5              # 檐口起点
Z_EAVE = 9.3                 # 檐口顶 / 屋顶起点
Z_ROOF = 12.0                # 屋顶平台
ROOF_IN = 1.9                # 屋顶平台内收


def footprint(inset=0.0):
    """主楼平面（逆时针）"""
    x0, y0, x1, y1 = BX0 - inset, BY0 - inset, BX0 + BW + inset, BY0 + BD + inset
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def deco_upper_window(P, M, F, cu, zb, w=1.7, h=2.55):
    """二层彩窗：切角窗 + 白石窗套 + 金色扇形/齿轮/竖线装饰"""
    c = 0.32
    inner = shift(coffin(w, h, c), cu, zb)
    P.fpoly(F, inner, -0.05, 0.02, M.glass)                              # 玻璃
    P.fring(F, shift(coffin(w, h, c, 0.07), cu, zb), inner, 0.0, 0.07, M.gold)   # 金色内框
    P.fring(F, shift(coffin(w, h, c, 0.26), cu, zb),
            shift(coffin(w, h, c, 0.07), cu, zb), 0.0, 0.16, M.stone)    # 白石窗套
    P.fbox(F, cu - w / 2 - 0.35, cu + w / 2 + 0.35, zb - 0.42, zb - 0.26, 0.0, 0.26, M.stone)  # 窗台
    g = 0.045
    D0, D1 = 0.02, 0.05
    # 竖向金线
    for du, z0, z1 in ((0, 0.45, h * 0.62), (-0.36, 0.3, h - 0.55), (0.36, 0.3, h - 0.55),
                       (-0.62, c + 0.1, h - c - 0.1), (0.62, c + 0.1, h - c - 0.1)):
        P.fbox(F, cu + du - g / 2, cu + du + g / 2, zb + z0, zb + z1, D0, D1, M.gold)
    # 顶部扇形
    fz = zb + h * 0.62
    for k in range(-3, 4):
        a = math.pi / 2 + k * 0.28
        P.fband(F, [(cu, fz), (cu + math.cos(a) * 0.8, fz + math.sin(a) * 0.72)], g, D0, D1, M.gold)
    P.farc(F, cu, fz, 0.52, 0.58, math.radians(12), math.radians(168), D0, D1, M.gold, 10)
    # 底部半个齿轮（枫丹的机械元素）
    gz = zb + 0.12
    P.farc(F, cu, gz, 0.26, 0.36, 0, math.pi, D0, D1, M.gold, 10)
    for k in range(7):
        a = math.pi * (k + 0.5) / 7
        P.fband(F, [(cu + math.cos(a) * 0.34, gz + math.sin(a) * 0.34),
                    (cu + math.cos(a) * 0.46, gz + math.sin(a) * 0.46)], 0.09, D0, D1, M.gold)
    # 横向金线 + 两侧阶梯纹
    P.fbox(F, cu - w / 2 + 0.1, cu + w / 2 - 0.1, zb + h * 0.33, zb + h * 0.33 + g, D0, D1, M.gold)
    for s in (-1, 1):
        P.fband(F, [(cu + s * 0.36, zb + h - 0.55), (cu + s * 0.62, zb + h - 0.85)], g, D0, D1, M.gold)


def deco_arch_window(P, M, F, cu, zb, w=2.4, h=3.0, rise=0.75):
    """一层大拱窗：金框 + 竖线与阶梯拱装饰"""
    inner = shift(arched(w, h, rise), cu, zb)
    P.fpoly(F, inner, -0.05, 0.02, M.glass)
    P.fring(F, shift(arched(w, h, rise, 0.12), cu, zb), inner, 0.0, 0.12, M.gold)
    P.fring(F, shift(arched(w, h, rise, 0.3), cu, zb),
            shift(arched(w, h, rise, 0.12), cu, zb), 0.0, 0.08, M.gold_dark)
    P.fbox(F, cu - w / 2 - 0.45, cu + w / 2 + 0.45, zb - 0.45, zb - 0.3, 0.0, 0.3, M.stone)
    g = 0.045
    D0, D1 = 0.02, 0.05
    for du in (-0.75, -0.45, -0.15, 0.15, 0.45, 0.75):
        P.fbox(F, cu + du - g / 2, cu + du + g / 2, zb + 0.1, zb + h - rise * 0.6 - abs(du) * 0.4, D0, D1, M.gold)
    # 两组小拱（左右各一）
    for s in (-1, 1):
        P.farc(F, cu + s * 0.6, zb + 1.1, 0.42, 0.47, 0, math.pi, D0, D1, M.gold, 8)
        P.farc(F, cu + s * 0.6, zb + 1.1, 0.25, 0.3, 0, math.pi, D0, D1, M.gold, 8)
    P.farc(F, cu, zb + h - rise - 0.1, 0.7, 0.75, 0, math.pi, D0, D1, M.gold, 10)
    # 底部阶梯
    for k in range(3):
        P.fbox(F, cu - 0.9 + k * 0.25, cu + 0.9 - k * 0.25, zb + 0.15 + k * 0.18,
               zb + 0.15 + k * 0.18 + g, D0, D1, M.gold)


def deco_door(P, M, F, cu, zb):
    """大门：石门套 + 拱形雨棚 + 青铜双开门"""
    dw, dh = 1.9, 3.0
    # 门洞（深色门板）
    P.fbox(F, cu - dw / 2, cu + dw / 2, zb, zb + dh, -0.3, 0.02, M.door)
    # 门上半部拱形玻璃
    P.fpoly(F, shift(arched(0.7, 1.3, 0.35), cu, zb + 1.35), 0.02, 0.05, M.glass)
    P.fring(F, shift(arched(0.7, 1.3, 0.35, 0.06), cu, zb + 1.35),
            shift(arched(0.7, 1.3, 0.35), cu, zb + 1.35), 0.02, 0.07, M.gold)
    # 门板竖向金线
    for du in (-0.65, -0.52, 0.52, 0.65):
        P.fbox(F, cu + du - 0.02, cu + du + 0.02, zb + 0.3, zb + dh - 0.3, 0.02, 0.06, M.gold)
    P.fbox(F, cu - 0.015, cu + 0.015, zb, zb + dh, 0.02, 0.05, M.metal)
    P.fbox(F, cu + 0.12, cu + 0.16, zb + 1.15, zb + 1.35, 0.03, 0.1, M.gold)       # 门把手
    # 门楣上的深色扇窗
    P.fbox(F, cu - dw / 2, cu + dw / 2, zb + dh, zb + dh + 0.55, -0.1, 0.02, M.glass_dark)
    P.fpoly(F, [(cu - 0.18, zb + dh + 0.05), (cu + 0.18, zb + dh + 0.05), (cu, zb + dh + 0.45)],
            0.02, 0.06, M.gold)
    # 门套：两侧壁柱
    for s in (-1, 1):
        x = cu + s * (dw / 2 + 0.22)
        P.fbox(F, x - 0.26, x + 0.26, zb, zb + dh + 0.6, 0.0, 0.38, M.stone)
        P.fbox(F, x - 0.32, x + 0.32, zb, zb + 0.35, 0.0, 0.44, M.stone)
        P.fbox(F, x - 0.32, x + 0.32, zb + dh + 0.45, zb + dh + 0.65, 0.0, 0.46, M.stone)
    # 拱形雨棚：白色弧形正面 + 石板瓦顶
    hw = dw / 2 + 0.75
    zs = zb + dh + 0.6
    n = 14
    top = []
    for i in range(n + 1):
        t = i / n
        u = -hw + 2 * hw * t
        top.append((cu + u, zs + 0.55 + 0.45 * math.sin(math.pi * t)))
    bot = [(u, z - 0.42) for u, z in top]
    for i in range(n):
        P.fpoly(F, [bot[i], bot[i + 1], top[i + 1], top[i]], 0.0, 1.05, M.stone)
    for i in range(n):
        (u0, z0), (u1, z1) = top[i], top[i + 1]
        P.fpoly(F, [(u0, z0), (u1, z1), (u1, z1 + 0.14), (u0, z0 + 0.14)], -0.05, 0.95, M.slate)
    # 雨棚两端的涡卷托
    for s in (-1, 1):
        P.fbox(F, cu + s * hw - 0.22, cu + s * hw + 0.22, zs - 0.35, zs + 0.55, 0.0, 1.0, M.stone)
    # 雨棚下的小装饰牌
    P.fpoly(F, [(cu - 0.25, zs + 0.2), (cu + 0.25, zs + 0.2), (cu + 0.12, zs - 0.15), (cu - 0.12, zs - 0.15)],
            0.0, 0.1, M.glass)


def pilaster(P, M, F, cu, w=0.9, corner=False):
    """白石壁柱：一层带拱形龛 + 圆肩，二层带金色竖线，檐口上方是金色“书本”柱头"""
    hw = w / 2
    d = 0.32
    # 一层
    P.fbox(F, cu - hw, cu + hw, 0.0, Z_BELT0, 0.0, d, M.stone_block)
    P.fbox(F, cu - hw - 0.08, cu + hw + 0.08, 0.0, 0.55, 0.0, d + 0.08, M.stone)
    niche = shift(arched(w * 0.42, 2.2, 0.21), cu, 1.4)
    P.fpoly(F, niche, d, d + 0.03, M.stone)
    P.fring(F, shift(arched(w * 0.42, 2.2, 0.21, 0.05), cu, 1.4), niche, d, d + 0.06, M.gold)
    P.fbox(F, cu - 0.02, cu + 0.02, 1.6, 3.2, d + 0.02, d + 0.06, M.gold)
    # 圆肩（腰线下）
    P.fpoly(F, shift(arched(w + 0.1, 0.75, 0.3), cu, Z_BELT0 - 0.45), 0.0, d + 0.1, M.stone)
    # 二层
    P.fbox(F, cu - hw * 0.85, cu + hw * 0.85, Z_BELT1, Z_CORNICE + 0.3, 0.0, d, M.stone)
    P.fbox(F, cu - 0.06, cu + 0.06, Z_BELT1 + 0.3, Z_CORNICE, d - 0.02, d + 0.03, M.gold)
    P.fbox(F, cu - hw * 0.85 - 0.06, cu + hw * 0.85 + 0.06, Z_BELT1, Z_BELT1 + 0.3, 0.0, d + 0.06, M.stone)
    # 檐口上方：金色书本柱头
    zc = Z_EAVE - 0.1
    P.fbox(F, cu - hw * 0.8, cu + hw * 0.8, zc, zc + 0.35, -0.2, d + 0.2, M.stone)
    P.fbox(F, cu - hw * 0.7, cu + hw * 0.7, zc + 0.35, zc + 1.05, -0.1, d + 0.12, M.gold)
    for s in (-1, 1):   # 两页翻开的书
        P.fpoly(F, [(cu, zc + 1.0), (cu + s * hw * 0.7, zc + 1.05), (cu + s * hw * 0.72, zc + 1.4),
                    (cu + s * 0.06, zc + 1.18)], -0.1, d + 0.12, M.gold)
    P.fbox(F, cu - 0.03, cu + 0.03, zc + 0.45, zc + 0.95, d + 0.12, d + 0.15, M.gold_dark)


def bookstall(P, M, F, cu, w=3.1):
    """粉色遮阳篷书报摊"""
    hw = w / 2
    # 青色铁架：两侧弯腿 + 底座书架
    for s in (-1, 1):
        u = cu + s * (hw - 0.08)
        P.fbox(F, u - 0.07, u + 0.07, 0.1, 2.75, 0.3, 0.44, M.teal_frame)
        P.fband(F, [(u, 0.1), (u + s * 0.25, 0.0)], 0.1, 0.25, 0.5, M.teal_frame)
        # 伸出的斜撑（篷布骨架）
        P.beam(F.p(u, 2.75, 0.4), F.p(u, 2.55, 1.85), 0.07, 0.07, M.teal_frame)
    # 书架：三层，每层一排书
    P.fbox(F, cu - hw + 0.15, cu + hw - 0.15, 0.1, 2.2, 0.02, 0.3, M.teal_frame)
    for k, z in enumerate((0.55, 1.1, 1.65)):
        P.fbox(F, cu - hw + 0.15, cu + hw - 0.15, z - 0.04, z, 0.02, 0.55, M.teal_frame)
        u = cu - hw + 0.22
        while u < cu + hw - 0.3:
            bw = RNG.uniform(0.18, 0.32)
            bh = RNG.uniform(0.3, 0.45)
            tilt = RNG.uniform(0.0, 0.15)
            P.fbox(F, u, u + bw, z, z + bh, 0.08 + tilt, 0.45 + tilt, RNG.choice(M.books))
            u += bw + 0.02
    # 前面的小书箱
    P.fbox(F, cu - 1.1, cu - 0.3, 0.0, 0.35, 0.6, 1.2, M.teal_frame)
    for i in range(4):
        P.fbox(F, cu - 1.05 + i * 0.18, cu - 0.9 + i * 0.18, 0.35, 0.75, 0.75, 1.1, RNG.choice(M.books))
    # 篷布：斜面（粉/浅粉条纹）
    top_z, low_z, out_d = 3.05, 2.45, 1.95
    n = 10
    for i in range(n):
        u0 = cu - hw - 0.15 + (w + 0.3) * i / n
        u1 = cu - hw - 0.15 + (w + 0.3) * (i + 1) / n
        mat = M.awning if i % 2 == 0 else M.awning_light
        P.prism([(u0, 0), (u1, 0), (u1, 1), (u0, 1)], mat,
                lambda u, t_, t, u0=u0: F.p(u, top_z + (low_z - top_z) * t_ + 0.06 * t,
                                            0.15 + (out_d - 0.15) * t_))
    # 篷布顶部卷筒
    P.beam(F.p(cu - hw - 0.2, top_z + 0.08, 0.2), F.p(cu + hw + 0.2, top_z + 0.08, 0.2), 0.32, 0.32, M.awning)
    P.fbox(F, cu - hw - 0.25, cu + hw + 0.25, top_z - 0.05, top_z + 0.3, 0.0, 0.1, M.gold)
    # 波浪垂边
    k = 9
    pts = [(cu - hw - 0.15, low_z + 0.05)]
    for i in range(k):
        a = cu - hw - 0.15 + (w + 0.3) * i / k
        b = cu - hw - 0.15 + (w + 0.3) * (i + 1) / k
        for j in range(5):
            t = j / 4
            pts.append((a + (b - a) * t, low_z - 0.38 - 0.12 * math.sin(math.pi * t)))
    pts.append((cu + hw + 0.15, low_z + 0.05))
    P.fpoly(F, pts, out_d - 0.03, out_d + 0.03, M.awning_light)
    for i in range(k):   # 垂边上的竖条纹
        a = cu - hw - 0.15 + (w + 0.3) * (i + 0.5) / k
        P.fbox(F, a - 0.03, a + 0.03, low_z - 0.45, low_z, out_d + 0.03, out_d + 0.05, M.gold)
    # 两侧挡板
    for s in (-1, 1):
        u = cu + s * (hw + 0.15)
        P.prism([(0.15, top_z), (out_d, low_z), (out_d, low_z - 0.3), (0.15, top_z - 0.3)], M.awning,
                lambda d, z, t, u=u: F.p(u - 0.02 + 0.04 * t, z, d))


def facade(P, M, F, width, bays, pil_w=0.9):
    """
    一整面立面。bays 是每个开间一层/二层放什么：
      ground: 'window' 大拱窗 / 'door' 大门 / 'stall' 拱窗+书报摊 / None
      upper : 'window' 二层彩窗 / None
    """
    n = len(bays)
    inner = width - 2 * pil_w * 0.6
    bw = inner / n
    centers = [pil_w * 0.6 + bw * (i + 0.5) for i in range(n)]
    pils = [pil_w * 0.5 + 0.1] + [pil_w * 0.6 + bw * i for i in range(1, n)] + [width - pil_w * 0.5 - 0.1]
    for i, (cu, bay) in enumerate(zip(centers, bays)):
        g, up = bay
        if g == "window":
            deco_arch_window(P, M, F, cu, 1.25)
        elif g == "stall":
            deco_arch_window(P, M, F, cu, 1.25)
            bookstall(P, M, F, cu)
        elif g == "door":
            deco_door(P, M, F, cu, Z_PLINTH)
        if up == "window":
            deco_upper_window(P, M, F, cu, Z_BELT1 + 0.55)
        # 二层窗上方的“眉线”
        l, r = cu - bw / 2 + pil_w * 0.3, cu + bw / 2 - pil_w * 0.3
        P.fband(F, [(l, Z_CORNICE - 0.95), (l + 0.55, Z_CORNICE - 0.4), (r - 0.55, Z_CORNICE - 0.4),
                    (r, Z_CORNICE - 0.95)], 0.1, 0.0, 0.1, M.stone)
        P.fbox(F, cu - 0.14, cu + 0.14, Z_CORNICE - 0.55, Z_CORNICE - 0.25, 0.0, 0.1, M.stone)
        # 一层墙裙
        P.fbox(F, cu - bw / 2 + 0.1, cu + bw / 2 - 0.1, Z_PLINTH, Z_PLINTH + 0.18, 0.0, 0.08, M.stone)
    for cu in pils:
        pilaster(P, M, F, cu, pil_w)
    return centers


def build_building(M, root):
    C = collection("主楼", root)

    # 墙体
    P = Part("主楼_墙体", C, bevel=0.03)
    P.box_mm(BX0, BX0 + BW, BY0, BY0 + BD, 0.0, Z_CORNICE + 0.2, M.sand)
    # 勒脚（沿四周扫掠的线脚）
    P.sweep(footprint(), [(0.0, 0.0), (0.22, 0.0), (0.22, Z_PLINTH - 0.12), (0.12, Z_PLINTH), (0.0, Z_PLINTH)],
            M.stone)
    # 腰线
    P.sweep(footprint(), [(0.0, Z_BELT0), (0.16, Z_BELT0), (0.16, Z_BELT0 + 0.08), (0.26, Z_BELT0 + 0.2),
                          (0.26, Z_BELT1 - 0.1), (0.2, Z_BELT1), (0.0, Z_BELT1)], M.stone)
    P.finish()

    # 檐口 + 芒萨尔屋顶 + 天窗
    P = Part("主楼_屋顶", C, bevel=0.02)
    P.sweep(footprint(), [(0.0, Z_CORNICE), (0.1, Z_CORNICE), (0.1, Z_CORNICE + 0.15), (0.28, Z_CORNICE + 0.3),
                          (0.3, Z_CORNICE + 0.5), (0.48, Z_CORNICE + 0.62), (0.48, Z_EAVE),
                          (0.0, Z_EAVE)], M.stone)
    roof_prof = [(0.25, Z_EAVE), (0.18, Z_EAVE + 0.25), (-0.25, Z_EAVE + 1.2), (-0.8, Z_EAVE + 1.95),
                 (-1.4, Z_EAVE + 2.45), (-ROOF_IN, Z_ROOF)]
    P.sweep(footprint(), roof_prof, M.slate)
    # 屋脊金属包边（四个角 + 顶部边缘）
    fp = footprint()
    for i, (x, y) in enumerate(fp):
        cx, cy = BX0 + BW / 2, BY0 + BD / 2
        out = Vector((x - cx, y - cy, 0))
        ox, oy = (1 if out.x > 0 else -1), (1 if out.y > 0 else -1)
        pts = [Vector((x + ox * d, y + oy * d, z)) for d, z in roof_prof]
        for a, b in zip(pts, pts[1:]):
            P.beam(a + Vector((0, 0, 0.04)), b + Vector((0, 0, 0.04)), 0.16, 0.1, M.metal)
    top = footprint(-ROOF_IN)
    P.sweep(top, [(0.05, Z_ROOF - 0.05), (0.05, Z_ROOF + 0.25), (-0.15, Z_ROOF + 0.25), (-0.15, Z_ROOF)], M.metal)
    P.box_mm(top[0][0], top[1][0], top[0][1], top[2][1], Z_ROOF - 0.1, Z_ROOF + 0.05, M.metal)
    # 玻璃天窗（四坡）
    sx0, sx1, sy0, sy1 = top[0][0] + 0.6, top[1][0] - 0.6, top[0][1] + 0.6, top[2][1] - 0.6
    zt, zh = Z_ROOF + 0.25, 1.0
    bm = P.bm
    base = [bm.verts.new(v) for v in ((sx0, sy0, zt), (sx1, sy0, zt), (sx1, sy1, zt), (sx0, sy1, zt))]
    rid = [bm.verts.new((sx0 + 1.4, (sy0 + sy1) / 2, zt + zh)), bm.verts.new((sx1 - 1.4, (sy0 + sy1) / 2, zt + zh))]
    fs = [bm.faces.new((base[0], base[1], rid[1], rid[0])), bm.faces.new((base[2], base[3], rid[0], rid[1])),
          bm.faces.new((base[1], base[2], rid[1])), bm.faces.new((base[3], base[0], rid[0])),
          bm.faces.new(list(reversed(base)))]
    P._tag_faces(fs, M.skylight)
    # 天窗铁框
    for i in range(1, 8):
        t = i / 8
        xb = sx0 + (sx1 - sx0) * t
        xr = min(max(xb, sx0 + 1.4), sx1 - 1.4)
        for y0 in (sy0, sy1):
            P.beam((xb, y0, zt), (xr, (sy0 + sy1) / 2, zt + zh), 0.06, 0.06, M.metal)
    P.beam(rid[0].co, rid[1].co, 0.12, 0.12, M.metal)
    for a, b in ((base[0], rid[0]), (base[1], rid[1]), (base[2], rid[1]), (base[3], rid[0])):
        P.beam(a.co, b.co, 0.1, 0.1, M.metal)
    for a, b in zip(base, base[1:] + base[:1]):
        P.beam(a.co, b.co, 0.12, 0.12, M.metal)
    P.finish()

    # 立面装饰
    P = Part("主楼_正立面", C, bevel=0.015)
    Ff = Frame((BX0, BY0, 0), (1, 0, 0), (0, -1, 0))
    facade(P, M, Ff, BW, [("window", "window"), ("door", "window"), ("stall", "window")])
    P.finish()

    P = Part("主楼_左立面", C, bevel=0.015)
    Fl = Frame((BX0, BY0 + BD, 0), (0, -1, 0), (-1, 0, 0))
    facade(P, M, Fl, BD, [("window", "window"), ("window", "window")])
    P.finish()

    P = Part("主楼_背立面", C, bevel=0.015)
    Fb = Frame((BX0 + BW, BY0 + BD, 0), (-1, 0, 0), (0, 1, 0))
    facade(P, M, Fb, BW, [(None, "window"), (None, "window"), (None, "window")])
    P.finish()

    # 门口的花盆
    P = Part("花盆", C, bevel=0.0)
    flower_pot(P, M, BX0 + 5.0, -0.75, 0.0)
    P.finish()


def flower_pot(P, M, x, y, z):
    P.lathe((x, y, z), [(0.28, 0.0), (0.34, 0.08), (0.24, 0.16), (0.2, 0.3), (0.3, 0.42), (0.52, 0.62),
                        (0.56, 0.72), (0.5, 0.74), (0.45, 0.68), (0.0, 0.68)], M.pot, seg=24)
    for i in range(9):
        a = RNG.uniform(0, math.tau)
        r = RNG.uniform(0.0, 0.3)
        tip = Vector((x + math.cos(a) * r * 1.4, y + math.sin(a) * r * 1.4, z + RNG.uniform(1.0, 1.45)))
        P.beam((x + math.cos(a) * r * 0.5, y + math.sin(a) * r * 0.5, z + 0.65), tip, 0.03, 0.03, M.hedge)
        P.ico(tip, 0.09, M.flower_white if i % 3 else M.flower, sub=1)
        for k in range(3):
            P.ico(tip + Vector((RNG.uniform(-0.15, 0.15), RNG.uniform(-0.15, 0.15), -0.25 - k * 0.12)),
                  0.1, M.hedge, scale=(1.4, 0.6, 0.3), sub=1)


# ===========================================================================
# 地面、台阶、喷泉、花园
# ===========================================================================

STAIR_X0, STAIR_X1 = -10.0, -26.0    # 台阶底 / 顶（向 -X 上升）
STAIR_Y0, STAIR_Y1 = -11.5, -3.2
STAIR_H = 5.0
CHANNEL_Y = (STAIR_Y0 + STAIR_Y1) / 2
GARDEN = (-22.0, -8.0, -3.0, 10.0)   # x0, x1, y0, y1
GARDEN_Z = 2.0


def stair_z(x):
    t = min(1.0, max(0.0, (STAIR_X0 - x) / (STAIR_X0 - STAIR_X1)))
    return t * STAIR_H


ROAD_Y = -14.5    # 广场与外侧六边形砖路的分界


def build_ground(M, root):
    C = collection("地面", root)
    P = Part("广场", C)
    P.box_mm(-60, 60, -60, 30, -0.6, 0.0, M.plaza)
    P.finish()

    P = Part("人行道", C, bevel=0.02)
    P.box_mm(-8.3, 16.5, -2.1, 0.0, 0.0, 0.07, M.sidewalk)
    P.box_mm(-8.3, 16.5, -2.4, -2.1, 0.0, 0.12, M.road_curb)       # 灰色路缘石
    # 广场外侧的六边形砖路 + 灰色路缘石（参考截图 7 下方）
    P.box_mm(-9.0, 40.0, -40.0, ROAD_Y - 0.3, 0.0, 0.03, M.road)
    P.box_mm(-9.0, 40.0, ROAD_Y - 0.3, ROAD_Y, 0.0, 0.12, M.road_curb)
    P.finish()

    # 井盖：砖路上一个、主楼门口人行道上一个
    P = Part("井盖", C, bevel=0.005)
    manhole(P, M, 6.0, -18.0, 0.03)
    manhole(P, M, -5.2, -1.05, 0.07, r=0.45)
    P.finish()


def manhole(P, M, x, y, z, r=0.55):
    """枫丹井盖：石圈 + 铸铁盖 + 金色同心环、放射筋、中间一对相背的弧形纹"""
    R = r + 0.2
    P.lathe((x, y, z), [(R, 0.0), (R, 0.02), (R - 0.03, 0.035), (r + 0.01, 0.035), (r + 0.01, 0.0)],
            M.curb, seg=40, caps=False)
    P.cyl((x, y, z + 0.015), r, 0.03, M.manhole, seg=40)
    for r0, r1 in ((r - 0.07, r - 0.02), (r * 0.52, r * 0.6)):
        P.lathe((x, y, z + 0.03), [(r1, 0.0), (r1, 0.012), (r0, 0.012), (r0, 0.0)], M.gold_dark, seg=40,
                caps=False)
    for k in range(16):                                   # 放射筋（齿轮感）
        a = k / 16 * math.tau
        p0 = (x + math.cos(a) * r * 0.64, y + math.sin(a) * r * 0.64, z + 0.036)
        p1 = (x + math.cos(a) * (r - 0.1), y + math.sin(a) * (r - 0.1), z + 0.036)
        P.beam(p0, p1, 0.035, 0.012, M.gold_dark)
    for side in (0, math.pi):                            # 中心一对弧
        pts = [(x + math.cos(side + t) * r * 0.3, y + math.sin(side + t) * r * 0.3)
               for t in (math.radians(v) for v in range(-70, 71, 20))]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            P.beam((ax, ay, z + 0.038), (bx, by, z + 0.038), 0.05, 0.014, M.gold_dark)
    P.beam((x - r * 0.3, y, z + 0.04), (x + r * 0.3, y, z + 0.04), 0.05, 0.014, M.gold_dark)


def build_stairs(M, root):
    C = collection("大台阶与喷泉", root)
    P = Part("大台阶", C, bevel=0.03)
    n = 26
    run = (STAIR_X0 - STAIR_X1) / n
    rise = STAIR_H / n
    for i in range(n):
        x = STAIR_X0 - i * run
        z = (i + 1) * rise
        P.box_mm(STAIR_X1, x, STAIR_Y0, STAIR_Y1, z - rise - 0.05, z - 0.05, M.step_riser)
        P.box_mm(x - run, x + 0.04, STAIR_Y0, STAIR_Y1, z - 0.06, z, M.step)
    P.box_mm(STAIR_X1 - 14, STAIR_X1, STAIR_Y0 - 12, STAIR_Y1 + 3, 0.0, STAIR_H, M.step)       # 顶部平台
    # 台阶实心底座
    P.prism([(STAIR_X0, 0), (STAIR_X1, STAIR_H - 0.3), (STAIR_X1, 0)], M.step_riser,
            lambda x, z, t: (x, STAIR_Y0 + (STAIR_Y1 - STAIR_Y0) * t, z))
    # 两侧矮墙（顺着台阶斜上）
    for y0, y1 in ((STAIR_Y0 - 0.6, STAIR_Y0), (STAIR_Y1, STAIR_Y1 + 0.35)):
        P.prism([(STAIR_X0 + 0.6, 0), (STAIR_X0 + 0.6, 0.8), (STAIR_X1, STAIR_H + 0.8), (STAIR_X1, 0)], M.curb,
                lambda x, z, t: (x, y0 + (y1 - y0) * t, z))
        P.prism([(STAIR_X0 + 0.6, 0.8), (STAIR_X1, STAIR_H + 0.8), (STAIR_X1, STAIR_H + 0.95),
                 (STAIR_X0 + 0.6, 0.95)], M.stone,
                lambda x, z, t: (x, y0 - 0.05 + (y1 - y0 + 0.1) * t, z))
    # 中间流水槽
    cw = 0.6
    for s in (-1, 1):
        y = CHANNEL_Y + s * (cw / 2 + 0.1)
        P.prism([(STAIR_X0 + 0.2, 0), (STAIR_X0 + 0.2, 0.3), (STAIR_X1, STAIR_H + 0.3), (STAIR_X1, STAIR_H)],
                M.curb, lambda x, z, t, y=y: (x, y - 0.1 + 0.2 * t, z))
    P.finish()

    P = Part("流水", C)
    P.prism([(STAIR_X0 + 0.2, 0.05), (STAIR_X0 + 0.2, 0.22), (STAIR_X1, STAIR_H + 0.22), (STAIR_X1, STAIR_H + 0.05)],
            M.water, lambda x, z, t: (x, CHANNEL_Y - cw / 2 + cw * t, z))
    P.finish()

    # 八角喷泉
    P = Part("八角喷泉", C, bevel=0.03)
    fx, fy = STAIR_X0 + 1.3, CHANNEL_Y
    R0, R1 = 1.5, 1.15
    oct_out = [(fx + R0 * math.cos(a), fy + R0 * math.sin(a)) for a in (math.pi / 8 + k * math.pi / 4 for k in range(8))]
    oct_in = [(fx + R1 * math.cos(a), fy + R1 * math.sin(a)) for a in (math.pi / 8 + k * math.pi / 4 for k in range(8))]
    P.sweep(oct_out, [(0.0, 0.0), (0.12, 0.0), (0.12, 0.12), (0.0, 0.2), (0.0, 0.42), (0.08, 0.5),
                      (-(R0 - R1) * 1.08, 0.5), (-(R0 - R1) * 1.08, 0.0)], M.curb)
    bm = P.bm
    fs = bm.faces.new([bm.verts.new((x, y, 0.36)) for x, y in oct_in])
    P._tag_faces([fs], M.water)
    P.finish()
    P = Part("喷泉水柱", C)
    P.lathe((fx, fy, 0.35), [(0.18, 0.0), (0.06, 0.3), (0.03, 1.1), (0.08, 1.3), (0.0, 1.35)], M.water_jet, seg=12)
    P.finish()


def build_garden(M, root):
    C = collection("高台花园", root)
    gx0, gx1, gy0, gy1 = GARDEN
    P = Part("花园挡土墙", C, bevel=0.03)
    # 土台
    P.box_mm(gx0, gx1, gy0, gy1, 0.0, GARDEN_Z, M.wall_relief)
    # 压顶石
    P.sweep([(gx0, gy0), (gx1, gy0), (gx1, gy1), (gx0, gy1)],
            [(0.0, GARDEN_Z - 0.1), (0.12, GARDEN_Z - 0.1), (0.12, GARDEN_Z + 0.55), (-0.4, GARDEN_Z + 0.55),
             (-0.4, GARDEN_Z)], M.stone)
    P.finish()

    P = Part("草地", C)
    P.box_mm(gx0 + 0.4, gx1 - 0.4, gy0 + 0.4, gy1 - 0.4, GARDEN_Z, GARDEN_Z + 0.25, M.grass)
    P.finish()

    # 修剪树篱（圆角 + 噪声置换，看起来像植物）
    P = Part("树篱", C)
    zt = GARDEN_Z + 0.25
    hedges = [((gx0 + 0.6, gx1 - 0.6, gy0 + 0.6, gy0 + 1.5), 0.9),
              ((gx1 - 1.5, gx1 - 0.6, gy0 + 1.5, gy1 - 0.6), 0.9),
              ((gx0 + 3.0, gx1 - 4.0, gy0 + 4.0, gy0 + 5.0), 0.8),
              ((gx0 + 3.0, gx0 + 4.0, gy0 + 5.0, gy1 - 2.0), 0.8)]
    for (x0, x1, y0, y1), h in hedges:
        P.box_mm(x0, x1, y0, y1, zt, zt + h, M.hedge)
    P.modifiers.append(organic(0.25, 0.12, 0.6))
    P.finish()

    P = Part("黄花丛", C)
    for x, y, r in ((gx1 - 1.2, gy0 + 1.3, 1.1), (gx0 + 5.0, gy0 + 1.2, 1.0), (gx1 - 1.3, gy0 + 6.0, 1.0),
                    (gx0 + 9.0, gy0 + 4.6, 0.9), (gx1 - 1.1, gy1 - 1.5, 0.9)):
        flower_bush(P, M, x, y, zt + 0.6, r)
    P.finish()

    P = Part("柏树", C)
    cypress(P, M, gx1 - 4.2, gy0 + 7.0, zt, 12.5, 2.0)
    cypress(P, M, gx0 + 6.0, gy0 + 8.5, zt, 14.0, 2.3)
    cypress(P, M, gx1 - 2.0, gy1 - 1.0, zt, 8.0, 1.4)
    P.finish()

    # 台阶另一侧的低花坛
    P = Part("右侧花坛", C, bevel=0.03)
    P.box_mm(STAIR_X1, STAIR_X0 - 0.5, STAIR_Y0 - 9.0, STAIR_Y0 - 0.6, 0.0, 1.0, M.wall_relief)
    P.sweep([(STAIR_X1, STAIR_Y0 - 9.0), (STAIR_X0 - 0.5, STAIR_Y0 - 9.0), (STAIR_X0 - 0.5, STAIR_Y0 - 0.6),
             (STAIR_X1, STAIR_Y0 - 0.6)], [(0.0, 0.9), (0.12, 0.9), (0.12, 1.25), (-0.3, 1.25), (-0.3, 1.0)], M.stone)
    P.finish()
    P = Part("右侧花坛_植物", C)
    P.box_mm(STAIR_X1 + 0.3, STAIR_X0 - 0.8, STAIR_Y0 - 8.7, STAIR_Y0 - 0.9, 1.0, 1.2, M.grass)
    P.box_mm(STAIR_X1 + 0.5, STAIR_X0 - 1.0, STAIR_Y0 - 1.9, STAIR_Y0 - 1.0, 1.2, 1.9, M.hedge)
    P.modifiers.append(organic(0.2, 0.1, 0.6))
    P.finish()
    P = Part("右侧花坛_花", C)
    for x in (STAIR_X0 - 2.5, STAIR_X0 - 7.0, STAIR_X0 - 12.0):
        flower_bush(P, M, x, STAIR_Y0 - 1.5, 1.8, 0.8)
    P.finish()


def organic(bevel, strength, size):
    """树篱用的修改器组合：倒圆 → 细分 → 云噪声置换"""
    def apply(ob):
        b = ob.modifiers.new("倒圆", "BEVEL")
        b.width = bevel
        b.segments = 3
        s = ob.modifiers.new("细分", "SUBSURF")
        s.subdivision_type = "SIMPLE"
        s.levels = s.render_levels = 3
        tex = bpy.data.textures.new("树篱噪声", "CLOUDS")
        tex.noise_scale = size
        d = ob.modifiers.new("置换", "DISPLACE")
        d.texture = tex
        d.strength = strength
        d.texture_coords = "GLOBAL"
    return apply


def flower_bush(P, M, x, y, z, r):
    P.ico((x, y, z), r, M.hedge, scale=(1, 1, 0.75), sub=2, jitter=r * 0.08)
    for i in range(int(40 * r * r)):
        a = RNG.uniform(0, math.tau)
        b = RNG.uniform(0.0, math.pi * 0.45)
        p = Vector((x + math.cos(a) * math.sin(b + 0.5) * r, y + math.sin(a) * math.sin(b + 0.5) * r,
                    z + math.cos(b + 0.5) * r * 0.75))
        P.ico(p, RNG.uniform(0.07, 0.11), M.flower, scale=(1, 1, 0.5), sub=1)


def cypress(P, M, x, y, z, h, r):
    """柏树：暗色树芯 + 几百个向外、略向下的尖叶簇 → 羽毛状剪影"""
    P.cyl((x, y, z + 0.8), 0.22, 1.6, M.bark, seg=8)
    P.ico((x, y, z + h * 0.52), 1.0, M.cypress_dark, scale=(r * 0.8, r * 0.8, h * 0.46), sub=2)
    n = int(h * 22)
    for i in range(n):
        t = RNG.uniform(0.08, 1.0)
        zz = z + 0.9 + (h - 0.9) * t
        prof = math.sin(math.pi * min(1.0, t * 1.05) ** 0.85) * (1.0 - 0.55 * t)
        rr = r * max(0.12, prof)
        a = RNG.uniform(0, math.tau)
        base = Vector((x + math.cos(a) * rr * 0.55, y + math.sin(a) * rr * 0.55, zz))
        d = Vector((math.cos(a), math.sin(a), RNG.uniform(0.2, 0.9) if t > 0.9 else RNG.uniform(-0.25, 0.35)))
        P.cone_dir(base, d, RNG.uniform(0.28, 0.42) * (1.2 - t * 0.5), rr * 0.9 + 0.35,
                   M.cypress if RNG.random() > 0.3 else M.cypress_dark)


# ===========================================================================
# 路灯
# ===========================================================================

def street_lamp(P, M, x, y, z=0.0, h=3.6):
    """枫丹金色路灯：八角底座 + 细杆 + 六角灯笼"""
    P.lathe((x, y, z), [(0.34, 0.0), (0.34, 0.12), (0.26, 0.2), (0.22, 0.45), (0.14, 0.55), (0.1, 0.8),
                        (0.13, 0.9), (0.06, 1.0), (0.0, 1.0)], M.gold, seg=8, smooth=False)
    P.cyl((x, y, z + (1.0 + h - 0.6) / 2), 0.05, h - 1.6, M.gold, seg=10)
    P.lathe((x, y, z + h - 0.75), [(0.06, 0.0), (0.12, 0.1), (0.09, 0.2), (0.16, 0.25), (0.0, 0.25)], M.gold, seg=10)
    # 灯笼
    zc = z + h - 0.5
    P.cyl((x, y, zc + 0.3), 0.17, 0.55, M.lamp_glass, seg=6, smooth=False)
    for k in range(6):
        a = k / 6 * math.tau
        P.box((x + math.cos(a) * 0.19, y + math.sin(a) * 0.19, zc + 0.3), (0.035, 0.035, 0.6), M.gold,
              rot=(0, 0, a))
    P.lathe((x, y, zc + 0.58), [(0.26, 0.0), (0.24, 0.08), (0.12, 0.25), (0.05, 0.35), (0.08, 0.42),
                                (0.0, 0.5)], M.gold, seg=6, smooth=False)
    P.lathe((x, y, zc - 0.05), [(0.0, -0.08), (0.12, 0.0), (0.22, 0.06), (0.0, 0.08)], M.gold, seg=6, smooth=False)


def build_props(M, root):
    C = collection("路灯", root)
    P = Part("路灯", C, bevel=0.01)
    street_lamp(P, M, GARDEN[1] - 0.2, GARDEN[2] + 0.4, 0.0)       # 花园转角
    street_lamp(P, M, BX0 + BW + 1.0, -1.7, 0.0)                    # 主楼右侧
    street_lamp(P, M, STAIR_X0 + 3.0, STAIR_Y0 - 2.0, 0.0)          # 喷泉旁
    street_lamp(P, M, -4.0, -12.6, 0.0)
    P.finish()


# ===========================================================================
# 远景：石塔、城墙、旗幡
# ===========================================================================

def build_backdrop(M, root):
    C = collection("石塔与城墙", root)
    # 主楼右边的巨大圆塔
    P = Part("圆塔", C, bevel=0.03)
    tx, ty, tr = BX0 + BW + 4.6, 1.9, 4.4
    P.lathe((tx, ty, 0), [(tr + 0.6, 0.0), (tr + 0.6, 1.0), (tr + 0.35, 1.25), (tr + 0.35, 2.4), (tr + 0.1, 2.6),
                          (tr, 2.6), (tr, 38.0), (tr + 0.4, 38.3), (tr + 0.4, 39.0), (0, 39.0)], M.tower, seg=48,
            smooth=True)
    # 塔身竖向装饰槽（刻纹）
    for k in range(10):
        a = math.pi * 0.45 + k * 0.33
        P.box((tx + math.cos(a) * (tr + 0.02), ty + math.sin(a) * (tr + 0.02), 18),
              (0.12, 0.35, 30), M.stone, rot=(0, 0, a))
    P.finish()

    # 背后的城墙（主楼后面 + 左后方高墙）
    P = Part("城墙", C, bevel=0.05)
    P.box_mm(-45, 45, 10.4, 16, 0.0, 34.0, M.bigwall)
    for x in range(-40, 44, 9):
        P.box_mm(x - 1.0, x + 1.0, 9.6, 10.4, 0.0, 34.0, M.bigwall)            # 扶壁
        P.box_mm(x - 1.2, x + 1.2, 9.4, 10.4, 0.0, 1.5, M.stone)
    for z in (13.0, 23.0):                                                   # 横向线脚
        P.box_mm(-45, 45, 10.0, 10.5, z, z + 0.6, M.stone)
    # 高处的拱形凹窗（阴影里的深色）
    for x in range(-36, 44, 9):
        P.prism(arched(4.2, 7.0, 2.1), M.bigwall_dark,
                lambda u, z, t, x=x: (x + 4.5 + u, 10.38 - 0.02 * t, 15.0 + z))
    # 左侧台阶顶上的高台和墙
    P.box_mm(-60, STAIR_X1 - 14, -40, 16, 0.0, 30.0, M.bigwall)
    P.box_mm(-60, -40, -40, 16, 0.0, 5.0, M.bigwall)
    # 右侧远处的高墙（塔后面）
    P.box_mm(19, 55, 5, 16, 0.0, 26.0, M.bigwall)
    for x in (24, 33, 42, 51):
        P.prism(arched(3.6, 8.0, 1.8), M.bigwall_dark, lambda u, z, t, x=x: (x + u, 4.98 - 0.02 * t, 2.0 + z))
    P.finish()

    # 蓝色旗幡（左上角那面）
    P = Part("旗幡", C, bevel=0.0)
    Fw = Frame((-14.0, 10.3, 0), (1, 0, 0), (0, -1, 0))
    kite = [(-1.3, 22.5), (1.3, 22.5), (1.5, 19.5), (0.0, 15.5), (-1.5, 19.5)]
    P.fpoly(Fw, kite, 0.1, 0.16, M.banner)
    for a, b in zip(kite, kite[1:] + kite[:1]):
        P.fband(Fw, [a, b], 0.1, 0.16, 0.2, M.gold)
    for pts in ([(0.0, 16.3), (0.0, 21.8)], [(-0.9, 21.8), (0.0, 19.0), (0.9, 21.8)],
                [(-0.6, 18.8), (0.0, 17.5), (0.6, 18.8)]):
        P.fband(Fw, pts, 0.07, 0.16, 0.19, M.gold)
    P.box_mm(-16.0, -12.0, 10.0, 10.3, 22.4, 22.8, M.gold)
    P.finish()

    # 右边远处的铜顶小楼（参考图1 最右侧）
    P = Part("铜顶小楼", C, bevel=0.03)
    x, y = 25.0, -2.0
    P.lathe((x, y, 0), [(3.2, 0.0), (3.2, 6.0), (3.5, 6.3), (3.5, 6.7), (2.9, 7.0), (2.9, 9.5), (3.2, 9.8),
                        (2.6, 11.0), (1.4, 12.2), (0.5, 12.8), (0.25, 13.6), (0.0, 13.6)], M.copper, seg=8,
            smooth=False)
    P.finish()


# ===========================================================================
# 世界 / 灯光 / 相机
# ===========================================================================

def setup_world():
    w = bpy.data.worlds.new("枫丹天空")
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
    ramp.color_ramp.elements[0].color = srgb("#DDEFFB")
    ramp.color_ramp.elements[1].color = srgb("#3F8FE0")
    L(mr.outputs[0], ramp.inputs["Fac"])
    sky = N("ShaderNodeBackground")
    L(ramp.outputs["Color"], sky.inputs["Color"])
    amb = N("ShaderNodeBackground")
    amb.inputs["Color"].default_value = srgb("#B4C4E4")
    amb.inputs["Strength"].default_value = 0.4
    lp = N("ShaderNodeLightPath")
    mix = N("ShaderNodeMixShader")
    L(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    L(amb.outputs[0], mix.inputs[1])
    L(sky.outputs[0], mix.inputs[2])
    out = N("ShaderNodeOutputWorld")
    L(mix.outputs[0], out.inputs["Surface"])


VIEWS = {
    # 参考图2：正对主楼立面
    "front": dict(loc=(0.6, -18.5, 4.0), target=(0.6, 0.0, 6.0), lens=30),
    # 参考图1：站在大台阶顶端往下看
    "stairs": dict(loc=(-30.0, -9.5, 13.0), target=(-3.0, -1.0, 0.5), lens=22),
    # 参考图3：从广场右前方斜俯视
    # 参考图 7：站在广场外侧路缘石上看主楼和地砖
    "ground": dict(loc=(1.0, -24.0, 7.0), target=(1.0, -5.0, 0.5), lens=26),
    # 参考图 9：门口俯视（人行道风车砖、路缘石、广场砖）
    "doorstep": dict(loc=(-1.2, -5.2, 4.6), target=(-0.3, -0.9, 0.0), lens=24),
    # 参考图 8：路缘石另一侧的砖路和井盖
    "road": dict(loc=(3.0, -12.8, 4.2), target=(5.2, -17.2, 0.0), lens=24),
    "plaza": dict(loc=(9.0, -20.0, 10.0), target=(-6.0, -4.0, 1.5), lens=24),
}


def setup_lights_camera(root, view):
    C = collection("灯光相机", root)
    sun = bpy.data.lights.new("太阳", "SUN")
    sun.energy = 4.0
    sun.color = (1.0, 0.97, 0.92)
    sun.angle = math.radians(1.5)
    ob = bpy.data.objects.new("太阳", sun)
    C.objects.link(ob)
    ob.rotation_euler = Vector((0.55, 0.45, -0.7)).normalized().to_track_quat("-Z", "Y").to_euler()
    ob.location = (-20, -30, 40)

    v = VIEWS[view]
    cam = bpy.data.cameras.new("相机")
    cam.lens = v["lens"]
    cam.clip_end = 1000
    co = bpy.data.objects.new("相机", cam)
    C.objects.link(co)
    co.location = v["loc"]
    co.rotation_euler = (Vector(v["target"]) - Vector(v["loc"])).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = co


def setup_render(engine, res, samples, outline):
    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = res
    s.render.resolution_percentage = 100
    if engine == "cycles":
        s.render.engine = "CYCLES"
        s.cycles.samples = samples
        s.cycles.device = "CPU"
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
        s.render.line_thickness = 0.8
        vl = s.view_layers[0]
        vl.use_freestyle = True
        for ls in vl.freestyle_settings.linesets:
            ls.linestyle.color = srgb("#6A5A55")[:3]
            ls.linestyle.thickness = 0.8
            ls.linestyle.alpha = 0.6


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                  bpy.data.worlds, bpy.data.textures, bpy.data.images):
        for item in list(block):
            block.remove(item)


def build(view="front", part="all"):
    clear_scene()
    M = Mats()
    root = collection("枫丹_芙宁娜家门口")
    build_building(M, root)
    build_ground(M, root)
    if part == "all":
        build_stairs(M, root)
        build_garden(M, root)
        build_props(M, root)
        build_backdrop(M, root)
    setup_world()
    setup_lights_camera(root, view)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="生成枫丹·芙宁娜家门口场景")
    p.add_argument("--render")
    p.add_argument("--save")
    p.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    p.add_argument("--res", default="1600x1000")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--outline", action="store_true")
    p.add_argument("--view", choices=tuple(VIEWS), default="front")
    p.add_argument("--part", choices=("all", "building"), default="all")
    p.add_argument("--export-textures", help="把生成的地砖贴图另存为 PNG 到这个文件夹")
    args, _ = p.parse_known_args(argv)
    return args


def main():
    args = parse_args()
    build(args.view, args.part)
    w, h = (int(v) for v in args.res.lower().split("x"))
    setup_render(args.engine, (w, h), args.samples, args.outline)
    if args.export_textures:
        export_images(os.path.abspath(args.export_textures),
                      *[i for i in bpy.data.images if i.name.startswith(("广场", "风车"))])
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
    if args.render:
        bpy.context.scene.render.filepath = os.path.abspath(args.render)
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
