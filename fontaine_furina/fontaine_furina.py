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
    elif plane == "WN":
        sub = N("ShaderNodeMath"); sub.operation = "SUBTRACT"
        L(sep.outputs["X"], sub.inputs[0]); L(sep.outputs["Y"], sub.inputs[1])
        L(sub.outputs[0], comb.inputs["X"])
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
            mp.inputs["Scale"].default_value = (1 / sx, 1 / sy, 1 / sx if image.get("box") else 1)
            mp.inputs["Rotation"].default_value = (0, 0, math.radians(image.get("rot", 0)))
            if image.get("plane", "XY") != "XY":        # 墙面：W = (x+y, z)，WN = (x−y, z)，CYL = (角度·半径, z)
                L(_coords(nt, image["plane"], image.get("radius", 1.0)), mp.inputs["Vector"])
            else:
                L(tc.outputs["Object"], mp.inputs["Vector"])
            uv = mp.outputs[0]
        it = N("ShaderNodeTexImage")
        it.image = image["color"]
        it.interpolation = "Cubic"
        if image.get("box"):                            # 六面投影（树篱、灌木、草地这类任意朝向的形体）
            it.projection = "BOX"
            it.projection_blend = 0.3
        L(uv, it.inputs["Vector"])
        base = it.outputs["Color"]
        if image.get("tint"):
            tn = N("ShaderNodeVectorMath"); tn.operation = "MULTIPLY"
            L(base, tn.inputs[0]); tn.inputs[1].default_value = image["tint"]
            base = tn.outputs[0]
        if image.get("height"):
            ht = N("ShaderNodeTexImage")
            ht.image = image["height"]
            if image.get("box"):
                ht.projection = "BOX"
                ht.projection_blend = 0.3
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
        # 墙面石块贴图（程序生成）：砂岩大块 + 白石小块
        sc_, sh_ = cached(gen_ashlar, seed=1)
        wc_, wh_ = cached(gen_ashlar, n=1024, W=4.0, H=2.4, rows=(0.3, 0.42), widths=(0.4, 0.9), base="#EEE7DB",
                              lighter="#F2ECE3", darker="#E8E0D3", joint="#D3C9BB", bevel=0.025, seed=2)
        self.wall_tex = [make_image("砂岩墙_颜色", sc_), make_image("砂岩墙_高度", sh_, True),
                         make_image("白石墙_颜色", wc_), make_image("白石墙_高度", wh_, True)]
        sand_img = lambda pl: dict(color=self.wall_tex[0], height=self.wall_tex[1], size=(8.0, 4.0), plane=pl, bump=0.5)
        white_img = lambda pl: dict(color=self.wall_tex[2], height=self.wall_tex[3], size=(4.0, 2.4), plane=pl, bump=0.4)
        self.sand = toon("砂岩墙", "#EAC28F", image=sand_img("W"))
        self.sand_n = toon("砂岩墙_斜", "#EAC28F", image=sand_img("WN"))
        self.stone = toon("白石", "#EEE7DA")
        self.stone_block = toon("白石_分块", "#EEE7DA", image=white_img("W"))
        self.stone_pier = toon("白石_墩", "#EDE5D8", image=white_img("W"))
        self.stone_pier_n = toon("白石_墩_斜", "#EDE5D8", image=white_img("WN"))
        self.scallop = toon("拱纹细石条", "#E6DED1", "scallop", 0.14, "W", 0.9)
        self.stone_relief = toon("白石_浮雕", "#DDD3C4")
        self.dado = toon("橙砂岩墙裙", "#D9A26C", "bricks", 0.8, "W", 0.88)
        self.door_frame = toon("青铜门框", "#3B625E", sheen=("#6F9C94", 0.35))
        self.door_dark = toon("青铜深色", "#2A4845")
        self.glass_light = toon("浅青彩窗", "#4CCAC9", sheen=("#B5F5EC", 0.8))
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
        self.papers = [toon("报纸_%d" % i, c, "hlines", 0.06, "XZ", 0.85) for i, c in
                       enumerate(("#B9C9A0", "#D8D2B4", "#9FBF9A", "#E3C9A2"))]
        # 地面
        # 地砖贴图（程序生成，见 gen_plaza_tiles / gen_hex_pavers）
        pc, ph = cached(gen_plaza_tiles)
        hc, hh, hu, hv = cached(gen_hex_pavers)
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
        self.water = toon("水", "#4FB0D0", emit=0.1, sheen=("#BFF3FF", 0.5))
        self.water_jet = toon("喷泉水柱", "#E8FBFF", emit=0.4, alpha=0.8)
        # 远景建筑
        big = lambda pl, r=1.0, tint=None: dict(color=self.wall_tex[2], height=self.wall_tex[3], size=(12.0, 7.2),
                                                plane=pl, radius=r, bump=0.6, tint=tint)
        self.tower = toon("圆塔石", "#E4E0D8", image=big("CYL", TOWER_R))
        self.tower_plinth = toon("圆塔底座", "#E6E1D8", image=big("CYL", TOWER_R + 0.5))
        self.bigwall = toon("城墙", "#D8D5CF", image=big("W", tint=(0.8, 0.8, 0.83)))
        self.bigwall_n = toon("城墙_斜", "#D8D5CF", image=big("WN", tint=(0.8, 0.8, 0.83)))
        self.scallop_big = toon("拱纹石板", "#E8E1D6", "scallop", 0.55, "W", 0.88)
        self.grate = toon("排水篦子", "#3E3A36")
        self.bench_wood = toon("长椅", "#3E6E6A")
        self.bigwall_dark = toon("城墙_背光", "#B9BCC9")
        self.banner = toon("蓝色旗幡", "#63B7EE", sheen=("#C8ECFF", 0.8), alpha=0.9)
        self.copper = toon("铜顶", "#B98A5A", sheen=("#E6C39A", 0.5))
        # 植物
        # 植物：叶片贴图（程序生成）+ 散布的叶片 / 花 / 草叶几何
        fc_, fh_ = cached(gen_foliage)
        gc_, gh_ = cached(gen_foliage, count=4200, length=(0.06, 0.14), ratio=0.16, bg="#4A8F2A",
                          palette=("#5FA935", "#6DB83B", "#7FC744", "#94D24F"), seed=4)
        self.plant_tex = [make_image("叶片_颜色", fc_), make_image("叶片_高度", fh_, True),
                          make_image("草地_颜色", gc_), make_image("草地_高度", gh_, True)]
        self.hedge = toon("树篱", "#4FA23A", image=dict(color=self.plant_tex[0], height=self.plant_tex[1],
                                                       size=(1.2, 1.2), box=True, bump=0.8))
        self.hedge_dark = toon("树篱_暗", "#3C8A33", image=dict(color=self.plant_tex[0], height=self.plant_tex[1],
                                                             size=(1.2, 1.2), box=True, bump=0.8,
                                                             tint=(0.75, 0.8, 0.75)))
        self.grass = toon("草地", "#78C03E", image=dict(color=self.plant_tex[2], height=self.plant_tex[3],
                                                       size=(1.5, 1.5), box=True, bump=0.5))
        self.leaves = [toon("叶片_%d" % i, c) for i, c in enumerate(("#3F8A2C", "#52A236", "#66B63F", "#7FC64B"))]
        self.blades = [toon("草叶_%d" % i, c) for i, c in enumerate(("#5FA935", "#72BC3E", "#8BCB4B"))]
        self.flower = toon("黄花", "#F9C42A")
        self.flower_deep = toon("黄花_深", "#F0A21C")
        self.flower_eye = toon("花心", "#E0781E")
        self.flower_white = toon("白花", "#F5F2E6")
        self.petal_cream = toon("乳白花瓣", "#F3EDD6")
        self.cypress = toon("柏树", "#1C8C74")
        self.cypress_light = toon("柏树_亮", "#28A88A")
        self.cypress_dark = toon("柏树_暗", "#0F5E50")
        self.bark = toon("树干", "#7A5238")
        self.pot = toon("花盆", "#DAD6CF")
        self.soil = toon("盆土", "#3A2A22")
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


def gen_ashlar(n=2048, W=8.0, H=4.0, rows=(0.42, 0.6), widths=(0.55, 1.35), base="#EAC28F",
               lighter="#EFCB9C", darker="#E1B682", joint="#C99D6B", bevel=0.035, seed=1):
    """
    错缝石块墙面贴图（砂岩 / 白石）：每行高度、每块宽度随机，行与行错缝；
    每块有色差、边缘倒角压暗、细灰缝，另外输出高度图做凹凸。横向 W 米、竖向 H 米无缝。
    """
    rng = np.random.RandomState(seed)
    nx, ny = n, int(round(n * H / W))
    hs = []
    while sum(hs) < H - rows[0] * 0.5:
        hs.append(rng.uniform(*rows))
    hs = np.array(hs) * H / sum(hs)
    redges = np.concatenate([[0], np.cumsum(hs)])
    x = (np.arange(nx) + 0.5) / nx * W
    y = (np.arange(ny) + 0.5) / ny * H
    X, Y = np.meshgrid(x, y)
    ri = np.clip(np.searchsorted(redges, Y, "right") - 1, 0, len(hs) - 1)
    dy = np.minimum(Y - redges[ri], redges[ri + 1] - Y)
    dx = np.zeros_like(X)
    bi = np.zeros_like(X)
    for r in range(len(hs)):
        ws = []
        while sum(ws) < W - widths[0] * 0.5:
            ws.append(rng.uniform(*widths))
        ws = np.array(ws) * W / sum(ws)
        ce = np.concatenate([[0], np.cumsum(ws)])
        off = rng.uniform(0, W)
        m = ri == r
        pos = np.mod(X[m] - off, W)
        k = np.clip(np.searchsorted(ce, pos, "right") - 1, 0, len(ws) - 1)
        dx[m] = np.minimum(pos - ce[k], ce[k + 1] - pos)
        bi[m] = k + r * 100
    d = np.minimum(dx, dy)
    h = _hash(bi, bi * 0.37 + 11)
    h2 = _hash(bi + 5, bi * 1.7)
    c0, cl, cd = _hex2rgb(base), _hex2rgb(lighter), _hex2rgb(darker)
    col = c0 + (cl - c0) * np.clip(h - 0.5, 0, 1)[..., None] * 2 + (cd - c0) * np.clip(0.5 - h, 0, 1)[..., None] * 2
    col *= (0.97 + 0.06 * h2)[..., None]
    col *= (1 + 0.04 * _lowfreq_noise(X / W, Y / H, seed + 3))[..., None]
    b = np.clip(d / bevel, 0, 1)
    col *= (0.9 + 0.1 * b ** 0.7)[..., None]
    px = W / nx
    jn = _cover(d, 0.008, px)
    col = col * (1 - jn[..., None]) + _hex2rgb(joint) * jn[..., None]
    height = (0.2 + 0.8 * b ** 0.5) * (1 - jn)
    return np.clip(col, 0, 1), np.clip(height, 0, 1)


def gen_foliage(n=1024, count=2600, length=(0.05, 0.11), ratio=0.42, bg="#2E6B22",
                palette=("#3F8A2C", "#4E9E34", "#5DAE3A", "#6FBE43", "#86CC4E"), seed=3):
    """
    无缝叶片贴图：在深色底上一层层叠几千片尖头叶子（随机方向、大小、颜色，叶尖亮叶根暗，
    带叶脉和深色描边），再输出高度图。length 是叶长占贴图边长的比例，ratio = 叶宽 / 叶长。
    """
    rng = np.random.RandomState(seed)
    col = np.tile(_hex2rgb(bg), (n, n, 1)) * (1 + 0.08 * _lowfreq_noise(*np.meshgrid(
        np.arange(n) / n, np.arange(n) / n), seed)[..., None])
    hgt = np.zeros((n, n))
    pal = [_hex2rgb(c) for c in palette]
    for _ in range(count):
        L = rng.uniform(*length) * n
        W = L * ratio * rng.uniform(0.8, 1.2)
        cx, cy = rng.uniform(0, n, 2)
        a = rng.uniform(0, math.tau)
        R = int(L / 2 + 3)
        ix = (np.arange(int(cx) - R, int(cx) + R + 1)) % n
        iy = (np.arange(int(cy) - R, int(cy) + R + 1)) % n
        X, Y = np.meshgrid(np.arange(int(cx) - R, int(cx) + R + 1) - cx, np.arange(int(cy) - R, int(cy) + R + 1) - cy)
        u = X * math.cos(a) + Y * math.sin(a)
        v = -X * math.sin(a) + Y * math.cos(a)
        t = np.clip(2 * u / L, -1, 1)
        half = W / 2 * (1 - t ** 2)
        d = half - np.abs(v)                              # >0 在叶内
        inside = (np.abs(2 * u / L) < 1) & (d > -1)
        if not inside.any():
            continue
        cov = np.clip(d + 0.5, 0, 1) * inside
        base = pal[rng.randint(len(pal))] * rng.uniform(0.9, 1.1)
        shade = (0.78 + 0.32 * (t + 1) / 2)[..., None]    # 叶根暗、叶尖亮
        c = base * shade
        c = c * np.where(np.abs(v) < 0.7, 0.85, 1.0)[..., None]                   # 叶脉
        c = c * np.where(d < 1.2, 0.72, 1.0)[..., None]                           # 描边
        sub_c = col[np.ix_(iy, ix)]
        col[np.ix_(iy, ix)] = sub_c * (1 - cov[..., None]) + c * cov[..., None]
        hh = np.clip(d / max(W / 2, 1), 0, 1) ** 0.5
        sub_h = hgt[np.ix_(iy, ix)]
        hgt[np.ix_(iy, ix)] = np.where(cov > 0.5, 0.4 + 0.6 * hh, sub_h)
    return np.clip(col, 0, 1), np.clip(hgt, 0, 1)


TEX_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "fontaine_furina_tex")


def cached(fn, *args, **kw):
    """贴图生成比较慢：按函数名 + 参数 + 生成代码缓存到磁盘，改了参数或代码会自动重新生成"""
    import hashlib
    import inspect
    import pickle
    key = hashlib.md5(repr((fn.__name__, args, sorted(kw.items()), inspect.getsource(fn))).encode()).hexdigest()
    path = os.path.join(TEX_CACHE, fn.__name__ + "_" + key + ".pkl")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        pass
    res = fn(*args, **kw)
    try:
        os.makedirs(TEX_CACHE, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(res, f, protocol=4)
    except Exception:
        pass
    return res


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

    def ffrustum(self, F, lo, hi, mat):
        """墙面坐标下两个矩形之间的台体（壁柱收分）：lo/hi = (u0, u1, d0, d1, z)"""
        bm = self.bm
        ring = lambda u0, u1, d0, d1, z: [bm.verts.new(F.p(u, z, d)) for u, d in
                                          ((u0, d0), (u1, d0), (u1, d1), (u0, d1))]
        a, b = ring(*lo), ring(*hi)
        faces = [bm.faces.new(list(reversed(a))), bm.faces.new(b)]
        for i in range(4):
            j = (i + 1) % 4
            faces.append(bm.faces.new((a[i], a[j], b[j], b[i])))
        self._tag_faces(faces, mat)

    # -- 生成对象 -----------------------------------------------------------
    def finish(self, origin=None):
        """origin：把对象原点放到这里（圆柱贴图坐标以对象原点为中心）"""
        bm = self.bm
        if origin is not None:
            bmesh.ops.translate(bm, vec=-Vector(origin), verts=bm.verts)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        me = bpy.data.meshes.new(self.name)
        bm.to_mesh(me)
        bm.free()
        for m in self.mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(self.name, me)
        if origin is not None:
            ob.location = Vector(origin)
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


def octo_top(w, h, c, t=0.0):
    """平底、上两角斜切的窗形（一层大窗）。t = 向外扩"""
    w, h, c = w + 2 * t, h + 2 * t, c + t * 0.414
    return [(-w / 2, -t), (w / 2, -t), (w / 2, h - c - t), (w / 2 - c, h - t), (-w / 2 + c, h - t), (-w / 2, h - c - t)]


def shift(poly, du, dz):
    return [(u + du, z + dz) for u, z in poly]


# ===========================================================================
# 主楼
# ===========================================================================

# 主楼尺寸（米）
BW, BD = 15.0, 10.0          # 宽、深
BX0, BY0 = -7.5, 0.0         # 左前角
Z_PLINTH = 0.18              # 勒脚
Z_BELT0, Z_BELT1 = 4.7, 5.2  # 腰线
Z_CORNICE = 8.5              # 檐口起点
Z_EAVE = 9.3                 # 檐口顶 / 屋顶起点
Z_ROOF = 12.0                # 屋顶平台
ROOF_IN = 1.9                # 屋顶平台内收


def footprint(inset=0.0):
    """主楼平面（逆时针）"""
    x0, y0, x1, y1 = BX0 - inset, BY0 - inset, BX0 + BW + inset, BY0 + BD + inset
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def ground_window_bay(P, M, F, l, r):
    """
    一层大窗整跨（参考截图 14 左）：
      白石凹板 + 上角同心圆浮雕 + 两侧折线纹石条
      平顶斜角金框彩窗（主干+喷泉杯、左右嵌套半圆拱、A 字斜线与圆钉、阶梯纹、底部波浪）
      窗下白石栏杆（窗台、短柱、下横杆）+ 橙色砂岩墙裙
    l, r = 两侧石墩内侧的 u 坐标
    """
    cu = (l + r) / 2
    ww = min(2.3, r - l - 0.75)
    wh, zb = 2.75, 1.45
    c = ww * 0.28
    # 墙裙与栏杆
    P.fbox(F, l, r, Z_PLINTH, 1.3, 0.0, 0.02, M.dado)
    P.fbox(F, l, r, 0.6, 0.72, 0.0, 0.2, M.stone)
    P.fbox(F, l, r, 1.28, 1.44, 0.0, 0.28, M.stone)
    for u in (l + 0.28, cu - ww * 0.3, cu + ww * 0.3, r - 0.28):
        P.fbox(F, u - 0.07, u + 0.07, 0.72, 1.28, 0.02, 0.18, M.stone)
    # 白石凹板
    P.fbox(F, l, r, 1.44, Z_BELT0, 0.0, 0.03, M.stone)
    for s_ in (-1, 1):
        u = cu + s_ * (ww / 2 + 0.22)                    # 折线纹石条
        P.fbox(F, u - 0.1, u + 0.1, 1.44, Z_BELT0, 0.03, 0.06, M.stone)
        zz = 1.55
        pts = []
        k = 0
        while zz < Z_BELT0 - 0.1:
            pts.append((u + (0.05 if k % 2 else -0.05), zz)); zz += 0.13; k += 1
        P.fband(F, pts, 0.025, 0.06, 0.075, M.stone_relief)
        ccu = cu + s_ * (ww / 2 - 0.05)                  # 上角同心圆
        for rr in (0.1, 0.19, 0.28):
            P.farc(F, ccu, Z_BELT0 - 0.4, rr - 0.025, rr, 0, math.tau, 0.03, 0.05, M.stone_relief, 20)
    # 窗
    win = shift(octo_top(ww, wh, c), cu, zb)
    P.fpoly(F, win, -0.02, 0.04, M.glass)
    P.fring(F, shift(octo_top(ww, wh, c, 0.12), cu, zb), win, 0.02, 0.14, M.gold)
    P.fring(F, shift(octo_top(ww, wh, c, 0.16), cu, zb), shift(octo_top(ww, wh, c, 0.12), cu, zb),
            0.02, 0.1, M.gold_dark)
    D0, D1, g = 0.04, 0.065, 0.035
    L = lambda pts, t=g: P.fband(F, [(cu + a, zb + b) for a, b in pts], t, D0, D1, M.gold)
    # 主干 + 喷泉杯
    L([(0, 0.4), (0, wh * 0.66)], 0.08)
    P.fpoly(F, shift([(-0.24, wh * 0.74), (0.24, wh * 0.74), (0.1, wh * 0.66), (-0.1, wh * 0.66)], cu, zb),
            D0, D1, M.gold)
    L([(0, wh * 0.74), (0, wh - 0.02)], 0.06)
    for s_ in (-1, 1):
        L([(s_ * 0.16, 0.35), (s_ * 0.16, wh * 0.85)])
        L([(s_ * 0.3, wh * 0.6), (s_ * 0.3, wh - 0.02 - (0.3 > ww / 2 - c) * 0)])
        # A 字斜线 + 圆钉
        a0, a1 = (s_ * 0.12, wh * 0.6), (s_ * (ww / 2 - 0.1), 0.42)
        L([a0, a1], 0.05)
        for t in (0.25, 0.45):
            P.cyl(F.p(cu + a0[0] + (a1[0] - a0[0]) * t, zb + a0[1] + (a1[1] - a0[1]) * t, D1),
                  0.04, 0.03, M.gold, seg=10, rot=F.n.to_track_quat("Z", "Y"))
        # 嵌套半圆拱 + 拱脚竖线
        ac = cu + s_ * ww * 0.27
        for r0 in (0.2, 0.32):
            P.farc(F, ac, zb + 1.2, r0, r0 + g, 0, math.pi, D0, D1, M.gold, 10)
            for sg in (-1, 1):
                P.fbox(F, ac + sg * (r0 + g / 2) - g / 2, ac + sg * (r0 + g / 2) + g / 2, zb + 0.35, zb + 1.2,
                       D0, D1, M.gold)
        P.fpoly(F, shift(arched(0.4, 0.8, 0.2), ac, zb + 0.6), D0 - 0.01, D0 + 0.005, M.glass_light)
        # 阶梯纹
        e = s_ * (ww / 2 - 0.08)
        L([(e, 0.62), (e - s_ * 0.22, 0.62), (e - s_ * 0.22, 0.48), (e - s_ * 0.42, 0.48),
           (e - s_ * 0.42, 0.34), (s_ * 0.3, 0.34)])
        # 波浪
        for k in range(3):
            zz = 0.1 + k * 0.07
            L([(e - s_ * (0.05 + 0.1 * m), zz + (0.02 if m % 2 else -0.02)) for m in range(5)], 0.018)
    P.fpoly(F, shift([(-0.5, 0.02), (0.5, 0.02), (0.22, 0.34), (-0.22, 0.34)], cu, zb), D0, D1 - 0.01, M.gold_dark)
    # 窗顶与腰线之间的小石块
    P.fbox(F, cu - 0.2, cu + 0.2, zb + wh + 0.16, Z_BELT0, 0.03, 0.1, M.stone)


def door_portal(P, M, F, cu):
    """
    大门门廊（参考截图 14 右、15）：
      两侧带柱头的方石柱；深青色厚门框；门楣竖向凹槽 + 阶梯轮廓 + 六边形“宝石”拱心石
      单扇门：上半拱形彩窗、下半两块圆头门板、金色把手；门两侧金色火炬形长杆
      篮柄拱雨棚：两端平、中间拱起的白石弧带 + 带竖肋的金属弧顶
    """
    # 石柱
    for s_ in (-1, 1):
        pu = cu + s_ * 1.3
        P.fbox(F, pu - 0.31, pu + 0.31, 0.0, 0.22, 0.0, 0.64, M.stone)
        P.fbox(F, pu - 0.26, pu + 0.26, 0.22, 3.6, 0.0, 0.56, M.stone_pier)
        P.fbox(F, pu - 0.32, pu + 0.32, 2.72, 2.92, 0.0, 0.64, M.stone)
        P.fbox(F, pu - 0.29, pu + 0.29, 2.66, 2.72, 0.0, 0.6, M.stone)
    # 深青色门框
    P.fbox(F, cu - 1.05, cu + 1.05, 0.0, 3.62, 0.0, 0.14, M.door_frame)
    for k in range(26):                                   # 门楣竖向凹槽
        u = cu - 1.0 + k * 0.08
        P.fbox(F, u - 0.018, u + 0.018, 2.9, 3.62, 0.14, 0.17, M.door_dark)
    for u0, u1, z0, z1 in ((0.98, 0.98, 2.6, 2.8), (0.64, 0.64, 2.8, 3.0), (0.38, 0.38, 3.0, 3.2)):
        P.fbox(F, cu - u0, cu + u1, z0, z1, 0.14, 0.2, M.door_frame)                 # 阶梯轮廓
    for s_ in (-1, 1):
        P.fband(F, [(cu + s_ * 0.96, 2.7), (cu + s_ * 0.62, 2.9), (cu + s_ * 0.3, 2.95)], 0.025, 0.2, 0.215, M.gold)
        P.fband(F, [(cu + s_ * 0.9, 2.64), (cu + s_ * 0.3, 2.82)], 0.02, 0.2, 0.215, M.gold)
    gem = [(-0.2, 3.3), (0.2, 3.3), (0.25, 3.1), (0.11, 2.7), (-0.11, 2.7), (-0.25, 3.1)]
    P.fring(F, shift([(x * 1.35, 3.0 + (z - 3.0) * 1.25) for x, z in gem], cu, 0), shift(gem, cu, 0),
            0.18, 0.27, M.door_frame)
    P.fpoly(F, shift(gem, cu, 0), 0.18, 0.24, M.glass)
    for k in range(3):                                   # 宝石里的金色 V 纹
        zz = 2.84 + k * 0.12
        P.fband(F, [(cu - 0.12, zz + 0.08), (cu, zz), (cu + 0.12, zz + 0.08)], 0.025, 0.24, 0.26, M.gold)
    # 门套（多层深青色框）
    rect = lambda hu, z1: [(cu - hu, 0.0), (cu + hu, 0.0), (cu + hu, z1), (cu - hu, z1)]
    P.fring(F, rect(0.86, 2.62), rect(0.62, 2.52), 0.14, 0.22, M.door_frame)
    P.fring(F, rect(0.7, 2.57), rect(0.62, 2.52), 0.22, 0.26, M.door_frame)
    for s_ in (-1, 1):                                   # 门套上角的阶梯“耳朵”
        P.fbox(F, cu + s_ * 0.86 - 0.12, cu + s_ * 0.86 + 0.12, 2.3, 2.62, 0.14, 0.24, M.door_frame)
    # 门扇
    P.fbox(F, cu - 0.6, cu + 0.6, 0.02, 2.52, 0.08, 0.16, M.door)
    arch = shift(arched(0.68, 1.3, 0.34), cu, 1.08)
    P.fpoly(F, arch, 0.15, 0.17, M.glass_light)
    P.fring(F, shift(arched(0.68, 1.3, 0.34, 0.08), cu, 1.08), arch, 0.16, 0.22, M.door)
    P.fbox(F, cu - 0.34, cu + 0.34, 1.55, 1.61, 0.17, 0.2, M.door)
    P.fbox(F, cu - 0.03, cu + 0.03, 1.08, 2.02, 0.17, 0.2, M.door)
    P.fbox(F, cu - 0.34, cu + 0.34, 1.97, 2.02, 0.17, 0.2, M.door)
    for s_ in (-1, 1):                                   # 玻璃里的金线
        P.fband(F, [(cu + s_ * 0.3, 1.65), (cu + s_ * 0.2, 1.65), (cu + s_ * 0.2, 1.9), (cu + s_ * 0.12, 1.9),
                    (cu + s_ * 0.12, 2.2)], 0.018, 0.17, 0.185, M.gold)
        P.fband(F, [(cu + s_ * 0.3, 1.15), (cu + s_ * 0.3, 1.5)], 0.018, 0.17, 0.185, M.gold)
    P.fpoly(F, [(cu - 0.07, 2.12), (cu + 0.07, 2.12), (cu, 1.98)], 0.19, 0.21, M.gold)
    P.fbox(F, cu - 0.6, cu + 0.6, 0.98, 1.06, 0.16, 0.2, M.door)
    for s_ in (-1, 1):                                   # 下半两块圆头门板
        pnl = shift(arched(0.3, 0.8, 0.15), cu + s_ * 0.2, 0.14)
        P.fring(F, shift(arched(0.3, 0.8, 0.15, 0.04), cu + s_ * 0.2, 0.14), pnl, 0.16, 0.2, M.door)
    P.fbox(F, cu + 0.42, cu + 0.47, 1.18, 1.42, 0.16, 0.22, M.gold)          # 把手
    P.fbox(F, cu + 0.41, cu + 0.48, 1.44, 1.52, 0.16, 0.2, M.gold)
    # 两侧金色火炬形长杆
    for s_ in (-1, 1):
        u0 = cu + s_ * 0.98
        P.fbox(F, u0 - 0.022, u0 + 0.022, 0.35, 2.35, 0.14, 0.18, M.gold)
        for sg in (-1, 1):
            P.fbox(F, u0 + sg * 0.07 - 0.012, u0 + sg * 0.07 + 0.012, 0.35, 1.9, 0.14, 0.17, M.gold)
        P.fpoly(F, [(u0 - 0.09, 2.5), (u0 + 0.09, 2.5), (u0, 2.3)], 0.14, 0.18, M.gold)
        P.farc(F, u0, 0.6, 0.09, 0.12, 0, math.pi, 0.14, 0.18, M.gold, 8)
        P.fbox(F, u0 - 0.12, u0 - 0.09, 0.3, 0.6, 0.14, 0.18, M.gold)
        P.fbox(F, u0 + 0.09, u0 + 0.12, 0.3, 0.6, 0.14, 0.18, M.gold)
    # 篮柄拱雨棚
    hw, zh, rise, th, dep = 1.8, 3.6, 0.55, 0.42, 1.0
    n = 28
    us = [-hw + 2 * hw * i / n for i in range(n + 1)]
    zbot = lambda u: zh + rise * max(0.0, 1 - (u / 1.15) ** 2) ** 0.6
    for a, b in zip(us, us[1:]):
        P.fpoly(F, [(cu + a, zbot(a)), (cu + b, zbot(b)), (cu + b, zbot(b) + th), (cu + a, zbot(a) + th)],
                0.0, dep, M.stone_pier)
        P.fpoly(F, [(cu + a, zbot(a) + th - 0.02), (cu + b, zbot(b) + th - 0.02), (cu + b, zbot(b) + th + 0.1),
                    (cu + a, zbot(a) + th + 0.1)], 0.0, dep + 0.08, M.stone)                     # 外层线脚
        P.fpoly(F, [(cu + a, zbot(a) - 0.03), (cu + b, zbot(b) - 0.03), (cu + b, zbot(b)), (cu + a, zbot(a))],
                0.14, dep - 0.08, M.sand)                                                      # 拱底（暖色）
        P.fpoly(F, [(cu + a, zbot(a) + th + 0.1), (cu + b, zbot(b) + th + 0.1), (cu + b, zbot(b) + th + 0.2),
                    (cu + a, zbot(a) + th + 0.2)], -0.05, dep + 0.02, M.slate)                  # 金属弧顶
    for k in range(1, 12):                               # 金属顶的竖肋
        u = -hw + 2 * hw * k / 12
        z = zbot(u) + th + 0.2
        P.beam(F.p(cu + u, z, -0.05), F.p(cu + u, z, dep + 0.02), 0.05, 0.05, M.metal)
    for s_ in (-1, 1):                                   # 雨棚两端的托块
        P.fbox(F, cu + s_ * hw - 0.2, cu + s_ * hw + 0.2, zh - 0.25, zh + 0.05, 0.0, dep - 0.1, M.stone)


def lower_pier(P, M, F, cu, w=0.95):
    """
    一层石墩（参考截图 14、13）：
      底座 → 宽墩身（拱形壁龛 + 金色浮雕板）→ 斜面收分 → 窄墩身（石块）
      → V 形线 + 喷泉状双拱浮雕 → 抹角“子弹头”墩顶（立在腰线前面）
    """
    hw, D0, D1 = w / 2, 0.36, 0.28
    hu = hw * 0.84
    zl1 = 1.95
    P.fbox(F, cu - hw - 0.07, cu + hw + 0.07, 0.0, 0.16, 0.0, D0 + 0.07, M.stone)
    # 宽墩身 + 壁龛
    nw, nz0, nz1 = w * 0.44, 0.32, 1.62
    P.fbox(F, cu - hw, cu - nw / 2, 0.16, zl1, 0.0, D0, M.stone_pier)
    P.fbox(F, cu + nw / 2, cu + hw, 0.16, zl1, 0.0, D0, M.stone_pier)
    P.fbox(F, cu - nw / 2, cu + nw / 2, 0.16, nz0, 0.0, D0, M.stone_pier)
    P.fbox(F, cu - nw / 2, cu + nw / 2, nz0, nz1, 0.0, 0.2, M.stone_pier)
    arc = [(cu + nw / 2 * math.cos(math.pi * i / 12), nz1 - nw / 2 + nw / 2 * math.sin(math.pi * i / 12))
           for i in range(13)]
    for (a, za), (b, zb_) in zip(arc, arc[1:]):
        P.fpoly(F, [(a, za), (b, zb_), (b, zl1), (a, zl1)], 0.0, D0, M.stone_pier)
    # 金色浮雕板
    gp = shift(arched(nw - 0.08, nz1 - nz0 - 0.06, nw / 2 - 0.04), cu, nz0 + 0.02)
    P.fpoly(F, gp, 0.2, 0.23, M.gold_dark)
    G = lambda pts, t=0.022: P.fband(F, [(cu + a, b) for a, b in pts], t, 0.23, 0.255, M.gold)
    G([(0, nz0 + 0.1), (0, nz1 - 0.25)], 0.03)
    for s_ in (-1, 1):
        G([(s_ * 0.1, nz0 + 0.1), (s_ * 0.1, nz1 - 0.35)])
        G([(s_ * 0.1, nz1 - 0.35), (0, nz1 - 0.45)])
    for k in range(5):                                   # 顶部小扇
        a = math.pi / 2 + (k - 2) * 0.35
        G([(0, nz1 - 0.3), (math.cos(a) * 0.15, nz1 - 0.3 + math.sin(a) * 0.15)], 0.018)
    for k in range(3):                                   # V 纹
        zz = nz0 + 0.35 + k * 0.14
        G([(-0.1, zz + 0.07), (0, zz), (0.1, zz + 0.07)], 0.018)
    # 收分
    P.ffrustum(F, (cu - hw, cu + hw, 0.0, D0, zl1), (cu - hu, cu + hu, 0.0, D1, zl1 + 0.3), M.stone)
    # 窄墩身
    zt = Z_BELT0 + 0.05
    P.fbox(F, cu - hu, cu + hu, zl1 + 0.3, zt, 0.0, D1, M.stone_pier)
    for s_ in (-1, 1):
        P.fbox(F, cu + s_ * (hu - 0.07) - 0.012, cu + s_ * (hu - 0.07) + 0.012, zl1 + 0.35, zt + 0.3,
               D1, D1 + 0.008, M.stone_relief)
    # 墩顶：抹角子弹头
    cap = shift(arched(2 * hu, 0.55, 0.3), cu, zt)
    P.fpoly(F, cap, 0.0, D1, M.stone)
    P.fring(F, shift(arched(2 * hu - 0.06, 0.52, 0.28), cu, zt), shift(arched(2 * hu - 0.12, 0.48, 0.25), cu, zt),
            D1, D1 + 0.015, M.stone_relief)
    # V 形线 + 喷泉双拱浮雕
    zv = Z_BELT0 - 1.2
    R = lambda pts, t=0.025: P.fband(F, [(cu + a, b) for a, b in pts], t, D1, D1 + 0.025, M.stone_relief)
    R([(-hu + 0.05, zv + 0.12), (0, zv), (hu - 0.05, zv + 0.12)])
    R([(-hu + 0.05, zv + 0.2), (hu - 0.05, zv + 0.2)], 0.02)
    R([(0, zv + 0.3), (0, zt + 0.2)], 0.035)
    for zz in (zv + 0.6, zv + 0.85):
        P.fbox(F, cu - 0.05, cu + 0.05, zz - 0.03, zz + 0.03, D1, D1 + 0.03, M.stone_relief)
    for s_ in (-1, 1):
        for r0, zc in ((0.1, zt + 0.1), (0.16, zt + 0.02)):
            P.farc(F, cu + s_ * 0.13, zc, r0, r0 + 0.025, 0, math.pi, D1, D1 + 0.025, M.stone_relief, 8)
            R([(s_ * 0.13 - r0 - 0.0125, zc), (s_ * 0.13 - r0 - 0.0125, zv + 0.35)], 0.025)
            R([(s_ * 0.13 + r0 + 0.0125, zc), (s_ * 0.13 + r0 + 0.0125, zv + 0.35)], 0.025)


def upper_pier(P, M, F, cu, w=0.95):
    """二层壁柱：窗台高度的挑出横板 + 短颈连到一层墩顶；柱身中间凹槽嵌宽金条；檐口上金色书本柱头"""
    hw = w / 2
    d = 0.32
    zs = Z_BELT1 + 0.35
    P.fbox(F, cu - 0.22, cu + 0.22, Z_BELT1 - 0.05, zs, 0.0, 0.22, M.stone)            # 短颈
    P.fbox(F, cu - hw * 1.3, cu + hw * 1.3, zs, zs + 0.18, 0.0, 0.55, M.stone)          # 挑出横板
    P.fbox(F, cu - hw * 1.25, cu + hw * 1.25, zs - 0.04, zs, 0.0, 0.5, M.stone_relief)
    z0, z1 = zs + 0.18, Z_CORNICE + 0.3
    for s_ in (-1, 1):
        P.fbox(F, cu + s_ * 0.15, cu + s_ * hw * 0.85, z0, z1, 0.0, d, M.stone)
    P.fbox(F, cu - 0.15, cu + 0.15, z0, z1, 0.0, d - 0.08, M.stone)
    P.fbox(F, cu - 0.1, cu + 0.1, z0 + 0.25, Z_CORNICE - 0.05, d - 0.08, d - 0.03, M.gold)
    zz = z0 + 0.5
    while zz < Z_CORNICE - 0.2:
        P.fbox(F, cu - 0.1, cu + 0.1, zz, zz + 0.03, d - 0.03, d - 0.025, M.gold_dark)
        zz += 0.6
    # 檐口上方：金色书本柱头
    zc = Z_EAVE - 0.1
    P.fbox(F, cu - hw * 0.8, cu + hw * 0.8, zc, zc + 0.35, -0.2, d + 0.2, M.stone)
    P.fbox(F, cu - hw * 0.7, cu + hw * 0.7, zc + 0.35, zc + 1.05, -0.1, d + 0.12, M.gold)
    for s_ in (-1, 1):
        P.fpoly(F, [(cu, zc + 1.0), (cu + s_ * hw * 0.7, zc + 1.05), (cu + s_ * hw * 0.72, zc + 1.4),
                    (cu + s_ * 0.06, zc + 1.18)], -0.1, d + 0.12, M.gold)
    P.fbox(F, cu - 0.03, cu + 0.03, zc + 0.45, zc + 0.95, d + 0.12, d + 0.15, M.gold_dark)


def bookstall(P, M, F, cu, w=2.75):
    """粉色遮阳篷书报摊（参考截图 15）：卷筒 + 斜篷布 + 方形条纹垂片 + 青色立柱与弧撑 + 两个报刊架 + 小书箱"""
    hw = w / 2
    zr, dr = 3.0, 0.36                 # 卷筒底高、卷筒外伸
    zt, zl, dout = 2.8, 2.4, 1.75      # 篷布上沿高、下沿高、外伸
    # 卷筒：粉色 + 金色竖带、上下金边、两端青色端盖
    P.fbox(F, cu - hw - 0.1, cu + hw + 0.1, zr, zr + 0.38, 0.0, dr, M.awning)
    P.fbox(F, cu - hw - 0.14, cu + hw + 0.14, zr - 0.06, zr, 0.0, dr + 0.04, M.teal_frame)
    for k in range(9):
        u = cu - hw + w * k / 8
        P.fbox(F, u - 0.03, u + 0.03, zr + 0.02, zr + 0.36, dr, dr + 0.012, M.gold)
    for z in (zr + 0.03, zr + 0.33):
        P.fbox(F, cu - hw - 0.1, cu + hw + 0.1, z, z + 0.025, dr, dr + 0.012, M.gold)
    for s_ in (-1, 1):
        P.fbox(F, cu + s_ * (hw + 0.1) - 0.08, cu + s_ * (hw + 0.1) + 0.08, zr - 0.06, zr + 0.44, 0.0, dr + 0.06,
               M.teal_frame)
    # 斜篷布
    slope = lambda u, t_, th: F.p(u, zt + (zl - zt) * t_ + th, 0.3 + (dout - 0.3) * t_)
    P.prism([(cu - hw - 0.05, 0), (cu + hw + 0.05, 0), (cu + hw + 0.05, 1), (cu - hw - 0.05, 1)], M.awning,
            lambda u, t_, t: slope(u, t_, 0.04 * t))
    for t_ in (0.25, 0.45, 0.65):                        # 浅色花纹线
        P.beam(slope(cu - hw + 0.1, t_, 0.045), slope(cu + hw - 0.1, t_, 0.045), 0.03, 0.01, M.awning_light)
    for k in range(10):
        u = cu - hw + 0.2 + (w - 0.4) * k / 9
        P.beam(slope(u, 0.2, 0.045), slope(u + 0.12, 0.7, 0.045), 0.025, 0.01, M.awning_light)
    P.beam(slope(cu - hw, 0.88, 0.045), slope(cu + hw, 0.88, 0.045), 0.04, 0.012, M.gold)
    for s_ in (-1, 1):
        P.beam(slope(cu + s_ * (hw - 0.05), 0.1, 0.045), slope(cu + s_ * (hw - 0.05), 0.88, 0.045), 0.04, 0.012,
               M.gold)
    # 方形条纹垂片
    ntab = 11
    tw = (w + 0.1) / ntab
    P.fbox(F, cu - hw - 0.05, cu + hw + 0.05, zl - 0.08, zl + 0.02, dout - 0.02, dout + 0.05, M.gold)
    for i in range(ntab):
        u0 = cu - hw - 0.05 + tw * i
        u1 = u0 + tw
        tab = [(u0 + 0.015, zl), (u1 - 0.015, zl), (u1 - 0.015, zl - 0.3), (u1 - 0.05, zl - 0.38),
               (u0 + 0.05, zl - 0.38), (u0 + 0.015, zl - 0.3)]
        P.fpoly(F, tab, dout, dout + 0.03, M.awning)
        um = (u0 + u1) / 2
        P.fbox(F, um - tw * 0.2, um + tw * 0.2, zl - 0.33, zl - 0.08, dout + 0.03, dout + 0.036, M.awning_light)
        P.fband(F, [(u0 + 0.03, zl - 0.3), (u0 + 0.06, zl - 0.36), (u1 - 0.06, zl - 0.36), (u1 - 0.03, zl - 0.3)],
                0.015, dout + 0.03, dout + 0.04, M.gold)
    # 两侧挡片
    for s_ in (-1, 1):
        u = cu + s_ * (hw + 0.06)
        P.prism([(0.3, zt), (dout, zl), (dout, zl - 0.35), (0.3, zt - 0.2)], M.awning,
                lambda d, z, t, u=u: F.p(u - 0.015 + 0.03 * t, z, d))
    # 青色前立柱 + 弧形撑
    for s_ in (-1, 1):
        u = cu + s_ * (hw - 0.05)
        P.fbox(F, u - 0.05, u + 0.05, 0.0, zl - 0.35, dout - 0.12, dout - 0.02, M.teal_frame)
        P.fbox(F, u - 0.09, u + 0.09, 0.0, 0.06, dout - 0.22, dout + 0.08, M.teal_frame)
        pts = [F.p(u, 0.9 + (zl - 1.3) * math.sin(math.pi / 2 * t), 0.1 + (dout - 0.2) * (1 - math.cos(math.pi / 2 * t)))
               for t in (i / 8 for i in range(9))]
        for a, b in zip(pts, pts[1:]):
            P.beam(a, b, 0.07, 0.07, M.teal_frame)
    # 报刊架 ×2
    papers = M.papers
    for rc in (cu - 0.42, cu + 0.5):
        P.fbox(F, rc - 0.36, rc + 0.36, 0.12, 1.85, 0.04, 0.1, M.teal_frame)
        for s_ in (-1, 1):
            P.fbox(F, rc + s_ * 0.36 - 0.04, rc + s_ * 0.36 + 0.04, 0.0, 1.95, 0.04, 0.6, M.teal_frame)
        P.fbox(F, rc - 0.4, rc + 0.4, 1.85, 1.95, 0.04, 0.6, M.teal_frame)
        P.fpoly(F, [(rc - 0.3, 1.95), (rc + 0.3, 1.95), (rc + 0.15, 2.1), (rc - 0.15, 2.1)], 0.3, 0.36, M.teal_frame)
        P.fbox(F, rc - 0.42, rc + 0.42, 0.0, 0.15, 0.04, 0.64, M.teal_frame)
        for z in (0.2, 0.62, 1.04, 1.46):
            P.beam(F.p(rc - 0.34, z, 0.55), F.p(rc + 0.34, z, 0.55), 0.06, 0.04, M.teal_frame)
            for k in range(2):
                u0 = rc - 0.33 + k * 0.34
                P.prism([(u0, 0), (u0 + 0.3, 0), (u0 + 0.3, 1), (u0, 1)], RNG.choice(papers),
                        lambda u, t_, t, z=z: F.p(u, z + 0.38 * t_, 0.5 - 0.3 * t_ + 0.02 * t))
    # 前面的小书箱
    P.fbox(F, cu - 1.25, cu - 0.55, 0.0, 0.38, 0.95, 1.45, M.teal_frame)
    for k in range(4):
        u0 = cu - 1.2 + k * 0.16
        P.prism([(u0, 0), (u0 + 0.14, 0), (u0 + 0.14, 1), (u0, 1)], RNG.choice(papers),
                lambda u, t_, t: F.p(u, 0.3 + 0.35 * t_, 1.05 + 0.25 * t_ + 0.02 * t))


def upper_bay(P, M, F, l, r):
    """
    二层一跨（参考截图 17 / 放大图）：
      切角彩窗（上下四角斜切）+ 白石窗套（顶部带台阶“肩”）+ 窗台
      彩窗：内金边、阶梯轮廓、喷泉扇 + 杯、主干与两侧竖线、向上张开的斜线、底部半齿轮（辐条 + 轮毂）
      窗顶两根拱纹细石条 + 两端斜落到壁柱的梯形眉线
    """
    cu = (l + r) / 2
    w, h, c = 1.3, 2.45, 0.3
    zb = Z_BELT1 + 0.32
    win = shift(coffin(w, h, c), cu, zb)
    P.fpoly(F, win, -0.04, 0.02, M.glass)
    P.fpoly(F, shift([(-0.2, 0.5), (0.2, 0.5), (0.2, h - 0.5), (0.12, h - 0.3), (-0.12, h - 0.3), (-0.2, h - 0.5)],
                     cu, zb), 0.02, 0.025, M.glass_light)
    # 窗套：白石内框 + 外台阶框 + 顶部“肩”
    P.fring(F, shift(coffin(w, h, c, 0.14), cu, zb), win, 0.0, 0.15, M.stone)
    P.fring(F, shift(coffin(w, h, c, 0.24), cu, zb), shift(coffin(w, h, c, 0.14), cu, zb), 0.0, 0.08, M.stone)
    for s_ in (-1, 1):
        P.fbox(F, cu + s_ * (w / 2 + 0.1) - 0.1, cu + s_ * (w / 2 + 0.1) + 0.1, zb + h - 0.06, zb + h + 0.24,
               0.0, 0.08, M.stone)
    P.fbox(F, cu - w / 2 - 0.3, cu + w / 2 + 0.3, Z_BELT1, zb - 0.14, 0.0, 0.14, M.stone)       # 窗台
    # 彩窗金线
    D0, D1, g = 0.02, 0.045, 0.03
    G = lambda pts, t=g: P.fband(F, [(cu + a, zb + b) for a, b in pts], t, D0, D1, M.gold)
    P.fring(F, shift(coffin(w, h, c, -0.05), cu, zb), shift(coffin(w, h, c, -0.08), cu, zb), D0, D1, M.gold)
    for s_ in (-1, 1):
        G([(s_ * 0.36, 0.55), (s_ * 0.36, h - 0.8), (s_ * 0.27, h - 0.8), (s_ * 0.27, h - 0.58),
           (s_ * 0.17, h - 0.58), (s_ * 0.17, h - 0.38), (0, h - 0.38)])
        G([(s_ * 0.09, 0.5), (s_ * 0.09, h * 0.56)])
        G([(s_ * 0.2, 0.52), (s_ * (w / 2 - 0.1), 0.95)])
        G([(s_ * 0.24, 0.72), (s_ * (w / 2 - 0.1), 1.12)])
    G([(0, 0.42), (0, h * 0.62)], 0.05)
    P.fpoly(F, shift([(-0.14, h * 0.66), (0.14, h * 0.66), (0.05, h * 0.6), (-0.05, h * 0.6)], cu, zb), D0, D1, M.gold)
    for k in range(-3, 4):                              # 喷泉扇
        a = math.pi / 2 + k * 0.22
        G([(0, h * 0.66), (math.cos(a) * 0.3, h * 0.66 + math.sin(a) * 0.36)], 0.022)
    gz = 0.1                                             # 半齿轮
    P.farc(F, cu, zb + gz, 0.2, 0.27, 0, math.pi, D0, D1, M.gold, 10)
    P.farc(F, cu, zb + gz, 0.0, 0.07, 0, math.pi, D0, D1, M.gold, 6)
    for k in range(7):
        a = math.pi * (k + 0.5) / 7
        G([(math.cos(a) * 0.25, gz + math.sin(a) * 0.25), (math.cos(a) * 0.35, gz + math.sin(a) * 0.35)], 0.07)
    for a in (math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        G([(math.cos(a) * 0.07, gz + math.sin(a) * 0.07), (math.cos(a) * 0.2, gz + math.sin(a) * 0.2)], 0.025)
    # 窗顶拱纹细石条 + 梯形眉线
    zt = zb + h + 0.24
    zc = min(zt + 0.35, Z_CORNICE - 0.12)
    for s_ in (-1, 1):
        u = cu + s_ * 0.34
        P.fbox(F, u - 0.07, u + 0.07, zt - 0.02, zc - 0.05, 0.0, 0.07, M.scallop)
    P.fband(F, [(l, zc - 0.5), (l + 0.42, zc), (r - 0.42, zc), (r, zc - 0.5)], 0.15, 0.0, 0.12, M.stone)
    P.fband(F, [(l, zc - 0.62), (l + 0.46, zc - 0.13), (r - 0.46, zc - 0.13), (r, zc - 0.62)], 0.03, 0.0, 0.06,
            M.stone_relief)


def facade(P, M, F, width, bays, pil_w=0.9):
    """
    一整面立面。bays 是每个开间一层/二层放什么：
      ground: 'window' 大拱窗 / 'door' 大门 / 'stall' 拱窗+书报摊 / None
      upper : 'window' 二层彩窗 / None
    """
    n = len(bays)
    edge = TURRET_IN + pil_w * 0.5          # 两端壁柱紧贴转角柱
    bw = (width - 2 * edge) / n
    centers = [edge + bw * (i + 0.5) for i in range(n)]
    pils = [edge + bw * i for i in range(n + 1)]
    for i, (cu, bay) in enumerate(zip(centers, bays)):
        g, up = bay
        l, r = cu - bw / 2 + pil_w * 0.45, cu + bw / 2 - pil_w * 0.45
        if g == "window":
            ground_window_bay(P, M, F, l, r)
        elif g == "stall":
            ground_window_bay(P, M, F, l, r)
            bookstall(P, M, F, cu)
        elif g == "door":
            door_portal(P, M, F, cu)
        else:
            P.fbox(F, l, r, Z_PLINTH, 1.3, 0.0, 0.02, M.dado)
        if up == "window":
            upper_bay(P, M, F, cu - bw / 2 + pil_w * 0.42, cu + bw / 2 - pil_w * 0.42)
    for cu in pils:
        lower_pier(P, M, F, cu, pil_w)
        upper_pier(P, M, F, cu, pil_w)
    return centers


TURRET_IN, TURRET_OUT, TURRET_CHAMFER = 1.6, 0.35, 1.0   # 转角柱：向内占 / 向外凸 / 斜切


def corner_turret(P, M, cx, cy, sx, sy):
    """
    楼角的八角形砂岩转角柱（参考截图 17、18）：切角的砂岩大石块体，夹在两个立面的白石壁柱之间；
    腰线、檐口高度各绕一圈多边形厚白石板，脚下一圈白石勒脚。
    (cx, cy) = 楼角，sx / sy = 楼角向外的方向（±1）
    """
    T, o, k = TURRET_IN, TURRET_OUT, TURRET_CHAMFER

    def poly(e):
        pts = [(-T, -T), (o + e, -T), (o + e, o - k + e * 0.41), (o - k + e * 0.41, o + e), (-T, o + e)]
        return [(cx + sx * a, cy + sy * b) for a, b in pts]

    def band(e, z0, z1, mat):
        P.prism(poly(e), mat, lambda x, y, t: (x, y, z0 + (z1 - z0) * t))

    # 斜面的方向决定用哪种贴图坐标，避免石块被拉伸
    diag = sx * sy > 0
    band(0.0, 0.0, Z_EAVE - 0.05, M.sand_n if diag else M.sand)
    band(0.1, 0.0, 0.25, M.stone)
    band(0.18, Z_BELT0 - 0.05, Z_BELT1 + 0.05, M.stone_pier_n if diag else M.stone_pier)
    band(0.24, Z_BELT0 - 0.12, Z_BELT0 - 0.05, M.stone)
    band(0.3, Z_CORNICE - 0.05, Z_EAVE, M.stone_pier_n if diag else M.stone_pier)
    band(0.36, Z_EAVE - 0.12, Z_EAVE + 0.02, M.stone)


def build_building(M, root):
    C = collection("主楼", root)

    # 墙体
    P = Part("主楼_墙体", C, bevel=0.03)
    P.box_mm(BX0, BX0 + BW, BY0, BY0 + BD, 0.0, Z_CORNICE + 0.2, M.sand)
    # 勒脚（沿四周扫掠的线脚）
    P.sweep(footprint(), [(0.0, 0.0), (0.1, 0.0), (0.1, Z_PLINTH - 0.05), (0.05, Z_PLINTH), (0.0, Z_PLINTH)],
            M.stone)
    # 腰线
    P.sweep(footprint(), [(0.0, Z_BELT0), (0.14, Z_BELT0), (0.14, Z_BELT0 + 0.06), (0.2, Z_BELT0 + 0.1),
                          (0.2, Z_BELT1 - 0.08), (0.16, Z_BELT1), (0.0, Z_BELT1)], M.stone_pier)
    P.finish()

    # 檐口 + 芒萨尔屋顶 + 天窗
    P = Part("主楼_屋顶", C, bevel=0.02)
    P.sweep(footprint(), [(0.0, Z_CORNICE), (0.1, Z_CORNICE), (0.1, Z_CORNICE + 0.15), (0.28, Z_CORNICE + 0.3),
                          (0.3, Z_CORNICE + 0.5), (0.48, Z_CORNICE + 0.62), (0.48, Z_EAVE),
                          (0.0, Z_EAVE)], M.stone_block)
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

    # 四个楼角的八角转角柱
    P = Part("主楼_转角柱", C, bevel=0.03)
    for cx, cy, sx_, sy_ in ((BX0, BY0, -1, -1), (BX0 + BW, BY0, 1, -1),
                             (BX0, BY0 + BD, -1, 1), (BX0 + BW, BY0 + BD, 1, 1)):
        corner_turret(P, M, cx, cy, sx_, sy_)
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
    """
    门口石花盆（参考截图 25）：十二棱石座 → 十二棱外敞盆身（腰上两道凸环夹交叉斜纹）→ 盆沿 → 深色土，
    里面一丛带对生长叶的茎，顶上开乳白尖瓣花、橙红花心
    """
    rng = random.Random(int(abs(x * 13 + y * 7)))
    seg = 12
    P.lathe((x, y, z), [(0.5, 0.0), (0.5, 0.08), (0.46, 0.13), (0.42, 0.13), (0.42, 0.15), (0.0, 0.15)],
            M.pot, seg=seg, smooth=False)
    prof = [(0.3, 0.15), (0.33, 0.2), (0.4, 0.34), (0.47, 0.5), (0.52, 0.62), (0.55, 0.66), (0.55, 0.72),
            (0.49, 0.72), (0.45, 0.66), (0.0, 0.66)]
    P.lathe((x, y, z), prof, M.pot, seg=seg, smooth=False)
    P.lathe((x, y, z), [(0.46, 0.66), (0.46, 0.68), (0.0, 0.68)], M.soil, seg=seg, smooth=False)
    r_at = lambda h: 0.4 + (0.47 - 0.4) * (h - 0.34) / 0.16
    for h in (0.33, 0.51):                              # 两道凸环
        P.lathe((x, y, z), [(r_at(h) - 0.01, h - 0.02), (r_at(h) + 0.025, h - 0.015), (r_at(h) + 0.025, h + 0.015),
                            (r_at(h) - 0.01, h + 0.02)], M.pot, seg=seg, smooth=False)
    for k in range(seg):                                # 交叉斜纹
        a0, a1 = k / seg * math.tau, (k + 1) / seg * math.tau
        for (h0, aa), (h1, bb) in (((0.36, a0), (0.48, a1)), ((0.36, a1), (0.48, a0))):
            P.beam((x + math.cos(aa) * (r_at(h0) + 0.012), y + math.sin(aa) * (r_at(h0) + 0.012), z + h0),
                   (x + math.cos(bb) * (r_at(h1) + 0.012), y + math.sin(bb) * (r_at(h1) + 0.012), z + h1),
                   0.018, 0.012, M.pot)
    # 植物
    zs = z + 0.68
    for _ in range(14):                                 # 贴着土的一圈叶子
        a = rng.uniform(0, math.tau)
        q = Vector((x + math.cos(a) * rng.uniform(0.05, 0.3), y + math.sin(a) * rng.uniform(0.05, 0.3), zs))
        leaf_card(P, q, Vector((math.cos(a), math.sin(a), rng.uniform(0.3, 1.0))).normalized(),
                  rng.uniform(0.14, 0.2), 0.4, rng.choice(M.leaves[1:]), rng)
    for k in range(7):
        a = k / 7 * math.tau + rng.uniform(-0.3, 0.3)
        r0 = rng.uniform(0.05, 0.22)
        base = Vector((x + math.cos(a) * r0, y + math.sin(a) * r0, zs))
        h = rng.uniform(0.3, 0.65)
        lean = Vector((math.cos(a), math.sin(a), 0)) * rng.uniform(0.08, 0.25)
        pts = [base + lean * (t ** 1.5) + Vector((0, 0, h * t)) for t in (0.0, 0.35, 0.7, 1.0)]
        for q0, q1 in zip(pts, pts[1:]):
            P.beam(q0, q1, 0.018, 0.018, M.leaves[1])
        for t in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):      # 对生长叶
            q = base + lean * (t ** 1.5) + Vector((0, 0, h * t))
            b = rng.uniform(0, math.tau)
            for s_ in (0, math.pi):
                d = Vector((math.cos(b + s_), math.sin(b + s_), rng.uniform(0.2, 0.7))).normalized()
                leaf_card(P, q, d, rng.uniform(0.12, 0.17) * (1.2 - t * 0.5), 0.42, rng.choice(M.leaves[1:]), rng)
        if k < 6:
            tip = pts[-1]
            petal_flower(P, tip, Vector((lean.x * 2, lean.y * 2, 1)).normalized(), rng.uniform(0.11, 0.15),
                         M.petal_cream, M.flower_eye, rng, n_pet=8, cup=0.4)
        else:
            P.ico(pts[-1], 0.035, M.petal_cream, scale=(1, 1, 1.5), sub=1)


def _frame(n):
    """法线 n → 两个切向量"""
    n = Vector(n).normalized()
    t = n.cross(Vector((0, 0, 1)) if abs(n.z) < 0.95 else Vector((1, 0, 0))).normalized()
    return t, n.cross(t).normalized()


def leaf_card(P, base, d, L, ratio, mat, rng, twist=None):
    """尖头叶片（6 个顶点的菱形），从 base 沿方向 d 伸出 L 米"""
    d = Vector(d).normalized()
    s1, _ = _frame(d)
    if twist is None:
        twist = rng.uniform(-0.6, 0.6)
    side = (s1 * math.cos(twist) + d.cross(s1) * math.sin(twist)) * (L * ratio / 2)
    bm = P.bm
    b = Vector(base)
    vs = [bm.verts.new(v) for v in (b, b + d * L * 0.3 + side, b + d * L * 0.7 + side * 0.7, b + d * L,
                                    b + d * L * 0.7 - side * 0.7, b + d * L * 0.3 - side)]
    P._tag_faces([bm.faces.new(vs)], mat)


def petal_flower(P, c, n, r, petal_mat, eye_mat, rng, n_pet=5, cup=0.25):
    """星形小花：n_pet 片尖花瓣 + 花心，朝向法线 n，cup = 花瓣向上翘的程度"""
    t1, t2 = _frame(n)
    n = Vector(n).normalized()
    c = Vector(c)
    bm = P.bm
    a0 = rng.uniform(0, math.tau)
    faces = []
    for k in range(n_pet):
        a = a0 + k / n_pet * math.tau
        w = math.pi / n_pet * 0.75
        dir_ = lambda ang, rr, lift: c + (t1 * math.cos(ang) + t2 * math.sin(ang)) * rr + n * lift
        faces.append(bm.faces.new([bm.verts.new(v) for v in (
            c + n * 0.005, dir_(a - w, r * 0.5, r * cup * 0.35), dir_(a, r, r * cup), dir_(a + w, r * 0.5, r * cup * 0.35))]))
    P._tag_faces(faces, petal_mat)
    P.ico(c + n * r * 0.12, r * 0.22, eye_mat, sub=1)


def grass_tuft(P, p, rng, mats, h=(0.18, 0.38)):
    """一簇 3~4 根弯曲的草叶"""
    bm = P.bm
    faces = []
    for _ in range(rng.randint(3, 4)):
        a = rng.uniform(0, math.tau)
        lean = Vector((math.cos(a), math.sin(a), 0))
        side = Vector((-math.sin(a), math.cos(a), 0)) * rng.uniform(0.018, 0.03)
        hh = rng.uniform(*h)
        bend = rng.uniform(0.05, 0.18)
        b = Vector(p) + lean * rng.uniform(0, 0.04)
        m = b + Vector((0, 0, hh * 0.55)) + lean * bend * 0.3
        t = b + Vector((0, 0, hh)) + lean * bend
        faces.append(bm.faces.new([bm.verts.new(v) for v in (b - side, b + side, m + side * 0.6, t, m - side * 0.6)]))
        P._tag_faces([faces[-1]], rng.choice(mats))


def bell_flower(P, M, p, rng):
    """草地里垂头的小白花：细茎 + 朝下的小钟形"""
    h = rng.uniform(0.35, 0.6)
    a = rng.uniform(0, math.tau)
    lean = Vector((math.cos(a), math.sin(a), 0))
    top = Vector(p) + lean * 0.08 + Vector((0, 0, h))
    P.beam(p, top, 0.01, 0.01, M.blades[0])
    P.beam(top, top + lean * 0.05 - Vector((0, 0, 0.03)), 0.008, 0.008, M.blades[0])
    P.cyl(top + lean * 0.055 - Vector((0, 0, 0.06)), 0.028, 0.045, M.flower_white, seg=6, r2=0.01, smooth=False)


def scatter_points(ob, density, rng, min_nz=-1.0):
    """在对象（含修改器后的）表面按面积随机撒点 → [(位置, 法线)]"""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    oe = ob.evaluated_get(dg)
    me = oe.to_mesh()
    me.calc_loop_triangles()
    mw = ob.matrix_world
    rot = mw.to_3x3()
    out = []
    vs = me.vertices
    for tri in me.loop_triangles:
        nrm = (rot @ tri.normal).normalized()
        if nrm.z < min_nz:
            continue
        cnt = density * tri.area
        k = int(cnt) + (1 if rng.random() < cnt - int(cnt) else 0)
        a, b, c = (vs[i].co for i in tri.vertices)
        for _ in range(k):
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            out.append((mw @ (a + (b - a) * u + (c - a) * v), nrm))
    oe.to_mesh_clear()
    return out


def leaf_skin(ob, C, M, density=90, seed=1):
    """给树篱 / 灌木表面撒一层外翻的叶片，让轮廓毛茸茸（参考截图 26）"""
    rng = random.Random(seed)
    P = Part(ob.name + "_叶片", C)
    for p, n in scatter_points(ob, density, rng, min_nz=-0.4):
        t1, t2 = _frame(n)
        a = rng.uniform(0, math.tau)
        d = (n * rng.uniform(0.3, 0.8) + (t1 * math.cos(a) + t2 * math.sin(a))).normalized()
        leaf_card(P, p - n * 0.03, d, rng.uniform(0.1, 0.17), 0.45, rng.choice(M.leaves), rng)
    return P.finish()


def grass_skin(ob, C, M, density=30, flowers=1.2, seed=2):
    """草地顶面撒草叶簇和垂头小白花"""
    rng = random.Random(seed)
    P = Part(ob.name + "_草叶", C)
    for p, n in scatter_points(ob, density, rng, min_nz=0.7):
        grass_tuft(P, p, rng, M.blades)
    if flowers:
        for p, n in scatter_points(ob, flowers, rng, min_nz=0.7):
            bell_flower(P, M, p, rng)
    return P.finish()


# ===========================================================================
# 地面、台阶、喷泉、花园
# ===========================================================================

STAIR_X0, STAIR_X1 = -10.0, -26.0    # 台阶底 / 顶（向 -X 上升）
STAIR_Y0, STAIR_Y1 = -11.5, -3.2
STAIR_H = 5.0
CHANNEL_Y = (STAIR_Y0 + STAIR_Y1) / 2
GARDEN = (-22.0, -8.0, -3.0, 10.0)   # x0, x1, y0, y1
GARDEN_Z = 2.0
SOUTH_BED_H = 2.0                    # 台阶另一侧花坛的高度


def stair_z(x):
    t = min(1.0, max(0.0, (STAIR_X0 - x) / (STAIR_X0 - STAIR_X1)))
    return t * STAIR_H


ROAD_Y = -14.5    # 广场与外侧六边形砖路的分界


def build_ground(M, root):
    C = collection("地面", root)
    P = Part("广场", C)
    P.box_mm(-60, 80, -60, 70, -0.6, 0.0, M.plaza)
    P.finish()

    P = Part("人行道", C, bevel=0.02)
    P.box_mm(-8.3, TOWER_X, -2.1, 0.0, 0.0, 0.07, M.sidewalk)
    P.box_mm(-8.3, TOWER_X - 3.4, -2.4, -2.1, 0.0, 0.12, M.road_curb)       # 灰色路缘石（到圆塔处转弯）
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


CH_W, CURB_W = 0.56, 0.32            # 流水槽水面宽 / 两侧石沿宽
CURB_TOP, WATER_TOP = 0.45, 0.3      # 石沿顶、水面 相对台阶斜线的高度
FOUNT_R = 1.65                       # 八角喷泉外接圆半径
FOUNT_X = STAIR_X0 + 1.0 + (FOUNT_R + 0.15) * math.cos(math.pi / 8)
N_STEPS = 26


def stair_line(x):
    """台阶鼻尖连线的高度（流水槽、石沿、侧墙都顺着它走）"""
    return max(0.0, min(STAIR_H, (STAIR_X0 - x) / (STAIR_X0 - STAIR_X1) * STAIR_H))


def slope_prism(P, x0, x1, y0, y1, zb0, zt0, zb1, zt1, mat):
    """x0→x1 之间、底 / 顶高度线性变化的斜块（x0 处 zb0/zt0，x1 处 zb1/zt1）"""
    P.prism([(x0, zb0), (x1, zb1), (x1, zt1), (x0, zt0)], mat, lambda x, z, t: (x, y0 + (y1 - y0) * t, z))


def stair_side_wall(P, M, y0, y1, face_y, top_min, x_end):
    """
    台阶一侧的挡土墙（参考截图 23）：墙顶 = max(top_min, 台阶线 + 1.1)，
    拱纹浮雕墙面、白石壁柱分段、斜压顶，底部在广场上以方墩收头
    """
    xs = [STAIR_X0 + 0.9] + [STAIR_X0 - k * 0.5 for k in range(0, 60) if STAIR_X0 - k * 0.5 > x_end] + [x_end]
    xs = sorted(set(xs), reverse=True)
    top = lambda x: max(top_min, stair_line(x) + 1.1)
    for a, b in zip(xs, xs[1:]):
        slope_prism(P, a, b, y0, y1, 0.0, top(a), 0.0, top(b), M.wall_relief)
        slope_prism(P, a, b, y0 - 0.06, y1 + 0.06, top(a), top(a) + 0.16, top(b), top(b) + 0.16, M.stone)
    sgn = 1 if face_y > (y0 + y1) / 2 else -1
    fy0, fy1 = (y1, y1 + 0.1) if sgn > 0 else (y0 - 0.1, y0)
    for k in range(6):                                  # 壁柱
        x = STAIR_X0 - 0.6 - k * 3.2
        if x < x_end + 0.3:
            break
        P.box_mm(x - 0.25, x + 0.25, fy0, fy1, 0.0, top(x) - 0.05, M.stone)
    P.box_mm(x_end, STAIR_X0 + 0.9, fy0, fy1, 0.0, 0.25, M.stone)       # 勒脚
    # 底部方墩
    P.box_mm(STAIR_X0 + 0.6, STAIR_X0 + 1.4, y0 - 0.12, y1 + 0.12, 0.0, top_min + 0.35, M.stone)
    P.box_mm(STAIR_X0 + 0.5, STAIR_X0 + 1.5, y0 - 0.2, y1 + 0.2, top_min + 0.35, top_min + 0.5, M.stone)


def build_stairs(M, root):
    """
    大台阶 + 流水槽 + 八角喷泉（参考截图 23、24）：
      两段台阶夹着中间的流水槽；每级踏面由错缝长石板拼成，鼻尖略挑出
      流水槽两侧是分段的斜石沿，水面有浅色波光；槽在台阶底部平走一小段后接进八角喷泉
      八角喷泉：外圈低台座 + 石沿（面向台阶的一边开口接水槽）+ 水池 + 中心喷头与水花
      两侧挡土墙带拱纹浮雕，花园一侧的土坡随台阶升高；顶部平台两股小喷泉
    """
    C = collection("大台阶与喷泉", root)
    rng = random.Random(5)
    run = (STAIR_X0 - STAIR_X1) / N_STEPS
    rise = STAIR_H / N_STEPS
    ch0, ch1 = CHANNEL_Y - CH_W / 2 - CURB_W, CHANNEL_Y + CH_W / 2 + CURB_W
    flights = ((STAIR_Y0, ch0), (ch1, STAIR_Y1))

    P = Part("大台阶", C, bevel=0.025)
    for i in range(N_STEPS):
        x = STAIR_X0 - i * run
        z = (i + 1) * rise
        P.box_mm(x - run, x - 0.02, STAIR_Y0, STAIR_Y1, 0.0, z - 0.07, M.step_riser)
        for fy0, fy1 in flights:                        # 踏面：错缝长石板
            y = fy0
            first = True
            while y < fy1 - 0.05:
                L = rng.uniform(0.9, 1.6) if not first else rng.uniform(0.4, 1.4)
                first = False
                y1 = min(fy1, y + L)
                if fy1 - y1 < 0.35:
                    y1 = fy1
                P.box_mm(x - run - 0.01, x + 0.05, y + 0.008, y1 - 0.008, z - 0.07, z, M.step)
                y = y1
    P.box_mm(STAIR_X1 - 14, STAIR_X1, STAIR_Y0 - 12, STAIR_Y1 + 3, 0.0, STAIR_H, M.step)       # 顶部平台
    P.finish()

    # ---- 两侧挡土墙
    P = Part("台阶挡土墙", C, bevel=0.03)
    stair_side_wall(P, M, STAIR_Y1, STAIR_Y1 + 0.4, STAIR_Y1, GARDEN_Z + 0.55, STAIR_X1)        # 花园一侧
    stair_side_wall(P, M, STAIR_Y0 - 0.5, STAIR_Y0, STAIR_Y0, SOUTH_BED_H + 0.5, STAIR_X1)      # 另一侧
    P.finish()

    # ---- 流水槽石沿（分段）+ 平走段
    P = Part("流水槽石沿", C, bevel=0.03)
    x_oct = FOUNT_X - FOUNT_R * math.cos(math.pi / 8)
    seg = [x_oct] + [STAIR_X0 - k * 1.9 for k in range(0, 9) if STAIR_X0 - k * 1.9 > STAIR_X1] + [STAIR_X1 - 1.5]
    for a, b in zip(seg, seg[1:]):
        for y0, y1 in ((ch0, ch0 + CURB_W), (ch1 - CURB_W, ch1)):
            slope_prism(P, a - 0.012, b + 0.012, y0, y1, 0.0, stair_line(a) + CURB_TOP,
                        0.0, stair_line(b) + CURB_TOP, M.curb)
    P.finish()

    P = Part("流水", C)
    w0, w1 = CHANNEL_Y - CH_W / 2, CHANNEL_Y + CH_W / 2
    slope_prism(P, x_oct + 0.3, STAIR_X0, w0, w1, 0.0, WATER_TOP, 0.0, WATER_TOP, M.water)
    slope_prism(P, STAIR_X0, STAIR_X1 - 1.5, w0, w1, 0.0, WATER_TOP, 0.0, STAIR_H + WATER_TOP, M.water)
    for k in range(22):                                 # 波光
        x = rng.uniform(STAIR_X1 + 0.5, x_oct)
        y = rng.uniform(w0 + 0.08, w1 - 0.08)
        L = rng.uniform(0.25, 0.7)
        z0, z1 = stair_line(x) + WATER_TOP + 0.005, stair_line(x - L) + WATER_TOP + 0.005
        P.beam((x, y, z0), (x - L, y + rng.uniform(-0.05, 0.05), z1), 0.025, 0.004, M.water_jet)
    P.finish()

    # ---- 八角喷泉
    P = Part("八角喷泉", C, bevel=0.035)
    fx, fy = FOUNT_X, CHANNEL_Y
    oct_ = lambda R: [(fx + R * math.cos(a), fy + R * math.sin(a)) for a in (math.pi / 8 + k * math.pi / 4
                                                                            for k in range(8))]
    O, I, B = oct_(FOUNT_R), oct_(FOUNT_R - 0.32), oct_(FOUNT_R + 0.15)
    for k in range(8):                                  # 外圈低台座
        j = (k + 1) % 8
        xy_prism(P, [B[k], B[j], O[j], O[k]], 0.0, 0.1, M.curb)
    for k in range(8):                                  # 石沿（朝台阶的那一边开口）
        j = (k + 1) % 8
        if k == 3:                                      # 这一边朝台阶：中间留出水槽宽度的开口
            xo, xi = O[k][0], I[k][0]
            for ya, yb in ((O[j][1], w0), (w1, O[k][1])):
                xy_prism(P, [(xo, ya), (xo, yb), (xi, yb), (xi, ya)], 0.0, CURB_TOP, M.curb)
            continue
        xy_prism(P, [O[k], O[j], I[j], I[k]], 0.0, CURB_TOP, M.curb)
    xy_prism(P, I, 0.0, 0.05, M.stone_relief)           # 池底
    P.finish()
    P = Part("喷泉水面", C)
    xy_prism(P, I, 0.05, WATER_TOP - 0.02, M.water)
    for rr in (0.35, 0.6, 0.85):                       # 水面涟漪
        a0 = rng.uniform(0, math.tau)
        ring_arc(P, fx, fy, rr, rr + 0.03, a0, a0 + rng.uniform(2.0, 4.0), WATER_TOP - 0.02, WATER_TOP - 0.015,
                 M.water_jet, 16)
    P.finish()
    P = Part("喷泉水柱", C)
    P.lathe((fx, fy, 0.0), [(0.22, 0.0), (0.22, WATER_TOP + 0.05), (0.15, WATER_TOP + 0.12), (0.0, WATER_TOP + 0.12)],
            M.curb, seg=12)
    P.lathe((fx, fy, WATER_TOP + 0.1), [(0.12, 0.0), (0.07, 0.3), (0.05, 0.9), (0.1, 1.1), (0.14, 1.15),
                                        (0.0, 1.2)], M.water_jet, seg=12)
    for k in range(14):                                 # 落下的水花
        a = k / 14 * math.tau + rng.uniform(-0.1, 0.1)
        rr = rng.uniform(0.25, 0.5)
        P.ico((fx + math.cos(a) * rr, fy + math.sin(a) * rr, WATER_TOP + rng.uniform(0.02, 0.12)),
              rng.uniform(0.06, 0.11), M.water_jet, scale=(1, 1, 0.6), sub=1)
    # 顶部平台两股小喷泉
    for s_ in (-1, 1):
        x, y = STAIR_X1 - 0.8, CHANNEL_Y + s_ * 1.3
        P.lathe((x, y, STAIR_H), [(0.3, 0.0), (0.3, 0.15), (0.2, 0.2), (0.0, 0.2)], M.curb, seg=12)
        P.lathe((x, y, STAIR_H + 0.18), [(0.08, 0.0), (0.05, 0.3), (0.1, 0.7), (0.0, 0.75)], M.water_jet, seg=10)
    P.finish()

    # ---- 花园一侧随台阶升高的土坡（树篱顺着坡走）
    P = Part("台阶旁土坡", C)
    xg = GARDEN[0]
    slope_prism(P, STAIR_X0 - 4.0, xg, STAIR_Y1 + 0.4, STAIR_Y1 + 3.5, 0.0, GARDEN_Z + 0.25, 0.0,
                max(GARDEN_Z, stair_line(xg) + 0.85), M.grass)
    grass_skin(P.finish(), C, M, seed=11)
    P = Part("台阶旁树篱", C)
    slope_prism(P, STAIR_X0 - 4.5, xg, STAIR_Y1 + 0.5, STAIR_Y1 + 1.3, GARDEN_Z + 0.2, GARDEN_Z + 0.95,
                stair_line(xg) + 0.8, stair_line(xg) + 1.6, M.hedge)
    P.modifiers.append(organic(0.25, 0.06, 0.6))
    leaf_skin(P.finish(), C, M, seed=12)


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
    grass_skin(P.finish(), C, M, seed=13)

    # 修剪树篱（圆角 + 噪声置换，看起来像植物）
    P = Part("树篱", C)
    zt = GARDEN_Z + 0.25
    hedges = [((gx0 + 0.6, gx1 - 0.6, gy0 + 0.6, gy0 + 1.5), 0.9),
              ((gx1 - 1.5, gx1 - 0.6, gy0 + 1.5, gy1 - 0.6), 0.9),
              ((gx0 + 3.0, gx1 - 4.0, gy0 + 4.0, gy0 + 5.0), 0.8),
              ((gx0 + 3.0, gx0 + 4.0, gy0 + 5.0, gy1 - 2.0), 0.8)]
    for (x0, x1, y0, y1), h in hedges:
        P.box_mm(x0, x1, y0, y1, zt, zt + h, M.hedge)
    P.modifiers.append(organic(0.25, 0.06, 0.6))
    leaf_skin(P.finish(), C, M, seed=14)

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
    H = SOUTH_BED_H
    P = Part("右侧花坛", C, bevel=0.03)
    P.box_mm(STAIR_X1, STAIR_X0 + 0.9, STAIR_Y0 - 9.0, STAIR_Y0 - 0.5, 0.0, H, M.wall_relief)
    P.sweep([(STAIR_X1, STAIR_Y0 - 9.0), (STAIR_X0 + 0.9, STAIR_Y0 - 9.0), (STAIR_X0 + 0.9, STAIR_Y0 - 0.5),
             (STAIR_X1, STAIR_Y0 - 0.5)], [(0.0, H - 0.1), (0.12, H - 0.1), (0.12, H + 0.25), (-0.3, H + 0.25),
                                           (-0.3, H)], M.stone)
    P.finish()
    P = Part("右侧花坛_植物", C)
    P.box_mm(STAIR_X1 + 0.3, STAIR_X0 + 0.6, STAIR_Y0 - 8.7, STAIR_Y0 - 0.8, H, H + 0.2, M.grass)
    grass_skin(P.finish(), C, M, flowers=0.6, seed=15)
    P = Part("右侧花坛_树篱", C)
    P.box_mm(STAIR_X1 + 0.5, STAIR_X0 + 0.4, STAIR_Y0 - 2.2, STAIR_Y0 - 0.9, H + 0.2, H + 1.0, M.hedge)
    P.modifiers.append(organic(0.2, 0.06, 0.6))
    leaf_skin(P.finish(), C, M, seed=16)
    P = Part("右侧花坛_花", C)
    for x in (STAIR_X0 - 1.5, STAIR_X0 - 7.0, STAIR_X0 - 12.0):
        flower_bush(P, M, x, STAIR_Y0 - 1.6, H + 0.9, 0.8)
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
    """
    黄花灌木（参考截图 26）：叶片贴图的圆顶 + 表面外翻的叶片 + 上半部缀满五瓣小黄花
    """
    rng = random.Random(int(abs(x * 31 + y * 17 + z)))
    P.ico((x, y, z), r, M.hedge, scale=(1, 1, 0.78), sub=3, jitter=r * 0.035, smooth=True)
    c = Vector((x, y, z))
    area = 2.6 * math.pi * r * r
    for _ in range(int(area * 70)):                     # 叶片
        d = Vector((rng.gauss(0, 1), rng.gauss(0, 1), abs(rng.gauss(0, 1)) - 0.25)).normalized()
        p = c + Vector((d.x * r, d.y * r, d.z * r * 0.78))
        a = rng.uniform(0, math.tau)
        t1, t2 = _frame(d)
        leaf_card(P, p - d * 0.03, (d * 0.6 + t1 * math.cos(a) + t2 * math.sin(a)).normalized(),
                  rng.uniform(0.09, 0.15), 0.45, rng.choice(M.leaves), rng)
    for _ in range(int(area * 42)):                     # 小黄花
        d = Vector((rng.gauss(0, 1), rng.gauss(0, 1), abs(rng.gauss(0, 1.2)) + 0.05)).normalized()
        p = c + Vector((d.x * r * 1.02, d.y * r * 1.02, d.z * r * 0.8))
        petal_flower(P, p, d, rng.uniform(0.06, 0.09), M.flower if rng.random() < 0.8 else M.flower_deep,
                     M.flower_eye, rng)


def cypress(P, M, x, y, z, h, r):
    """
    柏树（参考截图 26）：树干 + 深色火焰形树芯 + 几百片锯齿边的羽状叶片，
    向外、向上翘，越靠上越亮
    """
    rng = random.Random(int(abs(x * 7 + y * 11)))
    P.cyl((x, y, z + 0.9), 0.24, 1.8, M.bark, seg=8, r2=0.18)
    prof = lambda t: r * 0.95 * math.sin(math.pi * min(1.0, t * 1.08)) ** 0.75 * (1 - 0.45 * t)
    P.lathe((x, y, z + 1.1), [(max(0.05, prof(t) * 0.7), (h - 1.1) * t) for t in (i / 12 for i in range(13))]
            + [(0.0, h - 1.1)], M.cypress_dark, seg=10)
    n = int(h * 48)
    for _ in range(n):
        t = rng.uniform(0.02, 1.0) ** 0.85
        zz = z + 1.1 + (h - 1.1) * t
        R = max(0.15, prof(t))
        a = rng.uniform(0, math.tau)
        out = Vector((math.cos(a), math.sin(a), 0))
        base = Vector((x, y, zz)) + out * R * rng.uniform(0.35, 0.6)
        up = rng.uniform(0.35, 1.0) + t * 0.6
        d = (out + Vector((0, 0, up))).normalized()
        L = R * rng.uniform(0.7, 1.0) + 0.35
        if rng.random() < 0.15 + 0.5 * t:
            mat = M.cypress_light
        elif rng.random() < 0.3:
            mat = M.cypress_dark
        else:
            mat = M.cypress
        frond(P, base, d, L, L * rng.uniform(0.3, 0.42), mat, rng)


def frond(P, base, d, L, W, mat, rng, seg=5):
    """锯齿边羽状叶片：沿叶轴两侧各 seg 个尖齿，叶尖稍微上翘"""
    d = Vector(d).normalized()
    s1, _ = _frame(d)
    roll = rng.uniform(-0.7, 0.7)
    side = (s1 * math.cos(roll) + d.cross(s1) * math.sin(roll)).normalized()
    up = Vector((0, 0, 1))
    b = Vector(base)
    axis = lambda t: b + d * (L * t) + up * (0.12 * L * t * t)
    left, right = [], []
    for i in range(seg):
        t0 = (i + 0.2) / seg
        t1 = (i + 0.85) / seg
        w0 = W / 2 * math.sin(math.pi * t0) ** 0.7
        w1 = W / 2 * math.sin(math.pi * min(t1, 0.98)) ** 0.7
        left += [axis(t0) + side * w0 * 0.45, axis(t1) + side * w1]
        right += [axis(t0) - side * w0 * 0.45, axis(t1) - side * w1]
    bm = P.bm
    pts = [b] + left + [axis(1.0)] + list(reversed(right))
    P._tag_faces([bm.faces.new([bm.verts.new(v) for v in pts])], mat)


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
    street_lamp(P, M, STAIR_X0 + 3.4, STAIR_Y0 - 1.6, 0.0)          # 喷泉旁
    street_lamp(P, M, -4.0, -12.6, 0.0)
    P.finish()


# ===========================================================================
# 远景：石塔、城墙、旗幡
# ===========================================================================

TOWER_R = 2.8                                          # 主楼右侧圆塔半径
TOWER_GAP = 1.4                                        # 主楼转角柱与圆塔台座之间的空隙（人行道从这里通到后面）
TOWER_X, TOWER_Y = BX0 + BW + TURRET_OUT + TOWER_GAP + TOWER_R + 0.55, 0.9
WALL_ANG = math.radians(72)                            # 右侧大墙 / 街道相对主楼立面往后转的角度
SIDE_W = 1.35                                          # 圆塔脚下人行道宽度


def ring_arc(P, cx, cy, r0, r1, a0, a1, z0, z1, mat, n=24):
    """水平圆环的一段（弯曲的路缘石、弧形人行道）"""
    for i in range(n):
        t0 = a0 + (a1 - a0) * i / n
        t1 = a0 + (a1 - a0) * (i + 1) / n
        pts = [(cx + r0 * math.cos(t0), cy + r0 * math.sin(t0)), (cx + r0 * math.cos(t1), cy + r0 * math.sin(t1)),
               (cx + r1 * math.cos(t1), cy + r1 * math.sin(t1)), (cx + r1 * math.cos(t0), cy + r1 * math.sin(t0))]
        P.prism(pts, mat, lambda x, y, t: (x, y, z0 + (z1 - z0) * t))


def xy_prism(P, pts, z0, z1, mat):
    P.prism(pts, mat, lambda x, y, t: (x, y, z0 + (z1 - z0) * t))


def bench(P, M, p, d, n):
    """公园长椅：p = 中心，d = 长度方向，n = 朝前方向"""
    p, d, n = Vector(p), Vector(d), Vector(n)
    up = Vector((0, 0, 1))
    for s_ in (-1, 1):                                  # 两端铸铁腿
        q = p + d * s_ * 0.8
        P.beam(q + n * 0.25, q + n * 0.25 + up * 0.45, 0.08, 0.08, M.teal_frame)
        P.beam(q - n * 0.2, q - n * 0.2 + up * 0.9, 0.08, 0.08, M.teal_frame)
        P.beam(q + n * 0.28 + up * 0.43, q - n * 0.22 + up * 0.43, 0.08, 0.06, M.teal_frame)
    for k in range(3):                                  # 座板
        o = n * (0.2 - k * 0.16) + up * 0.46
        P.beam(p - d * 0.95 + o, p + d * 0.95 + o, 0.13, 0.04, M.bench_wood)
    for k in range(2):                                  # 靠背
        o = -n * 0.24 + up * (0.62 + k * 0.16)
        P.beam(p - d * 0.95 + o, p + d * 0.95 + o, 0.04, 0.12, M.bench_wood)


def build_tower_and_right_wall(M, C):
    """
    主楼右侧（参考截图 19、20）：
      巨大圆塔：大块白石错缝，脚下两层多边形台座（顶面抹斜角）
      人行道绕塔脚弯出去一段弧，路缘石跟着弯，再沿右侧大墙继续
      右侧大墙向后斜 22°：巨石、扶壁、拱形凹龛、远处第二座圆塔
      墙脚长花坛（树篱 + 黄花丛），花坛头路灯、长椅、排水篦子、路角花盆
      主楼与圆塔之间墙上的竖向拱纹石板
    """
    tx, ty, r = TOWER_X, TOWER_Y, TOWER_R
    # ---- 圆塔（对象原点放在塔心，贴图才能绕塔一圈）
    P = Part("圆塔", C, bevel=0.02)
    P.lathe((tx, ty, 0), [(r, 1.45), (r, 34.0), (r + 0.25, 34.2), (r + 0.25, 34.9), (r - 0.2, 35.1), (0, 35.1)],
            M.tower, seg=64, smooth=True)
    P.lathe((tx, ty, 0), [(r + 0.55, 0.0), (r + 0.55, 0.55), (r + 0.45, 0.65), (r + 0.3, 0.65), (r + 0.3, 1.35),
                          (r + 0.18, 1.45), (0, 1.45)], M.tower_plinth, seg=16, smooth=False)
    P.finish(origin=(tx, ty, 0))

    # ---- 弧形人行道 + 路缘石（绕塔脚，止于花坛头）；楼与塔之间的空隙也铺人行道
    Rs = r + 0.55 + SIDE_W
    C0 = Vector((tx, ty, 0))
    dW = Vector((math.cos(WALL_ANG), math.sin(WALL_ANG), 0))       # 大墙走向（往后）
    nO = Vector((math.sin(WALL_ANG), -math.cos(WALL_ANG), 0))      # 大墙朝街道的外法线
    depth = 1.5                                                    # 花坛进深
    wall_off = 0.5                                                 # 墙面离塔心线的距离
    ext = math.sqrt(max(Rs ** 2 - (wall_off + depth) ** 2, 0.0))
    vE = nO * (wall_off + depth) + dW * ext                        # 弧线终点（花坛前角）
    a0 = math.atan2(-2.25 - ty, -math.sqrt(max(Rs ** 2 - (-2.25 - ty) ** 2, 0)))
    a1 = math.atan2(vE.y, vE.x)
    while a1 < a0:
        a1 += math.tau
    P = Part("圆塔人行道", C, bevel=0.01)
    P.cyl((tx, ty, 0.035), Rs, 0.07, M.sidewalk, seg=64, smooth=False)
    ring_arc(P, tx, ty, Rs, Rs + 0.3, a0, a1, 0.0, 0.12, M.road_curb, 40)
    gx0 = BX0 + BW + TURRET_OUT
    P.box_mm(gx0, tx - r, -2.1, 10.4, 0.0, 0.07, M.sidewalk)       # 楼与塔之间的通道
    P.finish()

    # ---- 右侧大墙（从塔心往后斜）
    P = Part("右侧大墙", C, bevel=0.05)
    base = C0 - nO * wall_off
    back = base - nO * 8.0
    L = 60.0
    xy_prism(P, [base[:2], (base + dW * L)[:2], (back + dW * L)[:2], back[:2]], 0.0, 24.0, M.bigwall_n)
    for k in range(6):                                  # 扶壁
        q = base + dW * (8 + k * 9)
        xy_prism(P, [(q - dW * 1.1)[:2], (q + dW * 1.1)[:2], (q + dW * 1.1 + nO * 0.9)[:2],
                     (q - dW * 1.1 + nO * 0.9)[:2]], 0.0, 24.0, M.bigwall_n)
        xy_prism(P, [(q - dW * 1.35)[:2], (q + dW * 1.35)[:2], (q + dW * 1.35 + nO * 1.15)[:2],
                     (q - dW * 1.35 + nO * 1.15)[:2]], 0.0, 1.2, M.stone)
        if k < 5:                                       # 拱形凹龛
            m = q + dW * 4.5
            Fw = Frame(m - dW * 2.0, dW, nO)
            P.fpoly(Fw, arched(3.4, 9.0, 1.7), -0.02, 0.02, M.bigwall_dark, du=2.0, dz=3.0)
            P.fring(Fw, shift(arched(3.4, 9.0, 1.7, 0.25), 2.0, 3.0), shift(arched(3.4, 9.0, 1.7), 2.0, 3.0),
                    0.0, 0.2, M.stone)
    for z in (12.0, 20.0):                              # 横向线脚
        xy_prism(P, [base[:2], (base + dW * L)[:2], (base + dW * L + nO * 0.4)[:2], (base + nO * 0.4)[:2]],
                 z, z + 0.6, M.stone)
    P.finish()
    q2 = base + dW * 30 + nO * 1.0                      # 远处第二座圆塔
    P = Part("远处圆塔", C, bevel=0.02)
    P.lathe((q2.x, q2.y, 0), [(3.2, 0.0), (3.2, 30.0), (3.5, 30.3), (3.5, 31.0), (0, 31.0)], M.tower, seg=48)
    P.lathe((q2.x, q2.y, 0), [(3.8, 0.0), (3.8, 0.9), (3.6, 1.1), (0, 1.1)], M.tower_plinth, seg=16, smooth=False)
    P.finish(origin=(q2.x, q2.y, 0))

    # ---- 圆塔左侧（朝楼与塔之间的空隙）：白石框里的竖向拱纹石板
    P = Part("拱纹石板", C, bevel=0.01)
    Fg = Frame((tx - r - 0.02, ty - 0.3, 0), (0, 1, 0), (-1, 0, 0))
    for u0 in (0.0, 1.35):
        P.fbox(Fg, u0, u0 + 0.22, 1.45, 18.0, 0.0, 0.3, M.stone)
    P.fbox(Fg, 0.22, 1.35, 1.45, 18.0, -0.1, 0.12, M.scallop_big)
    P.finish()

    # ---- 墙脚长花坛（从弧线终点开始）+ 路灯 + 长椅 + 排水篦子 + 花盆 + 井盖
    P = Part("大墙花坛", C, bevel=0.03)
    p0 = base + dW * ext
    p1 = base + dW * 28.0
    xy_prism(P, [p0[:2], p1[:2], (p1 + nO * depth)[:2], (p0 + nO * depth)[:2]], 0.0, 0.6, M.stone)
    xy_prism(P, [(p0 - dW * 0.05)[:2], (p1 + dW * 0.05)[:2], (p1 + nO * (depth + 0.08) + dW * 0.05)[:2],
                 (p0 + nO * (depth + 0.08) - dW * 0.05)[:2]], 0.6, 0.7, M.curb)
    P.finish()
    P = Part("大墙花坛_树篱", C)
    h0, h1 = p0 + nO * 0.2 + dW * 0.2, p1 + nO * 0.2 - dW * 0.2
    xy_prism(P, [h0[:2], h1[:2], (h1 + nO * (depth - 0.4))[:2], (h0 + nO * (depth - 0.4))[:2]], 0.7, 1.45, M.hedge)
    P.modifiers.append(organic(0.25, 0.06, 0.6))
    leaf_skin(P.finish(), C, M, seed=17)
    P = Part("大墙花坛_花", C)
    for t in (2.5, 11.0, 20.0):
        q = p0 + dW * t + nO * (depth * 0.55)
        flower_bush(P, M, q.x, q.y, 1.6, 0.85)
    P.finish()
    P = Part("大墙前_路灯长椅", C, bevel=0.01)
    lamp_p = p0 + nO * (depth + 0.35) + dW * 0.6
    street_lamp(P, M, lamp_p.x, lamp_p.y, 0.0)
    bench(P, M, p0 + dW * 12.0 + nO * (depth + 0.7), dW, nO)
    gp = p0 + dW * 8.0 + nO * (depth + 0.6)
    P.beam(gp - dW * 0.7 + Vector((0, 0, 0.005)), gp + dW * 0.7 + Vector((0, 0, 0.005)), 0.45, 0.02, M.stone)
    for k in range(9):
        o = dW * (-0.6 + k * 0.15)
        P.beam(gp + o - nO * 0.17 + Vector((0, 0, 0.02)), gp + o + nO * 0.17 + Vector((0, 0, 0.02)),
               0.07, 0.012, M.grate)
    pot = C0 + vE * ((Rs + 0.6) / Rs)
    flower_pot(P, M, pot.x, pot.y, 0.0)
    P.finish()
    P = Part("圆塔井盖", C, bevel=0.005)
    mh = C0 + Vector((math.cos(a0 + 0.55 * (a1 - a0)), math.sin(a0 + 0.55 * (a1 - a0)), 0)) * (r + 0.55 + 0.75)
    manhole(P, M, mh.x, mh.y, 0.07, r=0.42)
    P.finish()

    # ---- 街道对面的弧形草坪（参考截图 22 右侧）
    P = Part("街对面草坪_石边", C, bevel=0.03)
    g0, g1, gn0, gn1 = 3.0, 45.0, 14.0, 30.0
    pts_out = [base + dW * (g0 + (g1 - g0) * i / 12) + nO * (gn0 + 1.8 * math.sin(math.pi * i / 12))
               for i in range(13)]
    pts_in = [p + nO * (gn1 - gn0) for p in pts_out]
    for a, b, c_, d_ in zip(pts_out, pts_out[1:], pts_in[1:], pts_in):
        xy_prism(P, [a[:2], b[:2], c_[:2], d_[:2]], 0.0, 0.55, M.stone)
    for a, b in zip(pts_out, pts_out[1:]):
        xy_prism(P, [(a - nO * 1.4)[:2], (b - nO * 1.4)[:2], b[:2], a[:2]], 0.0, 0.07, M.sidewalk)
        xy_prism(P, [(a - nO * 1.7)[:2], (b - nO * 1.7)[:2], (b - nO * 1.4)[:2], (a - nO * 1.4)[:2]], 0.0, 0.12,
                 M.road_curb)
    P.finish()
    P = Part("街对面草坪", C)
    for a, b, c_, d_ in zip(pts_out, pts_out[1:], pts_in[1:], pts_in):
        xy_prism(P, [(a + nO * 0.25)[:2], (b + nO * 0.25)[:2], c_[:2], d_[:2]], 0.55, 0.7, M.grass)
    P.finish()
    P = Part("街对面_路灯长椅", C, bevel=0.01)
    for i in (2, 6, 10):
        q = pts_out[i] - nO * 0.5
        street_lamp(P, M, q.x, q.y, 0.07)
    q = pts_out[4] - nO * 0.8
    bench(P, M, q + Vector((0, 0, 0.07)), dW, -nO)
    for i in (1, 8):
        q = pts_out[i] - nO * 0.7
        flower_pot(P, M, q.x, q.y, 0.07)
    P.finish()


def build_backdrop(M, root):
    C = collection("石塔与城墙", root)
    build_tower_and_right_wall(M, C)

    # 背后的城墙（主楼后面 + 左后方高墙）
    P = Part("城墙", C, bevel=0.05)
    P.box_mm(-45, 16, 10.4, 16, 0.0, 34.0, M.bigwall)
    for x in range(-40, 12, 9):
        P.box_mm(x - 1.0, x + 1.0, 9.6, 10.4, 0.0, 34.0, M.bigwall)            # 扶壁
        P.box_mm(x - 1.2, x + 1.2, 9.4, 10.4, 0.0, 1.5, M.stone)
    for z in (13.0, 23.0):                                                   # 横向线脚
        P.box_mm(-45, 16, 10.0, 10.5, z, z + 0.6, M.stone)
    # 高处的拱形凹窗（阴影里的深色）
    for x in range(-36, 8, 9):
        P.prism(arched(4.2, 7.0, 2.1), M.bigwall_dark,
                lambda u, z, t, x=x: (x + 4.5 + u, 10.38 - 0.02 * t, 15.0 + z))
    # 左侧台阶顶上的高台和墙
    P.box_mm(-60, STAIR_X1 - 14, -40, 16, 0.0, 30.0, M.bigwall)
    P.box_mm(-60, -40, -40, 16, 0.0, 5.0, M.bigwall)
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
    x, y = 25.0, -19.0
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
    # 参考图 14/15：正对大门和书报摊的近景
    "facade": dict(loc=(2.2, -8.2, 1.9), target=(2.2, 0.0, 2.6), lens=26),
    # 参考图 13：从书报摊旁斜着仰看立面
    # 参考图 17：从左边楼角往上看二层和转角柱
    "corner": dict(loc=(-12.5, -6.5, 3.6), target=(-1.5, 0.0, 6.6), lens=24),
    # 参考图 20：从广场右前方高处俯看主楼、圆塔和右侧大墙
    "tower": dict(loc=(22.0, -13.0, 16.0), target=(11.0, 5.0, 0.0), lens=24),
    # 参考图 19：书报摊旁看圆塔底座
    "tower_close": dict(loc=(6.2, -4.8, 6.4), target=(10.2, 1.5, 0.6), lens=24),
    # 参考图 23：广场上仰看台阶和流水槽
    "stairs_up": dict(loc=(-5.0, -2.2, 6.0), target=(-18.0, -9.0, 2.6), lens=22),
    # 参考图 24：高处俯看八角喷泉
    "fountain": dict(loc=(-5.8, -12.8, 13.0), target=(-7.3, -5.2, 0.0), lens=24),
    # 参考图 25：门口花盆近景
    "pot": dict(loc=(-1.3, -3.3, 1.8), target=(-2.45, -0.7, 0.75), lens=32),
    # 参考图 26：花园里的树篱、黄花丛、柏树
    "garden": dict(loc=(-15.5, -1.8, 3.6), target=(-10.0, 5.0, 2.8), lens=26),
    "facade_side": dict(loc=(5.6, -2.1, 2.6), target=(-4.0, 0.3, 5.3), lens=22),
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

    cam = bpy.data.cameras.new("相机")
    cam.clip_end = 1000
    co = bpy.data.objects.new("相机", cam)
    C.objects.link(co)
    bpy.context.scene.camera = co
    set_view(view)


def set_view(view):
    """把相机移到某个预设机位"""
    v = VIEWS[view]
    co = bpy.context.scene.camera
    co.data.lens = v["lens"]
    co.location = v["loc"]
    co.rotation_euler = (Vector(v["target"]) - Vector(v["loc"])).to_track_quat("-Z", "Y").to_euler()


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
                  bpy.data.worlds, bpy.data.textures, bpy.data.images):  # noqa
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
    p.add_argument("--view", default="front",
                   help="机位，可以用逗号连写多个一次渲染完：" + ",".join(VIEWS))
    p.add_argument("--part", choices=("all", "building"), default="all")
    p.add_argument("--export-textures", help="把生成的地砖贴图另存为 PNG 到这个文件夹")
    args, _ = p.parse_known_args(argv)
    return args


def main():
    args = parse_args()
    views = [v.strip() for v in args.view.split(",") if v.strip()]
    for v in views:
        if v not in VIEWS:
            raise SystemExit("未知机位 %s，可选：%s" % (v, ", ".join(VIEWS)))
    build(views[0], args.part)
    w, h = (int(v) for v in args.res.lower().split("x"))
    setup_render(args.engine, (w, h), args.samples, args.outline)
    if args.export_textures:
        export_images(os.path.abspath(args.export_textures),
                      *[i for i in bpy.data.images if i.name.startswith(("广场", "风车", "砂岩墙", "白石墙"))])
    if args.save:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save), compress=True)
    if args.render:
        # 多个机位时，输出路径里的 {view} 会替换成机位名（没有 {view} 就自动加在文件名后面）
        for v in views:
            set_view(v)
            out = args.render
            if len(views) > 1 and "{view}" not in out:
                root_, ext = os.path.splitext(out)
                out = root_ + "_{view}" + ext
            bpy.context.scene.render.filepath = os.path.abspath(out.replace("{view}", v))
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
