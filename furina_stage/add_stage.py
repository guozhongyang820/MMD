"""给芙宁娜 MMD 场景加一个舞台背景：酒红丝绒幕布 + 打蜡木地板 + 舞台灯光。

用法（Blender 5.2，与模型文件同版本）：
  命令行：blender -b furina.blend -P add_stage.py -- --save furina_stage.blend
  预览渲染：... -- --render "out/frame_{frame}.png" --frames 1,200,400 --percent 50
  全景机位：... -- --render "out/overview.png" --overview
  界面里：打开模型文件后在“脚本”工作区运行本文件即可（会在当前场景里生成/重建）。

脚本可以重复运行：每次先删掉上一次生成的“舞台背景”集合、灯光链接集合和世界，再重新生成。

人物的赛璐璐材质是 漫射 → Shader 转 RGB → 三阶色，场景里的任何灯光、环境光都会改变她的明暗。
为了不破坏原来调好的观感：
  * 舞台灯只照舞台（灯光链接 → “舞台_受光”），原来的 Cel Key / Cel Face Fill 只照人物（→ “角色_受光”）；
  * 世界改成很暗的舞台环境，原世界（Neutral studio background）提供的那份均匀环境光，
    在人物材质的“Shader 转 RGB”之后用一个“相加”节点补回来（节点名 STAGE_AMBIENT_NODE）。
"""

import math
import sys

import bpy
import numpy as np
from mathutils import Vector

STAGE_COLL = "舞台背景"
RECV_STAGE = "舞台_受光"
RECV_CHAR = "角色_受光"
RECV_FRAME = "檐幕边幕_受光"
RECV_BACK = "背幕_受光"
STAGE_WORLD = "舞台环境"
STAGE_AMBIENT_NODE = "舞台_原环境光补偿"
OVERVIEW_CAM = "舞台全景相机"

CHARACTER_OBJECTS = ("Furina_Mesh", "Cel Fine Outline")
CHARACTER_LIGHTS = ("Cel Key", "Cel Face Fill")
OLD_GROUND = "Studio_Ground"

# 舞台尺寸（米）。人物站在原点附近，相机在 -Y 方向往 +Y 看。
CURTAIN_Y = 2.2          # 背景大幕
CURTAIN_W = 13.0
CURTAIN_H = 7.5
PROSC_HALF = 4.3         # 台口半宽（两侧边幕、顶部檐幕的位置）
PROSC_Y = -2.1
VALANCE_TOP = 6.8
VALANCE_BOTTOM = 5.6
STAGE_FRONT_Y = -3.4     # 台口前沿
# 大幕的褶：mean_w 褶脊到褶谷的平均距离，depth 褶深，sigma 宽窄的随机程度，
# bunch_* 堆叠区（每隔多少米一处、挤多紧），drift 褶沿高度左右漂移，wobble 褶沿高度变深变浅
CURTAIN_DRAPE = dict(mean_w=0.12, depth=0.11, sigma=0.3, bunch_every=3.0, bunch_amp=(0.3, 0.9),
                     drift=0.06, wobble=0.18, n_creases=200, crease_amp=(0.0015, 0.005))
# 大幕落地堆布：forward 向台前摊出多远，height 在多高处开始弯下来，nose 最前面折边的圆弧半径，lumps 揉皱起伏（米）
# ridge：褶脊比褶谷多流出去多少（0 = 一样，1 = 褶脊约是褶谷的 3 倍）
HEM_POOL = dict(forward=0.15, forward_var=0.04, height=0.085, height_var=0.02, nose=0.02, lumps=0.01, ridge=1.0)
PLANK_W = 0.15           # 地板条宽
PLANK_L = 2.1            # 地板条长
# 木纹公式的输入参数，每块板在基准值上随机浮动的比例（0.08 = ±8%）
WOOD_JITTER = {
    "ring_scale": 0.08,       # 年轮疏密（基准 34）
    "ring_distortion": 0.12,  # 年轮扭曲（基准 5.0）
    "stretch": 0.10,          # 纹理沿板长的拉伸（基准 0.05）
    "fine_scale": 0.06,       # 细直纹疏密（基准 95）
    "ring_contrast": 0.10,    # 年轮深浅（基准 0.38）
}


# ----------------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------------

def srgb(hex_or_tuple):
    """sRGB 颜色 → 线性 RGBA。"""
    if isinstance(hex_or_tuple, str):
        h = hex_or_tuple.lstrip("#")
        c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    else:
        c = list(hex_or_tuple)
    lin = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return (*lin, 1.0)


def new_collection(name, parent):
    coll = bpy.data.collections.new(name)
    parent.children.link(coll)
    return coll


def link(obj, coll):
    coll.objects.link(obj)
    return obj


def grid_mesh(name, X, Y, Z, U=None, V=None):
    """(行, 列) 网格坐标 → 四边形网格，可选 UV。"""
    nr, nc = X.shape
    co = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    ii, jj = np.meshgrid(np.arange(nr - 1), np.arange(nc - 1), indexing="ij")
    a = (ii * nc + jj).ravel()
    quads = np.stack([a, a + 1, a + 1 + nc, a + nc], axis=-1)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", co.ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", quads.ravel().astype(np.int32))
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    if U is not None:
        uv = np.stack([U, V], axis=-1).reshape(-1, 2)[quads.ravel()].astype(np.float32)
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", uv.ravel())
    me.update(calc_edges=True)
    me.validate()
    me.shade_smooth()
    return me


class Nodes:
    """少写点 nodes.new / links.new 的小帮手。"""

    def __init__(self, tree):
        self.tree = tree
        self.nodes = tree.nodes
        self.links = tree.links
        self.x = 0

    def new(self, kind, label=None, **inputs):
        n = self.nodes.new(kind)
        n.location = (self.x, 0)
        self.x += 180
        if label:
            n.label = label
        for k, v in inputs.items():
            self.set(n, k, v)
        return n

    def set(self, node, key, value):
        sock = node.inputs[key] if isinstance(key, int) else node.inputs.get(key)
        if sock is None:
            setattr(node, key, value)
            return
        if isinstance(value, bpy.types.NodeSocket):
            self.links.new(value, sock)
        else:
            sock.default_value = value

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.set(n, i, v)
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        n = self.new("ShaderNodeVectorMath")
        n.operation = op
        if a is not None:
            self.set(n, 0, a)
        if b is not None:
            self.set(n, 1, b)
        if scale is not None:
            n.inputs["Scale"].default_value = scale
        return n.outputs[1] if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else n.outputs[0]

    def combine(self, x, y, z=0.0):
        n = self.new("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.set(n, i, v)
        return n.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        n = self.new("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        self.set(n, 0, fac)
        self.links.new(a, n.inputs[6]) if isinstance(a, bpy.types.NodeSocket) else setattr(n.inputs[6], "default_value", a)
        self.links.new(b, n.inputs[7]) if isinstance(b, bpy.types.NodeSocket) else setattr(n.inputs[7], "default_value", b)
        return n.outputs[2]

    def ramp(self, fac, stops):
        n = self.new("ShaderNodeValToRGB")
        self.set(n, 0, fac)
        stops = sorted(stops)
        el = n.color_ramp.elements
        el[0].position, el[1].position = stops[0][0], stops[-1][0]
        for pos, _ in stops[1:-1]:
            el.new(pos)
        for e, (_, col) in zip(el, stops):   # 色标按位置自动排序，和 stops 一一对应
            e.color = col
        return n.outputs[0]

    def map_range(self, v, a, b, c=0.0, d=1.0, interp="LINEAR"):
        n = self.new("ShaderNodeMapRange")
        n.interpolation_type = interp
        self.set(n, 0, v)
        for i, val in zip((1, 2, 3, 4), (a, b, c, d)):
            self.set(n, i, val)
        return n.outputs[0]


def new_material(name):
    mat = bpy.data.materials.get(name)
    if mat:
        bpy.data.materials.remove(mat)
    mat = bpy.data.materials.new(name)
    if not mat.node_tree:
        mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat


# ----------------------------------------------------------------------------
# 材质
# ----------------------------------------------------------------------------

def mat_velvet(name="酒红丝绒"):
    """酒红天鹅绒：正对视线时深、掠射角泛出偏粉的绒光；竖向绒毛纹 + 细密织物凹凸。"""
    mat = new_material(name)
    nb = Nodes(mat.node_tree)
    uv = nb.new("ShaderNodeTexCoord").outputs["UV"]
    sep = nb.new("ShaderNodeSeparateXYZ")
    nb.set(sep, 0, uv)
    # 竖向绒毛条纹：横向频率高、竖向频率低的噪声
    streak_co = nb.combine(nb.math("MULTIPLY", sep.outputs[0], 90.0), nb.math("MULTIPLY", sep.outputs[1], 1.6))
    streak = nb.new("ShaderNodeTexNoise", Scale=1.0, Detail=4.0, Roughness=0.55)
    nb.set(streak, "Vector", streak_co)
    # 大块的绒面“倒毛”明暗（丝绒被摸过的那种色块）
    patch = nb.new("ShaderNodeTexNoise", Scale=0.45, Detail=2.0, Roughness=0.5)
    nb.set(patch, "Vector", uv)
    patch_f = nb.map_range(patch.outputs["Fac"], 0.3, 0.7, 0.86, 1.1)
    streak_f = nb.map_range(streak.outputs["Fac"], 0.25, 0.75, 0.9, 1.08)

    lw = nb.new("ShaderNodeLayerWeight", Blend=0.42)
    facing = nb.math("POWER", lw.outputs["Facing"], 1.6)
    core = srgb("#4f0815")      # 正对时：深酒红
    rim = srgb("#a8203c")       # 掠射时：绒光，偏玫红
    base = nb.mix(facing, core, rim)
    base = nb.mix(1.0, base, nb.combine(patch_f, patch_f, patch_f), blend="MULTIPLY")
    # 褶谷轻微压暗（几何里存的 fold 属性：0 = 朝观众的褶脊，1 = 最深的褶谷）
    fold = nb.new("ShaderNodeAttribute")
    fold.attribute_type = "GEOMETRY"
    fold.attribute_name = "fold"
    cav = nb.map_range(nb.math("POWER", fold.outputs["Fac"], 1.6), 0.0, 1.0, 1.0, 0.65)
    base = nb.mix(1.0, base, nb.combine(cav, cav, cav), blend="MULTIPLY")
    base = nb.mix(1.0, base, nb.combine(streak_f, streak_f, streak_f), blend="MULTIPLY")

    # 织物细节凹凸
    weave = nb.new("ShaderNodeTexNoise", Scale=900.0, Detail=1.0)
    nb.set(weave, "Vector", uv)
    height = nb.math("ADD", nb.math("MULTIPLY", weave.outputs["Fac"], 0.5), nb.math("MULTIPLY", streak.outputs["Fac"], 0.5))
    bump = nb.new("ShaderNodeBump", Strength=0.12, Distance=0.002)
    nb.set(bump, "Height", height)

    bsdf = nb.new("ShaderNodeBsdfPrincipled")
    nb.set(bsdf, "Base Color", base)
    nb.set(bsdf, "Roughness", 0.82)
    nb.set(bsdf, "Specular IOR Level", 0.22)
    nb.set(bsdf, "Sheen Weight", 1.0)
    nb.set(bsdf, "Sheen Roughness", 0.32)
    nb.set(bsdf, "Sheen Tint", srgb("#ff8a9c"))
    nb.set(bsdf, "Normal", bump.outputs["Normal"])
    out = nb.new("ShaderNodeOutputMaterial")
    nb.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat



def mat_wood_floor(name="舞台木地板"):
    """打蜡胡桃木长条地板：错缝铺设，板缝、木纹、棕眼、半哑光清漆。

    所有板用同一个木纹公式，但每块板用自己的随机数把公式的输入参数轻微抖动
    （年轮尺度、扭曲、纵向拉伸、细纹尺度、年轮对比，幅度见 WOOD_JITTER），
    再加上随机的纹理偏移和一点色差，所以块块相似又块块不同。
    """
    mat = new_material(name)
    nb = Nodes(mat.node_tree)
    pos = nb.new("ShaderNodeNewGeometry").outputs["Position"]
    sep = nb.new("ShaderNodeSeparateXYZ")
    nb.set(sep, 0, pos)
    x, y = sep.outputs[0], sep.outputs[1]

    # 板条编号：沿 X 是第几条，沿 Y 按每条随机错缝后是第几段
    xs = nb.math("DIVIDE", x, PLANK_W)
    col = nb.math("FLOOR", xs)
    u = nb.math("SUBTRACT", xs, col)
    stagger = nb.new("ShaderNodeTexWhiteNoise")
    stagger.noise_dimensions = "2D"
    nb.set(stagger, "Vector", nb.combine(col, 3.7))
    along = nb.math("ADD", y, nb.math("MULTIPLY", stagger.outputs["Value"], PLANK_L))
    ys = nb.math("DIVIDE", along, PLANK_L)
    seg = nb.math("FLOOR", ys)
    v = nb.math("SUBTRACT", ys, seg)
    rid = nb.new("ShaderNodeTexWhiteNoise")
    rid.noise_dimensions = "3D"
    nb.set(rid, "Vector", nb.combine(col, seg, 1.3))
    rsep = nb.new("ShaderNodeSeparateColor")
    nb.set(rsep, 0, rid.outputs["Color"])
    r0, r1, r2 = rsep.outputs[0], rsep.outputs[1], rsep.outputs[2]
    # 第二组随机数：给木纹公式的参数做轻微抖动
    jid = nb.new("ShaderNodeTexWhiteNoise")
    jid.noise_dimensions = "3D"
    nb.set(jid, "Vector", nb.combine(col, seg, 5.1))
    jsep = nb.new("ShaderNodeSeparateColor")
    nb.set(jsep, 0, jid.outputs["Color"])
    j0, j1, j2 = jsep.outputs[0], jsep.outputs[1], jsep.outputs[2]

    def jitter(base, key, rnd):
        """base × (1 ± WOOD_JITTER[key])，每块板一个值。"""
        a = WOOD_JITTER[key]
        return nb.map_range(rnd, 0.0, 1.0, base * (1 - a), base * (1 + a))

    # 板缝：离板边的距离（米）
    du = nb.math("MULTIPLY", nb.math("MINIMUM", u, nb.math("SUBTRACT", 1.0, u)), PLANK_W)
    dv = nb.math("MULTIPLY", nb.math("MINIMUM", v, nb.math("SUBTRACT", 1.0, v)), PLANK_L)
    dist = nb.math("MINIMUM", du, nb.math("MULTIPLY", dv, 1.4))
    seam = nb.map_range(dist, 0.0004, 0.0016, 1.0, 0.0, interp="SMOOTHSTEP")
    bevel = nb.map_range(dist, 0.0004, 0.004, 1.0, 0.0, interp="SMOOTHSTEP")

    # 木纹坐标：每块板随机平移，沿板长方向拉长
    gx = nb.math("ADD", x, nb.math("MULTIPLY", r0, 7.0))
    gy = nb.math("ADD", nb.math("MULTIPLY", along, jitter(0.05, "stretch", j2)), nb.math("MULTIPLY", r1, 9.0))
    gco = nb.combine(gx, gy, nb.math("MULTIPLY", r2, 5.0))
    # 生长轮（较宽、扭曲大）
    rings = nb.new("ShaderNodeTexWave", Detail=3.0, **{"Detail Scale": 1.6, "Detail Roughness": 0.6})
    nb.set(rings, "Scale", jitter(34.0, "ring_scale", j0))
    nb.set(rings, "Distortion", jitter(5.0, "ring_distortion", j1))
    rings.wave_type = "BANDS"
    rings.bands_direction = "X"
    rings.wave_profile = "SAW"
    nb.set(rings, "Vector", gco)
    # 细直纹
    fine = nb.new("ShaderNodeTexWave", Distortion=2.0, Detail=2.0, **{"Detail Scale": 3.0})
    nb.set(fine, "Scale", jitter(95.0, "fine_scale", nb.math("FRACT", nb.math("MULTIPLY", j0, 7.31))))
    fine.wave_type = "BANDS"
    fine.bands_direction = "X"
    nb.set(fine, "Vector", gco)
    # 棕眼：沿纹理方向拉长的小黑点
    pore_co = nb.combine(nb.math("MULTIPLY", gx, 520.0), nb.math("MULTIPLY", gy, 140.0), r2)
    pores = nb.new("ShaderNodeTexNoise", Scale=1.0, Detail=0.0)
    nb.set(pores, "Vector", pore_co)
    pore = nb.map_range(pores.outputs["Fac"], 0.62, 0.72, 0.0, 1.0)
    # 板内低频色斑
    blotch = nb.new("ShaderNodeTexNoise", Scale=3.0, Detail=2.0)
    nb.set(blotch, "Vector", gco)

    ring_v = nb.math("POWER", rings.outputs["Fac"], 1.7)
    g = nb.math("ADD", nb.math("MULTIPLY", ring_v, jitter(0.38, "ring_contrast", nb.math("FRACT", nb.math("MULTIPLY", j1, 5.17)))),
                nb.math("MULTIPLY", fine.outputs["Fac"], 0.2))
    g = nb.math("ADD", g, nb.math("MULTIPLY", blotch.outputs["Fac"], 0.45))
    g = nb.math("SUBTRACT", g, 0.12, clamp=True)
    wood = nb.ramp(g, [
        (0.00, srgb("#9c6a40")),
        (0.35, srgb("#79492a")),
        (0.65, srgb("#53301b")),
        (1.00, srgb("#2e180d")),
    ])
    # 每块板：明度和冷暖都略有不同
    tone = nb.map_range(r0, 0.0, 1.0, 0.68, 1.22)
    wood = nb.mix(1.0, wood, nb.combine(tone, tone, tone), blend="MULTIPLY")
    warmer = nb.map_range(r1, 0.0, 1.0, 0.0, 0.35)
    wood = nb.mix(warmer, wood, srgb("#7a3a1c"), blend="OVERLAY")
    wood = nb.mix(nb.math("MULTIPLY", pore, 0.55), wood, srgb("#1a0c05"))
    wood = nb.mix(nb.math("MULTIPLY", seam, 0.92), wood, srgb("#0b0604"))

    # 粗糙度：清漆半哑光，带大块擦拭痕迹；板缝里粗糙
    smudge = nb.new("ShaderNodeTexNoise", Scale=0.9, Detail=5.0, Roughness=0.62)
    nb.set(smudge, "Vector", pos)
    rough = nb.map_range(smudge.outputs["Fac"], 0.3, 0.75, 0.28, 0.5)
    rough = nb.math("ADD", rough, nb.math("MULTIPLY", pore, 0.15))
    rough = nb.math("MAXIMUM", rough, nb.math("MULTIPLY", seam, 0.9))
    coat_r = nb.map_range(smudge.outputs["Fac"], 0.3, 0.75, 0.1, 0.24)

    # 凹凸：板缝凹进、板边小倒角、每块板轻微不平（反光会一块块断开）
    tilt = nb.math("MULTIPLY", nb.math("SUBTRACT", r2, 0.5), nb.math("SUBTRACT", u, 0.5))
    cup = nb.math("MULTIPLY", nb.math("POWER", nb.math("SUBTRACT", u, 0.5), 2.0), -0.6)
    h = nb.math("ADD", nb.math("MULTIPLY", bevel, -0.6), nb.math("MULTIPLY", tilt, 0.35))
    h = nb.math("ADD", h, cup)
    h = nb.math("ADD", h, nb.math("MULTIPLY", fine.outputs["Fac"], 0.04))
    bump = nb.new("ShaderNodeBump", Strength=1.0, Distance=0.0015)
    nb.set(bump, "Height", h)
    bump2 = nb.new("ShaderNodeBump", Strength=0.08, Distance=0.002)
    nb.set(bump2, "Height", pore)
    nb.set(bump2, "Normal", bump.outputs["Normal"])

    bsdf = nb.new("ShaderNodeBsdfPrincipled")
    nb.set(bsdf, "Base Color", wood)
    nb.set(bsdf, "Roughness", rough)
    nb.set(bsdf, "Specular IOR Level", 0.5)
    nb.set(bsdf, "Coat Weight", nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, seam), 0.3))
    nb.set(bsdf, "Coat Roughness", coat_r)
    nb.set(bsdf, "Coat Tint", srgb("#fff2e0"))
    nb.set(bsdf, "Normal", bump2.outputs["Normal"])
    nb.set(bsdf, "Coat Normal", bump.outputs["Normal"])
    out = nb.new("ShaderNodeOutputMaterial")
    nb.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat



def mat_gold(name="旧金"):
    mat = new_material(name)
    nb = Nodes(mat.node_tree)
    noise = nb.new("ShaderNodeTexNoise", Scale=40.0, Detail=4.0)
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Metallic=1.0)
    nb.set(bsdf, "Base Color", srgb("#c9973f"))
    nb.set(bsdf, "Roughness", nb.map_range(noise.outputs["Fac"], 0.3, 0.7, 0.22, 0.42))
    out = nb.new("ShaderNodeOutputMaterial")
    nb.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def mat_fringe(name="金色流苏"):
    """流苏：用竖条纹做透明遮罩，省掉成百上千根细线。"""
    mat = new_material(name)
    mat.surface_render_method = "DITHERED"
    nb = Nodes(mat.node_tree)
    uv = nb.new("ShaderNodeTexCoord").outputs["UV"]
    sep = nb.new("ShaderNodeSeparateXYZ")
    nb.set(sep, 0, uv)
    # 每厘米约 2 根，长短稍有参差
    strands = nb.new("ShaderNodeTexWave", Scale=1.0, Distortion=0.4, Detail=1.0)
    strands.wave_type = "BANDS"
    strands.bands_direction = "X"
    nb.set(strands, "Vector", nb.combine(nb.math("MULTIPLY", sep.outputs[0], 70.0), sep.outputs[1]))
    jitter = nb.new("ShaderNodeTexNoise", Scale=1.0, Detail=0.0)
    nb.set(jitter, "Vector", nb.combine(nb.math("MULTIPLY", sep.outputs[0], 300.0), 0.0))
    alpha = nb.math("GREATER_THAN", strands.outputs["Fac"], 0.42)
    tip = nb.math("LESS_THAN", sep.outputs[1], nb.math("ADD", 0.8, nb.math("MULTIPLY", jitter.outputs["Fac"], 0.3)))
    alpha = nb.math("MULTIPLY", alpha, tip)
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Metallic=1.0, Roughness=0.35)
    nb.set(bsdf, "Base Color", srgb("#b8862f"))
    nb.set(bsdf, "Alpha", alpha)
    out = nb.new("ShaderNodeOutputMaterial")
    nb.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def mat_apron(name="台口黑漆"):
    mat = new_material(name)
    nb = Nodes(mat.node_tree)
    bsdf = nb.new("ShaderNodeBsdfPrincipled", Roughness=0.45)
    nb.set(bsdf, "Base Color", srgb("#120b08"))
    out = nb.new("ShaderNodeOutputMaterial")
    nb.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


# ----------------------------------------------------------------------------
# 幕布几何
# ----------------------------------------------------------------------------

def _smooth_noise(rng, x0, x1, step):
    """一维平滑随机函数：每隔 step 一个随机值，余弦插值。"""
    kx = np.arange(x0 - 2 * step, x1 + 3 * step, step)
    kv = rng.normal(0.0, 1.0, len(kx))
    return lambda x: np.interp(x, kx, kv)


def fold_field(xs, t, rng, mean_w=0.2, depth=0.1, bunch_every=2.4, drift=0.45,
               sigma=0.38, wobble=0.9, bunch_amp=(0.7, 1.9)):
    """幕布的褶，一个一个“走”出来，而不是一条正弦波。

    * 褶的宽度按对数正态随机，再除以一个“疏密场”：大部分地方松散，隔一段有一处堆叠区，
      那里的褶又窄又挤、还更深；
    * 每个褶脊 / 褶谷有自己的深度，并且可以沿高度变深变浅（wobble）、左右漂移（drift）；
      大幕用很小的 wobble / drift，保持厚丝绒笔直的垂坠；
    * 截面前圆后尖（朝观众的褶脊宽而圆，褶谷窄），每段还带随机歪斜；
    * 顶部挂在杆上，褶子渐渐被拉整齐。
    xs: (nc,) 横向坐标；t: (nr,) 0=底 1=顶。返回 (前后偏移 (nr,nc), 褶谷程度 0..1 (nr,nc))。
    """
    x0, x1 = xs[0] - 0.8, xs[-1] + 0.8
    base = _smooth_noise(rng, x0, x1, 0.5)
    n_b = max(1, int((x1 - x0) / bunch_every))
    bc = rng.uniform(x0, x1, n_b)
    ba = rng.uniform(*bunch_amp, n_b)
    bs = rng.uniform(0.1, 0.32, n_b)

    def dens(x):
        d = np.exp(0.38 * base(x))
        d += (ba[None, :] * np.exp(-((np.atleast_1d(x)[:, None] - bc[None, :]) / bs[None, :]) ** 2)).sum(1)
        return d

    e = [x0]
    while e[-1] < x1:
        w = mean_w * math.exp(rng.normal(0.0, sigma)) / float(dens(e[-1])[0])
        e.append(e[-1] + min(max(w, 0.04), 0.7))
    e = np.array(e)
    n = len(e)
    gaps = np.diff(e)
    wavg = np.r_[gaps[0], (gaps[:-1] + gaps[1:]) / 2, gaps[-1]]
    sign = np.where((np.arange(n) + rng.integers(2)) % 2 == 0, -1.0, 1.0)   # -1 朝观众的褶脊，+1 褶谷
    dmag = depth * (wavg / mean_w) ** 0.75 * rng.uniform(0.45, 1.35, n) * dens(e) ** 0.45
    dmag = np.minimum(dmag, wavg * 1.05)                  # 太窄的褶不能太深，否则网格会翻折
    drift_i = rng.normal(0.0, drift, n) * wavg
    f_i = rng.uniform(0.25, 1.5, n)
    ph_i = rng.uniform(0.0, 2 * math.pi, n)
    skew = rng.uniform(-0.45, 0.45, n)

    nr, nc = len(t), len(xs)
    Y = np.empty((nr, nc))
    V = np.empty((nr, nc))
    for r, tr in enumerate(t):
        pos = e + drift_i * (1.0 - tr) ** 1.3
        pos = pos[0] + np.r_[0.0, np.cumsum(np.maximum(np.diff(pos), 0.035))]
        b = 0.5 * np.clip((tr - 0.8) / 0.2, 0, 1) ** 2
        pos = pos * (1 - b) + np.linspace(pos[0], pos[-1], n) * b
        env = np.clip(1.0 - wobble * (0.5 + 0.5 * np.sin(2 * math.pi * f_i * tr + ph_i)), 0.1, 1.0)
        amp = dmag * env * (1.0 + 0.4 * (1.0 - tr) ** 2) * (1.0 - 0.45 * b)
        ey = sign * amp
        i = np.clip(np.searchsorted(pos, xs) - 1, 0, n - 2)
        s = np.clip((xs - pos[i]) / (pos[i + 1] - pos[i]), 0.0, 1.0)
        s = s + skew[i] * s * (1.0 - s)
        y = ey[i] + (ey[i + 1] - ey[i]) * (1.0 - np.cos(math.pi * s)) / 2
        D = np.abs(ey[i]) * (1 - s) + np.abs(ey[i + 1]) * s + 1e-6
        y = y - 0.22 * (D - y * y / D)                   # 前圆后尖
        Y[r] = y
        V[r] = np.clip(0.5 + 0.5 * y / D, 0.0, 1.0)
    return Y, V


def add_creases(Y, xs, s, rng, n, x_range, s_max, amp=(0.004, 0.016), s_min=0.0):
    """细小折痕：几百条短的、大多接近竖直的细棱 / 细沟，越靠近底边越多、越斜。"""
    for _ in range(n):
        cx = rng.uniform(*x_range)
        cs = s_min + (s_max - s_min) * rng.uniform() ** 1.7
        theta = rng.normal(0.0, 0.22 if cs > 0.7 else 0.65)
        L = rng.uniform(0.08, 0.7)
        sig = rng.uniform(0.006, 0.024)
        a = rng.uniform(*amp) * rng.choice((-1.0, 1.0))
        hx = abs(math.sin(theta)) * L / 2 + 3 * sig
        hs = abs(math.cos(theta)) * L / 2 + 3 * sig
        ix = slice(*np.searchsorted(xs, (cx - hx, cx + hx)))
        js = slice(*np.searchsorted(s, (cs - hs, cs + hs)))
        dx = xs[ix][None, :] - cx
        ds = s[js][:, None] - cs
        along = dx * math.sin(theta) + ds * math.cos(theta)
        perp = dx * math.cos(theta) - ds * math.sin(theta)
        win = 0.5 * (1.0 + np.cos(math.pi * np.clip(2 * along / L, -1, 1)))
        Y[js, ix] += a * np.exp(-(perp / sig) ** 2) * win
    return Y


def hanging_curtain(name, x0, x1, y0, height, rng, mat, coll, mean_w=0.2, depth=0.1,
                    fine=(-2.7, 2.7), dx_fine=0.008, dx=0.02, ds_fine=0.015, s_fine=3.4, ds=0.05, n_creases=0,
                    crease_amp=(0.004, 0.016), **fold_kw):
    """一整幅垂到地面的幕：顶部挂在杆上，底部多出来的布“堆”在台面上。

    底边的截面（从上往下）：竖直垂下 → 向前弯出一个鼓包 → 在最前面绕一个小圆弧折回 →
    剩下的布平铺在鼓包下面往后藏。所以从台前看到的是圆润的折边（向前约 7–12 cm，处处不同），
    看不到布的末端。相机看得到的区域（fine × s_fine 以下）网格加密，细折痕只放在加密区里。
    """
    lo, hi = max(x0, fine[0]), min(x1, fine[1])
    if lo < hi:
        xs = np.unique(np.concatenate([np.arange(x0, lo, dx), np.arange(lo, hi, dx_fine), np.arange(hi, x1 + 1e-6, dx)]))
    else:
        xs = np.arange(x0, x1 + 1e-6, dx)
    nc = len(xs)

    # 落地堆布：鼓包向前伸 Ry、高 Rz，折边圆弧半径 r，藏在下面的布长 tail。
    # 先按固定顺序取随机数（保持褶子布局不变），再算褶子，最后让堆布跟着褶子走
    h = HEM_POOL
    n1, n2 = _smooth_noise(rng, x0, x1, 0.3), _smooth_noise(rng, x0, x1, 0.9)
    nz = _smooth_noise(rng, x0, x1, 0.45)
    nr = _smooth_noise(rng, x0, x1, 0.25)
    u_tail = rng.uniform(0.6, 1.0)
    n_pile = 6 + 18 + 16
    zn = np.concatenate([np.arange(0.2, min(s_fine, height), ds_fine), np.arange(min(s_fine, height), height + 1e-6, ds)])
    t = np.clip(np.concatenate([np.zeros(n_pile), (zn - 0.2) / (height - 0.2)]), 0.0, 1.0)
    off, valley = fold_field(xs, t, rng, mean_w, depth, **fold_kw)

    # 褶脊（valley≈0，朝观众）的布“流”得更远、弯得更高，褶谷几乎不往前出 → 前沿是跟着褶子走的花边
    ridge = 1.0 - valley[0]
    big = 0.5 * n1(xs) + 0.5 * n2(xs)
    k = h["ridge"]
    Ry = np.clip((h["forward"] + h["forward_var"] * big) * (1.0 - k / 2 + k * ridge), 0.02, h["forward"] * 2.2)
    r = np.clip(h["nose"] * (1.0 + 0.3 * nr(xs)) * (0.7 + 0.5 * ridge), h["nose"] * 0.5, h["nose"] * 1.6)
    Rz = h["height"] * (0.7 + 0.6 * ridge) + h["height_var"] * nz(xs) + 0.25 * (Ry - h["forward"])
    Rz = np.clip(Rz, 2 * r + 0.02, h["height"] * 2.0)
    tail = Ry * u_tail
    yn = -Ry                                             # 折边圆弧的圆心（相对 y0；台前是 -Y）

    rows_y, rows_z = [], []
    for a in np.linspace(1.0, 0.0, 6, endpoint=False):  # 末端：平铺在鼓包下面
        rows_y.append(yn + tail * a)
        rows_z.append(np.full(nc, 0.002))
    for k in np.linspace(0.0, 1.0, 18, endpoint=False):  # 折边：小圆弧从地面绕到上层
        b = math.radians(270.0 - 180.0 * k)
        rows_y.append(yn + r * math.cos(b))
        rows_z.append(0.002 + r + r * math.sin(b))
    for al in np.linspace(90.0, 0.0, 16, endpoint=False):  # 鼓包：从折边上沿圆滑地弯回竖直
        al = math.radians(al)
        rows_y.append(-Ry + Ry * math.cos(al))
        rows_z.append(0.002 + Rz - (Rz - 2 * r) * math.sin(al))
    assert len(rows_y) == n_pile
    for z in zn:                                         # 竖直部分
        rows_y.append(np.zeros(nc))
        rows_z.append(Rz + 0.002 + (z - 0.2) * (height - Rz) / (height - 0.2))
    BY = np.array([np.broadcast_to(v, (nc,)) for v in rows_y])
    BZ = np.array([np.broadcast_to(v, (nc,)) for v in rows_z])

    # 每列的弧长 → UV；各列平均 → 行参数 s（给褶子和折痕用）
    seglen = np.hypot(np.diff(BY, axis=0), np.diff(BZ, axis=0))
    S_col = np.vstack([np.zeros((1, nc)), np.cumsum(seglen, axis=0)])
    s = S_col.mean(axis=1)
    s_floor = s[n_pile]

    # 折痕沿截面法线方向（竖直部分就是前后，鼓包上是斜的）
    ty, tz = np.gradient(BY, axis=0), np.gradient(BZ, axis=0)
    tn = np.hypot(ty, tz) + 1e-9
    ny, nz = tz / tn, -ty / tn
    C = np.zeros_like(BY)
    if n_creases:
        C = add_creases(C, xs, s, rng, n_creases, (lo, hi), s_fine, amp=crease_amp)
        # 少量粗一点的长折痕铺满整幅
        C = add_creases(C, xs, s, rng, n_creases // 6, (x0, x1), height, amp=(crease_amp[0] * 1.5, crease_amp[1] * 1.2))
        # 鼓包上：短而乱的皱
        C = add_creases(C, xs, s, rng, 450, (lo, hi), s_floor + 0.05, amp=(0.003, 0.009),
                        s_min=s[6])
    # 鼓包整体再揉一揉：低频起伏，局部鼓得高一些
    kx, ks = rng.uniform(6, 30, 7), rng.uniform(8, 35, 7)
    ph = rng.uniform(0, 2 * math.pi, 7)
    lump = sum(np.sin(kx[k] * xs[None, :] + ks[k] * s[:, None] + ph[k]) for k in range(7)) / 3.0
    on_pile = np.zeros_like(BY)
    on_pile[6:n_pile] = np.sin(np.linspace(0, math.pi, n_pile - 6))[:, None]
    C = C + on_pile * HEM_POOL["lumps"] * lump

    Y = y0 + off + BY + ny * C
    Z = BZ + nz * C
    X = np.broadcast_to(xs[None, :], Z.shape)
    me = grid_mesh(name, X, Y, Z, X - x0, S_col)
    attr = me.attributes.new("fold", "FLOAT", "POINT")   # 褶谷程度，材质里用来压暗褶子深处
    attr.data.foreach_set("value", valley.ravel().astype(np.float32))
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    return link(ob, coll)


def valance(name, rng, mat_v, mat_gold_, mat_fr, coll):
    """顶部檐幕：一排垂花（下沿是弧形），沿下沿一根金色绳边 + 流苏。"""
    x0, x1 = -PROSC_HALF - 0.6, PROSC_HALF + 0.6
    xs = np.arange(x0, x1 + 0.006, 0.012)
    n_swag = 5
    span = (x1 - x0) / n_swag
    k = ((xs - x0) / span) % 1.0
    sag = 0.42 * np.sin(np.pi * k) ** 1.3
    zb = VALANCE_BOTTOM - sag
    t = np.linspace(0, 1, 48)
    Z = zb[None, :] + (VALANCE_TOP - zb)[None, :] * t[:, None]
    off, valley = fold_field(xs, t, rng, mean_w=0.15, depth=0.06, bunch_every=1.8, drift=0.3)
    bulge = -0.12 * np.sin(np.pi * np.clip(t * 1.1, 0, 1))[:, None] * np.sin(np.pi * k)[None, :]
    X = np.broadcast_to(xs[None, :], Z.shape)
    Y = PROSC_Y - 0.25 + off + bulge
    me = grid_mesh(name, X, Y, Z, X - x0, Z)
    me.attributes.new("fold", "FLOAT", "POINT").data.foreach_set("value", valley.ravel().astype(np.float32))
    me.materials.append(mat_v)
    link(bpy.data.objects.new(name, me), coll)

    # 下沿绳边（曲线 + 倒角）
    yb = Y[0]
    cu = bpy.data.curves.new(name + "_金绳", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.022
    cu.bevel_resolution = 4
    sp = cu.splines.new("POLY")
    pts = list(range(0, len(xs), 3))
    sp.points.add(len(pts) - 1)
    for p, i in zip(sp.points, pts):
        p.co = (xs[i], yb[i] - 0.025, zb[i] + 0.01, 1.0)
    cu.materials.append(mat_gold_)
    link(bpy.data.objects.new(name + "_金绳", cu), coll)

    # 流苏
    fl = 0.17
    tt = np.array([0.0, 1.0])
    FX = np.broadcast_to(xs[None, :], (2, len(xs)))
    FZ = zb[None, :] - fl * tt[:, None]
    FY = np.broadcast_to(yb[None, :] - 0.03, (2, len(xs)))
    FU = np.cumsum(np.r_[0, np.hypot(np.diff(xs), np.diff(zb))])   # 沿下沿的弧长（米）
    me = grid_mesh(name + "_流苏", FX, FY, FZ, np.broadcast_to(FU[None, :], FX.shape), np.broadcast_to(tt[:, None], FX.shape))
    me.materials.append(mat_fr)
    link(bpy.data.objects.new(name + "_流苏", me), coll)


# ----------------------------------------------------------------------------
# 地板与台口
# ----------------------------------------------------------------------------

def build_floor(coll, mat_floor, mat_black, mat_gold_):
    xs = np.array([-9.0, 9.0])
    ys = np.array([STAGE_FRONT_Y, CURTAIN_Y + 1.0])
    X, Y = np.meshgrid(xs, ys)
    me = grid_mesh("舞台地板", X, Y, np.zeros_like(X), X, Y)
    me.materials.append(mat_floor)
    link(bpy.data.objects.new("舞台地板", me), coll)

    # 台口前沿：圆鼻条 + 黑色台唇
    cu = bpy.data.curves.new("台沿圆鼻", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.03
    cu.bevel_resolution = 6
    sp = cu.splines.new("POLY")
    sp.points.add(1)
    sp.points[0].co = (-9.0, STAGE_FRONT_Y, -0.03, 1.0)
    sp.points[1].co = (9.0, STAGE_FRONT_Y, -0.03, 1.0)
    cu.materials.append(mat_floor)
    link(bpy.data.objects.new("台沿圆鼻", cu), coll)

    X, Z = np.meshgrid(np.array([-9.0, 9.0]), np.array([-1.1, -0.03]))
    me = grid_mesh("台唇", X, np.full_like(X, STAGE_FRONT_Y - 0.03), Z, X, Z)
    me.materials.append(mat_black)
    link(bpy.data.objects.new("台唇", me), coll)

    # 台唇上一道细金线
    cu = bpy.data.curves.new("台唇金线", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.008
    sp = cu.splines.new("POLY")
    sp.points.add(1)
    sp.points[0].co = (-9.0, STAGE_FRONT_Y - 0.035, -0.16, 1.0)
    sp.points[1].co = (9.0, STAGE_FRONT_Y - 0.035, -0.16, 1.0)
    cu.materials.append(mat_gold_)
    link(bpy.data.objects.new("台唇金线", cu), coll)


# ----------------------------------------------------------------------------
# 灯光
# ----------------------------------------------------------------------------

def spot(name, coll, loc, target, energy, size_deg, blend, color, soft=0.2, recv=None):
    ld = bpy.data.lights.new(name, "SPOT")
    ld.energy = energy
    ld.spot_size = math.radians(size_deg)
    ld.spot_blend = blend
    ld.color = color
    ld.shadow_soft_size = soft
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    if recv:
        ob.light_linking.receiver_collection = recv
    return link(ob, coll)


def build_lights(coll, recv_stage, recv_frame, recv_back):
    warm = (1.0, 0.78, 0.55)
    # 主追光：方向和人物的 Cel Key 一致，所以地上的影子和她身上的明暗是同一个方向
    spot("主追光", coll, (-3.6, -4.8, 7.2), (0.0, 0.0, 0.0), 3200, 36, 1.0, (1.0, 0.88, 0.74), 0.12, recv_stage)
    # 背幕光池：人物身后幕布上一团暖光，四周自然暗下去，把人和背景拉开
    # 只照大幕和地板（灯光链接到“背幕_受光”）：灯离檐幕很近，不排除的话檐幕上会被打出一块热斑
    spot("背幕光池", coll, (0.4, -3.0, 6.5), (-0.1, CURTAIN_Y, 1.15), 4200, 22, 1.0, warm, 0.4, recv_back)
    # 侧掠光：从台侧贴着幕布打过去，突出褶子的明暗节奏
    spot("左侧掠光", coll, (-5.6, CURTAIN_Y - 1.4, 4.2), (1.5, CURTAIN_Y + 0.1, 1.2), 550, 30, 1.0, (1.0, 0.7, 0.48), 0.3, recv_stage)
    spot("右侧掠光", coll, (5.6, CURTAIN_Y - 1.6, 3.6), (-1.5, CURTAIN_Y + 0.1, 1.0), 260, 30, 1.0, (0.78, 0.74, 1.0), 0.3, recv_stage)
    # 顶光：只照幕布上半段和檐幕（相机里基本看不到，全景机位用）
    spot("顶排光", coll, (0.0, PROSC_Y + 0.6, 7.5), (0.0, CURTAIN_Y, 4.8), 300, 55, 1.0, warm, 1.0, recv_stage)
    # 台口泛光：只照檐幕和两侧边幕（灯光链接到“檐幕边幕_受光”），全景机位里能看出台口的轮廓
    spot("檐幕光", coll, (0.0, -8.5, 3.6), (0.0, PROSC_Y, 6.0), 1500, 85, 0.6, warm, 1.0, recv_frame)
    for side, nm in ((-1, "左"), (1, "右")):
        spot(nm + "边幕光", coll, (side * 2.5, -11.0, 5.0), (side * (PROSC_HALF + 0.9), PROSC_Y, 2.8),
             2600, 34, 0.9, warm, 1.0, recv_frame)


def build_haze(coll, density=0.012):
    """很淡的舞台烟雾，让侧光、顶光在空中显出光束。人物灯的 volume_factor 设为 0，不参与。"""
    me = bpy.data.meshes.new("舞台薄雾")
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(me)
    bm.free()
    mat = new_material("舞台薄雾")
    nb = Nodes(mat.node_tree)
    pos = nb.new("ShaderNodeNewGeometry").outputs["Position"]
    noise = nb.new("ShaderNodeTexNoise", Scale=0.35, Detail=3.0)
    nb.set(noise, "Vector", pos)
    sep = nb.new("ShaderNodeSeparateXYZ")
    nb.set(sep, 0, pos)
    # 越高越浓一点、带一点缓慢的团块感
    dens = nb.math("MULTIPLY", nb.map_range(noise.outputs["Fac"], 0.3, 0.7, 0.5, 1.4), nb.map_range(sep.outputs[2], 0.0, 6.0, 0.6, 1.3))
    vol = nb.new("ShaderNodeVolumePrincipled", Anisotropy=0.45)
    nb.set(vol, "Color", srgb("#ffe7d2"))
    nb.set(vol, "Density", nb.math("MULTIPLY", dens, density))
    out = nb.new("ShaderNodeOutputMaterial")
    nb.links.new(vol.outputs[0], out.inputs["Volume"])
    me.materials.append(mat)
    ob = bpy.data.objects.new("舞台薄雾", me)
    ob.location = (0.0, (STAGE_FRONT_Y + CURTAIN_Y) / 2 - 1.0, 3.75)
    ob.scale = (2 * PROSC_HALF + 2.0, CURTAIN_Y - STAGE_FRONT_Y + 2.0, 7.5)
    ob.visible_shadow = False
    return link(ob, coll)


# ----------------------------------------------------------------------------
# 世界与人物
# ----------------------------------------------------------------------------

def stage_world(scene):
    old = scene.world
    ambient = (0.0, 0.0, 0.0)
    if old and old.name != STAGE_WORLD:
        old.use_fake_user = True                  # 原世界留着，想换回去随时可以
        bg = next((n for n in old.node_tree.nodes if n.bl_idname == "ShaderNodeBackground"), None)
        if bg and not bg.inputs["Color"].is_linked:
            c, s = bg.inputs["Color"].default_value, bg.inputs["Strength"].default_value
            ambient = (c[0] * s, c[1] * s, c[2] * s)
    elif old:
        ambient = tuple(old.get("stage_ambient_comp", (0.0, 0.0, 0.0)))
    w = bpy.data.worlds.get(STAGE_WORLD)
    if w:
        bpy.data.worlds.remove(w)
    w = bpy.data.worlds.new(STAGE_WORLD)
    if not w.node_tree:
        w.use_nodes = True
    bg = w.node_tree.nodes.get("Background") or w.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = srgb("#140a08")
    bg.inputs["Strength"].default_value = 0.35
    w["stage_ambient_comp"] = ambient
    scene.world = w
    return ambient


def compensate_character_ambient(ambient):
    """在人物材质的 Shader 转 RGB 后面加上原世界的环境光，让她的三阶色和原来一致。"""
    n_done = 0
    for mat in bpy.data.materials:
        nt = mat.node_tree
        if not nt:
            continue
        s2r = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeShaderToRGB"), None)
        if not s2r:
            continue
        add = nt.nodes.get(STAGE_AMBIENT_NODE)
        if add is None:
            targets = [l.to_socket for l in s2r.outputs["Color"].links]
            if not targets:
                continue
            add = nt.nodes.new("ShaderNodeMix")
            add.name = add.label = STAGE_AMBIENT_NODE
            add.data_type = "RGBA"
            add.blend_type = "ADD"
            add.inputs[0].default_value = 1.0
            add.location = s2r.location + Vector((180, -160))
            nt.links.new(s2r.outputs["Color"], add.inputs[6])
            for sock in targets:
                nt.links.new(add.outputs[2], sock)
        add.inputs[7].default_value = (*ambient, 1.0)
        n_done += 1
    return n_done


def receiver_collection(name, objects):
    coll = bpy.data.collections.get(name)
    if coll:
        bpy.data.collections.remove(coll)
    coll = bpy.data.collections.new(name)
    for ob in objects:
        coll.objects.link(ob)
    return coll


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def clear_previous(scene):
    coll = bpy.data.collections.get(STAGE_COLL)
    if coll:
        for ob in list(coll.all_objects):
            bpy.data.objects.remove(ob)
        for c in list(coll.children_recursive):
            bpy.data.collections.remove(c)
        bpy.data.collections.remove(coll)
    for block in (bpy.data.meshes, bpy.data.curves, bpy.data.lights, bpy.data.cameras):
        for d in list(block):
            if d.users == 0:
                block.remove(d)


def build(scene=None):
    scene = scene or bpy.context.scene
    clear_previous(scene)
    rng = np.random.default_rng(20261001)

    root = new_collection(STAGE_COLL, scene.collection)
    c_curtain = new_collection("幕布", root)
    c_floor = new_collection("地板", root)
    c_light = new_collection("舞台灯光", root)

    velvet = mat_velvet()
    gold = mat_gold()
    fringe = mat_fringe()
    floor = mat_wood_floor()
    black = mat_apron()

    # 厚丝绒的垂坠：褶窄而深、从上到下笔直；不规律只体现在横向的宽窄、深浅和一两处温和的堆叠
    hanging_curtain("大幕", -CURTAIN_W / 2, CURTAIN_W / 2, CURTAIN_Y, CURTAIN_H, rng, velvet, c_curtain,
                    **CURTAIN_DRAPE)
    for side in (-1, 1):
        x_in = side * PROSC_HALF
        x_out = side * (PROSC_HALF + 1.9)
        hanging_curtain("边幕_" + ("左" if side < 0 else "右"), min(x_in, x_out), max(x_in, x_out), PROSC_Y,
                        VALANCE_TOP, rng, velvet, c_curtain, mean_w=0.13, depth=0.085, drift=0.06, wobble=0.2,
                        sigma=0.3, bunch_amp=(0.3, 0.9), bunch_every=3.0,
                        fine=(0, 0), dx=0.012, ds=0.04)
    valance("檐幕", rng, velvet, gold, fringe, c_curtain)
    build_floor(c_floor, floor, black, gold)

    stage_objs = [ob for ob in root.all_objects if ob.type in ("MESH", "CURVE")]
    c_haze = new_collection("薄雾", root)
    haze = build_haze(c_haze)
    haze.hide_render = True                       # 默认关闭：想要光束感就在大纲里把它的渲染打开
    haze.hide_set(True)
    recv_stage = receiver_collection(RECV_STAGE, stage_objs)
    frame_objs = [ob for ob in c_curtain.objects if ob.name.startswith(("边幕", "檐幕"))]
    recv_frame = receiver_collection(RECV_FRAME, frame_objs)
    recv_back = receiver_collection(RECV_BACK, [ob for ob in stage_objs if ob not in frame_objs])
    build_lights(c_light, recv_stage, recv_frame, recv_back)

    # 原有的人物灯只照人物
    chars = [bpy.data.objects[n] for n in CHARACTER_OBJECTS if n in bpy.data.objects]
    recv_char = receiver_collection(RECV_CHAR, chars)
    for n in CHARACTER_LIGHTS:
        ob = bpy.data.objects.get(n)
        if ob:
            ob.light_linking.receiver_collection = recv_char
            ob.data.volume_factor = 0.0

    ground = bpy.data.objects.get(OLD_GROUND)
    if ground:
        ground.hide_render = True
        ground.hide_set(True)

    ambient = stage_world(scene)
    n_mat = compensate_character_ambient(ambient)

    e = scene.eevee
    e.use_raytracing = True                       # 地板上幕布和人物的倒影
    e.ray_tracing_method = "SCREEN"
    e.ray_tracing_options.resolution_scale = "1"
    e.ray_tracing_options.trace_max_roughness = 0.6
    e.use_fast_gi = True                          # 褶子深处的遮蔽
    e.fast_gi_method = "AMBIENT_OCCLUSION_ONLY"
    e.fast_gi_distance = 0.6
    e.use_shadows = True
    e.shadow_ray_count = 3
    e.shadow_step_count = 12
    e.volumetric_tile_size = "4"
    e.volumetric_samples = 96
    e.use_volumetric_shadows = True
    e.volumetric_end = 40.0

    # 全景机位（不改动原来的动画相机）
    cd = bpy.data.cameras.new(OVERVIEW_CAM)
    cd.lens = 24
    cam = link(bpy.data.objects.new(OVERVIEW_CAM, cd), c_light)
    cam.location = (0.0, -12.5, 2.4)
    cam.rotation_euler = (Vector((0, 0, 2.6)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    print(f"[stage] 舞台已生成；人物材质补偿 {n_mat} 个，补偿值 {tuple(round(a, 4) for a in ambient)}")
    return root


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"input": None, "save": None, "render": None, "frames": None, "percent": 100,
            "samples": None, "overview": False, "build": True}
    it = iter(argv)
    for a in it:
        if a == "--overview":
            opts["overview"] = True
        elif a == "--no-build":
            opts["build"] = False
        elif a.startswith("--"):
            opts[a[2:]] = next(it)
    return opts


def main():
    o = parse_args()
    if o["input"]:
        bpy.ops.wm.open_mainfile(filepath=o["input"])
    scene = bpy.context.scene
    if o["build"]:
        build(scene)
    if o["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=o["save"], compress=True)
    if o["render"]:
        r = scene.render
        r.resolution_percentage = int(o["percent"])
        if o["samples"]:
            scene.eevee.taa_render_samples = int(o["samples"])
        if o["overview"]:
            scene.camera = bpy.data.objects[OVERVIEW_CAM]
            r.resolution_x, r.resolution_y = 1920, 1080
        frames = [int(f) for f in o["frames"].split(",")] if o["frames"] else [scene.frame_current]
        for f in frames:
            scene.frame_set(f)
            r.filepath = o["render"].replace("{frame}", f"{f:04d}")
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
