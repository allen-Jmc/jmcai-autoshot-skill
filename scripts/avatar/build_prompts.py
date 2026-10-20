"""组装并校验每段的提示词。

  python build_prompts.py --workdir <项目目录>

输入：<workdir>/avatar/prompt_src.json（由 Claude 按 references/prompt-rules.md 写）：
  {"image_type": "图片类型，必填。A/B/C/D/E 之一，可加 +持物左/+持物右/+旁物/+非中性，如 "B+持物右+非中性"",
   "scene": "画面描述（看图写：人物的发型、衣着、佩戴物；位置和姿态；背景左右两侧有什么）",
   "hands": "双手的原姿态（分别写「画面左侧的手」「画面右侧的手」的位置、掌心朝向、手指状态）",
   "posture": "身体的原姿态（可选）",
   "lines": ["第1段的台词原文", "第2段的台词原文"],
   "motions": ["第1段的动作编排（若干行，每行一个节拍）", "第2段的……"]}
同目录的 plan.json 提供每段时长和识别文字。
输出：<workdir>/avatar/prompts/segment_NN.txt = 固定镜头行 + 画面描述 + 双手的原姿态 + 本段台词 + 动作编排。
镜头、视线、画面稳定等通用要求由云端应用自动加入，这里不写、也不需要写。

图片类型（image_type）决定这张图**能做**哪些动作，校验会按它放宽或收紧：
  * A 近景（胸口以上，手不在画面里或只露一点）：不要求 hands 写左右手，也不要求节拍写两只手；
  * B 半身无桌面 / D 站姿 / E 全身远景：提示要安排明显的身体动作（手没有看得见的落点）；
  * C 半身有桌面：手为主，有落点的动作最多；
  * +持物左 / +持物右：那只手不要求当主手，也不提示它「弱动作」；
  * +旁物：东西摆在桌面或身旁；
  * +非中性（图里嘴张开、手抬在半空或头偏着）：提示最后两拍要把头和上身回正写完整。

固定镜头首行（prompt_prefix）默认注入 `<固定镜头，无运镜>`，防模型把动作做成推拉变焦；
写成空字符串 "" 就不加。

校验（不通过就退出码 1，并说明哪一段哪一条）：
  * 每个节拍格式：`起–止「关键词」：动作描述`，时间是本段内的秒数，递增、不重叠、不超过本段时长；
  * 关键词必须出现在本段台词里；台词与识别文字相似度 ≥0.9；
  * 两只手的写法（每个节拍写两只手、副手有自己的动作、两只手轮流当主角、有起有落、不重复）只给提示，不拦截——
    云端模型不会逐字照做，没必要卡死；
  * **不写任何与嘴部有关的描述**：嘴、唇、口型、微笑/笑意、换气、表情、神态、露出、情绪形容词等一律禁止
    （画面描述、双手描述、动作编排都检查；台词原文不检查）；
  * 第一个节拍必须是保持原姿态、最后一个节拍必须回到原姿态附近，并写明手仍保持轻微自然晃动；
  * 动作描述里不能出现具体手指数、镜头运动、触碰脸或佩戴物、视线离开镜头；
  * hands 里必须分别描述"画面左侧的手"和"画面右侧的手"；每段不超过 2000 字。
"""
import argparse
import difflib
import json
import os
import re
import sys

BEAT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[–—\-~～]\s*(\d+(?:\.\d+)?)\s*(?:「([^」]*)」)?\s*[：:]?\s*(.+?)\s*$")
ORIGIN = re.compile(r"原位|原姿态|原图姿态|原有姿态|原来的姿态")
NATURAL_MOTION = re.compile(r"晃动|摆动")
FORBID = [
    (r"[一二三四五六七八九十两\d]\s*根\s*手指|[一二三四五六七八九十两\d]\s*指", "具体手指数（数字手势多次出错，改用手掌或食指示意）"),
    (r"推近|拉远|变焦|摇镜|运镜|跟拍|转身|站起|起身|走动|凑近|靠近镜头|特写|放大画面|缩小画面", "镜头运动或大幅位移（推近拉远一律留给后期做）"),
    (r"前倾|后仰|后靠|向后.{0,4}靠|往后.{0,4}靠|探身|探出|身体向前|上身向前|伸向镜头|向镜头", "身体或手朝镜头前后移动：模型会把它做成推近拉远，画面和背景跟着出错；身体只写耸肩、侧倾、左右轻晃"),
    (r"摸|触碰|扶正|摘下|整理|挠|揉|托腮", "触碰脸部或佩戴物"),
    (r"像|仿佛|好似|如同|似的", "比喻（“像拿着话筒”“像捧起东西”）：模型会把比喻里的东西画出来，直接写手怎么动"),
    (r"看向手|低头看|看着手|目光.{0,4}(移开|离开|下移|转向)|视线.{0,4}(移开|离开|下移|转向)|看向别处|看向画面外", "视线离开镜头"),
]
# 下面这些只给提示、不拦截：云端模型不会逐字照做，写法上的要求没必要卡死，提醒到了由写的人自己判断
HOVER = re.compile(r"停在原位|保持不动|一动不动|固定不动|悬停|纹丝|僵")
# 看得见位移的动作（主手、副手都该有一个）；只有晃动、摆动、一张一合这类词算“弱动作”
STRONG = re.compile(r"推|压|托|展开|摊|递|指|拍|敲|点|握|举|抬|落|放下|放到|放在|垂|搭|撑|按|叉|甩|挥|比|划|画|合|收|转|错动|振|伸")
REST = re.compile(r"落到|落在|落回|放下|放到|放在|垂到|垂下|搭在|撑在|叉在|收到")
BODY = re.compile(r"上身|肩|身体|重心|侧倾|挺")
# 与嘴部、表情有关的词一律不写（画面描述、双手描述、动作编排都检查；台词原文不检查）
MOUTH = re.compile(r"嘴|唇|口型|牙|笑|抿|咧|撇|开口|张口|闭合|换气|表情|神态|露出|亲切|惊喜|遗憾|无奈|皱眉|开心|高兴|愉快|满意|自信|明亮|热情")
LEFT, RIGHT = "画面左侧的手", "画面右侧的手"
MAX_CHARS = 2000

# 图片类型：景别决定哪一层能出动感（见 references/avatar/prompt-rules.md 第二节）
IMAGE_TYPES = {
    "A": "近景（胸口以上，手不在画面里或只露一点）",
    "B": "半身无桌面（腰或胸以上，双手在画面里，身前没有台面）",
    "C": "半身有桌面（坐在桌前，桌面在画面里）",
    "D": "站姿（站着，七分身或更大）",
    "E": "全身远景（人只占画面一小部分）",
}
EXTRA_TAGS = ["持物左", "持物右", "旁物", "非中性"]
# 手没有看得见的落点的类型：要靠身体动作补动感
BODY_NEEDED = ("B", "D", "E")
# 固定镜头首行：默认注入，防模型把动作做成推拉变焦；写成空字符串就不加
PREFIX_DEFAULT = "<固定镜头，无运镜>"


def parse_image_type(raw):
    """把 "B+持物右+非中性" 拆成 (字母, [附加条件])。返回 (None, []) 表示没填。"""
    s = (raw or "").strip()
    if not s:
        return None, []
    parts = [p.strip() for p in s.split("+") if p.strip()]
    letter = parts[0].upper() if parts else None
    return letter, parts[1:]


def check_image_type(raw, problems):
    """校验 image_type 必填且合法，返回 (字母, 附加条件集合)。"""
    letter, extras = parse_image_type(raw)
    if not raw or not str(raw).strip():
        problems.append(
            "image_type：必须填图片类型。看图判断景别，写 A/B/C/D/E 之一，"
            "可叠加 +持物左 / +持物右 / +旁物 / +非中性（例：B+持物右）。"
            "各类含义见 references/avatar/prompt-rules.md 第二节"
        )
        return None, set()
    if letter not in IMAGE_TYPES:
        problems.append(
            "image_type：类型字母「%s」不认识，只能是 A/B/C/D/E 之一（%s）"
            % (letter, "；".join("%s %s" % (k, v) for k, v in IMAGE_TYPES.items()))
        )
        return None, set()
    for t in extras:
        if t not in EXTRA_TAGS:
            problems.append(
                "image_type：附加条件「%s」不认识，只能是 %s" % (t, " / ".join(EXTRA_TAGS))
            )
    return letter, set(extras)



_CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def _cn_to_int(s):
    if not any(ch in _UNIT for ch in s):          # 没有十百千：逐位读（年份"一九五六"→1956）
        return int("".join(str(_CN[ch]) for ch in s))
    total, cur = 0, 0
    for ch in s:
        if ch in _CN:
            cur = _CN[ch]
        elif ch in _UNIT:
            total += (cur or 1) * _UNIT[ch]
            cur = 0
    return total + cur


def norm(s):
    """去标点空白，并把中文数字统一成阿拉伯数字（识别结果常把"三十"写成"30"）。"""
    s = re.sub(r"[零〇一二两三四五六七八九十百千]+", lambda m: str(_cn_to_int(m.group())), s)
    return re.sub(r"[\s，。、！？!?；;：:“”\"'（）()\-—…,.]", "", s).lower()


def richness_hints(tag, dur, beats, skip_hand=None):
    """手的动作够不够丰富：只给提示。看中间的节拍（首尾是回到原姿态，不算）

    skip_hand：拿东西的那只手（+持物左/+持物右），不提示它「弱动作」或「该当主角」。
    """
    mid = beats[1:-1]
    out = []
    if len(mid) < 2:
        return out
    weak = {LEFT: 0, RIGHT: 0}
    seen = {LEFT: 0, RIGHT: 0}
    lead = {LEFT: 0, RIGHT: 0}
    for _, _, desc, both in mid:
        if not both:
            continue
        i, j = desc.index(LEFT), desc.index(RIGHT)
        if abs(i - j) > len(LEFT) + 1:                      # 「左侧的手和右侧的手一起……」不算谁领头
            lead[LEFT if i < j else RIGHT] += 1
        for hand, a, b in ((LEFT, i, j), (RIGHT, j, i)):
            clause = desc[a + len(hand):b] if b > a else desc[a + len(hand):]
            if abs(i - j) <= len(LEFT) + 1:                 # 两只手连着写，共用后面那一句
                clause = desc[max(i, j) + len(hand):]
            seen[hand] += 1
            if not STRONG.search(clause):
                weak[hand] += 1
    for hand in (LEFT, RIGHT):
        if hand == skip_hand:                                   # 拿东西的那只手只做小幅动作，不提示
            continue
        if seen[hand] >= 3 and weak[hand] / seen[hand] > 0.6:
            out.append("%s：%s在多数节拍里只有晃动、摆动这类弱动作，成片里它多半不动。给它一件自己的事：落到某处、托着、叉腰、收到身前、在落点上轻点" % (tag, hand))
    n = lead[LEFT] + lead[RIGHT]
    if n >= 4 and min(lead.values()) / n < 0.25:
        idle = LEFT if lead[LEFT] < lead[RIGHT] else RIGHT
        if idle != skip_hand:
            out.append("%s：主要手势几乎都是另一只手在做，让%s也当几次主角" % (tag, idle))
    if dur >= 8 and not any(REST.search(d) for _, _, d, _ in mid):
        out.append("%s：这一段里手一直举着，没有落下再抬起。内容不是特别密的话，在停顿或平静的句子里让手落到某处歇一拍" % tag)
    if dur >= 6 and not any(BODY.search(d) for _, _, d, _ in mid):
        out.append("%s：这一段只写了手，没写身体。每两三个节拍带一次肩膀或上身的小动作（微微前倾、耸肩、略转向一侧、重心慢慢移），人才不像钉在原地" % tag)
    for (_, _, d1, _), (_, _, d2, _) in zip(mid, mid[1:]):
        if d1 == d2:
            out.append("%s：相邻两个节拍是同一个手势，换一个：%s" % (tag, d1[:24]))
            break
    return out


def feedback_hints(qa_path, segs):
    """读上一轮成片的质检结果，指出哪几段的哪只手没动起来。

    这是手势的反馈回路：qa_video.py 已经把"每段左右手各有几成帧近似静止"量出来了，
    以前这个数字只打印一次就没了。现在把它变成下一次写 prompt_src.json 时的具体修改点——
    不是让人写得更细，是让数据指出哪一段写得不够。
    """
    if not qa_path or not os.path.isfile(qa_path):
        return []
    try:
        qa = json.load(open(qa_path, encoding="utf-8"))
    except (ValueError, OSError):
        return []
    by_seg = {int(r["i"]): r for r in qa.get("rows", [])}
    out = []
    for s in segs:
        r = by_seg.get(s["i"])
        if not r:
            continue
        for side, key in (("画面左侧的手", "left_static"), ("画面右侧的手", "right_static")):
            v = r.get(key)
            if v is None or v <= 0.60:
                continue
            out.append("段%d：上一轮成片里%s有 %.0f%% 的帧近似静止（阈值 60%%）。"
                       "这一段要给它安排看得见位移的动作，并让它当两次主手；"
                       "现在多半只写了「托着」「停在原位」「轻轻晃动」这类没有落点的说法。"
                       % (s["i"], side, v * 100))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--feedback", nargs="?", const="auto", default="auto",
                    help="读上一轮质检结果来提示手势改写；默认自动找 avatar/qa.json，"
                         "给路径就指定文件，写 none 关掉")
    a = ap.parse_args()
    od = os.path.join(a.workdir, "avatar")
    plan = json.load(open(os.path.join(od, "plan.json"), encoding="utf-8"))
    src = json.load(open(os.path.join(od, "prompt_src.json"), encoding="utf-8"))
    segs = plan["segments"]
    problems, hints = [], []
    scene, hands = src.get("scene", "").strip(), src.get("hands", "").strip()
    posture = src.get("posture", "").strip()
    # 固定镜头首行：默认注入，写成空字符串就不加（防模型把动作做成推拉变焦）
    prefix = src.get("prompt_prefix", PREFIX_DEFAULT)
    prefix = prefix.strip() if isinstance(prefix, str) else PREFIX_DEFAULT
    if "head" in src and not scene:
        sys.exit("prompt_src.json 是旧格式（head）。现在改为 scene（画面描述）和 hands（双手的原姿态）两项，见 references/prompt-rules.md")
    # 图片类型：决定这张图能做哪些动作，校验按它放宽或收紧
    it_letter, it_extra = check_image_type(src.get("image_type"), problems)
    no_hand = it_letter == "A"                                   # A 类手不在画面里
    # 拿东西的那只手不要求当主手，也不提示它「弱动作」
    skip_hand = LEFT if "持物左" in it_extra else RIGHT if "持物右" in it_extra else None
    if not scene:
        problems.append("scene：缺少画面描述（人物的发型、衣着、佩戴物，位置和姿态，背景左右两侧）")
    if no_hand:
        if not hands:
            problems.append("hands：A 类近景也要写一句，例如「双手不在画面里」")
    elif not (LEFT in hands and RIGHT in hands):
        problems.append("hands：要分别描述「%s」和「%s」在图中的位置和姿态" % (LEFT, RIGHT))

    for name, txt in (("scene", scene), ("hands", hands), ("posture", posture)):
        m_ = MOUTH.search(txt)
        if m_:
            problems.append("%s：出现与嘴部或表情有关的词「%s」，一律不写" % (name, m_.group()))
    if not (len(src.get("lines", [])) == len(src.get("motions", [])) == len(segs)):
        sys.exit("lines / motions 的数量（%d / %d）必须和分段数 %d 一致" % (len(src.get("lines", [])), len(src.get("motions", [])), len(segs)))
    outs = []
    for s, line, motion in zip(segs, src["lines"], src["motions"]):
        tag = "段%d" % s["i"]
        dur = s["duration"]
        sim = difflib.SequenceMatcher(None, norm(line), norm(s["text_asr"])).ratio()
        if sim < 0.9:
            problems.append("%s：台词与本段识别文字相似度 %.2f（<0.9），请核对台词是不是这一段的内容" % (tag, sim))
        beats, prev_end = [], 0.0
        for ln in [x for x in motion.splitlines() if x.strip()]:
            m = BEAT.match(ln)
            if not m:
                problems.append("%s：节拍格式不对（应为 `起–止「关键词」：描述`）：%s" % (tag, ln[:40]))
                continue
            t0, t1, kw, desc = float(m[1]), float(m[2]), m[3], m[4]
            if t1 <= t0:
                problems.append("%s：节拍结束早于开始：%s" % (tag, ln[:40]))
            if t0 < prev_end - 0.05:
                problems.append("%s：节拍时间重叠或不递增：%s" % (tag, ln[:40]))
            if t1 > dur + 0.3:
                problems.append("%s：节拍 %.2f–%.2f 超出本段时长 %.1f 秒" % (tag, t0, t1, dur))
            if kw and norm(kw) not in norm(line):
                problems.append("%s：关键词「%s」不在本段台词里" % (tag, kw))
            for pat, why in FORBID:
                if re.search(pat, desc):
                    problems.append("%s：节拍里出现%s：%s" % (tag, why, desc[:30]))
            mm = MOUTH.search(desc)
            if mm:
                problems.append("%s：节拍里出现与嘴部或表情有关的词「%s」，一律不写：%s" % (tag, mm.group(), desc[:30]))
            both_sides = LEFT in desc and RIGHT in desc
            if not no_hand and not ("双手" in desc or both_sides):
                hints.append("%s：这个节拍没有写两只手（写「双手」，或同时写「%s」和「%s」），没写到的那只手容易僵住：%s" % (tag, LEFT, RIGHT, desc[:30]))
            if HOVER.search(desc):
                hints.append("%s：写了让手停在半空的说法，容易僵。想让手休息，就写它落到哪（桌面、腿上、腰侧……）：%s" % (tag, desc[:30]))
            prev_end = max(prev_end, t1)
            beats.append((t0, t1, desc, both_sides))
        if not beats:
            problems.append("%s：没有任何节拍" % tag)
        else:
            if not ORIGIN.search(beats[0][2]):
                problems.append("%s：第一个节拍要写明保持原姿态" % tag)
            if not ORIGIN.search(beats[-1][2]):
                problems.append("%s：最后一个节拍要写明回到原姿态附近" % tag)
            for k in (0, -1):
                if not NATURAL_MOTION.search(beats[k][2]):
                    hints.append("%s：第一个和最后一个节拍最好写明双手仍在轻微晃动，免得冻在原位" % tag)
                    break
            if not no_hand:
                side_share = sum(1 for b in beats if b[3]) / len(beats)
                if side_share < 0.5:
                    hints.append("%s：只有 %.0f%% 的节拍分别写了左右两只手，多数节拍最好写清两只手各自做什么" % (tag, side_share * 100))
                hints.extend(richness_hints(tag, dur, beats, skip_hand=skip_hand))
            else:
                # A 类：手不在画面里，动感全靠头肩，每个节拍都该带一个头的动作
                headless = [b[2] for b in beats[1:-1] if not re.search(r"头|肩", b[2])]
                if headless:
                    hints.append("%s：A 类近景手不在画面里，动感全靠头肩——有 %d 个节拍没写头的动作，每个节拍都配一个头的动作（偏、侧倾、轻点），隔一拍配一次肩" % (tag, len(headless)))
            if it_letter in BODY_NEEDED and not any(BODY.search(b[2]) for b in beats):
                hints.append("%s：%s 这类图手没有看得见的落点，动感一半靠身体——安排一两处明显的身体动作（重心移到一侧、上身侧倾、肩膀一耸），其余用小动作" % (tag, it_letter))
            if "非中性" in it_extra and len(beats) >= 2 and not re.search(r"回正|端正|摆正|复原|回中", beats[-2][2]):
                hints.append("%s：这张图是非中性（嘴张开／手悬在半空／头偏着），段尾回正要写完整——倒数第二个节拍就让头和上身回到端正，最后一拍只写回到原姿态；否则每个接缝都会跳一下" % tag)
            if beats[-1][1] < dur - 1.0:
                problems.append("%s：动作只编排到 %.1f 秒，本段 %.1f 秒，后面一截没有安排" % (tag, beats[-1][1], dur))
            density = len(beats) / max(dur, 0.1)
            if density > 1.3:
                print("提示：%s 节拍偏密（%.1f 个/秒），手势可能挤在一起" % (tag, density))
        parts = ["画面描述：%s\n双手的原姿态：%s" % (scene, hands) + ("\n身体的原姿态：%s" % posture if posture else ""),
                 "本段台词（时长约%.1f秒）：%s" % (dur, line),
                 "动作编排（按台词节拍，时间为本段内的秒数）：\n" + motion.strip()]
        full = "\n\n".join(parts)
        if len(full) > MAX_CHARS:                           # 字数只算主体，脚本加的首行不算
            problems.append("%s：提示词 %d 字，超过 %d 字" % (tag, len(full), MAX_CHARS))
        if prefix:
            full = prefix + "\n\n" + full
        outs.append((s["i"], full))
    for h in hints:
        print("提示：" + h)
    # ---- 手势反馈：把上一轮成片量到的"哪只手没动"变成这一轮的具体修改点
    fb_path = None
    if a.feedback == "auto":
        cand = os.path.join(od, "qa.json")
        fb_path = cand if os.path.isfile(cand) else None
    elif a.feedback != "none":
        fb_path = a.feedback if os.path.isfile(a.feedback) else None
    fb = feedback_hints(fb_path, segs)
    if fb:
        print()
        print("上一轮成片的手势反馈（改 prompt_src.json 里这几段，改完再重跑本脚本和 --only 重拍）：")
        for h in fb:
            print(" →", h)
        json.dump({"source": fb_path, "items": fb},
                  open(os.path.join(od, "prompt_feedback.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("   （已写入 %s）" % os.path.join(od, "prompt_feedback.json"))
    if problems:
        print("校验未通过：")
        for p in problems:
            print(" ✗", p)
        sys.exit(1)
    pd = os.path.join(od, "prompts")
    os.makedirs(pd, exist_ok=True)
    for i, full in outs:
        open(os.path.join(pd, "segment_%02d.txt" % i), "w", encoding="utf-8").write(full)
        print("segment_%02d.txt  %d 字" % (i, len(full)))
    if it_letter:
        print("图片类型：%s（%s）%s" % (src.get("image_type", "").strip(), IMAGE_TYPES[it_letter],
                                    "；首行已注入 " + prefix if prefix else "；未加固定镜头首行"))
    print("校验通过，提示词已写入", pd)


if __name__ == "__main__":
    main()
