// Showcase UI strings (the engineering drawer stays in English, like the
// desktop tool).

const zh = {
  tagline: "对数螺旋软体机器人 · 与鹦鹉螺、章鱼腕和旋涡星系同一条方程",
  modeFlex: "灵活性",
  modeSpeed: "速度",
  modeScale: "尺度",
  engineering: "工程",
  share: "分享",
  copied: "链接已复制",
  shape: "形状",
  length: "长度",
  units: "节数",
  dtheta: "每节角度 Δθ",
  turns: "螺旋圈数",
  taper: "锥度 b",
  centre: "中心线 p",
  cables: "线缆",
  gravity: "重力",
  stiffness: "刚度",
  more: "全部参数…",
  live: "实时",
  replay: "回放",
  play: "运行",
  pause: "暂停",
  resetPose: "复位",
  curl: "卷曲角",
  tipSpeed: "尖端速度",
  peakSpeed: "峰值速度",
  mass: "质量",
  timeScale: "时间倍率",
  realtime: "实时比",
  steer: "肌腱控制",
  steerHint: "拖动摇杆拉动线缆 · 双击归零",
  curlBtn: "卷起",
  relaxBtn: "放松",
  waveBtn: "摆动",
  circleBtn: "画圈",
  object: "抓取物体",
  none: "无",
  cylinder: "圆柱",
  sphere: "球",
  box: "方块",
  size: "尺寸",
  grasp: "抓取",
  release: "松开",
  dragHint: "在 3D 中拖动机器人可施加外力，拖动物体可移动它",
  heat: "弯曲热度",
  strike: "发射",
  force: "拉力",
  t90: "卷曲用时 t₉₀",
  peakCurl: "最大卷曲",
  ghosts: "残影",
  trail: "轨迹",
  slowmo: "慢放重播",
  strikeHint: "从伸直状态瞬间拉紧线缆，测量卷曲用时与尖端速度",
  measuring: "测量中…",
  scaleTitle: "整体尺寸",
  scaleHint: "同一设计从毫米级缩放到米级：材料不变，刚度∝L³、拉力∝L²",
  refs: "参照物",
  coin: "硬币",
  hand: "手掌",
  human: "潜水员",
  law: "尺度定律",
  lawText: "卷曲用时 ∝ 长度；尖端峰值速度几乎与尺寸无关",
  measureHere: "在此尺寸测速",
  loading: "正在加载物理引擎…",
  slowFactor: "慢放",
  points: "已测",
  joints: "关节",
  edges: "棱线",
  tendons: "线缆",
  fit: "取景",
  descFlex: "拉动线缆、推它一把、让它抓取物体",
  descSpeed: "从伸直到卷起，测量它有多快",
  descScale: "同一设计，从毫米到米",
  space: "宇宙",
  ocean: "海洋",
  toOcean: "切换到海洋环境",
  toSpace: "切换到宇宙环境",
  astronaut: "宇航员",
  collapse: "收起",
  expand: "展开控制",
  quality: "画质",
  auto: "自动",
  high: "高",
  medium: "中",
  low: "低",
} as const;

export type Key = keyof typeof zh;

const en: Record<Key, string> = {
  tagline: "Logarithmic-spiral soft robots: the curve of the nautilus, the octopus arm and spiral galaxies",
  modeFlex: "Flexibility",
  modeSpeed: "Speed",
  modeScale: "Scale",
  engineering: "Engineering",
  share: "Share",
  copied: "Link copied",
  shape: "Shape",
  length: "Length",
  units: "Units",
  dtheta: "Unit angle Δθ",
  turns: "Spiral turns",
  taper: "Taper b",
  centre: "Centre line p",
  cables: "Cables",
  gravity: "Gravity",
  stiffness: "Stiffness",
  more: "All parameters…",
  live: "Live",
  replay: "Replay",
  play: "Run",
  pause: "Pause",
  resetPose: "Reset",
  curl: "Curl",
  tipSpeed: "Tip speed",
  peakSpeed: "Peak speed",
  mass: "Mass",
  timeScale: "Time scale",
  realtime: "Real-time",
  steer: "Tendon control",
  steerHint: "Drag the stick to pull cables · double-click to centre",
  curlBtn: "Curl",
  relaxBtn: "Relax",
  waveBtn: "Wave",
  circleBtn: "Circle",
  object: "Object",
  none: "None",
  cylinder: "Cylinder",
  sphere: "Sphere",
  box: "Box",
  size: "Size",
  grasp: "Grasp",
  release: "Release",
  dragHint: "Drag the robot in 3D to push it; drag the object to move it",
  heat: "Bend heat",
  strike: "Strike",
  force: "Force",
  t90: "Curl time t₉₀",
  peakCurl: "Peak curl",
  ghosts: "Ghosts",
  trail: "Trail",
  slowmo: "Slow-motion replay",
  strikeHint: "Tension a cable instantly from straight; measures curl time and tip speed",
  measuring: "Measuring…",
  scaleTitle: "Overall size",
  scaleHint: "One design from millimetres to metres: same material, stiffness ∝ L³, force ∝ L²",
  refs: "References",
  coin: "Coin",
  hand: "Hand",
  human: "Diver",
  law: "Scaling law",
  lawText: "Curl time ∝ length; peak tip speed is almost size-independent",
  measureHere: "Measure at this size",
  loading: "Loading the physics engine…",
  slowFactor: "Slow-mo",
  points: "measured",
  joints: "Joints",
  edges: "Edges",
  tendons: "Cables",
  fit: "Fit",
  descFlex: "Pull the cables, push it, let it grasp",
  descSpeed: "From straight to curled, timed",
  descScale: "One design, millimetres to metres",
  space: "Space",
  ocean: "Ocean",
  toOcean: "Switch to the ocean scene",
  toSpace: "Switch to the space scene",
  astronaut: "Astronaut",
  collapse: "Hide",
  expand: "Controls",
  quality: "Quality",
  auto: "Auto",
  high: "High",
  medium: "Medium",
  low: "Low",
};

type Lang = "zh" | "en";
const LANG_KEY = "openspirobs.lang.v2";

/** The visitor's first preferred browser language, reduced to zh / en. */
function browserLang(): { lang: Lang; tag: string } {
  const tag = (navigator.languages?.[0] ?? navigator.language ?? "en").toLowerCase();
  return { lang: tag.startsWith("zh") ? "zh" : "en", tag };
}

// Follow the browser. The top-bar switch saves an override together with the
// browser language it was made under, so changing the browser language later
// takes effect again; picking the browser's own language clears the override.
let lang: Lang = (() => {
  const b = browserLang();
  try {
    const saved = JSON.parse(localStorage.getItem(LANG_KEY) ?? "null") as { lang?: string; tag?: string } | null;
    if (saved && (saved.lang === "zh" || saved.lang === "en") && saved.tag === b.tag) return saved.lang;
  } catch {
    /* fall back to the browser */
  }
  return b.lang;
})();

export const t = (k: Key): string => (lang === "zh" ? zh[k] : en[k]);
export const getLang = () => lang;
export function setLang(l: Lang): void {
  lang = l;
  const b = browserLang();
  try {
    localStorage.removeItem("openspirobs.lang"); // pre-v2 key: a plain choice that never expired
    if (l === b.lang) localStorage.removeItem(LANG_KEY);
    else localStorage.setItem(LANG_KEY, JSON.stringify({ lang: l, tag: b.tag }));
  } catch {
    /* ignore */
  }
}
