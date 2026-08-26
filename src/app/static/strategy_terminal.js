const terminalState = {
  symbol: null,
  data: null,
  busy: false,
  assetBusy: false,
  diagnosticsBusy: false,
  diagnostics: null,
  ontologyFilter: 'active',
  streamBusy: false,
  streamReceivedAt: 0,
  lastSecondBarTime: null,
  chartTransitionActive: false,
  chartMode: 'seconds',
  chartAnimationFrame: null,
  chartLastPaint: 0,
  gnnGraphBusy: false,
  gnnGraph: null,
  gnnStateBusy: false,
  gnnInference: null,
  gnnGraphRequestId: 0,
  gnnStateRequestId: 0,
  ontologySignature: null,
};

// Visualization is an operator preference only. Training and inference remain
// independent, while the expensive graph requests and render loop default off.
const GNN_VISUALIZATION_STORAGE_KEY = 'strategy-terminal-gnn-3d-enabled-v1';
let gnnVisualizationEnabled = readGnnVisualizationPreference();
const GNN_AUTO_ROTATION_STORAGE_KEY = 'strategy-terminal-gnn-auto-rotation-v1';
const GNN3D_AUTO_ROTATE_RAD_PER_SECOND = .075;
const GNN3D_AUTO_ROTATE_RESUME_DELAY_MS = 2500;
let gnnAutoRotationEnabled = readGnnAutoRotationPreference();

let gnn3dState = null;
let gnnThreePromise = null;

const gnnGraphView = {
  filter: 'all', zoom: 1, panX: 0, panY: 0, nodes: [], nodeMap: new Map(),
  hovered: null, selected: null, dragging: false, moved: false, lastX: 0, lastY: 0,
  frame: null, signature: null, lastPaint: 0,
  // Travelling waves currently crossing the threads. Each entry is one REAL
  // event -- see queueGnnWave. An empty list means nothing happened, and the
  // graph then sits still on purpose.
  waves: [],
  lastInferenceAt: null,
  lastCheckpointHash: null,
};

// Golden angle. Successive nodes placed at this turn never line up into rings or
// spokes, which is what makes a sunflower head look evenly filled.
const GNN_GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

// How long one wave takes to cross a thread, and how wide its packet is as a
// fraction of the span.
const GNN_WAVE_MS = 1150;
const GNN_WAVE_WIDTH = 0.17;

// Rope resolution in the 3D view. Six segments is enough for a droop to read as
// a curve; each one costs two vertices per edge, so this is the memory knob.
const GNN3D_EDGE_SEGMENTS = 6;
// Per-frame rope budget. While the system is in motion EVERY visible rope has to
// be re-solved, because a rope whose endpoint moved and whose vertices did not is
// visibly detached from its own node. Above this count the orbits stop instead of
// the ropes going stale: a still system is honest, a broken one is not.
//
// Sized against the DEPLOYED checkpoint, not an estimate: 23 strategies x 11 head
// channels over a 47x16 encoder is 7,491 links under the "all connections" filter.
// Measured solve cost is 0.61 ms/frame at 4,530 ropes, so 16,000 leaves the real
// graph animating with headroom for another strategy family, and the binding cost
// past that is the per-frame buffer upload rather than the solve.
const GNN3D_MAX_DYNAMIC_EDGES = 16000;

/**
 * Write one hanging rope into a LineSegments position buffer.
 *
 * The curve is a parabola through both endpoints with its low point at the
 * middle -- the standard small-sag approximation of a catenary, and visually
 * indistinguishable from one at this scale.
 *
 * Sag is a VECTOR, not a fixed -Y offset. Every rope hangs toward the body that
 * actually binds it (the message core for a system-wide edge, the parent strategy
 * for one of its own head moons), so the droop points where gravity points in
 * this layout and keeps doing so while the view is dragged. A negative sag
 * magnitude bows the rope the other way, which is how an inhibitory parameter --
 * a negative checkpoint weight, or a contrasting-methodology relation -- is drawn.
 */
function writeGnn3dEdgeCurve(positions, edgeIndex, source, target, sx, sy, sz) {
  const S = GNN3D_EDGE_SEGMENTS;
  const base = edgeIndex * S * 2 * 3;
  let px = source.x, py = source.y, pz = source.z;
  for (let segment = 0; segment < S; segment += 1) {
    const t = (segment + 1) / S;
    const droop = 4 * t * (1 - t);            // 0 at both ends, 1 at the middle
    const qx = source.x + (target.x - source.x) * t + sx * droop;
    const qy = source.y + (target.y - source.y) * t + sy * droop;
    const qz = source.z + (target.z - source.z) * t + sz * droop;
    const offset = base + segment * 6;
    positions[offset] = px; positions[offset + 1] = py; positions[offset + 2] = pz;
    positions[offset + 3] = qx; positions[offset + 4] = qy; positions[offset + 5] = qz;
    px = qx; py = qy; pz = qz;
  }
}

/*
 * The graph is laid out as ONE gravitationally bound system instead of four flat
 * discs stacked along an axis.
 *
 * The mapping is not decorative -- it is the checkpoint's own dependency
 * structure, which is radial: every path through the model passes through the
 * 16-D message vector, so that vector is the star, and everything else is bound
 * to it at a distance set by how strongly it is coupled.
 *
 *   core   <- hidden units      : the message vector, a churning nucleus
 *   planet <- strategy nodes    : orbit radius = 1 - learned head strength
 *   moon   <- output channels   : orbit their OWN strategy, radius = 1 - column norm
 *   belt   <- input features    : an outer belt, radius = 1 - encoder norm
 *
 * Radii therefore say "how tightly is this bound", inclination separates
 * methodology families onto their own orbital planes so 250-odd nodes fill the
 * volume instead of crowding four planes, and the reference plane is deliberately
 * wide and shallow (like a real system's ecliptic) because the panel is ~2:1.
 */
const GNN3D_SYSTEM = {
  core: { radius: 58, inclination: 1.45 },
  planet: { inner: 205, outer: 388, inclination: .52 },
  moon: { inner: 18, outer: 40, inclination: .72 },
  belt: { inner: 452, outer: 648, contextTilt: .17, identityTilt: .38 },
  // Kepler's third law: omega proportional to a^-1.5 about the message core, and
  // about a strategy for its own head moons. A tightly bound (strongly learned)
  // body visibly circulates faster, so the period is a second reading of the same
  // learned quantity rather than an arbitrary animation speed.
  keplerSystem: 372,
  keplerMoon: 47,
  maxOmega: .24,
  maxMoonOmega: .4,
};

const GNN3D_TWO_PI = Math.PI * 2;
// Deepest droop any rope may reach, as a fraction of its own span. Approached
// asymptotically, never hit, so a rope's bow can never swing back past its ends.
const GNN3D_SAG_LIMIT = .42;

// Home view. Tilted, because a head-on camera flattens the orbital planes that
// separate the methodology families back into the 2.5-D picture this replaced.
const GNN3D_HOME_DISTANCE = 1010;
const GNN3D_HOME_PITCH = -.34;
const GNN3D_HOME_YAW = .46;
// What the home view has to contain: the system's own extent (about 670 wide and
// 231 tall from the measured layout) plus the legend above it and the selection
// ribbon with its two caption rows below.
const GNN3D_FRAME_HALF_WIDTH = 700;
const GNN3D_FRAME_HALF_HEIGHT = 470;

/*
 * Orbital clock.
 *
 * How fast the whole system revolves is the one global thing the tape drives:
 * a quiet book barely drifts, a violent one visibly spins up, and a halted
 * session nearly stops. The floor is a property of the LAYOUT (a bound system
 * always moves a little) and not a claim about the market, so the resulting
 * multiplier is printed in the legend -- otherwise a fast graph and a fast market
 * would be indistinguishable.
 */
const GNN3D_CLOCK_FLOOR = .26;

function gnnSystemClock(forces) {
  const f = forces || GNN_NEUTRAL_FORCES;
  if (f.halted) return GNN3D_CLOCK_FLOOR * .3;
  return GNN3D_CLOCK_FLOOR
    + Number(f.activity || 0) * .55
    + Number(f.marketEnergy || 0) * 1.25
    + Math.max(0, Number(f.elasticity || 1) - 1) * .18;
}

/**
 * Market terms every rope is solved against, coerced once per frame.
 *
 * Per-rope code runs thousands of times a frame; reading and coercing the same six
 * fields inside that loop is work whose answer cannot change within the frame.
 */
function gnn3dMarketRopeTerms(forces) {
  const f = forces || GNN_NEUTRAL_FORCES;
  return {
    gravity: Number(f.gravity || 1),
    tension: Math.max(.35, Number(f.tension || 1)),
    energy: Number(f.marketEnergy || 0),
    damping: Number(f.damping || 1),
    stress: Number(f.liquidityStress || 0),
  };
}

/**
 * How far ONE rope sags right now: its own tension worked against market gravity.
 *
 *   tension   = this rope's stiffness (learned magnitude, relation norm, evidence,
 *               supervision, live net-edge forecast -- see edgeStiffnessFor)
 *               x market tension (tape activity, volatility, spread stress)
 *               x its own live traffic: a rope carrying data is pulled taut
 *   pull      = market gravity (macro regime + index trend) x the rope's own mass,
 *               which is the filled training rows behind its heavier endpoint
 *   swing     = a damped oscillator, rung faster by the change-point probability
 *               and harder by market energy, and driven further when the rope is
 *               active. Its clock is integrated by the caller so a change in rate
 *               does not teleport the phase.
 *   across    = how perpendicular the pull is to the rope; a radial spoke keeps a
 *               quarter of its droop so its tension stays readable
 *   polarity  = -1 for an inhibitory parameter, which bows the rope outward
 *
 * Two ropes therefore reach different lengths under the same tape, and one market
 * move bends the whole graph by a different amount edge by edge.
 */
function gnn3dRopeSag(spring, span, across, market, swingClock) {
  const tension = Math.max(.1, spring.stiffness * market.tension
    * (1 + spring.activation * .9) * (1 - market.stress * .28));
  const pull = market.gravity * (1 + spring.mass * .34);
  const swing = Math.sin(swingClock * spring.omega + spring.phase)
    * spring.amplitude * (.35 + market.energy * 1.7) * (1.3 - market.damping * .55)
    * (1 + spring.activation * .5);
  // The swing is a driving factor INSIDE the saturation, not a multiplier outside
  // it: a hot tape can drive a slack rope's oscillation past 1.0, and applying that
  // after the ceiling let the bow reach three quarters of the rope's own span. The
  // floor keeps an overshooting swing from inverting the rope -- only polarity, a
  // real property of the parameter, may do that.
  const raw = (.045 + .3 / (.55 + tension)) * pull * spring.seed * Math.max(.12, 1 + swing);
  // Soft ceiling instead of a hard clamp. A hard min() PINNED every slack rope at
  // the limit -- a weakly-learned edge sat at maximum droop in a calm market and
  // then did not move at all when the tape turned heavy, which is precisely the
  // reading the sag is supposed to carry. tanh keeps the response monotonic
  // forever: every rope always stretches a little more under more gravity, and none
  // can pass the limit that would bow it back over its own endpoints.
  return span * GNN3D_SAG_LIMIT * Math.tanh(raw / GNN3D_SAG_LIMIT)
    * (.25 + .75 * across) * spring.polarity;
}

/** Resolve one orbit's cartesian position from its current true anomaly. */
function gnn3dOrbitPosition(orbit) {
  const e = orbit.e;
  // Live activation does not invent a new lane: it makes the body's own measured
  // lane breathe by at most 4%. Connectivity controls the phase offset, so two
  // simultaneously active nodes do not pulse as a rigid sheet.
  const radialPulse = 1 + Number(orbit.liveActivation || 0) * .04
    * Math.sin(Number(orbit.activityPhase || 0) + Number(orbit.coupling || 0) * Math.PI);
  const a = orbit.a * radialPulse;
  const r = a * (1 - e * e) / (1 + e * Math.cos(orbit.theta));
  const inPlaneX = r * Math.cos(orbit.theta);
  const inPlaneY = r * Math.sin(orbit.theta);
  const cosNode = Math.cos(orbit.ascending), sinNode = Math.sin(orbit.ascending);
  const cosTilt = Math.cos(orbit.inclination), sinTilt = Math.sin(orbit.inclination);
  // Orthonormal basis of the orbital plane: the ascending-node direction lies in
  // the reference (XZ) plane, the second axis is tilted out of it by inclination.
  orbit.r = r;
  orbit.x = cosNode * inPlaneX - sinNode * cosTilt * inPlaneY + (orbit.parent ? orbit.parent.x : 0);
  orbit.y = sinTilt * inPlaneY + (orbit.parent ? orbit.parent.y : 0);
  orbit.z = sinNode * inPlaneX + cosNode * cosTilt * inPlaneY + (orbit.parent ? orbit.parent.z : 0);
}

/**
 * Advance every orbit by one frame. Parents are ordered before their children, so
 * a moon reads its planet's already-updated position in the same pass.
 */
function gnn3dAdvanceOrbits(orbits, dt, clock) {
  for (let index = 0; index < orbits.length; index += 1) {
    const orbit = orbits[index];
    if (dt > 0 && orbit.omega) {
      // Equal areas in equal times: d(theta) scales with 1/r^2, so an eccentric
      // orbit runs fast at perihelion and drags at aphelion. That is what makes a
      // strategy whose training payoff was unreliable look unsteady in its lane.
      const sweep = orbit.r > 0 ? Math.min(3, (orbit.a / orbit.r) ** 2) : 1;
      const activation = Number(orbit.liveActivation || 0);
      const coupling = Number(orbit.coupling || 0);
      const inertia = Number(orbit.inertialMass || 0);
      // Strongly connected bodies exchange more message mass and circulate
      // faster; heavily evidenced bodies resist acceleration. A live measured
      // activation temporarily increases throughput without moving an idle node.
      const relationshipSpeed = (.72 + coupling * .72) / (.82 + inertia * .34);
      const liveBoost = 1 + activation * 1.45;
      orbit.theta = (orbit.theta + orbit.omega * orbit.direction * dt * clock
        * sweep * relationshipSpeed * liveBoost) % GNN3D_TWO_PI;
      orbit.activityPhase = (Number(orbit.activityPhase || 0)
        + dt * (.9 + coupling * 1.8 + activation * 3.2)) % GNN3D_TWO_PI;
      // Relation diversity slowly precesses the orbital plane. This makes a node
      // linked through several semantic relation types trace a different path
      // from an equally heavy single-relation node.
      orbit.ascending = (orbit.ascending + Number(orbit.precession || 0) * dt
        * clock * (1 + activation)) % GNN3D_TWO_PI;
    }
    gnn3dOrbitPosition(orbit);
  }
}

const gnnClusterStyle = {
  // Labels carry no dimension: the deployed checkpoint is 47 features / 16 hidden /
  // 23 strategies x 11 head channels, and the hardcoded "41-D" / "104" went stale
  // the moment it was retrained. Counts are appended live from the payload where
  // these labels are drawn.
  input_context: { label: 'INPUT FEATURES', color: '#8178ff', x: 120, y: 335, radius: 145 },
  input_identity: { label: 'STRATEGY IDENTITY', color: '#a990ff', x: 120, y: 335, radius: 82 },
  hidden: { label: 'R-GCN MESSAGE', color: '#f6d778', x: 355, y: 335, radius: 105 },
  momentum: { label: 'MOMENTUM', color: '#ff537b', x: 590, y: 160, radius: 92 },
  breakout: { label: 'BREAKOUT', color: '#ffb861', x: 700, y: 245, radius: 92 },
  reversion: { label: 'REVERSION', color: '#20d9ff', x: 610, y: 455, radius: 102 },
  relative_strength: { label: 'RELATIVE', color: '#72e1bd', x: 745, y: 500, radius: 64 },
  specialist: { label: 'SPECIALIST', color: '#a78bfa', x: 670, y: 340, radius: 68 },
  output: { label: 'STRATEGY HEAD OUTPUTS', color: '#5eead4', x: 1010, y: 335, radius: 175 },
};

const gnnRelationStyle = {
  same_methodology_family: { color: '#b9d9d0', label: '동일 방법론 계열' },
  confirming_methodology: { color: '#ffd58a', label: '상호 확인 관계' },
  contrasting_methodology: { color: '#61c7d9', label: '대조 방법론 관계' },
  self_encoder_weight: { color: '#8278ff', label: '자기 특성 인코더 가중치' },
  strategy_head_weight: { color: '#5eead4', label: '전략 출력 헤드 가중치' },
  owns_output_head: { color: '#a7f3d0', label: '전략 출력 소유 관계' },
};

// The end-to-end pipeline, in the order a candidate actually walks it. Names and
// order come from SelectionStage in technical/selection_diagnostics.py; the
// counts come from /api/strategy-selection/diagnostics. Nothing here is invented
// on the client -- a stage with no measurement renders as unmeasured, not as
// zero, because "nothing was rejected here" and "we never looked" are different
// facts and only one of them is good news.
const GNN_PIPELINE_STAGES = [
  { id: 'RAW_CANDIDATE', label: '① 수집 · 원시 후보', color: 0x8178ff },
  { id: 'FEATURE_UNAVAILABLE', label: '② 특징 생성', color: 0x8fa4ff },
  { id: 'STRATEGY_TRIGGER_FALSE', label: '③ 전략 트리거', color: 0xf6d778 },
  { id: 'GROSS_EDGE_NON_POSITIVE', label: '④ 총엣지', color: 0xffc46b },
  { id: 'HORIZON_COST_UNVIABLE', label: '⑤ 호라이즌·비용', color: 0xffa457 },
  { id: 'MODEL_NOT_RELIABLE', label: '⑥ 모델 신뢰', color: 0xff8fb0 },
  { id: 'MODEL_DISAGREEMENT', label: '⑦ 모델 합치', color: 0xff7a9c },
  { id: 'FUSED_NET_NON_POSITIVE', label: '⑧ 순엣지', color: 0xff6f91 },
  { id: 'COST_FLOOR_REJECTED', label: '⑨ 비용 하한', color: 0xe86a9d },
  { id: 'PROFITABILITY_REJECTED', label: '⑩ 수익성 게이트', color: 0xc86ec0 },
  { id: 'MACRO_BLOCKED', label: '⑪ 거시 판정', color: 0x9f7bd8 },
  { id: 'ONTOLOGY_BLOCKED', label: '⑫ 온톨로지 게이트', color: 0x7d8ce8 },
  { id: 'SHADOW_ONLY', label: '⑬ 섀도우 전용', color: 0x5fa8e0 },
  { id: 'LIVE_NOT_AUTHORIZED', label: '⑭ 실거래 권한', color: 0x46c9d6 },
  { id: 'SELECTED', label: '⑮ 채택 · 주문', color: 0x5eead4 },
];

/**
 * Physics constants replaced by market state.
 *
 * Every term below is read from a field the engine already publishes; nothing is
 * synthesised. When a field is absent the force falls back to its neutral value
 * and is reported as unobserved, because a graph that hangs "normally" because
 * the market is calm and one that hangs normally because we could not read the
 * market look identical otherwise.
 *
 *   gravity   <- macro regime + index trend   : which way and how hard the tape pulls
 *   tension   <- learned edge strength        : a strong relation is a taut rope
 *   inertia   <- training rows behind a node  : evidence is mass, it resists moving
 *   elasticity<- change-point probability     : a regime about to break rings faster
 *   damping   <- 1 - change-point probability : a stable regime settles quickly
 */
const GNN_NEUTRAL_FORCES = {
  gravity: 1, tension: 1, inertia: 1, elasticity: 1, damping: 1,
  marketEnergy: 0, direction: 0, activity: 0, liquidityStress: 0,
  regime: null, indexTrend: null, changePoint: null, halted: false, observed: false,
};

// Regimes that pull the tape down. A rope in a falling market hangs heavier.
const GNN_HEAVY_REGIMES = new Set(['TREND_DOWN', 'STRONG_TREND_DOWN', 'HIGH_VOL_TRENDING_DOWN', 'RISK_OFF', 'BEAR']);
const GNN_LIGHT_REGIMES = new Set(['TREND_UP', 'STRONG_TREND_UP', 'HIGH_VOL_TRENDING_UP', 'RISK_ON', 'BULL']);

function gnnMarketForcesFrom(session, market = null) {
  if ((!session || typeof session !== 'object') && (!market || typeof market !== 'object')) {
    return { ...GNN_NEUTRAL_FORCES };
  }
  session = session && typeof session === 'object' ? session : {};
  market = market && typeof market === 'object' ? market : {};
  const regime = typeof session.macro_regime === 'string' ? session.macro_regime : null;
  const changePoint = Number.isFinite(Number(session.change_point_probability))
    ? Math.min(1, Math.max(0, Number(session.change_point_probability)))
    : null;
  // index_trend is carried inside the macro explanation the reasoner emitted.
  let indexTrend = null;
  (session.explanation_paths || []).forEach((path) => {
    const value = Number(path?.features?.index_trend);
    if (Number.isFinite(value)) indexTrend = value;
  });
  const halted = String(session.halt_level || 'NONE') !== 'NONE';
  const micro = market.microstructure || {};
  const finite = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
  const clamp = (value, low = 0, high = 1) => Math.max(low, Math.min(high, value));
  const return1s = finite(micro.return_1s);
  const return5s = finite(micro.return_5s);
  const return10s = finite(micro.return_10s);
  const aggressor = clamp(finite(micro.aggressor_imbalance_5s), -1, 1);
  const bookShift = clamp(finite(micro.orderbook_imbalance_change_5s), -1, 1);
  const spreadShift = Math.abs(finite(micro.spread_change_5s_bps));
  const activity = clamp(finite(micro.tick_count_5s) / 24);
  const volatility = clamp(
    Math.abs(return1s) * 220 + Math.abs(return5s) * 140
      + Math.abs(return10s) * 80 + spreadShift / 12,
  );
  const liquidityStress = clamp(spreadShift / 8 + (market.orderbook_stale ? .25 : 0));
  const direction = clamp(
    return5s * 180 + return10s * 90 + aggressor * .42 + bookShift * .28,
    -1,
    1,
  );
  const marketEnergy = clamp(volatility * .52 + Math.abs(aggressor) * .23 + activity * .25);

  // Trend is a small number (order 1e-3), so it is scaled into a visible range
  // and clamped: one violent print must not make every rope hit the floor.
  const trendPull = indexTrend === null ? -direction : Math.max(-1, Math.min(1, -indexTrend * 400));
  const regimePull = regime && GNN_HEAVY_REGIMES.has(regime) ? .45
    : regime && GNN_LIGHT_REGIMES.has(regime) ? -.3 : 0;
  return {
    gravity: Math.max(.45, Math.min(2.2, 1 + trendPull * .55 + regimePull + liquidityStress * .18)),
    tension: Math.max(.55, Math.min(2.1, .72 + activity * .42 + volatility * .38 + liquidityStress * .48)),
    inertia: 1,
    elasticity: 1 + (changePoint || 0) * 1.4 + marketEnergy * 1.25,
    damping: Math.max(.3, 1 - (changePoint || 0) * .55 - marketEnergy * .42),
    marketEnergy, direction, activity, liquidityStress,
    regime, indexTrend, changePoint, halted,
    observed: Boolean(regime || changePoint !== null || indexTrend !== null || micro.ready),
  };
}

async function fetchGnnMarketForces() {
  try {
    const response = await fetch('/api/realtime-trading/status', { cache: 'no-store' });
    if (!response.ok) return { ...GNN_NEUTRAL_FORCES };
    const payload = await response.json();
    return gnnMarketForcesFrom(
      (payload?.status || {}).strategy_session,
      terminalState.data?.market || null,
    );
  } catch (error) {
    // Unreadable market state must leave the graph neutral, not frozen.
    return { ...GNN_NEUTRAL_FORCES };
  }
}

async function fetchGnnPipeline() {
  try {
    const response = await fetch('/api/strategy-selection/diagnostics', { cache: 'no-store' });
    if (!response.ok) return null;
    const payload = await response.json();
    return payload && payload.available === false
      ? { unavailable: payload.reason || 'UNAVAILABLE', stages: payload.stages }
      : payload;
  } catch (error) {
    // A missing funnel must not take the graph down with it.
    return null;
  }
}

async function fetchGnnGraph() {
  if (!gnnVisualizationEnabled) return;
  if (terminalState.gnnGraphBusy) return;
  terminalState.gnnGraphBusy = true;
  const requestId = ++terminalState.gnnGraphRequestId;
  try {
    const response = await fetch('/api/account/gnn-graph', { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    if (!gnnVisualizationEnabled || requestId !== terminalState.gnnGraphRequestId) return;
    // Fetched with the graph so the funnel ribbon and the model layers always
    // describe the same moment.
    data.pipeline = await fetchGnnPipeline();
    data.forces = await fetchGnnMarketForces();
    if (!gnnVisualizationEnabled || requestId !== terminalState.gnnGraphRequestId) return;
    terminalState.gnnGraph = data;
    renderGnnGraphSummary(data);
    prepareGnnGraph(data);
  } catch (error) {
    if (requestId !== terminalState.gnnGraphRequestId) return;
    const status = document.getElementById('gnn-model-status');
    status.textContent = 'LOAD ERROR';
    status.className = 'status-chip blocked';
    document.getElementById('gnn-model-summary').textContent = `GNN 그래프를 불러오지 못했습니다: ${error.message}`;
  } finally {
    if (requestId === terminalState.gnnGraphRequestId) terminalState.gnnGraphBusy = false;
  }
}

async function fetchGnnState() {
  if (!gnnVisualizationEnabled) return;
  if (terminalState.gnnStateBusy) return;
  terminalState.gnnStateBusy = true;
  const requestId = ++terminalState.gnnStateRequestId;
  try {
    const response = await fetch('/api/account/gnn-state', { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const inference = await response.json();
    if (!gnnVisualizationEnabled || requestId !== terminalState.gnnStateRequestId) return;
    terminalState.gnnInference = inference;
    renderGnnLiveState(terminalState.gnnInference);
  } catch (_error) {
    if (requestId !== terminalState.gnnStateRequestId) return;
    renderGnnLiveState({ state: 'OFFLINE', active: false, age_seconds: null });
  } finally {
    if (requestId === terminalState.gnnStateRequestId) terminalState.gnnStateBusy = false;
  }
}

function refreshGnnMarketForces() {
  if (!gnnVisualizationEnabled || !terminalState.gnnGraph || !terminalState.data) return;
  terminalState.gnnGraph.forces = gnnMarketForcesFrom(
    terminalState.data.strategy_session || {},
    terminalState.data.market || {},
  );
  if (gnn3dState?.updateData) gnn3dState.updateData(terminalState.gnnGraph);
  renderGnnSystemHealth();
}

function renderGnnLiveState(state) {
  const badge = document.getElementById('gnn-inference-live');
  if (!badge) return;
  const active = Boolean(state?.active);
  const blocked = !active && state?.state === 'BLOCKED';
  badge.className = `gnn-inference-live ${active ? 'running' : blocked ? 'blocked' : 'idle'}`;
  document.getElementById('gnn-inference-live-state').textContent = active
    ? 'INFERENCE RUNNING'
    : blocked ? 'INFERENCE BLOCKED' : state?.state === 'OFFLINE' ? 'INFERENCE OFFLINE' : 'INFERENCE IDLE';
  // `Number(null)` is 0, so a missing age used to render as "마지막 추론 로그
  // 0.0초 전" — an absent reading displayed as a perfectly fresh one.
  const rawAge = state?.age_seconds;
  const age = rawAge === null || rawAge === undefined ? NaN : Number(rawAge);
  const activation = state?.activation || null;
  const evaluated = Number(activation?.layers?.strategy_election?.evaluated || 0);
  document.getElementById('gnn-inference-live-detail').textContent = active
    ? `${state.symbol || '-'} · ${state.path || 'GNN'} · ${evaluated}개 전략 평가`
    : Number.isFinite(age) ? `마지막 추론 로그 ${age < 60 ? `${age.toFixed(1)}초` : `${(age / 60).toFixed(1)}분`} 전` : '최근 추론 신호 없음';
  // Driven from the poll, not from the WebGL frame loop: these chips are DOM
  // state derived from data, they must be right whichever renderer is up (the 2D
  // fallback never entered the 3D loop, so they stayed blank there), and writing
  // them at 60fps was a per-frame DOM touch for a value that changes on poll.
  //
  // The layers survive a stale record instead of blanking. The inference log is
  // written per shadow cycle, so age crosses the freshness window between polls
  // and clearing on !active made the chips blink on and off — reintroducing by
  // accident exactly the pseudo-animation this panel was rid of. Stale is a
  // styling state, not an absence of knowledge.
  setGnnPhaseIndicator(activation?.layers || null, { stale: !active });
  renderGnnSystemHealth();
}

/*
 * Layer chips show which layers the inference record actually measured, not
 * which one a timer is currently on. Two of the four are never instrumented
 * (the encoder input and the hidden message vector are not logged), and they now
 * say so with a dedicated class instead of taking their turn in a sweep.
 */
function setGnnPhaseIndicator(layers, { stale = false } = {}) {
  document.querySelectorAll('[data-gnn-phase]').forEach((item) => {
    const layer = layers ? layers[item.dataset.gnnPhase] : null;
    const observed = Boolean(layer && layer.observed);
    item.classList.toggle('active', observed && !stale);
    item.classList.toggle('stale', observed && stale);
    item.classList.toggle('uninstrumented', Boolean(layer) && !observed);
    if (!layer) {
      item.removeAttribute('title');
      return;
    }
    // The count is the evidence: "16 arms evaluated", "4 channels decoded".
    if (observed && Number.isFinite(Number(layer.evaluated))) {
      item.title = `${layer.evaluated}개 전략 평가됨`;
    } else if (observed && Number.isFinite(Number(layer.channels))) {
      item.title = `${layer.channels}개 헤드 채널 디코딩됨`;
    } else if (layer.reason) {
      item.title = `계측되지 않음: ${layer.reason}`;
    }
  });
}

function renderGnnGraphSummary(data) {
  const model = data.model || {};
  const inference = data.inference || {};
  const counts = data.counts || {};
  const status = document.getElementById('gnn-model-status');
  const compatible = model.available && model.runtime_compatible;
  status.textContent = !model.available ? 'NO CHECKPOINT' : compatible ? 'LIVE COMPATIBLE' : 'CHECKPOINT STALE';
  status.className = `status-chip ${compatible ? 'passed' : 'blocked'}`;
  document.getElementById('gnn-model-summary').textContent = compatible
    ? '학습된 R-GCN 메시지 관계와 최근 실시간 추론을 함께 표시합니다.'
    : `학습 그래프는 표시되지만 현재 런타임 추론은 차단됨 · ${(model.runtime_reasons || []).join(' · ') || '호환성 확인 필요'}`;
  document.getElementById('gnn-training-size').textContent = `${formatInteger(model.training_rows)}행 · ${formatInteger(model.training_snapshots)} 스냅샷`;
  document.getElementById('gnn-validation-accuracy').textContent = Number.isFinite(Number(model.validation_accuracy))
    ? `${(Number(model.validation_accuracy) * 100).toFixed(1)}%`
    : '-';
  document.getElementById('gnn-graph-size').textContent = `${formatInteger(counts.nodes)} 노드 · ${formatInteger(counts.links)} 전체 연결`;
  document.getElementById('gnn-inference-size').textContent = `${formatInteger(inference.successful_decisions)} 성공 · ${formatInteger(inference.blocked_decisions)} 차단`;
  document.getElementById('gnn-model-provenance').textContent =
    `SOURCE ${data.source?.checkpoint || '-'} · 실제 체크포인트 텐서 기반 · 기존 시장 사실 온톨로지 그래프와 분리 · 최근 추론 ${shortDateTime(inference.latest_at)}`;
  renderGnnSystemHealth();
}

function renderGnnSystemHealth() {
  const panel = document.getElementById('gnn-system-health');
  if (!panel) return;
  const graph = terminalState.gnnGraph || null;
  const state = terminalState.gnnInference || {};
  const model = graph?.model || {};
  const forces = graph?.forces || GNN_NEUTRAL_FORCES;
  const pipeline = graph?.pipeline || null;
  const live = Boolean(state.active);
  const age = state.age_seconds === null || state.age_seconds === undefined
    ? NaN : Number(state.age_seconds);
  const activation = state.activation || {};
  const reasonCodes = Array.isArray(state.reason_codes) ? state.reason_codes : [];
  const authorizationBlocked = reasonCodes.some((reason) =>
    String(reason).startsWith('GNN_CHECKPOINT_NOT_LIVE_AUTHORIZED')
    || String(reason).startsWith('GNN_REALTIME_TRUST_NOT_READY')
    || String(reason).startsWith('GNN_TRUST_'));
  const strategies = activation.strategies || {};
  const activeStrategies = new Set(Object.entries(strategies)
    .filter(([, item]) => Number(item?.intensity || 0) > .02)
    .map(([id]) => id));
  const selected = activation.selected_strategy_id || null;
  const measuredChannels = new Set(Object.keys(activation.channels || {}));
  let activeNodes = live ? activeStrategies.size + measuredChannels.size : 0;
  let activeEdges = 0;
  if (live && graph) {
    (graph.links || []).forEach((link) => {
      const topology = (!link.kind || link.kind === 'topology')
        && activeStrategies.has(link.source) && activeStrategies.has(link.target);
      const decoded = link.relation === 'owns_output_head'
        && link.source === selected
        && measuredChannels.has((graph.nodes || []).find((node) => node.id === link.target)?.channel);
      if (topology || decoded) activeEdges += 1;
    });
  }

  const setCell = (id, level, value, detail) => {
    const cell = document.getElementById(id);
    if (!cell) return;
    cell.dataset.state = level;
    cell.querySelector('b').textContent = value;
    cell.querySelector('small').textContent = detail;
  };
  const apiReady = Boolean(graph);
  setCell(
    'gnn-health-data',
    !apiReady ? 'bad' : forces.observed ? 'good' : 'warn',
    !apiReady ? 'DISCONNECTED' : forces.observed ? 'LIVE FEED' : 'API READY',
    !apiReady ? '그래프 응답 없음' : forces.observed
      ? `활동 ${(Number(forces.activity || 0) * 100).toFixed(0)}% · 에너지 ${(Number(forces.marketEnergy || 0) * 100).toFixed(0)}%`
      : '시장 미관측 · 중립 물리값',
  );
  const compatible = Boolean(model.available && model.runtime_compatible);
  setCell(
    'gnn-health-model',
    compatible && !authorizationBlocked ? 'good' : model.available ? 'warn' : 'bad',
    !model.available ? 'NO CHECKPOINT' : !compatible ? 'INCOMPATIBLE'
      : authorizationBlocked ? 'SHADOW ONLY' : 'LIVE AUTHORIZED',
    compatible && !authorizationBlocked ? `${formatInteger(model.training_rows)}행 · ${formatInteger(model.relation_count)} 관계`
      : authorizationBlocked ? '실시간 신뢰/승격 조건 미충족'
      : (model.runtime_reasons || ['체크포인트 확인 필요']).join(' · '),
  );
  const flowLevel = live ? 'good' : state.state === 'BLOCKED' || state.state === 'OFFLINE' ? 'bad' : 'warn';
  setCell(
    'gnn-health-flow',
    flowLevel,
    live ? `${activeNodes} NODE · ${activeEdges} EDGE` : state.state || 'WAITING',
    live ? `${state.symbol || '-'} · ${state.action || 'EVALUATING'}`
      : Number.isFinite(age) ? `마지막 기록 ${age < 60 ? `${age.toFixed(1)}초` : `${(age / 60).toFixed(1)}분`} 전`
        : '최근 추론 계측 없음',
  );
  const stageCounts = pipeline?.stage_counts || null;
  let bottleneck = '파이프라인 미계측';
  if (stageCounts) {
    const highest = Object.entries(stageCounts).sort((a, b) => Number(b[1]) - Number(a[1]))[0];
    if (highest) bottleneck = `최대 탈락 ${highest[0]} ${formatInteger(highest[1])}건`;
  }
  setCell(
    'gnn-health-physics',
    forces.observed ? 'good' : 'warn',
    `G ${Number(forces.gravity || 1).toFixed(2)} · T ${Number(forces.tension || 1).toFixed(2)}`,
    `CLOCK ${gnnSystemClock(forces).toFixed(2)}× · ${bottleneck}`,
  );

  const overall = document.getElementById('gnn-health-overall');
  const level = !apiReady || !model.available || state.state === 'OFFLINE' ? 'bad'
    : !compatible || authorizationBlocked || state.state === 'BLOCKED' ? 'warn' : live ? 'good' : 'idle';
  overall.dataset.state = level;
  overall.textContent = level === 'good' ? 'RUNNING' : level === 'bad' ? 'DEGRADED'
    : level === 'warn' ? 'CHECK' : 'READY · IDLE';
}

function prepareGnnGraph(data) {
  // Only topology changes may recreate the WebGL scene. The previous signature
  // included the latest inference timestamp and live market forces, so every
  // normal poll destroyed the canvas and constructed it again. That looked like
  // data from two cycles flashing over each other. Dynamic facts are updated in
  // place by updateData() below.
  const nodeSignature = (data.nodes || []).map((node) => node.id).join(',');
  const linkSignature = (data.links || []).map((link) =>
    `${link.source}>${link.target}:${link.kind || ''}:${link.relation || ''}`).join(',');
  const signature = `${data.model?.checkpoint_hash || ''}|${nodeSignature}|${linkSignature}`;
  // Waves are events, not decoration. One fires only when the backend reports
  // something genuinely new: a fresh inference timestamp is a forward pass, a
  // new checkpoint hash is a retrain. If neither changed since the last poll the
  // threads just hang -- the honest picture of an idle model.
  const inferenceAt = data.inference?.latest_at || null;
  const checkpointHash = data.model?.checkpoint_hash || null;
  if (inferenceAt && gnnGraphView.lastInferenceAt && inferenceAt !== gnnGraphView.lastInferenceAt) {
    queueGnnWave('forward');
  }
  if (checkpointHash && gnnGraphView.lastCheckpointHash && checkpointHash !== gnnGraphView.lastCheckpointHash) {
    queueGnnWave('backward');
  }
  gnnGraphView.lastInferenceAt = inferenceAt;
  gnnGraphView.lastCheckpointHash = checkpointHash;
  const previous = new Map(gnnGraphView.nodes.map((node) => [node.id, node]));
  gnnGraphView.nodes = [];
  const groups = new Map();
  (data.nodes || []).forEach((node) => {
    const layoutGroup = node.kind === 'output' ? `output:${node.family || 'specialist'}` : node.cluster;
    if (!groups.has(layoutGroup)) groups.set(layoutGroup, []);
    groups.get(layoutGroup).push(node);
  });
  const outputCenters = {
    momentum: { x: 930, y: 115 }, breakout: { x: 1080, y: 245 },
    reversion: { x: 930, y: 485 }, relative_strength: { x: 1100, y: 515 },
    specialist: { x: 1070, y: 380 },
  };
  groups.forEach((items, layoutGroup) => {
    const isOutput = layoutGroup.startsWith('output:');
    const family = isOutput ? layoutGroup.slice(7) : layoutGroup;
    const style = isOutput
      ? { ...(gnnClusterStyle[family] || gnnClusterStyle.specialist), ...(outputCenters[family] || outputCenters.specialist), radius: items.length > 16 ? 82 : 58 }
      : (gnnClusterStyle[family] || gnnClusterStyle.specialist);
    // Phyllotaxis. sqrt(t) spaces nodes by equal AREA rather than equal radius,
    // so a 40-node cluster fills its disc as evenly as a 4-node one instead of
    // crowding the rim. The golden angle keeps successive nodes from lining up,
    // which removes the concentric banding the old two-ring rule made.
    const count = Math.max(1, items.length);
    items.forEach((node, index) => {
      const t = (index + .5) / count;
      const ring = style.radius * (.26 + .72 * Math.sqrt(t));
      const angle = index * GNN_GOLDEN_ANGLE + seededGraphUnit(node.id) * .22;
      const old = previous.get(node.id);
      gnnGraphView.nodes.push({ ...node, x: old?.x ?? style.x + Math.cos(angle) * ring, y: old?.y ?? style.y + Math.sin(angle) * ring });
    });
  });
  gnnGraphView.nodeMap = new Map(gnnGraphView.nodes.map((node) => [node.id, node]));
  gnnGraphView.signature = signature;
  bindGnnControls();
  startGnn3d(data, signature);
}

function bindGnnControls() {
  const reset = document.getElementById('gnn-reset-view');
  if (reset && reset.dataset.bound !== 'true') {
    reset.dataset.bound = 'true';
    reset.addEventListener('click', resetGnnGraphView);
  }
  document.querySelectorAll('[data-gnn-relation]').forEach((button) => {
    if (button.dataset.bound === 'true') return;
    button.dataset.bound = 'true';
    button.addEventListener('click', () => {
      gnnGraphView.filter = button.dataset.gnnRelation || 'all';
      document.querySelectorAll('[data-gnn-relation]').forEach((item) => item.classList.toggle('active', item === button));
      if (gnn3dState?.rebuildEdges) gnn3dState.rebuildEdges();
    });
  });
}

async function loadGnnThree() {
  if (window.__threeModule) return window.__threeModule;
  if (!gnnThreePromise) {
    gnnThreePromise = import('https://unpkg.com/three@0.165.0/build/three.module.js')
      .then((module) => { window.__threeModule = module; return module; })
      .catch((error) => { console.warn('3D GNN renderer unavailable; using 2D fallback.', error); return null; });
  }
  return gnnThreePromise;
}

async function startGnn3d(data, signature) {
  if (!gnnVisualizationEnabled) return;
  if (gnn3dState?.signature === signature) {
    gnn3dState.updateData(data);
    return;
  }
  const THREE = await loadGnnThree();
  if (!gnnVisualizationEnabled) return;
  if (!THREE) {
    bindGnnGraphCanvas();
    if (!gnnGraphView.frame) gnnGraphView.frame = requestAnimationFrame(drawGnnGraph);
    return;
  }
  if (gnn3dState?.cleanup) gnn3dState.cleanup();
  if (gnnGraphView.frame) { cancelAnimationFrame(gnnGraphView.frame); gnnGraphView.frame = null; }
  const canvas = document.getElementById('gnn-model-canvas');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x02050b);
  scene.fog = new THREE.FogExp2(0x02050b, 0.00105);
  const camera = new THREE.PerspectiveCamera(48, 1, .1, 4000);
  camera.position.set(0, 0, GNN3D_HOME_DISTANCE);
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  } catch (error) {
    console.warn('WebGL unavailable; using 2D GNN fallback.', error);
    bindGnnGraphCanvas();
    if (!gnnGraphView.frame) gnnGraphView.frame = requestAnimationFrame(drawGnnGraph);
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  const root = new THREE.Group();
  scene.add(root);
  scene.add(new THREE.HemisphereLight(0xbfe8ff, 0x080617, 1.25));
  const keyLight = new THREE.PointLight(0x38e8ff, 52, 1400); keyLight.position.set(80, 220, 420); scene.add(keyLight);
  const rimLight = new THREE.PointLight(0xb45cff, 40, 1200); rimLight.position.set(-360, -180, -250); scene.add(rimLight);
  // The message core lights the system it binds. Its output is driven by measured
  // inference liveness and tape energy in the frame loop, so the whole scene dims
  // when nothing is being decoded rather than staying evenly lit.
  const coreLight = new THREE.PointLight(0xffe6b0, 18, 1100);
  root.add(coreLight);
  const glowTexture = createGnn3dGlowTexture(THREE);
  const coreHalo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTexture, color: 0xffd58a, transparent: true, opacity: .16,
    blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  coreHalo.scale.set(320, 320, 1);
  root.add(coreHalo);

  const starPositions = [];
  for (let index = 0; index < 700; index += 1) {
    const radius = 900 + seededGraphUnit(`gnn-star:${index}`) * 1100;
    const theta = seededGraphUnit(`gnn-star-t:${index}`) * Math.PI * 2;
    const phi = Math.acos(2 * seededGraphUnit(`gnn-star-p:${index}`) - 1);
    starPositions.push(radius * Math.sin(phi) * Math.cos(theta), radius * Math.sin(phi) * Math.sin(theta), radius * Math.cos(phi));
  }
  const starsGeometry = new THREE.BufferGeometry();
  starsGeometry.setAttribute('position', new THREE.Float32BufferAttribute(starPositions, 3));
  scene.add(new THREE.Points(starsGeometry, new THREE.PointsMaterial({ color: 0x315597, size: 1.25, transparent: true, opacity: .5, depthWrite: false })));

  const nodeMap = new Map();
  // id -> position in `meshes`. Resolving activation walks every edge, and
  // meshes.indexOf() inside that loop would be 252 comparisons per edge — about
  // 1.5M per pass on this graph.
  const nodeIndex = new Map();
  const meshes = [];
  // Parallel to `meshes`: the orbit each body rides, or null for a body the layout
  // could not place. The frame loop copies orbit -> mesh.position through this, so
  // no per-frame Map lookup is needed for 250-odd bodies.
  const meshOrbits = [];
  const labels = [];
  const halos = [];
  const orbitRings = [];
  // Two shared unit spheres instead of one geometry per node. The previous build
  // allocated a SphereGeometry per node — 252 separate vertex buffers for a graph
  // whose nodes differ only in radius, which mesh.scale already expresses.
  const sharedSphere = {
    detailed: new THREE.SphereGeometry(1, 18, 18),
    simple: new THREE.SphereGeometry(1, 10, 10),
  };
  const contextLayer = new THREE.Group();
  root.add(contextLayer);
  const pipelineSignatureFor = (payload) => payload?.stage_counts
    ? GNN_PIPELINE_STAGES.map((stage) => payload.stage_counts[stage.id] || 0).join(',')
    : (payload?.unavailable || 'none');
  const forceSignatureFor = (payload) => {
    const f = payload || GNN_NEUTRAL_FORCES;
    return `${f.regime || '-'}|${Number(f.gravity || 1).toFixed(2)}|${Number(f.tension || 1).toFixed(2)}`
      + `|${Number(f.elasticity || 1).toFixed(2)}|${Number(f.damping || 1).toFixed(2)}`
      + `|${Number(f.marketEnergy || 0).toFixed(2)}|${f.halted ? 'halt' : 'run'}`;
  };
  let contextSignature = '';
  function disposeObject(object) {
    object.traverse?.((child) => {
      child.geometry?.dispose();
      if (child.material) (Array.isArray(child.material) ? child.material : [child.material]).forEach((material) => {
        material.map?.dispose(); material.dispose();
      });
    });
  }
  function rebuildContext(payload) {
    while (contextLayer.children.length) disposeObject(contextLayer.children.pop());
    contextLayer.add(buildGnnPipelineRibbon(THREE, payload.pipeline));
    // The physics legend. Without it the ropes are just pretty: an operator has
    // no way to distinguish a heavy market from an arbitrary visual setting.
    const f = payload.forces || GNN_NEUTRAL_FORCES;
    const text = f.observed
      ? `중력 ${f.gravity.toFixed(2)} · 장력 ${f.tension.toFixed(2)} · 탄성 ${f.elasticity.toFixed(2)}`
        + ` · 감쇠 ${f.damping.toFixed(2)} · 시장에너지 ${(f.marketEnergy * 100).toFixed(0)}%`
        + ` · 궤도클럭 ×${gnnSystemClock(f).toFixed(2)}`
        + ` (${f.regime || '-'}${f.indexTrend === null ? '' : `, 지수추세 ${(f.indexTrend * 100).toFixed(3)}%`})`
        + `${f.changePoint === null ? '' : ` (국면전환 ${(f.changePoint * 100).toFixed(1)}%)`}`
        + `${f.halted ? ' · HALT' : ''}`
      : `시장 물리 · 측정 없음 (중립값으로 렌더링 · 궤도클럭 ×${gnnSystemClock(f).toFixed(2)})`;
    const legend = createGnn3dLabel(THREE, text, f.observed ? 0xffd58a : 0x8aa1b7, { height: 30 });
    legend.position.set(0, 316, 0);
    legend.material.opacity = f.observed ? .78 : .45;
    contextLayer.add(legend);
    // What the physics MEANS. Every quantity below is a measurement, and an
    // operator who cannot tell which is which is looking at decoration.
    const mapping = createGnn3dLabel(
      THREE,
      '궤도반경 = 학습 결합도 · 속도 = 연결도 ÷ 증거관성 · 궤도 세차 = 관계 다양성 · 활성 = 펄스/가속 · 늘어짐 = 시장중력 ÷ 엣지장력',
      0x8aa1b7,
      { height: 24 },
    );
    mapping.position.set(0, 286, 0);
    mapping.material.opacity = .5;
    contextLayer.add(mapping);
    contextSignature = `${pipelineSignatureFor(payload.pipeline)}|${forceSignatureFor(f)}`;
  }
  rebuildContext(data);
  const layout = gnn3dLayout(data.nodes || [], data.links || [], data.model || null);
  const orbits = layout.orbits;

  (data.nodes || []).forEach((node) => {
    const position = gnn3dNodePosition(node, layout);
    const metric = layout.metrics.get(node.id) || { size: 2.4, mass: 0, strength: null };
    const color = new THREE.Color(gnn3dNodeColor(node));
    const radius = metric.size;
    // An unsupervised head is not a trained one. Its emissive floor is cut so it
    // reads as an unlit body even while the arm around it is being evaluated --
    // the runtime suppresses its positive-edge forecast, and the graph should not
    // present it as a working part of the model.
    const unsupervised = node.kind === 'strategy' && node.upside_supervised === false;
    const emissive = (node.kind === 'strategy' ? .3 + .28 * (metric.strength || 0) : .13)
      * (unsupervised ? .35 : 1);
    const material = new THREE.MeshStandardMaterial({
      color, emissive: color, emissiveIntensity: emissive,
      roughness: node.kind === 'strategy' ? .3 : .42, metalness: .16,
    });
    const mesh = new THREE.Mesh(
      node.kind === 'strategy' ? sharedSphere.detailed : sharedSphere.simple,
      material,
    );
    mesh.scale.setScalar(radius);
    mesh.position.set(position.x, position.y, position.z);
    mesh.userData = {
      ...node, baseEmissive: emissive, baseRadius: radius,
      orbitRadius: layout.placements.get(node.id)?.a ?? null,
      orbitEccentricity: layout.placements.get(node.id)?.e ?? null,
      orbitAngularSpeed: layout.placements.get(node.id)?.omega ?? null,
      orbitPrecession: layout.placements.get(node.id)?.precession ?? null,
      inertialMass: layout.placements.get(node.id)?.inertialMass ?? null,
      parameterNorm: metric.strength,
    };
    nodeIndex.set(node.id, meshes.length);
    meshOrbits.push(layout.placements.get(node.id) || null);
    root.add(mesh); meshes.push(mesh); nodeMap.set(node.id, mesh);
    if (node.kind === 'strategy') {
      const label = createGnn3dLabel(THREE, shortGnnLabel(node.label), color.getHex(), { height: 17 });
      label.userData = node;
      root.add(label);
      labels.push({ sprite: label, meshIndex: meshes.length - 1 });
    }
    // Every compute kind gets an additive halo. Only measured activation gives it
    // material opacity, so output-channel activity is visible while uninstrumented
    // input/hidden nodes remain honestly dark.
    const halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTexture, color, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    root.add(halo);
    halos.push({ sprite: halo, meshIndex: meshes.length - 1, unsupervised });
  });

  // Real orbit paths, drawn from each planet's own elements. These replace the
  // four translucent stage discs: a ring here is where a body actually travels,
  // and its brightness tracks that arm's live activation.
  layout.planetOrbits.forEach((orbit, strategyId) => {
    const mesh = nodeMap.get(strategyId);
    if (!mesh) return;
    const ring = buildGnn3dOrbitPath(THREE, orbit, gnn3dNodeColor(mesh.userData), .12);
    root.add(ring);
    orbitRings.push({ line: ring, meshIndex: nodeIndex.get(strategyId), base: .12 });
  });
  layout.beltRings.forEach((belt) => {
    const ring = buildGnn3dOrbitPath(
      THREE,
      { a: belt.a, e: 0, inclination: belt.inclination, ascending: 0, theta: 0, parent: null },
      belt.cluster === 'input_identity' ? gnnClusterStyle.input_identity.color : gnnClusterStyle.input_context.color,
      belt.cluster === 'input_identity' ? .14 : .08,
    );
    root.add(ring);
    orbitRings.push({ line: ring, meshIndex: null, base: belt.cluster === 'input_identity' ? .14 : .08 });
  });

  // Shell captions, stacked clear of the bodies on the right. Each carries its own
  // population count, so an empty or truncated layer is visible as such.
  layout.shells.forEach((shell, index) => {
    const caption = createGnn3dLabel(THREE, `${shell.label} · ${shell.count}`, shell.color, { height: 24 });
    caption.position.set(596, 224 - index * 32, 0);
    caption.material.opacity = .58;
    root.add(caption);
  });

  const edgeLayer = new THREE.Group();
  const glowLayer = new THREE.Group();
  root.add(edgeLayer); root.add(glowLayer);
  let visibleEdges = [];
  // One glow mesh whose per-vertex alpha carries activation, replacing four
  // full-graph LineSegments (one per pretend phase) that each duplicated every
  // edge's positions and were switched on by a clock.
  let glowLine = null;
  let glowColors = null;
  let activeEdgeIndexes = [];
  // Parallel to activeEdgeIndexes: the measured intensity of each active rope,
  // which is what the packet budget is shared out by.
  const activeEdgeIntensity = [];
  // Rope state, rebuilt with the edge set. `edgePositions` is the shared buffer
  // both the base and glow lines read, so animating a rope moves its glow too.
  let edgeSprings = [];
  let edgePositions = null;
  let edgePositionAttribute = null;
  // Whether the bodies are allowed to move this build. Motion requires re-solving
  // EVERY visible rope each frame; over the budget the system holds still instead
  // of leaving ropes hanging off nodes that have drifted away from them.
  let orbitsAnimated = true;
  // Render/solve cadence. Measured at 0.6 ms to solve the full 4,500-rope build, so
  // 30fps is comfortable there and the slower beat is reserved for a graph several
  // times denser, where the per-frame buffer upload rather than the solve is the
  // limit. A slower beat is always preferable to dropping a subset of the ropes.
  let framePace = 2;
  // Declared here because rebuildEdges() runs before the animation block and
  // marks activation stale; a `let` further down would still be in its TDZ.
  let activationDirty = true;
  let activationSignature = '';
  const nodeIntensity = new Float32Array(meshes.length);
  const glowColor = new THREE.Color();
  const tooltipElement = document.getElementById('gnn-model-tooltip');
  function filteredLinks() {
    return (gnn3dState?.data?.links || data.links || []).filter((link) => {
      if (gnnGraphView.filter === 'all') return true;
      if (gnnGraphView.filter === 'learned_parameter') return link.kind === 'learned_parameter';
      if (gnnGraphView.filter === 'strategy_topology') return !link.kind || link.kind === 'topology';
      return link.relation === gnnGraphView.filter;
    });
  }
  function clearGroup(group) {
    while (group.children.length) {
      const child = group.children.pop(); child.geometry?.dispose(); child.material?.dispose();
    }
  }
  function edgeSignatureFor(payload) {
    return `${gnnGraphView.filter}|${(payload.links || []).map((link) =>
      `${link.source}>${link.target}:${Number(link.learned_strength || 0).toFixed(6)}`).join(',')}`;
  }
  let edgeSignature = '';
  /*
   * Per-rope tension. This is where the graph stops being decoration.
   *
   * Sag used to be one global expression: every rope took the same market gravity
   * over the same market tension, and the only thing that differed between two
   * ropes was the learned weight and a hash. So the whole picture moved as one
   * sheet, and "this connection is strong" and "the tape is heavy" produced the
   * same droop.
   *
   * Each rope now carries its OWN stiffness, assembled from what is actually known
   * about that specific connection:
   *
   *   learned_strength   the parameter's normalised magnitude
   *   relation_strength  the learned Frobenius norm of that relation's tensor
   *   prior_weight       1 / in-degree: a message split many ways pulls less
   *   filled rows        evidence behind the endpoints (also its inertia)
   *   upside_supervised  an unsupervised head cannot hold a rope taut
   *   latest_probability the arm's most recent decoded success probability
   *   expected_net_bps   its most recent net-edge forecast -- signed, so a
   *                      negative forecast visibly SLACKENS its own ropes
   *
   * Market terms then scale that per-rope stiffness rather than replacing it, so a
   * regime change bends 4,500 ropes by 4,500 different amounts.
   */
  function edgeStiffnessFor(link, sourceNode, targetNode, rows) {
    const model = (gnn3dState?.data?.model || data.model) || {};
    const relationStrength = model.relation_strength || {};
    const strength = Math.min(1, Math.max(0, Number(link.learned_strength || 0)));
    const relationKey = String(link.relation || '').startsWith('relation_encoder:')
      ? String(link.relation).slice('relation_encoder:'.length)
      : String(link.relation || '');
    const relationNorm = Number.isFinite(Number(relationStrength[relationKey]))
      ? Math.min(1, Math.max(0, Number(relationStrength[relationKey])))
      : null;
    const evidence = rows > 0 ? Math.min(1, Math.log10(1 + rows) / 4.2) : 0;
    const owner = sourceNode.kind === 'strategy' ? sourceNode
      : targetNode.kind === 'strategy' ? targetNode
        : nodeMap.get(String(link.strategy_id || targetNode.strategy_id || ''))?.userData || null;
    const probability = owner && owner.latest_probability !== null && owner.latest_probability !== undefined
      ? Math.min(1, Math.max(0, Number(owner.latest_probability)))
      : null;
    const netBps = owner && owner.latest_expected_net_bps !== null && owner.latest_expected_net_bps !== undefined
      ? Number(owner.latest_expected_net_bps)
      : null;
    const netUnit = netBps === null || !Number.isFinite(netBps)
      ? 0
      : Math.max(-1, Math.min(1, netBps / 14));
    const supervised = owner ? owner.upside_supervised : null;
    let stiffness = .26
      + strength * 1.15
      + (relationNorm === null ? 0 : relationNorm * .55)
      + evidence * .42
      + (probability === null ? 0 : (probability - .5) * .5)
      + netUnit * .3
      - (supervised === false ? .34 : 0);
    // prior_weight is 1 / in-degree for that relation: a rope carrying a smaller
    // share of the incoming message is a slacker rope.
    const prior = Number(link.prior_weight);
    if (Number.isFinite(prior)) stiffness *= .62 + Math.min(1, prior) * .75;
    return Math.max(.12, Math.min(3.4, stiffness));
  }
  function rebuildEdges() {
    clearGroup(edgeLayer); clearGroup(glowLayer);
    glowLine = null; glowColors = null; activeEdgeIndexes = [];
    visibleEdges = filteredLinks().filter((link) => nodeMap.has(link.source) && nodeMap.has(link.target));
    // Positions are written once into a typed array sized exactly for the edge
    // set, and the glow layer REUSES them: same buffer, second material.
    // Each edge is a hanging rope, not a chord: SEGMENTS+1 sampled points joined
    // as segment pairs. A straight line between two spheres carries no sense of
    // connection strength; a rope's droop does.
    const S = GNN3D_EDGE_SEGMENTS;
    const vertexCount = visibleEdges.length * S * 2;
    const positions = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);
    const reusable = new THREE.Color();
    edgeSprings = new Array(visibleEdges.length);
    orbitsAnimated = visibleEdges.length <= GNN3D_MAX_DYNAMIC_EDGES;
    framePace = visibleEdges.length > 12000 ? 3 : 2;
    visibleEdges.forEach((link, index) => {
      const sourceMesh = nodeMap.get(link.source);
      const targetMesh = nodeMap.get(link.target);
      const strength = Math.min(1, Math.max(0, Number(link.learned_strength || 0)));
      const seed = seededGraphUnit(`${link.source}->${link.target}:${link.relation || ''}`);
      // Inertia is EVIDENCE: a node trained on many filled rows resists being
      // swung around, one trained on almost nothing whips about. Read off the
      // heavier of the two ends, since a rope is only as sluggish as its anchor.
      const rows = Math.max(
        Number(sourceMesh.userData.training_filled_rows || 0),
        Number(targetMesh.userData.training_filled_rows || 0),
      );
      const inertia = 1 + Math.min(1.6, Math.log10(1 + rows) * .55);
      const weight = Number(link.weight);
      edgeSprings[index] = {
        source: sourceMesh.position,
        target: targetMesh.position,
        // A head moon is bound to its planet, not to the system core, so its rope
        // hangs toward the strategy that owns it.
        center: link.relation === 'owns_output_head' ? sourceMesh.position : null,
        stiffness: edgeStiffnessFor(link, sourceMesh.userData, targetMesh.userData, rows),
        mass: rows > 0 ? Math.min(1, Math.log10(1 + rows) / 4.2) : 0,
        // An inhibitory parameter pushes instead of pulling. A negative checkpoint
        // weight and a contrasting-methodology relation both bow outward, away from
        // the body that binds the rope.
        polarity: (Number.isFinite(weight) && weight < 0) || link.relation === 'contrasting_methodology' ? -1 : 1,
        seed: .78 + seed * .44,
        // ELASTICITY (change-point probability) scales this in the solver, INERTIA
        // slows it here.
        omega: (.0011 + seed * .0016) / inertia,
        phase: seed * Math.PI * 2,
        amplitude: .1 + .22 * (1 - strength),
        activation: 0,
        sx: 0, sy: 0, sz: 0,
      };
      reusable.set(gnn3dEdgeColor(link)).multiplyScalar(.22 + strength * .78);
      for (let vertex = 0; vertex < S * 2; vertex += 1) {
        const offset = (index * S * 2 + vertex) * 3;
        colors[offset] = reusable.r; colors[offset + 1] = reusable.g; colors[offset + 2] = reusable.b;
      }
    });
    edgePositions = positions;
    const positionAttribute = new THREE.Float32BufferAttribute(positions, 3);
    positionAttribute.setUsage(THREE.DynamicDrawUsage);
    edgePositionAttribute = positionAttribute;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', positionAttribute);
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    edgeLayer.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: .18, depthWrite: false,
    })));

    glowColors = new Float32Array(vertexCount * 3);
    const glowGeometry = new THREE.BufferGeometry();
    glowGeometry.setAttribute('position', positionAttribute);
    glowGeometry.setAttribute('color', new THREE.BufferAttribute(glowColors, 3));
    glowLine = new THREE.LineSegments(glowGeometry, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    glowLayer.add(glowLine);
    edgeSignature = edgeSignatureFor(gnn3dState?.data || data);
    activationDirty = true;
  }

  function updateData(nextData) {
    gnn3dState.data = nextData;
    (nextData.nodes || []).forEach((node) => {
      const mesh = nodeMap.get(node.id);
      if (mesh) Object.assign(mesh.userData, node);
    });
    const nextContextSignature = `${pipelineSignatureFor(nextData.pipeline)}|${forceSignatureFor(nextData.forces)}`;
    if (nextContextSignature !== contextSignature) rebuildContext(nextData);
    if (edgeSignatureFor(nextData) !== edgeSignature) rebuildEdges();
    activationDirty = true;
  }

  const particleCount = 260;
  const particleArray = new Float32Array(particleCount * 3);
  // Phase offsets are fixed per particle, so they are hashed ONCE. The previous
  // loop rebuilt a string key and hashed it for every particle on every frame —
  // ~13k string allocations a second whose result never changed.
  const particleOffsets = new Float32Array(particleCount);
  for (let index = 0; index < particleCount; index += 1) {
    particleOffsets[index] = seededGraphUnit(`particle:${index}`);
  }
  // Which rope each packet rides and how fast. Both are assigned from measured
  // edge intensity when activation is resolved, not round-robin: the rope the
  // inference actually ran down carries visibly more traffic than one lit only by
  // ambient tape energy. -1 parks the packet.
  const particleEdge = new Int32Array(particleCount).fill(-1);
  const particleSpeed = new Float32Array(particleCount).fill(1);
  const particleGeometry = new THREE.BufferGeometry();
  particleGeometry.setAttribute('position', new THREE.BufferAttribute(particleArray, 3));
  const particles = new THREE.Points(particleGeometry, new THREE.PointsMaterial({
    color: 0x66fbff, size: 3.4, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  root.add(particles);

  let dragging = false, moved = false, lastX = 0, lastY = 0;
  let autoRotateResumeAt = 0;
  // A tilted home view. The system is volumetric now, and a near-flat camera hid
  // the orbital planes that separate the methodology families.
  let rotationX = GNN3D_HOME_PITCH, rotationY = GNN3D_HOME_YAW, cameraTarget = GNN3D_HOME_DISTANCE;
  const pointer = new THREE.Vector2(9, 9), raycaster = new THREE.Raycaster();
  const controller = new AbortController();
  function updatePointer(event) {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }
  function resumeAutoRotationAfterManualControl() {
    dragging = false;
    autoRotateResumeAt = performance.now() + GNN3D_AUTO_ROTATE_RESUME_DELAY_MS;
  }
  canvas.addEventListener('pointerdown', (event) => {
    updatePointer(event); dragging = true; moved = false; autoRotateResumeAt = Infinity;
    lastX = event.clientX; lastY = event.clientY; canvas.setPointerCapture(event.pointerId);
  }, { signal: controller.signal });
  canvas.addEventListener('pointermove', (event) => {
    updatePointer(event);
    if (!dragging) return;
    const dx = event.clientX - lastX, dy = event.clientY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 1) moved = true;
    rotationY += dx * .006; rotationX += dy * .006; lastX = event.clientX; lastY = event.clientY;
  }, { signal: controller.signal });
  canvas.addEventListener('pointerup', (event) => {
    updatePointer(event); resumeAutoRotationAfterManualControl();
    if (!moved) {
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(meshes, false)[0];
      if (hit) renderGnnInspector(hit.object.userData);
    }
  }, { signal: controller.signal });
  canvas.addEventListener('pointercancel', resumeAutoRotationAfterManualControl, {
    signal: controller.signal,
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault(); cameraTarget = Math.max(180, Math.min(2100, cameraTarget + event.deltaY * .7));
    autoRotateResumeAt = performance.now() + GNN3D_AUTO_ROTATE_RESUME_DELAY_MS;
  }, { passive: false, signal: controller.signal });

  function resize() {
    const rect = canvas.getBoundingClientRect();
    renderer.setSize(rect.width, rect.height, false); camera.aspect = rect.width / Math.max(1, rect.height); camera.updateProjectionMatrix();
  }
  /**
   * Distance at which the whole system fits THIS canvas. A fixed distance cropped
   * the outer belt on a tall-narrow panel, where the limit is width rather than
   * height. Resizing does not call this: once the operator has zoomed, the zoom is
   * theirs to keep.
   */
  function homeDistance() {
    const rect = canvas.getBoundingClientRect();
    const aspect = rect.width > 0 && rect.height > 0 ? rect.width / rect.height : 1.9;
    const halfTan = Math.tan((camera.fov / 2) * Math.PI / 180);
    return Math.max(GNN3D_HOME_DISTANCE, Math.min(2100, Math.max(
      GNN3D_FRAME_HALF_HEIGHT / halfTan,
      GNN3D_FRAME_HALF_WIDTH / (halfTan * Math.max(.5, aspect)),
    )));
  }
  function resetView() {
    rotationX = GNN3D_HOME_PITCH; rotationY = GNN3D_HOME_YAW; cameraTarget = homeDistance();
  }
  function cleanup() {
    controller.abort(); window.removeEventListener('resize', resize); gnn3dState.stop = true;
    scene.traverse((object) => { object.geometry?.dispose(); if (object.material) (Array.isArray(object.material) ? object.material : [object.material]).forEach((material) => { material.map?.dispose(); material.dispose(); }); });
    renderer.dispose();
  }
  gnn3dState = { signature, data, renderer, scene, root, stop: false, rebuildEdges, updateData, resetView, cleanup };
  rebuildEdges(); resize(); resetView(); window.addEventListener('resize', resize, { passive: true });

  /*
   * Activation is RESOLVED, not sequenced. Every node's intensity comes from the
   * inference record (which arms the router evaluated, which gates it shut, which
   * head channels it decoded), so glow and pulse amplitude are a measurement.
   * A node the pass never touched has intensity 0 and therefore does not move.
   *
   * What this replaces: a 3.6s wall-clock carousel that swept the four layers in
   * order and lit whatever layer the clock was on. It read as staged inference
   * while carrying exactly one bit of real information — whether the log was
   * fresh — and it lit the input and hidden layers, which are not instrumented
   * at all.
   */
  function resolveActivation() {
    const state = terminalState.gnnInference || {};
    const activation = state.activation || {};
    const strategies = activation.strategies || {};
    const channels = activation.channels || {};
    const selected = activation.selected_strategy_id || null;
    const live = Boolean(state.active);
    let peak = 0;

    for (let index = 0; index < meshes.length; index += 1) {
      const node = meshes[index].userData;
      let intensity = 0;
      if (node.kind === 'strategy') {
        intensity = Number(strategies[node.id]?.intensity || 0);
      } else if (node.kind === 'output') {
        // A decoded channel only describes the arm it was decoded for, so only
        // that arm's output nodes light. Channels the record does not carry stay
        // dark rather than borrowing a neighbour's value.
        const value = channels[node.channel];
        if (value !== undefined && node.strategy_id === selected) {
          intensity = 0.35 + 0.65 * Math.min(1, Math.abs(Number(value)) / 2);
        }
      }
      // feature and hidden nodes are deliberately 0: ENCODER_INPUT_NOT_LOGGED /
      // HIDDEN_STATE_NOT_LOGGED. An un-instrumented layer must look it.
      nodeIntensity[index] = live ? intensity : intensity * 0.25;
      if (nodeIntensity[index] > peak) peak = nodeIntensity[index];
    }

    activeEdgeIndexes.length = 0;
    activeEdgeIntensity.length = 0;
    if (glowColors) {
      glowColors.fill(0);
      const forces = gnn3dState?.data?.forces || GNN_NEUTRAL_FORCES;
      const ambientEnergy = Number(forces.marketEnergy || 0);
      visibleEdges.forEach((link, index) => {
        const sourceIntensity = nodeIntensity[nodeIndex.get(link.source)] || 0;
        const targetIntensity = nodeIntensity[nodeIndex.get(link.target)] || 0;
        // An edge is only as active as the quieter end: a live strategy node does
        // not make its whole upstream fan light up.
        const inferenceIntensity = Math.min(sourceIntensity, targetIntensity)
          * (.35 + Number(link.learned_strength || 0) * .65);
        // Market glow is deliberately restricted to topology edges. It shows
        // measured tape energy without pretending that every learned parameter
        // was traversed by the GNN inference.
        const marketIntensity = (!link.kind || link.kind === 'topology')
          ? ambientEnergy * (.16 + Number(link.learned_strength || 0) * .24)
          : 0;
        const edgeIntensity = Math.max(inferenceIntensity, marketIntensity);
        // Activation is also a MECHANICAL term, not only a colour: a rope the pass
        // is pushing data down is pulled taut in the solver. Recorded for every
        // edge, including the quiet ones, so a rope that just went quiet relaxes.
        const spring = edgeSprings[index];
        if (spring) spring.activation = Math.min(1, edgeIntensity);
        if (edgeIntensity <= 0.02) return;
        activeEdgeIndexes.push(index);
        activeEdgeIntensity.push(edgeIntensity);
        glowColor.set(gnn3dEdgeColor(link)).multiplyScalar(Math.min(1, edgeIntensity));
        // An edge owns SEGMENTS*2 vertices now that it is a rope rather than a
        // chord. Writing only the first pair would light one sixth of the rope
        // AND scribble into the next edge's vertices.
        const vertices = GNN3D_EDGE_SEGMENTS * 2;
        for (let vertex = 0; vertex < vertices; vertex += 1) {
          const offset = (index * vertices + vertex) * 3;
          glowColors[offset] = glowColor.r;
          glowColors[offset + 1] = glowColor.g;
          glowColors[offset + 2] = glowColor.b;
        }
      });
      glowLine.geometry.attributes.color.needsUpdate = true;
    }
    allocateParticles();
    return { live, peak };
  }

  /*
   * Hand out the packet budget in proportion to measured edge intensity.
   *
   * Round-robin over the active set spread the same traffic down a rope the
   * inference actually traversed and a rope merely lit by ambient tape energy, so
   * "data is moving here" was the one thing the exchange animation could not say.
   */
  function allocateParticles() {
    if (!activeEdgeIndexes.length) { particleEdge.fill(-1); return; }
    let total = 0;
    for (let index = 0; index < activeEdgeIntensity.length; index += 1) total += activeEdgeIntensity[index];
    if (total <= 0) { particleEdge.fill(-1); return; }
    let cursor = 0;
    let carried = activeEdgeIntensity[0];
    for (let index = 0; index < particleCount; index += 1) {
      const target = ((index + .5) / particleCount) * total;
      while (carried < target && cursor < activeEdgeIntensity.length - 1) {
        cursor += 1;
        carried += activeEdgeIntensity[cursor];
      }
      particleEdge[index] = activeEdgeIndexes[cursor];
      // A busier rope also moves its packets faster, so throughput reads as speed
      // as well as density.
      particleSpeed[index] = .55 + Math.min(1, activeEdgeIntensity[cursor]) * 1.7;
    }
  }

  /*
   * Solve every visible rope for this instant.
   *
   * Both terms of the droop are measurements. The magnitude is this rope's own
   * stiffness (see edgeStiffnessFor) worked against the market's gravity, and the
   * DIRECTION is the line to whatever body binds the rope -- the message core, or
   * the parent strategy for one of its head moons -- so the whole picture hangs
   * radially like a bound system instead of everything drooping toward -Y.
   */
  function solveRopes(swingClock, forces) {
    if (!edgePositions || !edgeSprings.length) return;
    const market = gnn3dMarketRopeTerms(forces);
    for (let index = 0; index < edgeSprings.length; index += 1) {
      const spring = edgeSprings[index];
      if (!spring) continue;
      const source = spring.source, target = spring.target;
      const dx = target.x - source.x, dy = target.y - source.y, dz = target.z - source.z;
      const span = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      // Direction of the pull: the line from the rope's midpoint to the body that
      // binds it. `center` null means the message core at the origin.
      const cx = spring.center ? spring.center.x : 0;
      const cy = spring.center ? spring.center.y : 0;
      const cz = spring.center ? spring.center.z : 0;
      let gx = cx - (source.x + target.x) * .5;
      let gy = cy - (source.y + target.y) * .5;
      let gz = cz - (source.z + target.z) * .5;
      const reach = Math.sqrt(gx * gx + gy * gy + gz * gz);
      if (reach < 1e-3) { gx = 0; gy = -1; gz = 0; } else { gx /= reach; gy /= reach; gz /= reach; }
      // A rope cannot sag along its own axis, so only the component of the pull
      // ACROSS the rope bends it. That is why a spoke running straight out from the
      // core stays taut while a rope strung sideways between two bodies droops: the
      // length of this projection is the sine of the angle between them.
      const ux = dx / span, uy = dy / span, uz = dz / span;
      const along = gx * ux + gy * uy + gz * uz;
      let px = gx - along * ux, py = gy - along * uy, pz = gz - along * uz;
      let across = Math.sqrt(px * px + py * py + pz * pz);
      if (across < 1e-3) {
        // Perfectly radial: gravity has no across-rope component to bend it with.
        // Bow it along the tangential direction instead, at a fraction of the
        // droop, so its tension is still readable rather than the rope going dead.
        // cross(rope, Z), or cross(rope, X) when the rope is itself along Z.
        px = uy; py = -ux; pz = 0;
        let tangent = Math.sqrt(px * px + py * py);
        if (tangent < 1e-3) { px = 0; py = uz; pz = -uy; tangent = Math.sqrt(py * py + pz * pz) || 1; }
        px /= tangent; py /= tangent; pz /= tangent;
        across = 0;
      } else {
        px /= across; py /= across; pz /= across;
      }
      const magnitude = gnn3dRopeSag(spring, span, across, market, swingClock);
      spring.sx = px * magnitude; spring.sy = py * magnitude; spring.sz = pz * magnitude;
      writeGnn3dEdgeCurve(edgePositions, index, source, target, spring.sx, spring.sy, spring.sz);
    }
    if (edgePositionAttribute) edgePositionAttribute.needsUpdate = true;
  }

  let resolved = { live: false, peak: 0 };
  let frame = 0;
  let lastSolvedAt = 0;
  let ropesDirty = true;
  let lastForceSignature = '';
  let pulsePhase = 0;
  // Integrated clocks. Every rate in this loop is market-driven, so none of them may
  // be applied as a multiplier on the wall clock: each is accumulated at its current
  // rate, which lets a rate change alter the speed without moving the phase.
  let swingClock = 0;
  let flowClock = 0;

  function animate(now) {
    if (!gnn3dState || gnn3dState.stop) return;
    requestAnimationFrame(animate);
    frame += 1;
    // Solve and draw on the same beat. Solving a rope we will not draw is pure
    // cost, and solving on a different beat than we draw makes the sag stutter.
    // The beat is fixed rather than raised while the model is busy: a live pass is
    // exactly when thousands of ropes are moving and the frame is most expensive.
    if (frame % framePace !== 0) return;
    const dt = lastSolvedAt ? Math.min(.25, (now - lastSolvedAt) / 1000) : 0;
    lastSolvedAt = now;

    const state = terminalState.gnnInference || {};
    // Re-resolve only when the record actually changed. The signature is the
    // record's identity, so a still market costs no work at all.
    const signature = `${state.updated_at || ''}|${state.strategy_id || ''}|${state.action || ''}|${gnnGraphView.filter}`;
    if (activationDirty || signature !== activationSignature) {
      activationSignature = signature;
      activationDirty = false;
      resolved = resolveActivation();
      ropesDirty = true;
    }
    const { live, peak } = resolved;
    const forces = gnn3dState?.data?.forces || GNN_NEUTRAL_FORCES;
    const marketEnergy = Number(forces.marketEnergy || 0);
    const forceSignature = forceSignatureFor(forces);
    if (forceSignature !== lastForceSignature) { lastForceSignature = forceSignature; ropesDirty = true; }

    // Auto-rotation changes only the observer's yaw. The data-driven body orbits,
    // rope tension and inference pulses remain independent measurements. Direct
    // manipulation wins immediately and gets a short reading pause before the
    // overview resumes.
    if (gnnAutoRotationEnabled && !dragging && now >= autoRotateResumeAt && dt > 0) {
      rotationY = (rotationY + dt * GNN3D_AUTO_ROTATE_RAD_PER_SECOND) % GNN3D_TWO_PI;
    }
    root.rotation.x += (rotationX - root.rotation.x) * .12;
    root.rotation.y += (rotationY - root.rotation.y) * .12;
    camera.position.z += (cameraTarget - camera.position.z) * .09;

    const clock = gnnSystemClock(forces);
    if (orbitsAnimated && dt > 0) {
      for (let index = 0; index < meshOrbits.length; index += 1) {
        if (meshOrbits[index]) meshOrbits[index].liveActivation = nodeIntensity[index] || 0;
      }
      gnn3dAdvanceOrbits(orbits, dt, clock);
      for (let index = 0; index < meshes.length; index += 1) {
        const orbit = meshOrbits[index];
        if (orbit) meshes[index].position.set(orbit.x, orbit.y, orbit.z);
      }
      ropesDirty = true;
    }

    // One oscillator, scaled per node by that node's own intensity. The pulse is
    // the carrier; the amplitude is the data. Its RATE is the tape, integrated
    // rather than divided into the clock so a change in market energy speeds the
    // breathing up smoothly instead of jumping its phase.
    pulsePhase = (pulsePhase + dt * (1.35 + marketEnergy * 3.4)) % GNN3D_TWO_PI;
    const pulse = .5 + Math.sin(pulsePhase) * .5;
    for (let index = 0; index < meshes.length; index += 1) {
      const mesh = meshes[index];
      const intensity = nodeIntensity[index];
      if (intensity <= 0) {
        const ambient = marketEnergy * (.08 + pulse * .08);
        mesh.material.emissiveIntensity = mesh.userData.baseEmissive + ambient;
        mesh.scale.setScalar(mesh.userData.baseRadius * (1 + ambient * .025));
        continue;
      }
      mesh.material.emissiveIntensity = mesh.userData.baseEmissive + intensity * (1.8 + pulse * 2.4);
      mesh.scale.setScalar(mesh.userData.baseRadius * (1 + intensity * (.1 + pulse * .24)));
    }
    // Elastic time: the change-point probability sets how fast the ropes ring, and
    // integrating it here is what lets that rate change without shifting the phase
    // of 4,500 ropes at once.
    swingClock += dt * 1000 * Number(forces.elasticity || 1);
    if (orbitsAnimated || ropesDirty) { solveRopes(swingClock, forces); ropesDirty = false; }
    if (glowLine) glowLine.material.opacity = activeEdgeIndexes.length
      ? Math.min(1, .38 + marketEnergy * .42 + pulse * (.24 + peak * .26))
      : 0;
    for (let index = 0; index < labels.length; index += 1) {
      const { sprite, meshIndex } = labels[index];
      const mesh = meshes[meshIndex];
      const intensity = nodeIntensity[meshIndex] || 0;
      sprite.position.set(mesh.position.x + 11, mesh.position.y + mesh.userData.baseRadius + 9, mesh.position.z);
      sprite.material.opacity = .5 + Math.min(.45, intensity * .45);
    }
    for (let index = 0; index < halos.length; index += 1) {
      const halo = halos[index];
      const mesh = meshes[halo.meshIndex];
      const intensity = nodeIntensity[halo.meshIndex] || 0;
      halo.sprite.position.copy(mesh.position);
      const size = mesh.userData.baseRadius * (5.5 + intensity * 10 + pulse * intensity * 7);
      halo.sprite.scale.set(size, size, 1);
      halo.sprite.material.opacity = Math.min(.92, .025 + intensity * (.58 + pulse * .28) + marketEnergy * .045)
        * (halo.unsupervised ? .3 : 1);
    }
    // Orbit paths brighten with the arm that rides them, which makes the lane the
    // election is currently working in readable at a glance.
    for (let index = 0; index < orbitRings.length; index += 1) {
      const ring = orbitRings[index];
      const intensity = ring.meshIndex === null ? 0 : nodeIntensity[ring.meshIndex] || 0;
      ring.line.material.opacity = ring.base + intensity * .38 + marketEnergy * .05;
    }
    // The core is the system's star: it brightens with decoded output and tape
    // energy, and goes dim when nothing is being inferred.
    coreHalo.material.opacity = .08 + (live ? .12 : 0) + marketEnergy * .2 + peak * .14;
    coreLight.intensity = 14 + marketEnergy * 24 + peak * 22;

    particles.material.opacity = activeEdgeIndexes.length
      ? Math.min(.95, .22 + marketEnergy * (.3 + pulse * .42) + peak * .42)
      : 0;
    // Flow rate is measured tape speed: tick activity plus energy. A still book
    // leaves the packets nearly stationary rather than implying throughput. Also
    // integrated, so a quickening tape accelerates the packets already in flight.
    flowClock += dt * (.16 + Number(forces.activity || 0) * .4 + marketEnergy * .55);
    if (activeEdgeIndexes.length) {
      for (let index = 0; index < particleCount; index += 1) {
        const edgeIndex = particleEdge[index];
        const spring = edgeIndex >= 0 ? edgeSprings[edgeIndex] : null;
        const offset = index * 3;
        if (!spring) { particleArray[offset] = particleArray[offset + 1] = particleArray[offset + 2] = 0; continue; }
        const t = (flowClock * particleSpeed[index] + particleOffsets[index]) % 1;
        const droop = 4 * t * (1 - t);
        // Ride the rope, not the chord, using the sag the solver just applied: a
        // packet travelling straight between two endpoints visibly leaves its own
        // edge, which makes the droop look drawn rather than travelled.
        particleArray[offset] = spring.source.x + (spring.target.x - spring.source.x) * t + spring.sx * droop;
        particleArray[offset + 1] = spring.source.y + (spring.target.y - spring.source.y) * t + spring.sy * droop;
        particleArray[offset + 2] = spring.source.z + (spring.target.z - spring.source.z) * t + spring.sz * droop;
      }
      particleGeometry.attributes.position.needsUpdate = true;
    }

    // Picking on every solved frame allocated an intersection array for a pointer
    // that moves far slower than that.
    if (frame % 8 === 0) {
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(meshes, false)[0];
      if (hit && !dragging) updateGnn3dTooltip(canvas, hit.object.userData, pointer);
      else if (tooltipElement) tooltipElement.hidden = true;
    }
    renderer.render(scene, camera);
  }
  requestAnimationFrame(animate);
}

/**
 * One shared radial-gradient sprite texture for every halo in the scene.
 *
 * A halo per body would otherwise mean a texture per body; this is allocated once
 * and tinted per sprite, which is what makes a per-planet glow affordable.
 */
function createGnn3dGlowTexture(THREE) {
  const canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 128;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(255,255,255,.95)');
  gradient.addColorStop(.28, 'rgba(255,255,255,.34)');
  gradient.addColorStop(.62, 'rgba(255,255,255,.08)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/**
 * The closed path a body actually travels, sampled from its own orbital elements.
 *
 * Only bodies orbiting the system barycentre get one. A moon's path would have to
 * follow its moving parent, and 104 of them would bury the graph they are meant to
 * explain.
 */
function buildGnn3dOrbitPath(THREE, orbit, color, opacity) {
  const steps = 128;
  const positions = new Float32Array(steps * 3);
  const probe = { ...orbit, x: 0, y: 0, z: 0, r: orbit.a, parent: null };
  for (let index = 0; index < steps; index += 1) {
    probe.theta = (index / steps) * GNN3D_TWO_PI;
    gnn3dOrbitPosition(probe);
    positions[index * 3] = probe.x;
    positions[index * 3 + 1] = probe.y;
    positions[index * 3 + 2] = probe.z;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  return new THREE.LineLoop(geometry, new THREE.LineBasicMaterial({
    color, transparent: true, opacity, depthWrite: false,
  }));
}

// Functional layers, now shells of one bound system rather than planes along an
// axis. `kinds` maps a payload node onto its layer, which is what the inspector
// and the phase chips read; `label` is the caption the 3D view draws.
const GNN3D_LAYERS = [
  { id: 'ingest', kinds: ['feature'], color: 0x8178ff, label: '① 입력 특징 벨트 · 인코더 노름 = 반경' },
  { id: 'message', kinds: ['hidden'], color: 0xf6d778, label: '② R-GCN 메시지 코어 (계 중심)' },
  { id: 'strategy', kinds: ['strategy'], color: 0xff8fb0, label: '③ 전략 행성 · 헤드 강도 = 결합도' },
  { id: 'head', kinds: ['output'], color: 0x5eead4, label: '④ 출력 헤드 위성 · 전략에 종속' },
];

/**
 * Deterministic orbital layout: id -> orbit, plus the per-node measurements the
 * renderer sizes and lights each body by.
 *
 * Replaces four flat discs stacked along X. That layout wasted the volume (every
 * node sat within 26 units of its own plane, so the view was 2.5-D whichever way
 * you dragged it) and, worse, spent its one free spatial axis on a stage index the
 * colour and the captions already carried.
 *
 * Here every distance is a measurement:
 *
 *   - a feature's orbit radius is its ENCODER NORM, summed from the checkpoint's
 *     own self- and relation-weight tensors, so a feature the model leans on sits
 *     closer in. Identity features keep a separate tilted band.
 *   - a strategy's orbit radius is 1 - its learned head strength, and its
 *     eccentricity is 1 - the rate at which its filled training rows actually paid.
 *   - an output channel orbits its OWN strategy at 1 - its head column norm, and
 *     runs retrograde when that column sums negative.
 *   - a hidden unit's depth in the core is how much encoder mass it receives.
 *
 * Anything the checkpoint does not carry is placed mid-band and flagged
 * `observed: false` rather than being given an invented extreme.
 */
function gnn3dLayout(nodes, links, model = null) {
  const list = Array.isArray(nodes) ? nodes : [];
  const orbits = [];
  const placements = new Map();
  const metrics = new Map();

  // Per-node parameter norms, accumulated once from the tensors the payload
  // already ships. Encoder and head weights live on different scales, so they are
  // normalised within their own tensor family and never mixed.
  const featureNorm = new Map();
  const hiddenNorm = new Map();
  const outputNorm = new Map();
  const outputSigned = new Map();
  (links || []).forEach((link) => {
    const weight = Number(link.weight);
    if (!Number.isFinite(weight) || weight === 0) return;
    const relation = String(link.relation || '');
    const magnitude = Math.abs(weight);
    if (relation === 'self_encoder_weight' || relation.startsWith('relation_encoder:')) {
      featureNorm.set(link.source, (featureNorm.get(link.source) || 0) + magnitude);
      hiddenNorm.set(link.target, (hiddenNorm.get(link.target) || 0) + magnitude);
    } else if (relation === 'strategy_head_weight') {
      outputNorm.set(link.target, (outputNorm.get(link.target) || 0) + magnitude);
      outputSigned.set(link.target, (outputSigned.get(link.target) || 0) + weight);
    }
  });
  // Min-max within the kind, not value-over-maximum. A parameter norm is a sum of
  // 16-64 weights, so the raw norms of 41 features sit within a few percent of each
  // other: dividing by the maximum put every feature on the inner rim of its belt
  // and threw away the band it was given. Scaling across the observed range spends
  // the whole band on the spread that actually exists, and a node's position still
  // means "where this one sits between the weakest and strongest of its kind".
  const unitOf = (map) => {
    let max = -Infinity, min = Infinity;
    map.forEach((value) => {
      if (value > max) max = value;
      if (value < min) min = value;
    });
    const range = max - min;
    return (id) => {
      if (!map.has(id)) return null;
      if (!(range > 1e-12)) return .5;   // one member, or all identical
      return Math.min(1, Math.max(0, (map.get(id) - min) / range));
    };
  };
  const featureUnit = unitOf(featureNorm);
  const hiddenUnit = unitOf(hiddenNorm);
  const outputUnit = unitOf(outputNorm);

  // Mass is EVIDENCE: the filled training rows behind a node, log-scaled against
  // the best-evidenced node in this checkpoint.
  let heaviestRows = 0;
  list.forEach((node) => {
    const rows = Number(node.training_filled_rows || 0);
    if (rows > heaviestRows) heaviestRows = rows;
  });
  const massOf = (node) => {
    const rows = Number(node.training_filled_rows || 0);
    if (rows <= 0 || heaviestRows <= 0) return 0;
    return Math.min(1, Math.log10(1 + rows) / Math.log10(1 + heaviestRows));
  };

  // Connectivity is computed by the API from normalized checkpoint weights,
  // separately within each node kind. It is the gravitational coupling used by
  // the orbit solver. Relation diversity controls slow plane precession, while
  // evidence mass provides inertia. These values are also exposed in the node
  // inspector, so motion is auditable instead of decorative.
  const dynamicsOf = (node, inertialMass = 0) => {
    const coupling = Math.min(1, Math.max(0, Number(node.connectivity || 0)));
    const diversity = Math.min(1, Math.max(0, Number(node.relation_diversity || 0) / 5));
    const precessionSign = seededGraphUnit(`${node.id}:precession`) > .5 ? 1 : -1;
    return {
      coupling,
      inertialMass: Math.min(1, Math.max(0, inertialMass)),
      relationDiversity: diversity,
      precession: precessionSign * (.002 + diversity * .012 + coupling * .004),
      liveActivation: 0,
      activityPhase: seededGraphUnit(`${node.id}:activity`) * GNN3D_TWO_PI,
    };
  };

  const makeOrbit = (spec) => {
    const orbit = { ...spec, direction: spec.direction || 1, x: 0, y: 0, z: 0, r: spec.a };
    gnn3dOrbitPosition(orbit);
    orbits.push(orbit);
    return orbit;
  };
  const keplerAbout = (a) => Math.min(GNN3D_SYSTEM.maxOmega, GNN3D_SYSTEM.keplerSystem / (a ** 1.5));

  // ---- the core: hidden message units, a churning nucleus ----
  const coreOmega = keplerAbout(GNN3D_SYSTEM.core.radius);
  list.filter((node) => node.kind === 'hidden').forEach((node) => {
    const unit = hiddenUnit(node.id);
    const depth = unit === null ? .5 : unit;
    const orbit = makeOrbit({
      a: GNN3D_SYSTEM.core.radius * (.58 + .55 * (1 - depth)),
      e: 0,
      inclination: (seededGraphUnit(`${node.id}:i`) * 2 - 1) * GNN3D_SYSTEM.core.inclination,
      ascending: seededGraphUnit(node.id) * GNN3D_TWO_PI,
      theta: seededGraphUnit(`${node.id}:t`) * GNN3D_TWO_PI,
      omega: coreOmega,
      // Mixed directions: the nucleus churns instead of turning as one dial.
      direction: seededGraphUnit(`${node.id}:d`) > .5 ? 1 : -1,
      parent: null,
      ...dynamicsOf(node, depth),
    });
    placements.set(node.id, orbit);
    metrics.set(node.id, {
      size: 3 + 2.3 * depth, strength: unit, mass: depth,
    });
  });

  // ---- the belt: input features ----
  const belt = GNN3D_SYSTEM.belt;
  list.filter((node) => node.kind === 'feature').forEach((node, index) => {
    const unit = featureUnit(node.id);
    const bound = unit === null ? .5 : unit;
    const identity = node.cluster === 'input_identity';
    const a = belt.outer - (belt.outer - belt.inner) * bound;
    const orbit = makeOrbit({
      a,
      e: .02 + seededGraphUnit(`${node.id}:e`) * .06,
      // Identity features share one clearly tilted plane; context features spread
      // through a shallow band. The eight identity slots are the known leak
      // channel, and a separate plane is the cheapest way to keep them countable.
      inclination: identity
        ? belt.identityTilt * (.82 + .3 * seededGraphUnit(`${node.id}:i`))
        : (seededGraphUnit(`${node.id}:i`) * 2 - 1) * belt.contextTilt,
      ascending: (index * GNN_GOLDEN_ANGLE) % GNN3D_TWO_PI,
      theta: seededGraphUnit(`${node.id}:t`) * GNN3D_TWO_PI,
      omega: keplerAbout(a),
      parent: null,
      ...dynamicsOf(node, bound),
    });
    placements.set(node.id, orbit);
    metrics.set(node.id, {
      size: 1.7 + 2.2 * bound, strength: unit, mass: bound,
    });
  });

  // ---- the planets: strategy nodes ----
  const strategies = list.filter((node) => node.kind === 'strategy');
  const familyKeys = [...new Set(strategies.map((node) => node.family || node.cluster || 'specialist'))].sort();
  const planetOrbits = new Map();
  strategies.forEach((node, index) => {
    const seed = seededGraphUnit(node.id);
    const strength = Math.min(1, Math.max(0, Number(node.learned_strength || 0)));
    const familyIndex = Math.max(0, familyKeys.indexOf(node.family || node.cluster || 'specialist'));
    const lane = familyKeys.length > 1 ? (familyIndex + .5) / familyKeys.length * 2 - 1 : 0;
    const rate = node.training_positive_net_rate;
    const reliability = rate === null || rate === undefined
      ? null
      : Math.min(1, Math.max(0, Number(rate)));
    const a = GNN3D_SYSTEM.planet.inner
      + (GNN3D_SYSTEM.planet.outer - GNN3D_SYSTEM.planet.inner) * (1 - strength);
    const orbit = makeOrbit({
      a,
      // Eccentricity is how reliable the payoff was. An arm that paid on most of
      // its filled rows runs a near-circle; one that rarely did visibly wobbles.
      e: reliability === null ? .12 + seed * .06 : .05 + .28 * (1 - reliability),
      // One orbital plane per methodology family, so a family reads as a family
      // from any camera angle instead of only from the front.
      inclination: lane * GNN3D_SYSTEM.planet.inclination + (seed - .5) * .1,
      ascending: (index * GNN_GOLDEN_ANGLE) % GNN3D_TWO_PI,
      theta: seededGraphUnit(`${node.id}:t`) * GNN3D_TWO_PI,
      omega: keplerAbout(a),
      parent: null,
      ...dynamicsOf(node, massOf(node)),
    });
    planetOrbits.set(node.id, orbit);
    placements.set(node.id, orbit);
    metrics.set(node.id, {
      size: 5.2 + 5.2 * massOf(node),
      strength, mass: massOf(node),
    });
  });

  // ---- the moons: output head channels, bound to their own strategy ----
  const heads = Math.max(1, Number(model?.head_channels || 0) || 8);
  list.filter((node) => node.kind === 'output').forEach((node) => {
    const parent = planetOrbits.get(node.strategy_id) || null;
    const unit = outputUnit(node.id);
    const bound = unit === null ? .5 : unit;
    const seed = seededGraphUnit(node.id);
    const channel = Number(node.channel_index || 0);
    const a = GNN3D_SYSTEM.moon.inner
      + (GNN3D_SYSTEM.moon.outer - GNN3D_SYSTEM.moon.inner) * (1 - bound);
    const orbit = makeOrbit({
      a,
      e: .04 + seed * .14,
      // The eight channels fan out in inclination, so a strategy's heads form a
      // shell around it and stay individually pickable at any camera angle.
      inclination: (heads > 1 ? (channel + .5) / heads * 2 - 1 : 0) * GNN3D_SYSTEM.moon.inclination,
      ascending: (channel * GNN_GOLDEN_ANGLE + seed * .4) % GNN3D_TWO_PI,
      theta: seed * GNN3D_TWO_PI,
      omega: Math.min(GNN3D_SYSTEM.maxMoonOmega, GNN3D_SYSTEM.keplerMoon / (a ** 1.5)),
      // A head column that sums negative subtracts from its channel, and is drawn
      // retrograde: the sign of the parameter is visible in the motion.
      direction: (outputSigned.get(node.id) || 0) < 0 ? -1 : 1,
      parent,
      ...dynamicsOf(node, bound),
    });
    placements.set(node.id, orbit);
    metrics.set(node.id, {
      size: 1.5 + 2 * bound, strength: unit, mass: bound,
    });
  });

  // Reference rings for the two feature bands, at each band's mean radius and tilt.
  // Planet orbits are drawn from their own real elements; moons are not (their ring
  // would have to follow a moving parent, and 104 of them would bury the graph).
  const beltRings = ['input_context', 'input_identity'].map((cluster) => {
    const members = list.filter((node) => node.kind === 'feature' && node.cluster === cluster);
    if (!members.length) return null;
    let sumA = 0, sumTilt = 0;
    members.forEach((node) => {
      const orbit = placements.get(node.id);
      sumA += orbit.a; sumTilt += orbit.inclination;
    });
    return { cluster, a: sumA / members.length, inclination: sumTilt / members.length };
  }).filter(Boolean);

  const shells = GNN3D_LAYERS.map((layer) => ({
    label: layer.label, color: layer.color,
    count: list.filter((node) => layer.kinds.includes(node.kind)).length,
  })).filter((shell) => shell.count > 0);

  return { placements, orbits, metrics, shells, beltRings, planetOrbits };
}

function gnn3dNodePosition(node, layout) {
  const placed = layout?.placements?.get(node.id);
  return placed ? { x: placed.x, y: placed.y, z: placed.z } : { x: 0, y: 0, z: 0 };
}

/**
 * The selection funnel as a ribbon under the model graph.
 *
 * One marker per pipeline stage, left to right in the order a candidate walks
 * them, joined by hanging ropes. Marker size is how many candidates STOPPED at
 * that stage, so it shows where the pipeline actually kills things -- which is
 * what "why did nothing trade" reduces to.
 *
 * Unmeasured is drawn as a hollow outline, never as a zero-size marker: a stage
 * nobody reached and a stage that rejected nothing have nothing in common.
 */
function buildGnnPipelineRibbon(THREE, pipeline) {
  const group = new THREE.Group();
  // Clear of the system's southern bodies: the outer belt reaches about -150 in Y
  // and an inclined planet with its moons about -230.
  const y = -352;
  const spanX = 1120;
  const stageCounts = (pipeline && pipeline.stage_counts) || null;
  const unavailable = !pipeline || pipeline.unavailable || !stageCounts;
  // The SERVER's stage order wins when it sends one. SelectionStage is the
  // authority on what the pipeline is; a hardcoded client copy would silently
  // desync the moment a stage is added. The local table only supplies labels
  // and colours, and an unknown stage is still drawn under its raw id.
  const order = Array.isArray(pipeline?.stages) && pipeline.stages.length
    ? pipeline.stages.map((id) => (
      GNN_PIPELINE_STAGES.find((stage) => stage.id === id) || { id, label: id, color: 0x8aa1b7 }
    ))
    : GNN_PIPELINE_STAGES;
  const maxCount = stageCounts
    ? Math.max(1, ...order.map((stage) => Number(stageCounts[stage.id] || 0)))
    : 1;

  const banner = createGnn3dLabel(
    THREE,
    unavailable
      ? `수집→판별 파이프라인 · 측정 없음 (${(pipeline && pipeline.unavailable) || 'NO_DATA'})`
      : '수집 → 전략 판별 파이프라인 (단계별 탈락 수)',
    unavailable ? 0x8aa1b7 : 0x5eead4,
    { height: 30 },
  );
  banner.position.set(0, y + 78, 0);
  banner.material.opacity = .72;
  group.add(banner);

  const points = [];
  order.forEach((stage, index) => {
    const x = -spanX / 2 + (spanX * index) / Math.max(1, order.length - 1);
    const count = stageCounts ? Number(stageCounts[stage.id] || 0) : null;
    // sqrt keeps one huge stage from flattening every other marker to a dot.
    const radius = count === null ? 13 : 9 + 26 * Math.sqrt(count / maxCount);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(radius * .82, radius, 40),
      new THREE.MeshBasicMaterial({
        color: stage.color, transparent: true,
        opacity: count === null ? .22 : .55, side: THREE.DoubleSide, depthWrite: false,
      }),
    );
    ring.position.set(x, y, 0);
    group.add(ring);
    if (count) {
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(radius * .82, 32),
        new THREE.MeshBasicMaterial({ color: stage.color, transparent: true, opacity: .16, side: THREE.DoubleSide, depthWrite: false }),
      );
      disc.position.set(x, y, 0);
      group.add(disc);
    }
    const caption = createGnn3dLabel(
      THREE,
      count === null ? `${stage.label} · —` : `${stage.label} · ${count}`,
      stage.color,
      { height: 22 },
    );
    // Alternate the caption height so 15 labels in a row stay readable.
    caption.position.set(x, y - 40 - (index % 2) * 24, 0);
    caption.material.opacity = count === null ? .4 : .72;
    group.add(caption);
    points.push({ x, y, z: 0 });
  });

  const S = GNN3D_EDGE_SEGMENTS;
  const positions = new Float32Array(Math.max(1, points.length - 1) * S * 2 * 3);
  for (let index = 0; index < points.length - 1; index += 1) {
    writeGnn3dEdgeCurve(positions, index, points[index], points[index + 1], 0, -16, 0);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  group.add(new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
    color: 0x5eead4, transparent: true, opacity: unavailable ? .1 : .3, depthWrite: false,
  })));
  return group;
}

function gnn3dNodeColor(node) {
  return (gnnClusterStyle[node.family || node.cluster] || gnnClusterStyle.specialist).color;
}
function gnn3dEdgeColor(link) {
  const key = String(link.relation || '').startsWith('relation_encoder:') ? 'self_encoder_weight' : link.relation;
  return (gnnRelationStyle[key] || { color: '#7187a0' }).color;
}
const GNN3D_LABEL_FONT = 'bold 21px Inter, sans-serif';
let gnn3dLabelMeasure = null;

/**
 * A caption sprite whose texture is sized to the text it holds.
 *
 * The fixed 320px canvas silently clipped every long caption -- the market-physics
 * legend has always run past it -- and callers then stretched a 320px texture to an
 * arbitrary width, so identical text rendered at different letter widths depending
 * on which call site drew it. Width now follows the measured string and `height`
 * sets the scale, so every caption in the scene has the same glyph size.
 */
function createGnn3dLabel(THREE, text, color, { height = 20 } = {}) {
  const label = String(text ?? '');
  if (!gnn3dLabelMeasure) {
    gnn3dLabelMeasure = document.createElement('canvas').getContext('2d');
  }
  gnn3dLabelMeasure.font = GNN3D_LABEL_FONT;
  const width = Math.max(72, Math.ceil(gnn3dLabelMeasure.measureText(label).width) + 22);
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(3,7,13,.78)'; ctx.fillRect(0, 8, width, 46);
  ctx.strokeStyle = `#${color.toString(16).padStart(6, '0')}`; ctx.strokeRect(1, 9, width - 2, 44);
  ctx.fillStyle = '#e9fbff'; ctx.font = GNN3D_LABEL_FONT; ctx.fillText(label, 11, 39);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: .72, depthWrite: false }));
  sprite.scale.set((width / 64) * height, height, 1);
  return sprite;
}
function updateGnn3dTooltip(canvas, node, pointer) {
  const tooltip = document.getElementById('gnn-model-tooltip');
  if (!tooltip) return;
  const rect = canvas.getBoundingClientRect();
  tooltip.hidden = false;
  tooltip.style.left = `${Math.min(rect.width - 245, Math.max(8, (pointer.x + 1) * rect.width / 2 + 12))}px`;
  tooltip.style.top = `${Math.min(rect.height - 108, Math.max(8, (-pointer.y + 1) * rect.height / 2 + 12))}px`;
  tooltip.innerHTML = `<b>${escapeHtml(node.label)}</b><br>${gnn3dHoverDetail(node)}`;
}

/**
 * What a hovered body actually is. The tooltip used to say "3D compute node",
 * which told an operator nothing they could not see -- while every number the
 * layout positioned that body by was already on the node.
 */
function gnn3dHoverDetail(node) {
  const lines = [];
  const norm = node.parameterNorm;
  const radius = node.orbitRadius === null || node.orbitRadius === undefined
    ? '-'
    : Number(node.orbitRadius).toFixed(0);
  if (node.kind === 'strategy') {
    lines.push(`전략 행성 · ${(gnnClusterStyle[node.cluster] || gnnClusterStyle.specialist).label}`);
    lines.push(`헤드 강도 ${Number(node.learned_strength || 0).toFixed(3)} → 궤도 ${radius}`);
    lines.push(`학습 체결 ${formatInteger(node.training_filled_rows)}행 (질량)`);
    lines.push(node.training_positive_net_rate == null
      ? '학습 양수 순효율 미기록 → 이심률 기본값'
      : `양수 순효율 ${(Number(node.training_positive_net_rate) * 100).toFixed(1)}% → 이심률 ${Number(node.orbitEccentricity || 0).toFixed(2)}`);
    if (node.upside_supervised === false) lines.push('상승 학습 부족 → 양엣지 예보 억제');
  } else if (node.kind === 'output') {
    lines.push(`출력 헤드 위성 · ${node.channel || '-'}`);
    lines.push(`소속 전략 ${node.strategy_id || '-'}`);
    lines.push(norm === null || norm === undefined
      ? '헤드 열 노름 미측정 → 중간 궤도'
      : `헤드 열 노름 ${Number(norm).toFixed(3)} → 궤도 ${radius}`);
  } else if (node.kind === 'hidden') {
    lines.push('R-GCN 메시지 코어 유닛');
    lines.push(norm === null || norm === undefined
      ? '수신 인코더 노름 미측정'
      : `수신 인코더 노름 ${Number(norm).toFixed(3)}`);
    lines.push('활성 미계측 (HIDDEN_STATE_NOT_LOGGED)');
  } else {
    lines.push(node.cluster === 'input_identity' ? '입력 특징 · 종목 정체성 밴드' : '입력 특징 · 컨텍스트 밴드');
    lines.push(norm === null || norm === undefined
      ? '인코더 노름 미측정 → 중간 궤도'
      : `인코더 노름 ${Number(norm).toFixed(3)} → 궤도 ${radius}`);
    lines.push('활성 미계측 (ENCODER_INPUT_NOT_LOGGED)');
  }
  return lines.map((line) => escapeHtml(line)).join('<br>');
}

function queueGnnWave(direction) {
  gnnGraphView.waves.push({ direction, start: performance.now() });
  // A tab left open for hours polls continuously; bound the list so waves that
  // will never be drawn cannot accumulate.
  if (gnnGraphView.waves.length > 6) {
    gnnGraphView.waves.splice(0, gnnGraphView.waves.length - 6);
  }
}

function gnnQuadPoint(a, c, b, t) {
  const mt = 1 - t;
  return {
    x: mt * mt * a.x + 2 * mt * t * c.x + t * t * b.x,
    y: mt * mt * a.y + 2 * mt * t * c.y + t * t * b.y,
  };
}

function drawGnnThreadWave(ctx, a, c, b, waves, color, timestamp, gain) {
  const STEPS = 22;
  waves.forEach((wave) => {
    const progress = (timestamp - wave.start) / GNN_WAVE_MS;
    if (progress < 0 || progress > 1) return;
    // Backpropagation runs the other way. This reversal is the ONLY visual
    // difference between "data arrived" and "the model was corrected", and the
    // two mean opposite things, so it must not be cosmetic.
    const head = wave.direction === 'backward' ? 1 - progress : progress;
    const envelope = Math.sin(Math.PI * progress);
    const lo = Math.max(0, head - GNN_WAVE_WIDTH * 3);
    const hi = Math.min(1, head + GNN_WAVE_WIDTH * 3);
    if (hi - lo < 1e-3) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.globalAlpha = Math.min(1, .9 * envelope * gain);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let step = 0; step <= STEPS; step += 1) {
      const t = lo + (hi - lo) * (step / STEPS);
      const base = gnnQuadPoint(a, c, b, t);
      const mt = 1 - t;
      const tx = 2 * (mt * (c.x - a.x) + t * (b.x - c.x));
      const ty = 2 * (mt * (c.y - a.y) + t * (b.y - c.y));
      const length = Math.hypot(tx, ty) || 1;
      const offset = t - head;
      const packet = Math.exp(-(offset * offset) / (2 * GNN_WAVE_WIDTH * GNN_WAVE_WIDTH));
      // The thread is plucked, not lit: displacement, not just brightness.
      const amplitude = packet * Math.sin(offset * 42) * 7.5 * envelope * gain;
      const x = base.x - (ty / length) * amplitude;
      const y = base.y + (tx / length) * amplitude;
      if (step === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  });
}

function gnnMeasuredNodeIntensity(node) {
  const state = terminalState.gnnInference || {};
  if (!state.active) return 0;
  const activation = state.activation || {};
  if (node.kind === 'strategy') {
    return Math.min(1, Math.max(0, Number(activation.strategies?.[node.id]?.intensity || 0)));
  }
  if (node.kind === 'output' && node.strategy_id === activation.selected_strategy_id) {
    const value = activation.channels?.[node.channel];
    return value === undefined ? 0 : .35 + .65 * Math.min(1, Math.abs(Number(value)) / 2);
  }
  return 0;
}

function drawGnnGraph(timestamp) {
  if (!gnnVisualizationEnabled) { gnnGraphView.frame = null; return; }
  gnnGraphView.frame = requestAnimationFrame(drawGnnGraph);
  if (timestamp - gnnGraphView.lastPaint < 32) return;
  gnnGraphView.lastPaint = timestamp;
  const data = terminalState.gnnGraph;
  const canvas = document.getElementById('gnn-model-canvas');
  if (!data || !canvas) return;
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const baseScale = Math.min(width / 1200, height / 650);
  const scale = baseScale * gnnGraphView.zoom;
  const ox = (width - 1200 * baseScale) / 2 + gnnGraphView.panX;
  const oy = (height - 650 * baseScale) / 2 + gnnGraphView.panY;
  const point = (node) => ({ x: ox + node.x * scale, y: oy + node.y * scale });

  ctx.fillStyle = '#03060b';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = 'rgba(58, 82, 160, .28)';
  for (let x = 18; x < width; x += 38) for (let y = 18; y < height; y += 38) {
    ctx.beginPath(); ctx.arc(x, y, 1.2, 0, Math.PI * 2); ctx.fill();
  }

  Object.entries(gnnClusterStyle).filter(([clusterId]) => clusterId !== 'output').forEach(([clusterId, style]) => {
    if (!gnnGraphView.nodes.some((node) => node.cluster === clusterId)) return;
    const center = point(style);
    const radius = style.radius * scale;
    const glow = ctx.createRadialGradient(center.x, center.y, 0, center.x, center.y, radius);
    glow.addColorStop(0, `${style.color}22`); glow.addColorStop(.58, `${style.color}0b`); glow.addColorStop(1, `${style.color}00`);
    ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(center.x, center.y, radius, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `${style.color}aa`; ctx.font = `${Math.max(8, 9 * scale)}px Consolas, monospace`;
    // Count from the payload rather than the label, so a retrain that changes the
    // model's shape is reflected here instead of contradicting it.
    const members = gnnGraphView.nodes.filter((node) => node.cluster === clusterId).length;
    ctx.fillText(`${style.label} · ${members}`, center.x - radius * .55, center.y - radius * .72);
  });
  ctx.fillStyle = '#5eead4aa';
  ctx.font = `${Math.max(8, 9 * scale)}px Consolas, monospace`;
  const outputCount = gnnGraphView.nodes.filter((node) => node.cluster === 'output').length;
  ctx.fillText(
    `${gnnClusterStyle.output.label} · ${outputCount}`,
    ox + 855 * scale,
    oy + 30 * scale,
  );

  // Expire finished waves once per frame rather than per link, so every thread
  // in this frame is plucked by exactly the same set.
  gnnGraphView.waves = gnnGraphView.waves.filter((wave) => timestamp - wave.start < GNN_WAVE_MS);
  const waves = gnnGraphView.waves;

  const visibleLinks = (data.links || []).filter((link) => {
    if (gnnGraphView.filter === 'all') return true;
    if (gnnGraphView.filter === 'learned_parameter') return link.kind === 'learned_parameter';
    if (gnnGraphView.filter === 'strategy_topology') return !link.kind || link.kind === 'topology';
    return link.relation === gnnGraphView.filter;
  });
  visibleLinks.forEach((link) => {
    const source = gnnGraphView.nodeMap.get(link.source), target = gnnGraphView.nodeMap.get(link.target);
    if (!source || !target) return;
    const a = point(source), b = point(target);
    const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1;
    const seed = seededGraphUnit(`${link.source}:${link.target}`);
    const strength = Math.min(1, Math.max(0, Number(link.learned_strength || 0)));
    // A thread hangs. Sag is always toward screen +y and grows with the span,
    // which is the cue that reads as physical rope rather than a drawn arc. It
    // also carries meaning: slack is the INVERSE of learned strength. (A
    // quadratic dips half its control offset, so this is doubled to make the
    // number mean the visible sag.)
    const slack = 1 - strength;
    const sag = Math.min(58, distance * (.05 + .17 * slack)) * (.75 + .5 * seed) * 2;
    const sway = Math.sin(timestamp / 1600 + seed * 6.283) * sag * .1;
    // Kept from the old bend, reduced: threads sharing both endpoints would
    // otherwise be drawn exactly on top of one another.
    const spread = (seed > .5 ? 1 : -1) * Math.min(16, distance * .05);
    const cx = (a.x + b.x) / 2 - dy / distance * spread;
    const cy = (a.y + b.y) / 2 + dx / distance * spread + sag + sway;
    const relationKey = String(link.relation || '').startsWith('relation_encoder:') ? 'self_encoder_weight' : link.relation;
    const relation = gnnRelationStyle[relationKey] || { color: '#8aa1b7' };
    const sourceIntensity = gnnMeasuredNodeIntensity(source);
    const targetIntensity = gnnMeasuredNodeIntensity(target);
    const inferenceIntensity = Math.min(sourceIntensity, targetIntensity)
      * (.35 + strength * .65);
    const marketIntensity = (!link.kind || link.kind === 'topology')
      ? Number(data.forces?.marketEnergy || 0) * (.16 + strength * .24) : 0;
    const edgeIntensity = Math.max(inferenceIntensity, marketIntensity);
    const active = edgeIntensity > .02;
    ctx.save();
    ctx.strokeStyle = relation.color;
    const parameter = link.kind === 'learned_parameter';
    ctx.globalAlpha = active
      ? Math.min(1, .34 + edgeIntensity * .66)
      : (parameter ? .018 + strength * .075 : .055 + strength * .16);
    ctx.lineWidth = active ? 1.4 + edgeIntensity * 2.2
      : parameter ? .25 + strength * .48 : .45 + strength * 1.05;
    if (active) { ctx.shadowBlur = 18 + edgeIntensity * 16; ctx.shadowColor = relation.color; }
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(cx, cy, b.x, b.y); ctx.stroke();
    ctx.restore();
    // Waves ride the topology and whatever is currently firing. Plucking every
    // parameter edge each frame would cost far more than it shows.
    if (waves.length && (active || !parameter)) {
      drawGnnThreadWave(
        ctx, a, { x: cx, y: cy }, b, waves, relation.color, timestamp,
        active ? 1 : .45 + strength * .35,
      );
    }
  });

  const pulse = .5 + Math.sin(timestamp / 240) * .5;
  gnnGraphView.nodes.forEach((node) => {
    const p = point(node);
    const family = node.family || node.cluster;
    const style = gnnClusterStyle[family] || gnnClusterStyle.specialist;
    const baseRadius = node.kind === 'strategy' ? 7.5 : node.kind === 'hidden' ? 4.2 : node.kind === 'feature' ? 3.2 : 2.8;
    const intensity = gnnMeasuredNodeIntensity(node);
    const radius = (baseRadius + Number(node.learned_strength || 0) * (node.kind === 'strategy' ? 5.5 : 1.8)
      + intensity * (2.5 + pulse * 4.2)) * Math.max(.72, scale);
    ctx.save();
    ctx.fillStyle = style.color; ctx.shadowColor = style.color; ctx.shadowBlur = intensity > 0 ? 28 + pulse * 22 : 7;
    ctx.globalAlpha = node === gnnGraphView.hovered ? 1 : .9;
    ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fill();
    ctx.lineWidth = node === gnnGraphView.selected ? 2.4 : 1;
    ctx.strokeStyle = node === gnnGraphView.selected ? '#ffffff' : '#071019'; ctx.stroke();
    ctx.shadowBlur = 0; ctx.fillStyle = '#dce9f4'; ctx.globalAlpha = .92;
    ctx.font = `${Math.max(7, 8 * scale)}px Inter, sans-serif`;
    if (node.kind === 'strategy' || node === gnnGraphView.hovered || node === gnnGraphView.selected || gnnGraphView.zoom > 1.75) {
      ctx.fillText(shortGnnLabel(node.label), p.x + radius + 4, p.y + 3);
    }
    ctx.restore();
  });
  gnnGraphView.transform = { scale, ox, oy, width, height };
}

function bindGnnGraphCanvas() {
  const canvas = document.getElementById('gnn-model-canvas');
  if (!canvas || canvas.dataset.gnnBound === 'true') return;
  canvas.dataset.gnnBound = 'true';
  const localPoint = (event) => {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };
  const pick = (screen) => {
    const t = gnnGraphView.transform;
    if (!t) return null;
    let result = null, best = Infinity;
    gnnGraphView.nodes.forEach((node) => {
      const x = t.ox + node.x * t.scale, y = t.oy + node.y * t.scale;
      const distance = Math.hypot(screen.x - x, screen.y - y);
      if (distance < 15 && distance < best) { result = node; best = distance; }
    });
    return result;
  };
  canvas.addEventListener('pointerdown', (event) => {
    gnnGraphView.dragging = true; gnnGraphView.moved = false;
    gnnGraphView.lastX = event.clientX; gnnGraphView.lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', (event) => {
    if (gnnGraphView.dragging) {
      const dx = event.clientX - gnnGraphView.lastX, dy = event.clientY - gnnGraphView.lastY;
      if (Math.abs(dx) + Math.abs(dy) > 1) gnnGraphView.moved = true;
      gnnGraphView.panX += dx; gnnGraphView.panY += dy;
      gnnGraphView.lastX = event.clientX; gnnGraphView.lastY = event.clientY;
      return;
    }
    gnnGraphView.hovered = pick(localPoint(event));
    updateGnnTooltip(event, gnnGraphView.hovered);
  });
  canvas.addEventListener('pointerup', (event) => {
    if (!gnnGraphView.moved) {
      const node = pick(localPoint(event));
      if (node) { gnnGraphView.selected = node; renderGnnInspector(node); }
    }
    gnnGraphView.dragging = false;
  });
  canvas.addEventListener('pointerleave', () => {
    gnnGraphView.dragging = false; gnnGraphView.hovered = null;
    document.getElementById('gnn-model-tooltip').hidden = true;
  });
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault();
    gnnGraphView.zoom = Math.max(.55, Math.min(2.8, gnnGraphView.zoom * (event.deltaY > 0 ? .9 : 1.1)));
  }, { passive: false });
}

function resetGnnGraphView() {
  gnnGraphView.zoom = 1; gnnGraphView.panX = 0; gnnGraphView.panY = 0;
  if (gnn3dState?.resetView) gnn3dState.resetView();
}
function updateGnnTooltip(event, node) {
  const tooltip = document.getElementById('gnn-model-tooltip');
  if (!node) { tooltip.hidden = true; return; }
  const stage = document.querySelector('.gnn-model-stage').getBoundingClientRect();
  tooltip.hidden = false;
  tooltip.style.left = `${Math.min(stage.width - 245, Math.max(8, event.clientX - stage.left + 12))}px`;
  tooltip.style.top = `${Math.min(stage.height - 70, Math.max(8, event.clientY - stage.top + 12))}px`;
  const detail = node.kind === 'strategy'
    ? `학습 강도 ${Number(node.learned_strength || 0).toFixed(3)} · 연결도 ${(Number(node.connectivity || 0) * 100).toFixed(0)}% · 추론 ${formatInteger(node.inference_count)}회`
    : `${String(node.layer || node.kind || '').toUpperCase()} · 연결도 ${(Number(node.connectivity || 0) * 100).toFixed(0)}% · ${formatInteger(node.edge_count)}개 엣지`;
  tooltip.innerHTML = `<b>${escapeHtml(node.label)}</b><br>${escapeHtml(detail)}`;
}
function renderGnnInspector(node) {
  if (node.kind !== 'strategy') {
    const details = node.kind === 'feature'
      ? [['계산 계층', '입력 특징'], ['특징 인덱스', node.feature_index], ['입력 차원', terminalState.gnnGraph?.model?.feature_dim]]
      : node.kind === 'hidden'
        ? [['계산 계층', 'R-GCN 메시지 은닉층'], ['은닉 인덱스', node.hidden_index], ['은닉 차원', terminalState.gnnGraph?.model?.hidden_dim]]
        : [['계산 계층', '전략별 출력 헤드'], ['전략', node.strategy_id], ['출력 채널', node.channel], ['채널 인덱스', node.channel_index]];
    details.push(
      ['가중 연결도', `${(Number(node.connectivity || 0) * 100).toFixed(1)}%`],
      ['연결 / 관계 종류', `${formatInteger(node.edge_count)} / ${formatInteger(node.relation_diversity)}`],
      ['기본 각속도', node.orbitAngularSpeed == null ? '-' : `${Number(node.orbitAngularSpeed).toFixed(4)} rad/s`],
    );
    document.getElementById('gnn-model-inspector').innerHTML = `
      <span>CHECKPOINT COMPUTE NODE</span><h3>${escapeHtml(node.label)}</h3>
      <p>저장된 체크포인트 텐서에서 복원한 실제 계산 노드입니다.</p>
      <dl>${details.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value ?? '-')}</dd></div>`).join('')}</dl>`;
    return;
  }
  const cluster = gnnClusterStyle[node.cluster] || gnnClusterStyle.specialist;
  document.getElementById('gnn-model-inspector').innerHTML = `
    <span>TRAINED STRATEGY NODE</span><h3>${escapeHtml(node.label)}</h3>
    <p>${cluster.label} 군집 · 체크포인트 전략 인덱스 ${Number(node.checkpoint_index) + 1}</p>
    <dl><div><dt>학습 헤드 강도</dt><dd>${Number(node.learned_strength || 0).toFixed(4)}</dd></div>
    <div><dt>가중 연결도</dt><dd>${(Number(node.connectivity || 0) * 100).toFixed(1)}% · ${formatInteger(node.edge_count)} 엣지</dd></div>
    <div><dt>관계 다양성 / 세차</dt><dd>${formatInteger(node.relation_diversity)}종 · ${node.orbitPrecession == null ? '-' : Number(node.orbitPrecession).toFixed(4)}</dd></div>
    <div><dt>기본 각속도 / 관성</dt><dd>${node.orbitAngularSpeed == null ? '-' : Number(node.orbitAngularSpeed).toFixed(4)} · ${node.inertialMass == null ? '-' : Number(node.inertialMass).toFixed(3)}</dd></div>
    <div><dt>학습 라벨</dt><dd>${formatInteger(node.training_labels)} <small>(체결 ${formatInteger(node.training_filled_rows)})</small></dd></div>
    <div><dt>상승(MFE) 학습</dt><dd>${gnnUpsideSupervisionLabel(node)}</dd></div>
    <div><dt>학습 양수 순효율</dt><dd>${node.training_positive_net_rate == null ? '-' : `${(Number(node.training_positive_net_rate) * 100).toFixed(1)}%`}</dd></div>
    <div><dt>최근 GNN 추론</dt><dd>${formatInteger(node.inference_count)}회</dd></div>
    <div><dt>최근 효용</dt><dd>${node.latest_utility == null ? '-' : Number(node.latest_utility).toFixed(3)}</dd></div>
    <div><dt>예상 순효율</dt><dd>${node.latest_expected_net_bps == null ? '-' : `${Number(node.latest_expected_net_bps).toFixed(2)} bp`}</dd></div></dl>`;
}
function gnnUpsideSupervisionLabel(node) {
  // ``학습 라벨`` counts snapshots, so it reads high for every strategy including
  // ones that never triggered. The rows that trained the MFE channel are the only
  // ones behind a positive net-edge forecast, and when they are short the runtime
  // suppresses that forecast entirely — say so here rather than letting a big
  // label count imply the node is ready.
  const rows = node.training_upside_rows;
  if (rows == null) return '알 수 없음 (체크포인트 미기록)';
  const minimum = Number(node.minimum_upside_rows || 20);
  return node.upside_supervised
    ? `${formatInteger(rows)}/${minimum} · 학습됨`
    : `${formatInteger(rows)}/${minimum} · 부족 → 양엣지 예보 억제`;
}
function shortGnnLabel(value) {
  const words = String(value || '').split(' ');
  return words.length > 2 ? `${words[0]} ${words[1]}` : String(value || '');
}
function seededGraphUnit(value) {
  let hash = 2166136261;
  for (const char of String(value)) { hash ^= char.charCodeAt(0); hash = Math.imul(hash, 16777619); }
  return (hash >>> 0) / 4294967295;
}

const strategyLabels = {
  intraday_momentum: '장중 모멘텀',
  breakout_volume: '거래량 돌파',
  vwap_mean_reversion: 'VWAP 평균회귀',
  bar_confirmed_vwap_recovery: '1분봉 확인 VWAP 회복',
  liquidity_shock_reversal: '유동성 충격 반전',
  event_momentum: '이벤트 모멘텀',
  cross_sectional_relative_strength: '횡단면 상대강도',
  gap_context: '갭 컨텍스트',
};

async function fetchMarketView(symbol = null) {
  if (terminalState.busy) return;
  terminalState.busy = true;
  const refresh = document.getElementById('terminal-refresh');
  if (refresh) refresh.classList.add('loading');
  try {
    const query = symbol ? `?symbol=${encodeURIComponent(symbol)}&limit=180` : '?limit=180';
    const response = await fetch(`/api/refactor/market-view${query}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json();
    const symbolChanged = terminalState.symbol && terminalState.symbol !== data.symbol;
    terminalState.data = data;
    terminalState.symbol = data.symbol;
    registerSecondBarArrival(data.market || {}, Boolean(symbolChanged));
    refreshGnnMarketForces();
    renderTerminal(data);
  } catch (error) {
    document.getElementById('feed-state').textContent = `데이터 오류 · ${error.message}`;
    document.querySelector('.connection')?.classList.add('stale');
  } finally {
    terminalState.busy = false;
    if (refresh) refresh.classList.remove('loading');
  }
}

async function fetchAssetSummary() {
  if (terminalState.assetBusy) return;
  terminalState.assetBusy = true;
  try {
    const response = await fetch('/api/account/summary', { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    renderAssetSummary(await response.json());
  } catch (error) {
    renderAssetSummary({
      status: 'unavailable',
      message: `계좌 정보를 불러오지 못했습니다: ${error.message}`,
      snapshot: null,
    });
  } finally {
    terminalState.assetBusy = false;
  }
}

async function fetchMarketStream() {
  if (terminalState.streamBusy || !terminalState.symbol || !terminalState.data) return;
  terminalState.streamBusy = true;
  try {
    const response = await fetch(
      `/api/refactor/market-stream?symbol=${encodeURIComponent(terminalState.symbol)}&limit=30`,
      { cache: 'no-store' },
    );
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const stream = await response.json();
    if (stream.symbol !== terminalState.symbol || !stream.market) return;
    terminalState.data.market = {
      ...(terminalState.data.market || {}),
      ...stream.market,
    };
    terminalState.data.generated_at = stream.generated_at || terminalState.data.generated_at;
    registerSecondBarArrival(terminalState.data.market);
    refreshGnnMarketForces();
    renderHeader(terminalState.data, terminalState.data.market);
    renderInstrument(
      terminalState.symbol,
      terminalState.data.market,
      terminalState.data.selection || {},
      terminalState.data.algorithm || null,
    );
    renderTradingChart(terminalState.data.market, terminalState.data.algorithm || null);
    renderTradingLayers(terminalState.data);
    renderSecondAnalysis(terminalState.data.market);
    renderExecution(
      terminalState.data.execution || {},
      terminalState.data.market.latest_orderbook,
    );
    setChartStreamState(true, terminalState.data.market);
  } catch (error) {
    setChartStreamState(false, null, error.message);
  } finally {
    terminalState.streamBusy = false;
  }
}

async function fetchSystemDiagnostics() {
  if (terminalState.diagnosticsBusy) return;
  terminalState.diagnosticsBusy = true;
  try {
    const response = await fetch('/api/system-diagnostics', { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    renderSystemDiagnostics(await response.json());
  } catch (error) {
    renderSystemDiagnostics({
      headline: '시스템 진단 정보를 불러오지 못했습니다.',
      summary: error.message,
      workers: [],
      blockers: [{ label: '진단 API', message: '상태 조회 실패', detail: error.message }],
      files: {},
      recent_activity: [],
    });
  } finally {
    terminalState.diagnosticsBusy = false;
  }
}

function formatOperatingModelMetric(activeModel, metric, digits = 4) {
  const rawDirect = activeModel?.metrics?.[metric];
  const direct = rawDirect == null ? Number.NaN : Number(rawDirect);
  if (Number.isFinite(direct)) return direct.toFixed(digits);

  const marketValues = Object.entries(activeModel?.market_models || {})
    .map(([market, report]) => {
      const rawValue = report?.metrics?.[metric];
      const value = rawValue == null ? Number.NaN : Number(rawValue);
      return Number.isFinite(value) ? `${market} ${value.toFixed(digits)}` : null;
    })
    .filter(Boolean);
  return marketValues.length ? marketValues.join(' · ') : 'N/A';
}

function renderSystemDiagnostics(data) {
  terminalState.diagnostics = data;
  const score = Number(data.score || 0);
  const threshold = Number(data.threshold || .9);
  const scorePercent = Math.max(0, Math.min(100, score * 100));
  document.getElementById('diagnostics-summary').textContent =
    `${data.headline || '상태 확인 중'} ${data.summary || ''}`.trim();
  document.getElementById('diagnostics-mode').textContent =
    String(data.mode || 'unknown').toUpperCase();
  document.getElementById('diagnostics-score').textContent =
    `${scorePercent.toFixed(0)}%`;
  document.getElementById('diagnostics-threshold').textContent =
    `인프라 신뢰도 기준 ${(threshold * 100).toFixed(0)}%`;
  document.getElementById('diagnostics-progress-bar').style.width = `${scorePercent}%`;

  const workers = data.workers || [];
  const runningWorkers = workers.filter((worker) => worker.running).length;
  document.getElementById('diagnostics-worker-count').textContent =
    `${runningWorkers}/${workers.length} 실행`;
  document.getElementById('diagnostics-workers').innerHTML = workers.length
    ? workers.map((worker) => `
      <div class="worker-card ${worker.running ? 'running' : ''}">
        <span class="worker-state">${worker.running ? 'RUNNING' : 'WAITING'}</span>
        <b>${escapeHtml(worker.label || worker.key || '-')}</b>
        <small>${escapeHtml(worker.detail || '-')}</small>
      </div>
    `).join('')
    : '<div class="blocker-clear">워커 상태가 없습니다.</div>';

  const blockers = data.blockers || [];
  document.getElementById('diagnostics-blocker-count').textContent =
    blockers.length ? `${blockers.length}개 차단` : '모두 통과';
  document.getElementById('diagnostics-blockers').innerHTML = blockers.length
    ? blockers.map((item) => `
      <div class="blocker-card">
        <b>${escapeHtml(item.label || item.code || '-')}</b>
        <span>${escapeHtml(item.message || '-')}</span>
        <small>${escapeHtml(item.detail || item.code || '-')}</small>
      </div>
    `).join('')
    : '<div class="blocker-clear">인프라 신뢰도 차단 사유가 없습니다. 실제 진입 가능 여부는 별도 단계에서 확인합니다.</div>';

  const flows = data.flows || {};
  const researchCounts = (flows.research_collection || {}).counts || {};
  const training = flows.training || {};
  const metrics = training.metrics || {};
  const activeModel = training.active_model || {};
  const deployment = training.deployment || {};
  const market = flows.market_data || {};
  const healthy = market.healthy || {};
  const account = data.account_context || {};
  const files = data.files || {};
  const evidence = [
    [
      '운영 모델',
      `AUC ${formatOperatingModelMetric(activeModel, 'auc')}`,
      `Precision@K ${formatOperatingModelMetric(activeModel, 'precision_at_k')} · 실거래 적용 중`,
    ],
    [
      '모델 갱신 상태',
      training.training_skipped ? '새 라벨 대기' : (deployment.deployed ? '개선 모델 교체' : '기존 모델 유지'),
      deployment.reason || (training.training_skipped ? '동일 데이터 반복 학습 생략' : '후보 모델 검증 중'),
    ],
    [
      '누적 학습 행',
      formatInteger(metrics.materialized_training_rows || metrics.training_rows || metrics.example_count),
      `모델 윈도우 ${formatInteger(metrics.training_rows || metrics.example_count)}행 · 이번 주기 성숙 라벨 ${formatInteger(metrics.fresh_training_rows || 0)}개`,
    ],
    [
      '연구 데이터 수집',
      flows.research_collection?.active ? '진행 중' : '중지',
      `이벤트 ${formatInteger(researchCounts.events || researchCounts.events_seen)} · 원문 ${formatInteger(researchCounts.raw_records || researchCounts.raw_records_seen)}`,
    ],
    [
      '시장 스냅샷',
      formatInteger(researchCounts.market_snapshots || researchCounts.market_snapshots_seen),
      `연구 DB ${formatFileSize(files.research_store?.size_bytes)} · ${formatAge(files.research_store?.age_seconds)}`,
    ],
    [
      '새 실시간 학습 프레임',
      formatInteger(training.feature_frames_built),
      `저널 ${formatFileSize(files.feature_journal?.size_bytes)} · ${formatAge(files.feature_journal?.age_seconds)}`,
    ],
    [
      '최신 모델',
      `AUC ${Number(metrics.auc || 0).toFixed(4)}`,
      `Precision@K ${Number(metrics.precision_at_k || 0).toFixed(4)} · ${metrics.live_eligible ? '승격 가능' : '기준 미달'}`,
    ],
    [
      '거래급 시세 종목',
      `KRX ${(healthy.KRX || []).length} · US ${(healthy.US || []).length}`,
      `시장별 최소 ${market.minimum_per_market || '-'}개 필요`,
    ],
    [
      '주문 가능 현금',
      `KRW ${formatCompactNumber(account.krw_orderable)}`,
      `USD ${formatCompactNumber(account.usd_orderable)}${account.us_collection_limited_by_cash ? ' · 미국 후보 제한' : ''}`,
    ],
    [
      '실시간 저장소',
      formatFileSize(files.realtime_store?.size_bytes),
      formatAge(files.realtime_store?.age_seconds),
    ],
    [
      '다음 연구 수집',
      data.next_collection_at ? shortDateTime(data.next_collection_at) : '-',
      `상태 생성 ${shortDateTime(data.generated_at)}`,
    ],
  ];
  document.getElementById('diagnostics-evidence').innerHTML = evidence.map(([label, value, note]) => `
    <div class="evidence-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(note)}</small>
    </div>
  `).join('');

  const activity = [...(data.recent_activity || [])].reverse().slice(0, 8);
  document.getElementById('diagnostics-activity').innerHTML = activity.length
    ? activity.map((item) => `
      <div class="activity-row">
        <time>${escapeHtml(shortClock(item.timestamp))}</time>
        <b>${escapeHtml(item.status || '-')}</b>
        <span title="${escapeHtml(item.message || '')}">${escapeHtml(item.message || '-')}</span>
      </div>
    `).join('')
    : '<div class="blocker-clear">최근 활동 기록이 없습니다.</div>';
  document.getElementById('diagnostics-generated-at').textContent =
    `갱신 ${shortClock(data.generated_at)}`;
  document.getElementById('diagnostics-next-run').textContent =
    data.next_collection_at ? `다음 ${shortDateTime(data.next_collection_at)}` : '다음 일정 없음';
  renderTrainingMonitor((data.flows || {}).training || {});
}

function renderTrainingMonitor(training) {
  const history = training.history || {};
  const points = Array.isArray(history.points) ? history.points : [];
  const latest = history.latest || points.at(-1) || {};
  const change = history.change || {};
  const optimizer = history.optimizer || {};
  const state = document.getElementById('training-cycle-state');
  if (!state) return;

  const statusLabels = {
    waiting: 'WAITING',
    collecting: 'COLLECTING',
    promoted: 'MODEL PROMOTED',
    evaluated_improved: 'IMPROVED',
    evaluated: 'EVALUATED',
  };
  state.textContent = statusLabels[history.status] || String(history.status || 'WAITING').toUpperCase();
  document.getElementById('training-last-cycle').textContent =
    latest.timestamp ? `최근 학습 ${shortDateTime(latest.timestamp)}` : '학습 기록 없음';
  document.getElementById('training-monitor-note').textContent =
    history.note || '완료된 실제 학습 사이클을 표시합니다.';

  const learningRate = Number(optimizer.classification_learning_rate);
  document.getElementById('training-learning-rate').textContent =
    Number.isFinite(learningRate) ? learningRate.toFixed(4) : '-';
  document.getElementById('training-optimizer').textContent = optimizer.classification_family
    ? `${String(latest.training_mode || 'full').toUpperCase()} · ${optimizer.classification_family} · ${formatInteger(optimizer.classification_epochs)} epochs · L2 ${optimizer.l2}`
    : '옵티마이저 기록 없음';

  const aucChange = Number(change.auc || 0);
  const aucChangeNode = document.getElementById('training-auc-change');
  aucChangeNode.textContent = points.length > 1 ? formatSignedMetric(aucChange, 4) : '-';
  aucChangeNode.className = aucChange > 0 ? 'positive' : aucChange < 0 ? 'negative' : '';
  document.getElementById('training-current-auc').textContent =
    `현재 AUC ${Number(latest.auc || 0).toFixed(4)} · Precision ${Number(latest.precision_at_k || 0).toFixed(4)}`;

  const rowsPerHour = Number(history.rows_per_hour || 0);
  document.getElementById('training-row-rate').textContent =
    `${rowsPerHour.toFixed(rowsPerHour >= 10 ? 0 : 1)} 행/시간`;
  document.getElementById('training-new-rows').textContent =
    `${latest.training_mode === 'incremental' ? '증분 학습' : '전체 학습'} ${formatInteger(latest.incremental_rows || latest.training_rows || 0)}행 · 누적 ${formatInteger(latest.materialized_rows || latest.training_rows || 0)}행 · 모델 윈도우 ${formatInteger(latest.training_rows || 0)}행`;

  const promoted = Boolean(latest.promoted);
  const eligible = Boolean(latest.live_eligible);
  const deploymentNode = document.getElementById('training-deployment-state');
  deploymentNode.textContent = promoted ? '교체 완료' : eligible ? '승격 후보' : '기존 모델 유지';
  deploymentNode.className = promoted || eligible ? 'positive' : '';
  document.getElementById('training-deployment-reason').textContent =
    latest.deployment_reason || '평가 기록 없음';

  drawTrainingPerformanceChart(points);
  drawTrainingDataChart(points);
  const cycles = points.slice(-10).reverse();
  document.getElementById('training-cycle-list').innerHTML = cycles.length
    ? cycles.map((point) => `
      <article class="training-cycle ${point.promoted ? 'promoted' : point.live_eligible ? 'eligible' : ''}">
        <time>${escapeHtml(shortClock(point.timestamp))}</time>
        <b>AUC ${Number(point.auc || 0).toFixed(4)}</b>
        <small>${String(point.training_mode || 'full').toUpperCase()} ${formatInteger(point.incremental_rows || point.training_rows || 0)}행 · P@K ${Number(point.precision_at_k || 0).toFixed(3)}</small>
        <small>${point.promoted ? '모델 교체' : point.live_eligible ? '승격 가능' : escapeHtml(point.deployment_reason || '평가 완료')}</small>
      </article>
    `).join('')
    : '<div class="blocker-clear">아직 완료된 학습 사이클이 없습니다.</div>';
}

function drawTrainingPerformanceChart(points) {
  const canvas = document.getElementById('training-performance-chart');
  if (!canvas) return;
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 34, right: 36, top: 12, bottom: 22 };
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = Math.max(1, height - pad.top - pad.bottom);
  drawTrainingGrid(ctx, width, height, pad, ['1.00', '0.75', '0.50', '0.25', '0.00']);
  if (!points.length) {
    drawEmptyTrainingChart(ctx, width, height);
    return;
  }
  const xAt = (index) => pad.left + (points.length === 1 ? plotWidth / 2 : index * plotWidth / (points.length - 1));
  const scoreY = (value) => pad.top + (1 - Math.max(0, Math.min(1, Number(value) || 0))) * plotHeight;
  drawTrainingLine(ctx, points, xAt, (point) => scoreY(point.auc), '#39d7e7');
  drawTrainingLine(ctx, points, xAt, (point) => scoreY(point.precision_at_k), '#9b8cff');

  const returns = points.map((point) => Number(point.top_return_bps || 0));
  let returnMin = Math.min(0, ...returns);
  let returnMax = Math.max(0, ...returns);
  if (returnMax === returnMin) returnMax = returnMin + 1;
  const returnY = (value) => pad.top + (returnMax - Number(value || 0)) / (returnMax - returnMin) * plotHeight;
  drawTrainingLine(ctx, points, xAt, (point) => returnY(point.top_return_bps), '#f3b95f', [4, 3]);
  ctx.fillStyle = '#708196';
  ctx.font = '7px Consolas';
  ctx.textAlign = 'left';
  ctx.fillText(`${returnMax.toFixed(1)}bp`, width - pad.right + 3, pad.top + 3);
  ctx.fillText(`${returnMin.toFixed(1)}bp`, width - pad.right + 3, height - pad.bottom);
  drawTrainingTimeLabels(ctx, points, xAt, height);
}

function drawTrainingDataChart(points) {
  const canvas = document.getElementById('training-data-chart');
  if (!canvas) return;
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 42, right: 12, top: 12, bottom: 22 };
  const plotWidth = Math.max(1, width - pad.left - pad.right);
  const plotHeight = Math.max(1, height - pad.top - pad.bottom);
  if (!points.length) {
    drawEmptyTrainingChart(ctx, width, height);
    return;
  }
  const rowValues = points.map((point) => Number(point.materialized_rows || point.training_rows || 0));
  const newValues = points.map((point) => Number(point.new_rows || 0));
  const rowMin = Math.min(...rowValues);
  const rowMax = Math.max(...rowValues);
  const rowSpan = Math.max(1, rowMax - rowMin);
  const newMax = Math.max(1, ...newValues);
  const xAt = (index) => pad.left + (points.length === 1 ? plotWidth / 2 : index * plotWidth / (points.length - 1));
  const rowY = (value) => pad.top + (rowMax - Number(value || 0)) / rowSpan * plotHeight;
  const barWidth = Math.max(2, Math.min(9, plotWidth / Math.max(1, points.length) * .55));
  ctx.fillStyle = 'rgba(57, 215, 231, .25)';
  points.forEach((point, index) => {
    const barHeight = Number(point.new_rows || 0) / newMax * plotHeight * .42;
    ctx.fillRect(xAt(index) - barWidth / 2, pad.top + plotHeight - barHeight, barWidth, barHeight);
  });
  drawTrainingLine(ctx, points, xAt, (point) => rowY(point.materialized_rows || point.training_rows), '#42d392');
  ctx.fillStyle = '#708196';
  ctx.font = '7px Consolas';
  ctx.textAlign = 'right';
  ctx.fillText(formatInteger(rowMax), pad.left - 4, pad.top + 3);
  ctx.fillText(formatInteger(rowMin), pad.left - 4, height - pad.bottom);
  ctx.strokeStyle = '#1c2937';
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();
  drawTrainingTimeLabels(ctx, points, xAt, height);
}

function drawTrainingGrid(ctx, width, height, pad, labels) {
  const plotHeight = height - pad.top - pad.bottom;
  ctx.font = '7px Consolas';
  ctx.textAlign = 'right';
  labels.forEach((label, index) => {
    const y = pad.top + index * plotHeight / (labels.length - 1);
    ctx.strokeStyle = '#172330';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillStyle = '#66778c';
    ctx.fillText(label, pad.left - 5, y + 2);
  });
}

function drawTrainingLine(ctx, points, xAt, yAt, color, dash = []) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.setLineDash(dash);
  ctx.beginPath();
  points.forEach((point, index) => {
    const x = xAt(index);
    const y = yAt(point);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function drawTrainingTimeLabels(ctx, points, xAt, height) {
  const indices = [...new Set([0, Math.floor((points.length - 1) / 2), points.length - 1])];
  ctx.fillStyle = '#66778c';
  ctx.font = '7px Consolas';
  indices.forEach((index) => {
    ctx.textAlign = index === 0 ? 'left' : index === points.length - 1 ? 'right' : 'center';
    ctx.fillText(shortClock(points[index]?.timestamp), xAt(index), height - 6);
  });
}

function drawEmptyTrainingChart(ctx, width, height) {
  ctx.fillStyle = '#66778c';
  ctx.font = '9px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('학습 이력 대기 중', width / 2, height / 2);
}

function formatSignedMetric(value, digits = 4) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return `${number > 0 ? '+' : ''}${number.toFixed(digits)}`;
}

function renderAssetSummary(data) {
  const snapshot = data.snapshot || null;
  const live = data.status === 'live' && data.authoritative;
  const lastKnown = data.status === 'last_known';
  const status = document.getElementById('asset-status');
  status.textContent = live ? '실계좌 LIVE' : lastKnown ? '마지막 확인값' : '확인 불가';
  status.className = live ? 'status-chip' : 'status-chip blocked';
  document.getElementById('asset-verified-at').textContent = snapshot
    ? `마지막 확인 ${formatTime(data.last_verified_at)}`
    : '마지막 확인 -';
  document.getElementById('asset-source').textContent = data.message || '계좌 상태를 확인할 수 없습니다.';
  document.getElementById('asset-total').textContent = snapshot ? formatKrw(snapshot.total_asset_krw) : '-';
  document.getElementById('asset-cash').textContent = snapshot ? formatKrw(snapshot.cash_equivalent_krw) : '-';
  document.getElementById('asset-cash-detail').textContent = snapshot
    ? `원화 ${formatKrw(snapshot.krw_cash)} · 외화 ${formatKrw(snapshot.foreign_cash_krw)}`
    : '원화 - · 외화 -';
  const stockValue = snapshot
    ? Number(snapshot.domestic_stock_value_krw || 0) + Number(snapshot.overseas_stock_value_krw || 0)
    : null;
  document.getElementById('asset-stocks').textContent = snapshot ? formatKrw(stockValue) : '-';
  document.getElementById('asset-stock-detail').textContent = snapshot
    ? `국내 ${formatKrw(snapshot.domestic_stock_value_krw)} · 해외 ${formatKrw(snapshot.overseas_stock_value_krw)}`
    : '국내 - · 해외 -';
  const pnl = snapshot ? Number(snapshot.total_pnl_krw || 0) : null;
  const pnlElement = document.getElementById('asset-pnl');
  pnlElement.textContent = snapshot ? formatSignedKrw(pnl) : '-';
  pnlElement.className = snapshot ? (pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : '') : '';
  document.getElementById('asset-pnl-rate').textContent = snapshot
    ? `수익률 ${formatPercent(snapshot.total_pnl_rate)}`
    : '수익률 -';
  const warning = document.getElementById('asset-warning');
  warning.textContent = live ? '' : (data.message || '실계좌 연결 상태를 확인해 주세요.');
  warning.hidden = live;
  document.querySelector('.asset-overview')?.classList.toggle('asset-degraded', !live);
}

function renderTerminal(data) {
  const market = data.market || {};
  const selection = data.selection || {};
  const algorithm = data.algorithm || null;
  renderHeader(data, market);
  renderLiveOwner(data);
  renderCandidates(data.candidates || [], data.strategy_session || {});
  renderInstrument(data.symbol, market, selection, algorithm);
  renderTradingChart(market, algorithm);
  renderTradingLayers(data);
  renderSecondAnalysis(market);
  renderDecisionOntology(data.decision_ontology || {});
  renderOntology(selection, market);
  renderExecution(data.execution || {}, market.latest_orderbook);
  renderSafety(data);
}

function renderLiveOwner(data) {
  const live = data.live_trading || {};
  const session = data.strategy_session || {};
  const execution = data.execution || {};
  const events = execution.events || [];
  const latestEvent = events[0] || {};
  const latestPayload = latestEvent.payload || {};
  const adopted = Boolean(live.adopted && live.symbol && live.strategy_id);
  const running = Boolean(live.engine_running);
  const capable = Boolean(data.live_order_capable);
  const container = document.querySelector('.live-owner-strip');
  container?.classList.toggle('scanning', running && !adopted);
  container?.classList.toggle('blocked', !running || !capable);

  document.getElementById('live-owner-state').textContent = running
    ? (adopted ? String(live.phase || 'ACTIVE') : 'SCANNING')
    : 'ENGINE OFFLINE';
  document.getElementById('live-owner-session').textContent = adopted
    ? `SESSION ${String(live.session_id || '-').slice(-12)}`
    : (live.last_reason || live.buy_disabled_reason || '채택 가능한 신호 탐색 중');
  document.getElementById('live-owner-symbol').textContent = adopted ? live.symbol : '채택 대기';
  document.getElementById('live-owner-market').textContent = adopted
    ? `${session.macro_regime || 'MACRO'} · ${session.micro_regime || 'MICRO'}`
    : `현재 화면 ${data.symbol || '-'}`;
  document.getElementById('live-owner-strategy').textContent = adopted
    ? (strategyLabels[live.strategy_id] || live.strategy_id)
    : '채택 대기';
  document.getElementById('live-owner-source').textContent = adopted
    ? `${live.selection_source || 'ONTOLOGY'} 선택 · ${live.execution_authority || live.strategy_id} 단독 실행`
    : '후보 판단은 주문 권한과 분리';
  const orderState = String(latestPayload.status || latestEvent.event_type || '').toUpperCase();
  document.getElementById('live-owner-order').textContent =
    orderState || (capable ? 'KIS ARMED' : 'ORDER BLOCKED');
  document.getElementById('live-owner-cycle').textContent =
    `마지막 주기 ${shortClock(live.last_cycle_at)} · ${execution.event_count || 0} events`;
}

function renderHeader(data, market) {
  const stale = Boolean(market.stale);
  const connection = document.querySelector('.connection');
  connection?.classList.toggle('stale', stale);
  document.getElementById('feed-state').textContent = stale
    ? '시세 정체 · 자동 재연결 중'
    : market.feed_state === 'LIVE_QUOTE_ONLY'
      ? '실시간 호가 수신 · 체결 대기'
      : '실시간 체결·호가 수신';
  const mode = document.getElementById('terminal-mode');
  mode.textContent = `${String(data.mode || 'unknown').toUpperCase()} · ${data.live_order_capable ? 'ORDER ARMED' : 'ORDER BLOCKED'}`;
  mode.className = data.live_order_capable ? 'status-chip' : 'status-chip blocked';
}

function setChartStreamState(connected, market = null, error = '') {
  const target = document.getElementById('chart-stream-state');
  if (!target) return;
  const fresh = connected && market && !market.stale;
  target.classList.toggle('connected', Boolean(fresh));
  target.classList.toggle('stale', !fresh);
  const label = target.querySelector('span');
  if (label) {
    label.textContent = fresh
      ? `${market.feed_state === 'LIVE_QUOTE_ONLY' ? 'QUOTE MID' : 'LIVE TRADE'} · ${shortClock(market.last_event_at)}`
      : connected
        ? `스트림 대기 · ${market?.microstructure?.block_reason || 'STALE'}`
        : `연결 재시도 · ${error || 'WAITING'}`;
  }
}

function registerSecondBarArrival(market, reset = false) {
  const bars = market.second_bars || [];
  const latestTime = bars[bars.length - 1]?.time || null;
  if (reset || terminalState.lastSecondBarTime === null) {
    terminalState.lastSecondBarTime = latestTime;
    terminalState.chartTransitionActive = false;
    terminalState.streamReceivedAt = performance.now() - 1000;
    return;
  }
  if (latestTime && latestTime !== terminalState.lastSecondBarTime) {
    terminalState.lastSecondBarTime = latestTime;
    terminalState.chartTransitionActive = true;
    terminalState.streamReceivedAt = performance.now();
  }
}

function renderCandidates(rows, session = {}) {
  const container = document.getElementById('candidate-list');
  const rankings = Array.isArray(session.bandit_evaluations)
    ? session.bandit_evaluations.slice(0, 12)
    : [];
  const selectedSymbol = String(session.selected_symbol || '');
  const selectedStrategy = String(session.selected_strategy || '');
  const selectedDirection = String(session.selected_direction || 'LONG');
  if (rankings.length) {
    container.innerHTML = rankings.map((row, index) => {
      const strategyId = String(row.arm || '').split(':', 1)[0];
      const direction = String(row.direction || 'LONG');
      const edge = Number(row.conservative_edge_bps);
      const selected = row.symbol === selectedSymbol
        && strategyId === selectedStrategy
        && direction === selectedDirection;
      const verdict = selected
        ? 'SELECTED'
        : row.shadow_only
          ? 'SHADOW'
          : row.admissible
            ? 'ELIGIBLE'
            : 'REJECTED';
      return `
    <button type="button" class="candidate joint-candidate ${selected ? 'selected' : ''} ${row.admissible ? 'admissible' : 'rejected'}" data-symbol="${escapeHtml(row.symbol)}">
      <span class="candidate-rank">#${index + 1}</span>
      <strong>${escapeHtml(row.symbol || '-')}</strong>
      <em>${escapeHtml(verdict)}</em>
      <small>${escapeHtml(strategyLabels[strategyId] || strategyId || 'NO_TRADE')}</small>
      <span class="candidate-meta">${escapeHtml(direction)} · ${Number.isFinite(edge) ? `${edge >= 0 ? '+' : ''}${edge.toFixed(1)}bp` : '-'}</span>
    </button>
  `;
    }).join('');
  } else {
    container.innerHTML = rows.length ? rows.map((row) => `
      <button type="button" class="candidate ${row.selected ? 'selected' : ''}" data-symbol="${escapeHtml(row.symbol)}">
        <strong>${escapeHtml(row.symbol)}</strong>
        <em>${row.ontology_allowed ? 'ALLOWED' : 'WATCH'}</em>
        <small>${escapeHtml(strategyLabels[row.strategy_id] || row.strategy_id || row.action || 'NO_TRADE')}</small>
      </button>
    `).join('') : '<span class="tape-empty">공동 순위에 올릴 종목·전략 조합을 기다리고 있습니다.</span>';
  }
  container.querySelectorAll('.candidate').forEach((button) => {
    button.addEventListener('click', () => fetchMarketView(button.dataset.symbol));
  });
}

function renderInstrument(symbol, market, selection, algorithm) {
  const session = terminalState.data?.strategy_session || {};
  const live = terminalState.data?.live_trading || {};
  document.getElementById('instrument-symbol').textContent = symbol || '선택 대기';
  document.getElementById('instrument-price').textContent = formatPrice(market.last_price);
  const change = document.getElementById('instrument-change');
  const rate = Number(market.change_rate);
  change.textContent = Number.isFinite(rate) ? `${rate >= 0 ? '+' : ''}${(rate * 100).toFixed(2)}% / chart range` : '-';
  change.className = Number.isFinite(rate) ? (rate >= 0 ? 'positive' : 'negative') : '';
  const bars = market.bars || [];
  const latest = bars[bars.length - 1] || {};
  const tick = market.latest_tick || {};
  const book = market.latest_orderbook || {};
  const stats = [
    ['OPEN', latest.open],
    ['HIGH', latest.high],
    ['LOW', latest.low],
    ['VOLUME', latest.volume],
    ['SPREAD', book.spread_bps === null || book.spread_bps === undefined ? '-' : `${Number(book.spread_bps).toFixed(1)} bps`],
    ['LATENCY', tick.latency_ms === null || tick.latency_ms === undefined ? '-' : `${Number(tick.latency_ms).toFixed(1)} ms`],
  ];
  document.getElementById('instrument-stats').innerHTML = stats.map(([label, value]) => `
    <div class="instrument-stat"><span>${label}</span><strong>${typeof value === 'number' ? formatCompact(value) : escapeHtml(value ?? '-')}</strong></div>
  `).join('');

  const adoptedStrategyId = live.adopted ? (live.strategy_id || session.selected_strategy) : null;
  const active = Boolean(live.adopted && adoptedStrategyId);
  document.getElementById('algorithm-state').textContent = String(live.phase || session.phase || 'SCANNING');
  document.getElementById('algorithm-name').textContent = active
    ? (strategyLabels[adoptedStrategyId] || adoptedStrategyId)
    : '실거래 전략 채택 대기';
  document.getElementById('algorithm-thesis').textContent = active
    ? `${algorithm?.thesis || '온톨로지·GNN 채택 전략'} · ${session.last_reason || '실시간 감시 중'}`
    : `${session.last_reason || '매수·매도 조건을 평가 중입니다.'} · 후보 모델의 판단은 실제 채택 전까지 주문되지 않습니다.`;
  const tags = algorithm ? [
    ...(algorithm.visual_indicators || []),
    `STOP ${algorithm.stop_bps}bps`,
    `TARGET ${algorithm.profit_bps}bps`,
    `HOLD ${algorithm.max_holding_seconds}s`,
  ] : ['CLOSED-WORLD', 'NoTrade', 'OWNER-LOCK'];
  if (session.session_id) {
    tags.unshift(
      `OWNER ${String(session.session_id).slice(-8)}`,
      session.selection_source || 'ONTOLOGY',
    );
    if (Number(session.target_price) > 0) tags.push(`목표 ${formatPrice(session.target_price)}`);
    if (Number(session.stop_loss_rate) > 0) tags.push(`전략 손절 ${(Number(session.stop_loss_rate) * 100).toFixed(2)}%`);
    if (Number(session.trailing_stop_rate) > 0) tags.push(`추적 손절 ${(Number(session.trailing_stop_rate) * 100).toFixed(2)}%`);
    if (session.exit_reason) tags.push(`EXIT ${session.exit_reason}`);
  } else if (session.phase) {
    tags.unshift(session.phase, session.macro_regime || 'MACRO WAITING');
    const candidateReason = session.candidate_diagnostics?.[0]?.reason_codes?.[0];
    if (candidateReason) tags.push(candidateReason);
  }
  document.getElementById('algorithm-tags').innerHTML = tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join('');
}

function renderTradingChart(market, algorithm = null) {
  const secondBars = market.second_bars || [];
  const useSeconds = terminalState.chartMode === 'seconds' && secondBars.length > 0;
  const bars = useSeconds ? secondBars.slice(-90) : (market.bars || []);
  renderChart(bars, algorithm, market.last_price, useSeconds ? 'seconds' : 'minutes');
}

function renderTradingLayers(data) {
  const container = document.getElementById('trade-layer-grid');
  const counter = document.getElementById('trade-layer-count');
  if (!container || !counter) return;
  const layers = Array.isArray(data?.trading_layers) ? data.trading_layers : [];
  counter.textContent = `${layers.length} ACTIVE`;
  if (!layers.length) {
    container.innerHTML = '<div class="trade-layer-empty">전략이 종목을 채택하면 진입부터 청산까지 독립 레이어가 생성됩니다.</div>';
    return;
  }
  container.innerHTML = layers.map((layer, index) => {
    const realized = String(layer.pnl_kind || '').startsWith('REALIZED');
    const pnl = Number(realized ? layer.realized_pnl : layer.unrealized_pnl || 0);
    const rate = Number(layer.return_rate || 0);
    const pnlClass = pnl > 0 ? 'profit' : pnl < 0 ? 'loss' : '';
    const valueClass = pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : '';
    const currency = String(layer.currency || 'KRW');
    const timeline = Array.isArray(layer.timeline) && layer.timeline.length
      ? layer.timeline
      : [
        { kind: 'SELECTED', at: layer.opened_at, detail: layer.strategy_id },
        { kind: layer.phase || 'OWNED', at: layer.market_data?.last_event_at, detail: '실시간 가격·청산 조건 감시' },
      ];
    return `
      <article class="trade-layer ${pnlClass}" data-trade-layer="${index}">
        <div class="trade-layer-summary">
          <div class="trade-layer-head">
            <div class="trade-layer-title">
              <strong>${escapeHtml(layer.symbol || '-')} · ${formatInteger(layer.quantity)}주</strong>
              <span>${escapeHtml(strategyLabels[layer.strategy_id] || layer.strategy_id || '전략 복구')} · ${escapeHtml(layer.market_data?.feed_state || 'FEED WAITING')}</span>
            </div>
            <span class="trade-layer-phase">${escapeHtml(layer.phase || 'OWNED')}</span>
          </div>
          <div class="trade-layer-stats">
            <div><span>ENTRY</span><b>${formatPrice(layer.entry_price)}</b></div>
            <div><span>LIVE</span><b>${formatPrice(layer.current_price)}</b></div>
            <div><span>TARGET / STOP</span><b>${formatPrice(layer.target_price)} / ${formatPrice(layer.stop_price)}</b></div>
            <div><span>${realized ? 'REALIZED GROSS' : 'UNREALIZED'}</span><b class="${valueClass}">${formatLayerPnl(pnl, currency)} · ${rate >= 0 ? '+' : ''}${(rate * 100).toFixed(2)}%</b></div>
          </div>
        </div>
        <canvas class="trade-layer-chart" data-trade-layer-chart="${index}"></canvas>
        <div class="trade-layer-timeline">
          ${timeline.map((event) => `
            <div class="trade-layer-event">
              <b>${escapeHtml(event.kind || 'EVENT')}</b>
              <small>${escapeHtml(shortClock(event.at))}</small>
              <small title="${escapeHtml(event.detail || '')}">${escapeHtml(event.detail || '-')}</small>
            </div>`).join('')}
        </div>
      </article>`;
  }).join('');
  layers.forEach((layer, index) => {
    const canvas = container.querySelector(`[data-trade-layer-chart="${index}"]`);
    if (!canvas) return;
    const market = layer.symbol === data.symbol
      ? { ...(layer.market_data || {}), ...(data.market || {}) }
      : (layer.market_data || {});
    const secondBars = market.second_bars || [];
    const bars = terminalState.chartMode === 'seconds' && secondBars.length
      ? secondBars.slice(-90)
      : (market.bars || []).slice(-90);
    drawTradingLayerChart(canvas, bars, layer);
  });
}

function drawTradingLayerChart(canvas, bars, layer) {
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#091018';
  ctx.fillRect(0, 0, width, height);
  const pad = { left: 10, right: 58, top: 12, bottom: 19 };
  const entry = Number(layer.entry_price);
  const live = Number(layer.current_price);
  const target = Number(layer.target_price);
  const stop = Number(layer.stop_price);
  const prices = bars.flatMap((bar) => [Number(bar.high), Number(bar.low)])
    .concat([entry, live, target, stop]).filter((value) => Number.isFinite(value) && value > 0);
  if (!prices.length) {
    ctx.fillStyle = '#687b90';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('가격 데이터 대기', width / 2, height / 2);
    return;
  }
  const rawMin = Math.min(...prices);
  const rawMax = Math.max(...prices);
  const span = Math.max(rawMax - rawMin, rawMax * .001, 1e-6);
  const min = rawMin - span * .08;
  const max = rawMax + span * .08;
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const y = (value) => pad.top + (max - Number(value)) / (max - min) * plotHeight;
  ctx.strokeStyle = '#182534';
  ctx.fillStyle = '#62758a';
  ctx.font = '7px Consolas';
  for (let line = 0; line <= 3; line += 1) {
    const yy = pad.top + plotHeight * line / 3;
    ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(width - pad.right, yy); ctx.stroke();
    ctx.fillText(formatPrice(max - (max - min) * line / 3), width - pad.right + 5, yy + 3);
  }
  const step = plotWidth / Math.max(1, bars.length - 1);
  ctx.beginPath();
  bars.forEach((bar, index) => {
    const x = pad.left + step * index;
    const yy = y(Number(bar.close));
    if (index === 0) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
  });
  ctx.strokeStyle = '#39d7e7';
  ctx.lineWidth = 1.35;
  ctx.stroke();
  if (Number.isFinite(entry) && entry > 0) drawLevel(ctx, width, height, pad, y, entry, '#9b8cff', 'ENTRY');
  if (Number.isFinite(target) && target > 0) drawLevel(ctx, width, height, pad, y, target, '#42d392', 'TARGET');
  if (Number.isFinite(stop) && stop > 0) drawLevel(ctx, width, height, pad, y, stop, '#ff6678', 'STOP');
  if (Number.isFinite(live) && live > 0) {
    const liveY = y(live);
    ctx.fillStyle = live >= entry ? '#42d392' : '#ff6678';
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = 9;
    ctx.beginPath(); ctx.arc(width - pad.right, liveY, 3.2, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
  }
  ctx.fillStyle = '#61758a';
  ctx.font = '7px Consolas';
  ctx.textAlign = 'left';
  ctx.fillText(`OPEN ${shortClock(layer.opened_at)} · ${bars.length} BARS`, pad.left, height - 5);
}

function formatLayerPnl(value, currency) {
  const amount = Number(value || 0);
  if (String(currency).toUpperCase() === 'USD') {
    return `${amount >= 0 ? '+' : '-'}$${Math.abs(amount).toFixed(2)}`;
  }
  return formatSignedKrw(amount);
}

function renderChart(bars, algorithm = null, referencePrice = null, timeframe = 'minutes') {
  const empty = document.getElementById('chart-empty');
  empty.style.display = bars.length ? 'none' : 'grid';
  const live = timeframe === 'seconds';
  const progress = live && terminalState.chartTransitionActive
    ? Math.min(1, Math.max(0, (performance.now() - terminalState.streamReceivedAt) / 420))
    : 1;
  if (progress >= 1) terminalState.chartTransitionActive = false;
  drawPriceChart(
    document.getElementById('price-chart'),
    bars,
    algorithm,
    referencePrice,
    { live, progress, maximumPoints: live ? 90 : bars.length },
  );
  drawVolumeChart(
    document.getElementById('volume-chart'),
    bars,
    { live, progress, maximumPoints: live ? 90 : bars.length },
  );
  document.getElementById('chart-range').textContent = live
    ? `실시간 이동 창 · 최근 ${bars.length}개 1초봉 · 체결봉 + 실제 호가 중간값봉 · MA5 / MA20 / VWAP`
    : `최근 ${bars.length}개 1분봉 · MA5 / MA20 / VWAP`;
  const last = bars[bars.length - 1];
  document.getElementById('chart-updated').textContent = `${live ? (last?.bar_source === 'quote_mid' ? 'QUOTE MID' : 'LIVE TRADE') : '마지막 이벤트'} ${formatTime(last?.time)}`;
}

function drawPriceChart(canvas, bars, algorithm, referencePrice, options = {}) {
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0d131c';
  ctx.fillRect(0, 0, width, height);
  if (!bars.length) return;
  const pad = { left: 12, right: 62, top: 15, bottom: 24 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const lows = bars.map((bar) => Number(bar.low));
  const highs = bars.map((bar) => Number(bar.high));
  const reference = Number(referencePrice || bars[bars.length - 1]?.close);
  if (algorithm && Number.isFinite(reference)) {
    lows.push(reference * (1 - Number(algorithm.stop_bps || 0) / 10000));
    highs.push(reference * (1 + Number(algorithm.profit_bps || 0) / 10000));
  }
  const priceMin = Math.min(...lows);
  const priceMax = Math.max(...highs);
  const span = Math.max(priceMax - priceMin, Math.abs(priceMax) * .001, 1e-6);
  const min = priceMin - span * .05;
  const max = priceMax + span * .05;
  const y = (value) => pad.top + (max - Number(value)) / (max - min) * plotHeight;
  const maximumPoints = Math.max(1, Number(options.maximumPoints || bars.length));
  const step = plotWidth / maximumPoints;
  const progress = Number(options.progress || 0);
  const xForIndex = (index) => options.live
    ? width - pad.right - step * (bars.length - index - .5) + step * (1 - progress)
    : pad.left + step * (index + .5);

  ctx.strokeStyle = '#1c2734';
  ctx.fillStyle = '#6f8095';
  ctx.font = '9px Consolas';
  ctx.textAlign = 'left';
  for (let i = 0; i <= 5; i += 1) {
    const yy = pad.top + plotHeight * i / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(width - pad.right, yy);
    ctx.stroke();
    ctx.fillText(formatPrice(max - (max - min) * i / 5), width - pad.right + 7, yy + 3);
  }
  for (let i = 0; i < bars.length; i += 1) {
    const bar = bars[i];
    const x = xForIndex(i);
    const quoteOnly = bar.bar_source === 'quote_mid';
    const up = Number(bar.close) >= Number(bar.open);
    const color = quoteOnly ? '#39d7e7' : (up ? '#42d392' : '#ff6678');
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y(bar.high));
    ctx.lineTo(x, y(bar.low));
    ctx.stroke();
    const bodyTop = Math.min(y(bar.open), y(bar.close));
    const bodyHeight = Math.max(1, Math.abs(y(bar.open) - y(bar.close)));
    const bodyWidth = Math.max(1, Math.min(7, step * .66));
    if (quoteOnly) {
      ctx.strokeRect(x - bodyWidth / 2, bodyTop - 1, bodyWidth, Math.max(2, bodyHeight + 2));
    } else {
      ctx.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);
    }
  }
  drawLine(ctx, movingAverage(bars.map((bar) => Number(bar.close)), 5), y, pad.left, step, '#39d7e7', 1.3, [], xForIndex);
  drawLine(ctx, movingAverage(bars.map((bar) => Number(bar.close)), 20), y, pad.left, step, '#9b8cff', 1.3, [], xForIndex);
  drawLine(ctx, bars.map((bar) => Number(bar.vwap) || null), y, pad.left, step, '#f3b95f', 1, [4, 3], xForIndex);
  if (algorithm && Number.isFinite(reference)) {
    drawLevel(ctx, width, height, pad, y, reference * (1 + Number(algorithm.profit_bps || 0) / 10000), '#42d392', 'TARGET');
    drawLevel(ctx, width, height, pad, y, reference * (1 - Number(algorithm.stop_bps || 0) / 10000), '#ff6678', 'STOP');
  }

  ctx.fillStyle = '#6f8095';
  ctx.textAlign = 'center';
  const labels = Math.min(6, bars.length);
  for (let i = 0; i < labels; i += 1) {
    const index = Math.round(i * (bars.length - 1) / Math.max(1, labels - 1));
    const x = xForIndex(index);
    ctx.fillText(options.live ? shortClock(bars[index].time) : shortTime(bars[index].time), x, height - 7);
  }
  if (options.live && bars.length) {
    const lastIndex = bars.length - 1;
    const livePrice = Number(referencePrice || bars[lastIndex].close);
    const liveX = xForIndex(lastIndex);
    const liveY = y(livePrice);
    const pulse = .55 + Math.sin(performance.now() / 170) * .25;
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.strokeStyle = `rgba(57,215,231,${pulse})`;
    ctx.beginPath();
    ctx.moveTo(pad.left, liveY);
    ctx.lineTo(width - pad.right, liveY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#39d7e7';
    ctx.shadowColor = '#39d7e7';
    ctx.shadowBlur = 8 + pulse * 7;
    ctx.beginPath();
    ctx.arc(liveX, liveY, 2.8 + pulse * 1.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.font = '700 9px Consolas';
    ctx.textAlign = 'right';
    ctx.fillText(`LIVE ${formatPrice(livePrice)}`, width - pad.right - 4, liveY - 6);
    ctx.restore();
  }
}

function drawLevel(ctx, width, height, pad, y, value, color, label) {
  const yy = y(value);
  if (yy < pad.top || yy > height - pad.bottom) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, yy);
  ctx.lineTo(width - pad.right, yy);
  ctx.stroke();
  ctx.font = '8px Consolas';
  ctx.textAlign = 'right';
  ctx.fillText(`${label} ${formatPrice(value)}`, width - pad.right - 4, yy - 3);
  ctx.restore();
}

function drawLine(ctx, values, y, left, step, color, width, dash = [], xForIndex = null) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  let started = false;
  values.forEach((value, index) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return;
    const x = xForIndex ? xForIndex(index) : left + step * (index + .5);
    if (!started) {
      ctx.moveTo(x, y(value));
      started = true;
    } else {
      ctx.lineTo(x, y(value));
    }
  });
  if (started) ctx.stroke();
  ctx.restore();
}

function drawVolumeChart(canvas, bars, options = {}) {
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#0d131c';
  ctx.fillRect(0, 0, width, height);
  if (!bars.length) return;
  const left = 12;
  const right = 62;
  const plotWidth = width - left - right;
  const maxVolume = Math.max(...bars.map((bar) => Number(bar.volume || 0)), 1);
  const maximumPoints = Math.max(1, Number(options.maximumPoints || bars.length));
  const step = plotWidth / maximumPoints;
  const progress = Number(options.progress || 0);
  bars.forEach((bar, index) => {
    const value = Number(bar.volume || 0);
    const h = value / maxVolume * (height - 10);
    const x = options.live
      ? width - right - step * (bars.length - index) + step * (1 - progress)
      : left + step * index;
    if (bar.bar_source !== 'quote_mid') {
      ctx.fillStyle = Number(bar.close) >= Number(bar.open) ? 'rgba(66,211,146,.5)' : 'rgba(255,102,120,.5)';
      ctx.fillRect(x, height - h, Math.max(1, step * .7), h);
    }
  });
}

function renderSecondAnalysis(market) {
  const bars = market.second_bars || [];
  const micro = market.microstructure || {};
  const status = document.getElementById('second-data-status');
  if (!status) return;
  status.textContent = micro.ready ? 'SECOND DATA READY' : `BLOCKED · ${micro.block_reason || 'WAITING'}`;
  status.className = micro.ready ? 'status-chip' : 'status-chip blocked';
  const metrics = [
    ['표시 피드', micro.display_feed_state || 'STALE', 'display_feed_state'],
    ['체결/호가 봉', `${Number(micro.trade_bar_count || 0)} / ${Number(micro.quote_bar_count || 0)}`, 'trade_bar_count'],
    ['1초 수익률', formatSecondReturn(micro.return_1s), 'return_1s'],
    ['5초 수익률', formatSecondReturn(micro.return_5s), 'return_5s'],
    ['10초 수익률', formatSecondReturn(micro.return_10s), 'return_10s'],
    ['1초 체결', `${Number(micro.tick_count_1s || 0)}건`, 'tick_count_1s'],
    ['5초 체결', `${Number(micro.tick_count_5s || 0)}건`, 'tick_count_5s'],
    ['5초 거래량', formatCompact(micro.volume_5s), 'volume_5s'],
    ['5초 체결강도', formatSignedDecimal(micro.aggressor_imbalance_5s), 'aggressor_imbalance_5s'],
    ['스프레드 변화', micro.spread_change_5s_bps === null || micro.spread_change_5s_bps === undefined ? '-' : `${formatSignedDecimal(micro.spread_change_5s_bps)} bps`, 'spread_change_5s_bps'],
    ['호가 불균형 변화', formatSignedDecimal(micro.orderbook_imbalance_change_5s), 'orderbook_imbalance_change_5s'],
  ];
  document.getElementById('second-metrics').innerHTML = metrics.map(([label, value, key]) => {
    const numeric = Number(micro[key]);
    const direction = Number.isFinite(numeric) ? (numeric > 0 ? 'positive' : numeric < 0 ? 'negative' : '') : '';
    return `<article><span>${escapeHtml(label)}</span><strong class="${direction}">${escapeHtml(value)}</strong></article>`;
  }).join('');
  document.getElementById('second-analysis-time').textContent = `최근 초단위 이벤트 ${formatTime(micro.as_of)}`;
  document.getElementById('second-analysis-gate').textContent = micro.ready
    ? '초단위 진입 게이트 통과 · 모델과 온톨로지가 이 데이터로 진입 시점을 재평가합니다.'
    : `실매수 차단: ${micro.block_reason || 'SECOND_LEVEL_DATA_NOT_READY'}`;
  const empty = document.getElementById('second-chart-empty');
  empty.style.display = bars.length ? 'none' : 'grid';
  drawSecondPriceChart(document.getElementById('second-price-chart'), bars);
}

function drawSecondPriceChart(canvas, allBars) {
  const bars = allBars.slice(-90);
  const { ctx, width, height } = prepareCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#090f17';
  ctx.fillRect(0, 0, width, height);
  if (!bars.length) return;
  const pad = { left: 10, right: 58, top: 13, bottom: 24 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const prices = bars.flatMap((bar) => [Number(bar.high), Number(bar.low)]).filter(Number.isFinite);
  const minimum = Math.min(...prices);
  const maximum = Math.max(...prices);
  const span = Math.max(maximum - minimum, Math.abs(maximum) * .0001, 1e-6);
  const low = minimum - span * .08;
  const high = maximum + span * .08;
  const y = (value) => pad.top + (high - Number(value)) / (high - low) * plotHeight;
  const step = plotWidth / Math.max(1, bars.length);
  ctx.strokeStyle = '#1b2836';
  ctx.fillStyle = '#66788e';
  ctx.font = '8px Consolas';
  for (let index = 0; index <= 4; index += 1) {
    const yy = pad.top + plotHeight * index / 4;
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(width - pad.right, yy);
    ctx.stroke();
    ctx.fillText(formatPrice(high - (high - low) * index / 4), width - pad.right + 6, yy + 3);
  }
  bars.forEach((bar, index) => {
    const x = pad.left + step * (index + .5);
    const quoteOnly = bar.bar_source === 'quote_mid';
    const up = Number(bar.close) >= Number(bar.open);
    const imbalance = Number(bar.aggressor_imbalance || 0);
    ctx.strokeStyle = quoteOnly ? '#39d7e7' : (up ? '#42d392' : '#ff6678');
    ctx.fillStyle = quoteOnly ? '#39d7e7' : (up ? '#42d392' : '#ff6678');
    ctx.beginPath();
    ctx.moveTo(x, y(bar.high));
    ctx.lineTo(x, y(bar.low));
    ctx.stroke();
    const bodyTop = Math.min(y(bar.open), y(bar.close));
    const bodyWidth = Math.max(1, step * .56);
    const bodyHeight = Math.max(1, Math.abs(y(bar.open) - y(bar.close)));
    if (quoteOnly) {
      ctx.strokeRect(x - Math.max(1, step * .28), bodyTop - 1, bodyWidth, Math.max(2, bodyHeight + 2));
    } else {
      ctx.fillRect(x - Math.max(1, step * .28), bodyTop, bodyWidth, bodyHeight);
    }
    const flowHeight = Math.min(12, Math.abs(imbalance) * 12);
    ctx.fillStyle = imbalance >= 0 ? 'rgba(66,211,146,.5)' : 'rgba(255,102,120,.5)';
    ctx.fillRect(x - Math.max(1, step * .28), height - pad.bottom - flowHeight, Math.max(1, step * .56), flowHeight);
  });
  ctx.fillStyle = '#66788e';
  ctx.textAlign = 'center';
  const labelIndexes = [0, Math.floor((bars.length - 1) / 2), bars.length - 1];
  labelIndexes.forEach((index) => {
    const x = pad.left + step * (index + .5);
    ctx.fillText(shortClock(bars[index].time), x, height - 7);
  });
}

function formatSecondReturn(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number >= 0 ? '+' : ''}${(number * 100).toFixed(3)}%` : '-';
}

function formatSignedDecimal(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number >= 0 ? '+' : ''}${number.toFixed(3)}` : '-';
}

function renderOntology(selection, market) {
  const evidenceStatus = String(
    selection.evidence_status || (selection.as_of ? 'CURRENT' : 'MISSING'),
  ).toUpperCase();
  const evidenceCurrent = evidenceStatus === 'CURRENT';
  const allowed = Boolean(selection.ontology_allowed) && evidenceCurrent;
  const ontologyStrategyId = selection.ontology_strategy_id || selection.strategy_id || null;
  const finalAction = String(selection.action || 'NO_TRADE').toUpperCase();
  const utilityPassed = evidenceCurrent && finalAction !== 'NO_TRADE';
  const reasons = selection.reason_codes || [];
  const utility = Number(selection.utility);
  const hasUtility = selection.utility !== null
    && selection.utility !== undefined
    && Number.isFinite(utility);
  const netEdgeReason = reasons.find((reason) => String(reason).includes('NON_POSITIVE_NET_EDGE'));
  const uncertaintyReason = reasons.find((reason) => String(reason).includes('UNCERTAINTY'));
  const utilityReason = reasons.find((reason) => String(reason).includes('UTILITY'));
  const trustReason = reasons.find((reason) => String(reason).includes('GNN_REALTIME_TRUST'));
  const contractReason = reasons.find((reason) => (
    String(reason).includes('SCHEMA_MISMATCH')
    || String(reason).includes('CATALOG_MISMATCH')
    || String(reason).includes('CHECKPOINT')
    || String(reason).includes('NOT_LIVE_AUTHORIZED')
  ));
  const status = document.getElementById('ontology-status');
  status.textContent = !evidenceCurrent
    ? `EVIDENCE ${evidenceStatus}`
    : allowed ? 'ONTOLOGY ALLOWED' : 'NO TRADE';
  status.className = allowed ? 'status-chip' : 'status-chip blocked';
  const nodes = [
    ['DECISION EVIDENCE', `${evidenceStatus} · ${shortClock(selection.as_of)}`, evidenceCurrent],
    ['데이터 신선도', market.stale ? 'STALE · 신규 진입 차단' : 'FRESH', !market.stale],
    ['운영 사실 검증', allowed ? '필수 사실 충족' : '필수 사실 미충족', allowed],
    ['전략 호환성', ontologyStrategyId || '허용 전략 없음', allowed],
    ['모델 계약', contractReason || '스키마·전략 카탈로그 일치', !contractReason],
    [
      '실시간 GNN 신뢰도',
      trustReason
        ? `${trustReason} · score ${Number(selection.realtime_trust_score || 0).toFixed(3)} · n=${selection.realtime_trust_samples || 0}`
        : '실시간 검증 결과 없음',
      trustReason === 'GNN_REALTIME_TRUST_PASSED',
    ],
    ['순수익', netEdgeReason || (hasUtility ? '양의 순수익 조건 통과' : '모델 평가 없음'), !netEdgeReason && hasUtility],
    ['불확실성', uncertaintyReason || (hasUtility ? '허용 범위' : '모델 평가 없음'), !uncertaintyReason && hasUtility],
    ['순효용', utilityReason || (hasUtility ? `utility ${utility.toFixed(3)}` : '모델 평가 없음'), !utilityReason && hasUtility && utilityPassed],
    ['최종 라우팅', finalAction, utilityPassed],
  ];
  document.getElementById('ontology-flow').innerHTML = nodes.map(([label, detail, pass]) => `
    <div class="ontology-node ${pass ? 'pass' : 'block'}"><i></i><b>${escapeHtml(label)}</b><small>${escapeHtml(detail)}</small></div>
  `).join('');
  document.getElementById('ontology-reasons').innerHTML = reasons.length
    ? reasons.map((reason) => `<span class="reason-chip">${escapeHtml(reason)}</span>`).join('')
    : `<span class="reason-chip">${allowed ? 'ALL_REQUIRED_FACTS_VALID' : 'NO_ADMISSIBLE_STRATEGY'}</span>`;
  document.getElementById('decision-compare').innerHTML = (selection.all_decisions || []).map((decision) => `
    <div class="decision-row">
      <b>${escapeHtml(decision.path || '-')}</b>
      <span>${escapeHtml(decision.action || '-')}</span>
      <span>${escapeHtml(strategyLabels[decision.strategy_id] || decision.strategy_id || 'NoTrade')}</span>
    </div>
  `).join('');
}

function renderDecisionOntology(trace) {
  const svg = document.getElementById('decision-ontology-graph');
  if (!svg) return;
  const sources = trace.sources || [];
  const indicators = trace.indicators || [];
  const algorithms = trace.algorithms || [];
  const ontology = trace.ontology_selection || {};
  const finalDecision = trace.final_decision || {};
  // generated_at/fresh change on every poll but do not change the graph. Keep
  // those badges live without destroying and recreating the SVG node tree.
  const graphSignature = JSON.stringify({
    filter: terminalState.ontologyFilter,
    sources: sources.map(({ updated_at: _updatedAt, ...source }) => source),
    indicators,
    algorithms,
    ontology,
    finalDecision,
  });
  renderDecisionOntologyMeta(trace, ontology, finalDecision);
  if (terminalState.ontologySignature === graphSignature) return;
  terminalState.ontologySignature = graphSignature;
  const activeAlgorithms = algorithms.filter((item) => item.ontology_selected || item.final_selected);
  const activeAlgorithm = activeAlgorithms.find((item) => item.ontology_selected) || activeAlgorithms[0];
  const activeIndicatorIds = new Set(
    (activeAlgorithm?.requirements || []).map((item) => item.indicator_id),
  );
  const showAllRelationships = terminalState.ontologyFilter === 'all';
  const visibleIndicators = showAllRelationships
    ? indicators
    : indicators.filter((item) => activeIndicatorIds.has(item.id));
  const visibleSources = showAllRelationships
    ? sources
    : sources.filter((source) => visibleIndicators.some((item) => item.source_id === source.id));
  // An empty active path is an authoritative state, not a cue to mix arbitrary
  // available facts with every strategy. That old fallback produced orphaned
  // cards and made a valid NO_TRADE decision look like a broken graph.
  const graphSources = visibleSources;
  const graphIndicators = visibleIndicators;
  const reasonCodes = finalDecision.reason_codes || [];
  const noCandidateSymbols = reasonCodes.includes('NO_CANDIDATE_SYMBOLS');
  const emptyPathGate = {
    id: 'decision_gate',
    label: noCandidateSymbols ? '후보 종목 탐색 게이트' : '전략 선택 게이트',
    thesis: noCandidateSymbols
      ? '실시간 후보 탐색 결과가 0건이어서 전략 평가 단계로 진입하지 않았습니다.'
      : '이번 판단 주기에는 선택되거나 라우팅된 전략이 없습니다.',
    gate: true,
  };
  const displayedAlgorithms = showAllRelationships
    ? algorithms
    : activeAlgorithms.length ? activeAlgorithms : [emptyPathGate];
  const height = Math.max(
    510,
    78 + graphSources.length * 72,
    78 + graphIndicators.length * 62,
    78 + displayedAlgorithms.length * 72,
  );
  const width = 1100;
  const positions = new Map();
  const nodeWidth = 188;
  const layerX = { source: 30, indicator: 300, algorithm: 635, decision: 905 };
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('height', String(height));

  const addText = (x, y, value, className) => {
    const textNode = createSvgElement('text', { x, y, class: className });
    textNode.textContent = value;
    svg.appendChild(textNode);
  };
  addText(layerX.source, 27, '01  SOURCE DATA', 'graph-layer-label');
  addText(layerX.indicator, 27, '02  INDICATORS / FACTS', 'graph-layer-label');
  addText(layerX.algorithm, 27, '03  STRATEGY / GATES', 'graph-layer-label');
  addText(layerX.decision, 27, '04  ROUTING', 'graph-layer-label');

  const layoutNodes = (items, layer, gap, formatter) => {
    const total = (items.length - 1) * gap;
    const start = Math.max(52, (height - total) / 2 - 27);
    items.forEach((item, index) => {
      const y = start + index * gap;
      positions.set(`${layer}:${item.id}`, { x: layerX[layer], y, width: nodeWidth, height: 48 });
      drawOntologyNode(svg, {
        ...formatter(item),
        x: layerX[layer],
        y,
        width: nodeWidth,
        height: 48,
      });
    });
  };
  layoutNodes(graphSources, 'source', 72, (item) => ({
    id: `source:${item.id}`,
    title: item.label,
    value: String(item.status || '').toUpperCase() === 'PARTIAL'
      ? `PARTIAL · ${item.samples || 0} fields`
      : item.available ? `${item.samples || 0} samples` : 'DATA MISSING',
    className: String(item.status || '').toUpperCase() === 'PARTIAL'
      ? 'partial' : item.available ? '' : 'unavailable',
    detail: {
      kind: 'SOURCE DATA',
      title: item.label,
      value: String(item.status || '').toUpperCase() === 'PARTIAL'
        ? '부분 가용' : item.available ? '수신 중' : '데이터 없음',
      description: item.available ? '실시간 지표 계산에 사용되는 원천 데이터입니다.' : '현재 시장 응답에서 이 데이터가 제공되지 않습니다.',
      rows: [
        ['ID', item.id],
        ['상태', item.status || (item.available ? 'AVAILABLE' : 'MISSING')],
        ['샘플', item.samples || 0],
        ['갱신', item.updated_at || '-'],
      ],
    },
  }));
  layoutNodes(graphIndicators, 'indicator', 62, (item) => ({
    id: `indicator:${item.id}`,
    title: item.label,
    value: item.available ? `score ${formatOntologyScore(item.score)}` : 'UNKNOWN',
    className: item.available ? '' : 'unavailable',
    detail: {
      kind: 'RECONSTRUCTED INDICATOR',
      title: item.label,
      value: item.available ? formatOntologyScore(item.score) : 'N/A',
      description: item.detail,
      rows: [
        ['원시값', formatOntologyRaw(item.raw_value)],
        ['출처', item.source_id],
        ['계보', item.provenance || 'reconstructed'],
      ],
    },
  }));
  layoutNodes(displayedAlgorithms, 'algorithm', 72, (item) => ({
    id: `algorithm:${item.id}`,
    title: strategyLabels[item.id] || item.label || item.id,
    value: item.gate
      ? noCandidateSymbols ? '0 CANDIDATES' : 'NO STRATEGY SELECTED'
      : item.ontology_selected ? 'ONTOLOGY SELECTED' : item.final_selected ? 'FINAL SELECTED' : 'CANDIDATE',
    className: item.gate ? 'blocked' : item.ontology_selected || item.final_selected ? 'selected' : '',
    detail: {
      kind: item.gate ? 'DECISION GATE' : 'STRATEGY EXPERT',
      title: strategyLabels[item.id] || item.label || item.id,
      value: item.gate
        ? noCandidateSymbols ? '후보 0건' : '선택 전략 없음'
        : item.ontology_selected ? '온톨로지 선택' : item.final_selected ? '최종 선택' : '후보',
      description: item.thesis,
      rows: item.gate
        ? [['판단 경로', finalDecision.path || '-'], ['차단 사유', reasonCodes.join(' · ') || 'NO_SELECTED_STRATEGY']]
        : (item.requirements || []).map((rule) => [
          indicators.find((indicator) => indicator.id === rule.indicator_id)?.label || rule.indicator_id,
          `${rule.operator} ${formatOntologyScore(rule.threshold)} · ${rule.passed === null ? 'UNKNOWN' : rule.passed ? 'PASS' : 'FAIL'}`,
        ]),
    },
  }));

  const decisionId = 'decision:final';
  const decisionY = Math.max(52, height / 2 - 24);
  positions.set(decisionId, { x: layerX.decision, y: decisionY, width: 165, height: 55 });
  drawOntologyNode(svg, {
    id: decisionId,
    x: layerX.decision,
    y: decisionY,
    width: 165,
    height: 55,
    title: finalDecision.action || 'NO_TRADE',
    value: `${String(finalDecision.path || 'ontology').toUpperCase()} ROUTER`,
    className: finalDecision.action === 'NO_TRADE' ? 'blocked' : 'selected',
    detail: {
      kind: 'AUTHORITATIVE DECISION',
      title: finalDecision.action || 'NO_TRADE',
      value: finalDecision.strategy_id || '주문 없음',
      description: (finalDecision.reason_codes || []).join(' · ') || '기록된 차단 사유가 없습니다.',
      rows: [
        ['경로', finalDecision.path || '-'],
        ['효용', finalDecision.utility ?? '-'],
        ['기록', trace.provenance?.decision || '-'],
      ],
    },
  });

  const edges = [];
  graphIndicators.forEach((indicator) => {
    if (positions.has(`source:${indicator.source_id}`)) {
      edges.push({
        from: `source:${indicator.source_id}`,
        to: `indicator:${indicator.id}`,
        className: indicator.available ? 'pass' : 'unknown',
        detail: {
          kind: 'DATA → INDICATOR',
          title: `${sources.find((item) => item.id === indicator.source_id)?.label || indicator.source_id} → ${indicator.label}`,
          value: indicator.available ? formatOntologyScore(indicator.score) : 'UNKNOWN',
          description: indicator.detail,
          rows: [['출처', indicator.source_id], ['산출 방식', 'dashboard reconstruction']],
        },
      });
    }
  });
  displayedAlgorithms.forEach((algorithm) => {
    if (algorithm.gate) {
      edges.push({
        from: `algorithm:${algorithm.id}`,
        to: decisionId,
        className: 'block',
        detail: {
          kind: 'GATE → ROUTER',
          title: `${algorithm.label} → ${finalDecision.action || 'NO_TRADE'}`,
          value: noCandidateSymbols ? '후보 없음' : '전략 미선택',
          description: algorithm.thesis,
          rows: [['결과', finalDecision.action || 'NO_TRADE'], ['사유', reasonCodes.join(' · ') || 'NO_SELECTED_STRATEGY']],
        },
      });
      return;
    }
    (algorithm.requirements || []).forEach((rule) => {
      if (!positions.has(`indicator:${rule.indicator_id}`)) return;
      const indicator = indicators.find((item) => item.id === rule.indicator_id);
      const isActive = algorithm.ontology_selected && rule.passed !== false;
      edges.push({
        from: `indicator:${rule.indicator_id}`,
        to: `algorithm:${algorithm.id}`,
        className: isActive ? 'active' : rule.passed === true ? 'pass' : rule.passed === false ? 'block' : 'unknown',
        detail: {
          kind: 'ONTOLOGY REQUIREMENT',
          title: `${indicator?.label || rule.indicator_id} → ${strategyLabels[algorithm.id] || algorithm.id}`,
          value: `${formatOntologyScore(indicator?.score)} ${rule.operator} ${formatOntologyScore(rule.threshold)}`,
          description: rule.passed === null ? '필요 데이터가 없어 이 화면에서는 조건을 재검증할 수 없습니다.' : rule.passed ? '화면 재구성값 기준으로 조건을 충족합니다.' : '화면 재구성값 기준으로 조건을 충족하지 않습니다.',
          rows: [['결과', rule.passed === null ? 'UNKNOWN' : rule.passed ? 'PASS' : 'FAIL'], ['선택 기록', algorithm.ontology_selected ? 'SELECTED' : 'NOT SELECTED']],
        },
      });
    });
    const blockedByUtility = (finalDecision.reason_codes || []).some((reason) => String(reason).includes(algorithm.id));
    if (algorithm.ontology_selected || algorithm.final_selected || showAllRelationships) {
      edges.push({
        from: `algorithm:${algorithm.id}`,
        to: decisionId,
        className: algorithm.final_selected ? 'active' : blockedByUtility || finalDecision.action === 'NO_TRADE' ? 'block' : 'unknown',
        detail: {
          kind: 'STRATEGY → ROUTER',
          title: `${strategyLabels[algorithm.id] || algorithm.id} → ${finalDecision.action || 'NO_TRADE'}`,
          value: algorithm.final_selected ? 'ROUTED' : blockedByUtility ? 'NON-POSITIVE NET EDGE' : 'NOT ROUTED',
          description: algorithm.ontology_selected && !algorithm.final_selected
            ? '온톨로지는 전략을 허용했지만 최종 효용 라우터가 주문을 차단했습니다.'
            : '최종 의사결정 로그에 기록된 라우팅 관계입니다.',
          rows: [['온톨로지', algorithm.ontology_selected ? 'SELECTED' : 'NOT SELECTED'], ['최종', algorithm.final_selected ? 'SELECTED' : 'BLOCKED']],
        },
      });
    }
  });
  edges.forEach((edge) => drawOntologyEdge(svg, positions, edge));
  [...svg.querySelectorAll('.graph-node')].forEach((node) => svg.appendChild(node));

  document.getElementById('algorithm-catalog').innerHTML = algorithms.map((item) => {
    const failed = (item.requirements || []).filter((rule) => rule.passed === false).length;
    const unknown = (item.requirements || []).filter((rule) => rule.passed === null).length;
    const state = item.ontology_selected ? 'ONTOLOGY SELECTED' : failed ? `${failed} RULE BLOCKED` : unknown ? `${unknown} DATA UNKNOWN` : 'RULES PASS';
    return `<article class="algorithm-catalog-card ${item.ontology_selected ? 'selected' : failed ? 'blocked' : ''}">
      <b>${escapeHtml(strategyLabels[item.id] || item.label || item.id)}</b>
      <small>${escapeHtml((item.visual_indicators || []).join(' · '))}</small>
      <em>${escapeHtml(state)}</em>
    </article>`;
  }).join('');
  document.querySelectorAll('.graph-filter').forEach((button) => {
    button.classList.toggle('active', button.dataset.graphFilter === terminalState.ontologyFilter);
    button.onclick = () => {
      terminalState.ontologyFilter = button.dataset.graphFilter;
      renderDecisionOntology(trace);
    };
  });
}

function renderDecisionOntologyMeta(trace, ontology, finalDecision) {
  const allowedName = strategyLabels[ontology.strategy_id] || ontology.strategy_id || '선택 없음';
  const primaryReason = (finalDecision.reason_codes || [])[0] || '판단 근거 기록 없음';
  document.getElementById('decision-ontology-summary').textContent =
    `온톨로지: ${allowedName} ${ontology.allowed ? '허용' : '차단'} · 최종: ${finalDecision.action || 'NO_TRADE'} (${String(finalDecision.path || '-').toUpperCase()}) · ${primaryReason}`;
  const liveBadge = document.getElementById('decision-ontology-live');
  liveBadge.textContent = `${trace.fresh ? 'LIVE' : 'STALE'} · ${shortClock(trace.generated_at)}`;
  liveBadge.className = trace.fresh ? 'status-chip' : 'status-chip blocked';
  document.getElementById('ontology-provenance').textContent =
    `${trace.provenance?.warning || ''} 결정 출처: ${trace.provenance?.decision || '-'} · 지표 출처: ${trace.provenance?.indicators || '-'}`;
}

function createSvgElement(name, attributes = {}) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function drawOntologyNode(svg, node) {
  const group = createSvgElement('g', {
    class: `graph-node ${node.className || ''}`,
    transform: `translate(${node.x} ${node.y})`,
    tabindex: 0,
    role: 'button',
  });
  group.appendChild(createSvgElement('rect', { width: node.width, height: node.height, rx: 7 }));
  const title = createSvgElement('text', { x: 11, y: 20, class: 'node-title' });
  title.textContent = truncateGraphText(node.title, 25);
  const value = createSvgElement('text', { x: 11, y: 36, class: 'node-value' });
  value.textContent = truncateGraphText(node.value, 29);
  group.append(title, value);
  group.addEventListener('click', () => showOntologyInspector(node.detail));
  group.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') showOntologyInspector(node.detail);
  });
  svg.appendChild(group);
}

function drawOntologyEdge(svg, positions, edge) {
  const from = positions.get(edge.from);
  const to = positions.get(edge.to);
  if (!from || !to) return;
  const x1 = from.x + from.width;
  const y1 = from.y + from.height / 2;
  const x2 = to.x;
  const y2 = to.y + to.height / 2;
  const curve = Math.max(45, (x2 - x1) * .46);
  const path = createSvgElement('path', {
    d: `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`,
    class: `graph-edge ${edge.className || ''}`,
  });
  path.addEventListener('click', () => showOntologyInspector(edge.detail));
  svg.insertBefore(path, svg.firstChild);
}

function showOntologyInspector(detail = {}) {
  const inspector = document.getElementById('ontology-inspector');
  inspector.innerHTML = `
    <span>${escapeHtml(detail.kind || 'ONTOLOGY')}</span>
    <h3>${escapeHtml(detail.title || '-')}</h3>
    <div class="inspector-value">${escapeHtml(String(detail.value ?? '-'))}</div>
    <p>${escapeHtml(detail.description || '')}</p>
    <div class="inspector-table">${(detail.rows || []).map(([label, value]) => `
      <div><span>${escapeHtml(String(label))}</span><b>${escapeHtml(String(value ?? '-'))}</b></div>
    `).join('')}</div>
  `;
}

function formatOntologyScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : 'N/A';
}

function formatOntologyRaw(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString('ko-KR', { maximumFractionDigits: 6 }) : 'N/A';
}

function truncateGraphText(value, maximum) {
  const text = String(value ?? '-');
  return text.length > maximum ? `${text.slice(0, maximum - 1)}…` : text;
}

function renderExecution(execution, book) {
  const stages = execution.stages || [];
  document.getElementById('execution-count').textContent = `${execution.event_count || 0} EVENTS`;
  document.getElementById('execution-track').innerHTML = stages.map((stage, index) => `
    <div class="execution-stage ${escapeHtml(stage.status)}">
      <span class="stage-number">${String(index + 1).padStart(2, '0')}</span>
      <b>${escapeHtml(stage.label)}</b>
      <small>${escapeHtml(stage.detail)}</small>
    </div>
  `).join('');
  const events = execution.events || [];
  document.getElementById('execution-tape').innerHTML = events.length ? events.map((event) => `
    <div class="tape-row">
      <b>${escapeHtml(shortClock(event.recorded_at))} · ${escapeHtml(event.event_type || '-')}</b>
      <span>${escapeHtml(compactPayload(event.payload))}</span>
    </div>
  `).join('') : '<div class="tape-empty">실제 주문 기록이 없습니다. 전략과 종목이 채택되면 매수·매도 판단부터 KIS 전송, 접수, 체결까지 여기에 표시됩니다.</div>';
  renderOrderbookAnalysis(book);
}

function renderOrderbookAnalysis(book) {
  const container = document.getElementById('orderbook-card');
  const ageSeconds = Number(book?.age_seconds);
  const stale = Boolean(book?.stale) || (Number.isFinite(ageSeconds) && ageSeconds > 90);
  const valid = Boolean(book?.valid)
    && Number(book?.best_bid) > 0
    && Number(book?.best_ask) >= Number(book?.best_bid);
  if (!book || !valid || stale) {
    const state = stale ? 'STALE ORDERBOOK' : 'NO VALID ORDERBOOK';
    const detail = stale
      ? `마지막 호가 ${formatAge(ageSeconds)} 전 · 오래된 잔량은 분석에 사용하지 않습니다.`
      : '유효한 매수·매도 호가가 들어오면 깊이 그래프가 표시됩니다.';
    container.className = `orderbook-card ${stale ? 'stale' : 'invalid'}`;
    container.innerHTML = `
      <div class="book-title">${state}</div>
      <div class="book-state-detail">${escapeHtml(detail)}</div>
      ${book ? `<div class="book-row last-known"><span>LAST BID ${formatPrice(book.best_bid)}</span><span>LAST ASK ${formatPrice(book.best_ask)}</span></div>` : ''}
    `;
    return;
  }

  const bidTotal = Math.max(0, Number(book.total_bid_volume) || 0);
  const askTotal = Math.max(0, Number(book.total_ask_volume) || 0);
  const total = bidTotal + askTotal;
  const bidShare = total > 0 ? Math.max(0, Math.min(100, bidTotal / total * 100)) : 50;
  const levels = (Array.isArray(book.levels) ? book.levels : []).slice(0, 10);
  const maximumDepth = Math.max(
    1,
    ...levels.flatMap((level) => [
      Math.max(0, Number(level.bid_size) || 0),
      Math.max(0, Number(level.ask_size) || 0),
    ]),
  );
  const depthRows = levels.map((level, index) => {
    const bidSize = Math.max(0, Number(level.bid_size) || 0);
    const askSize = Math.max(0, Number(level.ask_size) || 0);
    const bidWidth = bidSize / maximumDepth * 100;
    const askWidth = askSize / maximumDepth * 100;
    return `
      <div class="book-depth-row">
        <div class="book-depth-side bid">
          <i style="width:${bidWidth.toFixed(2)}%"></i>
          <span>${formatPrice(level.bid_price)}</span><b>${formatCompact(bidSize)}</b>
        </div>
        <em>L${index + 1}</em>
        <div class="book-depth-side ask">
          <i style="width:${askWidth.toFixed(2)}%"></i>
          <span>${formatPrice(level.ask_price)}</span><b>${formatCompact(askSize)}</b>
        </div>
      </div>
    `;
  }).join('');
  const spread = Number(book.spread_bps);
  const spreadClass = Number.isFinite(spread) && spread > 80 ? 'wide' : '';
  container.className = 'orderbook-card live';
  container.innerHTML = `
    <div class="book-title">LIVE ORDERBOOK DEPTH <small>${levels.length || 1} LEVELS</small></div>
    <div class="book-spread ${spreadClass}">SPREAD ${Number.isFinite(spread) ? `${spread.toFixed(2)} bps` : '-'}</div>
    <div class="book-row"><span>BID ${formatPrice(book.best_bid)}</span><span>ASK ${formatPrice(book.best_ask)}</span></div>
    <div class="book-row"><span>${formatCompact(bidTotal)} (${bidShare.toFixed(1)}%)</span><span>${formatCompact(askTotal)} (${(100 - bidShare).toFixed(1)}%)</span></div>
    <div class="book-imbalance"><i style="width:${bidShare}%"></i></div>
    <div class="book-depth-head"><span>매수 가격 · 잔량</span><span>매도 가격 · 잔량</span></div>
    <div class="book-depth">${depthRows || '<div class="book-state-detail">단계별 잔량 없음</div>'}</div>
  `;
}

function renderSafety(data) {
  const passed = (data.promotion_gates || []).filter((gate) => gate.passed);
  const failed = (data.promotion_gates || []).filter((gate) => !gate.passed);
  document.getElementById('safety-state').textContent = data.live_order_capable ? '제한 실행 가능' : '실주문 차단';
  document.getElementById('safety-reason').textContent = data.live_order_capable
    ? '모든 승격 게이트 통과'
    : `${passed.length}/${passed.length + failed.length} 게이트 통과 · ${failed.map((gate) => gate.label).join(' / ')}`;
}

function movingAverage(values, window) {
  return values.map((_, index) => {
    if (index + 1 < window) return null;
    const slice = values.slice(index + 1 - window, index + 1);
    return slice.reduce((sum, value) => sum + value, 0) / window;
  });
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function formatPrice(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  const digits = Math.abs(number) >= 1000 ? 0 : Math.abs(number) >= 10 ? 2 : 4;
  return number.toLocaleString('ko-KR', { maximumFractionDigits: digits });
}
function formatCompact(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return Intl.NumberFormat('ko-KR', { notation: 'compact', maximumFractionDigits: 2 }).format(number);
}
function formatKrw(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return `${Math.round(number).toLocaleString('ko-KR')}원`;
}
function formatSignedKrw(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return `${number > 0 ? '+' : ''}${Math.round(number).toLocaleString('ko-KR')}원`;
}
function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '-';
  return `${number > 0 ? '+' : ''}${(number * 100).toFixed(2)}%`;
}
function formatInteger(value) {
  return Math.round(Number(value || 0)).toLocaleString('ko-KR');
}
function formatCompactNumber(value) {
  return Number(value || 0).toLocaleString('ko-KR', { maximumFractionDigits: 2 });
}
function formatFileSize(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** index)).toFixed(index >= 3 ? 2 : 1)} ${units[index]}`;
}
function formatAge(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return '갱신 기록 없음';
  if (seconds < 60) return `${Math.round(seconds)}초 전 갱신`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}분 전 갱신`;
  return `${(seconds / 3600).toFixed(1)}시간 전 갱신`;
}
function shortClock(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value).slice(11, 19)
    : date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}
function shortDateTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
}
function formatTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('ko-KR', { hour12: false });
}
function shortTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value).slice(11, 16)
    : date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false });
}
function compactPayload(payload) {
  if (!payload || typeof payload !== 'object') return String(payload ?? '-');
  return Object.entries(payload)
    .filter(([key]) => [
      'symbol',
      'ticker',
      'action',
      'side',
      'quantity',
      'filled_quantity',
      'limit_price',
      'average_fill_price',
      'reason_code',
      'status',
      'intent_id',
      'verdict_id',
      'broker_order_id',
    ].includes(key))
    .map(([key, value]) => `${key}=${value}`)
    .join(' · ') || JSON.stringify(payload).slice(0, 180);
}
function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function syncDecisionOntologyFullscreen() {
  const panel = document.getElementById('decision-ontology-panel');
  const toggle = document.getElementById('decision-ontology-fullscreen');
  if (!panel || !toggle) return;
  const active = document.fullscreenElement === panel
    || panel.classList.contains('is-viewport-fullscreen');
  document.body.classList.toggle('decision-ontology-fullscreen', active);
  toggle.classList.toggle('active', active);
  toggle.setAttribute('aria-pressed', String(active));
  toggle.textContent = active ? '⛶ 원래 크기' : '⛶ 전체화면';
  toggle.title = active ? '네트워크 시각화 원래 크기로 복귀' : '네트워크 시각화 전체화면';
  // The SVG uses a viewBox, but dispatching resize also lets every browser
  // recalculate the fullscreen element before the next paint.
  requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
}

async function toggleDecisionOntologyFullscreen() {
  const panel = document.getElementById('decision-ontology-panel');
  if (!panel) return;
  const fallbackActive = panel.classList.contains('is-viewport-fullscreen');
  const nativeActive = document.fullscreenElement === panel;
  if (nativeActive && document.exitFullscreen) {
    await document.exitFullscreen();
    return;
  }
  if (fallbackActive) {
    panel.classList.remove('is-viewport-fullscreen');
    syncDecisionOntologyFullscreen();
    return;
  }
  if (panel.requestFullscreen) {
    try {
      await panel.requestFullscreen();
      return;
    } catch (_error) {
      // Embedded browsers can deny the native API. The viewport overlay keeps
      // the same graph-only behavior without making the control a dead button.
    }
  }
  panel.classList.add('is-viewport-fullscreen');
  syncDecisionOntologyFullscreen();
}

function bindDecisionOntologyFullscreen() {
  const panel = document.getElementById('decision-ontology-panel');
  const toggle = document.getElementById('decision-ontology-fullscreen');
  if (!panel || !toggle || toggle.dataset.bound === 'true') return;
  toggle.dataset.bound = 'true';
  toggle.addEventListener('click', () => {
    toggleDecisionOntologyFullscreen().catch(() => {
      panel.classList.add('is-viewport-fullscreen');
      syncDecisionOntologyFullscreen();
    });
  });
  document.addEventListener('fullscreenchange', syncDecisionOntologyFullscreen);
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !panel.classList.contains('is-viewport-fullscreen')) return;
    panel.classList.remove('is-viewport-fullscreen');
    syncDecisionOntologyFullscreen();
  });
  syncDecisionOntologyFullscreen();
}

function syncGnnModelFullscreen() {
  const panel = document.getElementById('gnn-model-panel');
  const toggle = document.getElementById('gnn-model-fullscreen');
  if (!panel || !toggle) return;
  const active = document.fullscreenElement === panel
    || panel.classList.contains('is-viewport-fullscreen');
  document.body.classList.toggle('gnn-model-fullscreen', active);
  toggle.classList.toggle('active', active);
  toggle.setAttribute('aria-pressed', String(active));
  toggle.textContent = active ? '⛶ 원래 크기' : '⛶ 전체화면';
  toggle.title = active
    ? '학습·추론 GNN 네트워크 원래 크기로 복귀'
    : '학습·추론 GNN 네트워크 전체화면';
  // Fullscreen changes the WebGL canvas box without changing its backing
  // buffer. Two frames let the browser finish layout before the renderer's
  // existing resize listener reads the new dimensions.
  requestAnimationFrame(() => requestAnimationFrame(() => {
    window.dispatchEvent(new Event('resize'));
  }));
}

async function toggleGnnModelFullscreen() {
  const panel = document.getElementById('gnn-model-panel');
  if (!panel) return;
  const fallbackActive = panel.classList.contains('is-viewport-fullscreen');
  const nativeActive = document.fullscreenElement === panel;
  if (nativeActive && document.exitFullscreen) {
    await document.exitFullscreen();
    return;
  }
  if (fallbackActive) {
    panel.classList.remove('is-viewport-fullscreen');
    syncGnnModelFullscreen();
    return;
  }
  if (panel.requestFullscreen) {
    try {
      await panel.requestFullscreen();
      return;
    } catch (_error) {
      // Some embedded browsers reject the native API. Use a viewport-fixed
      // surface so the graph control still behaves as fullscreen.
    }
  }
  panel.classList.add('is-viewport-fullscreen');
  syncGnnModelFullscreen();
}

function bindGnnModelFullscreen() {
  const panel = document.getElementById('gnn-model-panel');
  const toggle = document.getElementById('gnn-model-fullscreen');
  if (!panel || !toggle || toggle.dataset.bound === 'true') return;
  toggle.dataset.bound = 'true';
  toggle.addEventListener('click', () => {
    toggleGnnModelFullscreen().catch(() => {
      panel.classList.add('is-viewport-fullscreen');
      syncGnnModelFullscreen();
    });
  });
  document.addEventListener('fullscreenchange', syncGnnModelFullscreen);
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || !panel.classList.contains('is-viewport-fullscreen')) return;
    panel.classList.remove('is-viewport-fullscreen');
    syncGnnModelFullscreen();
  });
  syncGnnModelFullscreen();
}

function readGnnVisualizationPreference() {
  try {
    return localStorage.getItem(GNN_VISUALIZATION_STORAGE_KEY) === 'true';
  } catch (_error) {
    return false;
  }
}

function readGnnAutoRotationPreference() {
  try {
    const saved = localStorage.getItem(GNN_AUTO_ROTATION_STORAGE_KEY);
    return saved === null ? true : saved === 'true';
  } catch (_error) {
    return true;
  }
}

function applyGnnAutoRotationState() {
  const toggle = document.getElementById('gnn-auto-rotation-toggle');
  if (!toggle) return;
  toggle.classList.toggle('active', gnnAutoRotationEnabled);
  toggle.setAttribute('aria-pressed', String(gnnAutoRotationEnabled));
  toggle.textContent = gnnAutoRotationEnabled ? '자동 회전 끄기' : '자동 회전 켜기';
  toggle.title = gnnAutoRotationEnabled
    ? '3D 네트워크 자동 회전 끄기'
    : '3D 네트워크 자동 회전 켜기';
}

function bindGnnAutoRotationToggle() {
  const toggle = document.getElementById('gnn-auto-rotation-toggle');
  if (!toggle || toggle.dataset.bound === 'true') return;
  toggle.dataset.bound = 'true';
  toggle.addEventListener('click', () => {
    gnnAutoRotationEnabled = !gnnAutoRotationEnabled;
    try {
      localStorage.setItem(GNN_AUTO_ROTATION_STORAGE_KEY, String(gnnAutoRotationEnabled));
    } catch (_error) {
      /* Private mode: the choice applies until the page is closed. */
    }
    applyGnnAutoRotationState();
  });
  applyGnnAutoRotationState();
}

function applyGnnVisualizationState() {
  const toggle = document.getElementById('gnn-visualization-toggle');
  const paused = document.getElementById('gnn-visualization-paused');
  const status = document.getElementById('gnn-model-status');
  const summary = document.getElementById('gnn-model-summary');
  if (toggle) {
    toggle.classList.toggle('active', gnnVisualizationEnabled);
    toggle.setAttribute('aria-pressed', String(gnnVisualizationEnabled));
    toggle.textContent = gnnVisualizationEnabled ? '3D 시각화 끄기' : '3D 시각화 켜기';
  }
  if (paused) paused.hidden = gnnVisualizationEnabled;
  if (gnnVisualizationEnabled) {
    if (status) { status.textContent = 'LOADING'; status.className = 'status-chip waiting'; }
    if (summary) summary.textContent = '학습 체크포인트와 최근 추론 기록을 불러오는 중입니다.';
    fetchGnnGraph();
    fetchGnnState();
    return;
  }
  // Invalidate responses that were started before the operator switched the
  // visualization off. They must never repaint a newly enabled, newer scene.
  terminalState.gnnGraphRequestId += 1;
  terminalState.gnnStateRequestId += 1;
  terminalState.gnnGraphBusy = false;
  terminalState.gnnStateBusy = false;
  if (gnn3dState?.cleanup) gnn3dState.cleanup();
  gnn3dState = null;
  if (gnnGraphView.frame) cancelAnimationFrame(gnnGraphView.frame);
  gnnGraphView.frame = null;
  const tooltip = document.getElementById('gnn-model-tooltip');
  if (tooltip) tooltip.hidden = true;
  setGnnPhaseIndicator(null);
  if (status) { status.textContent = 'VISUALIZATION OFF'; status.className = 'status-chip'; }
  if (summary) summary.textContent = '3D 시각화만 꺼져 있습니다. GNN 학습과 추론은 계속 실행됩니다.';
  renderGnnLiveState({ state: 'OFFLINE', active: false, age_seconds: null });
}

function bindGnnVisualizationToggle() {
  const toggle = document.getElementById('gnn-visualization-toggle');
  if (!toggle || toggle.dataset.bound === 'true') return;
  toggle.dataset.bound = 'true';
  toggle.addEventListener('click', () => {
    gnnVisualizationEnabled = !gnnVisualizationEnabled;
    try {
      localStorage.setItem(GNN_VISUALIZATION_STORAGE_KEY, String(gnnVisualizationEnabled));
    } catch (_error) {
      /* Private mode: the choice applies until the page is closed. */
    }
    applyGnnVisualizationState();
  });
}

// Layout density. The dense grid is the default so a fresh browser lands on the
// one-screen view; the preference sticks per browser because an operator who
// wants the stacked view wants it on every reload, not once.
const layoutToggle = document.getElementById('layout-toggle');
if (layoutToggle) {
  const readLayout = () => {
    try {
      return localStorage.getItem('terminalLayout');
    } catch (error) {
      return null;
    }
  };
  const applyLayout = (classic) => {
    document.body.classList.toggle('classic-layout', classic);
    layoutToggle.textContent = classic ? '⤡' : '⤢';
    layoutToggle.title = classic ? '밀집(한 화면) 레이아웃으로 전환' : '기본(세로) 레이아웃으로 전환';
    // Canvases size themselves from their box, and a class flip fires no resize.
    window.dispatchEvent(new Event('resize'));
  };
  applyLayout(readLayout() === 'classic');
  layoutToggle.addEventListener('click', () => {
    const classic = !document.body.classList.contains('classic-layout');
    try {
      localStorage.setItem('terminalLayout', classic ? 'classic' : 'dense');
    } catch (error) {
      /* private mode: the choice just does not persist. */
    }
    applyLayout(classic);
  });
}
document.getElementById('terminal-refresh').addEventListener('click', () => {
  fetchMarketView();
  fetchAssetSummary();
  fetchSystemDiagnostics();
  fetchGnnGraph();
  fetchGnnState();
});
bindGnnVisualizationToggle();
applyGnnVisualizationState();
bindGnnAutoRotationToggle();
bindDecisionOntologyFullscreen();
bindGnnModelFullscreen();
// Every canvas sizes its backing store from its box, so a box that changed
// without a repaint shows a stretched or clipped chart. Frames are now
// resizable and movable, which makes that a routine event rather than a
// once-a-session one: repaint the canvas-backed panels whenever the layout
// reports a size change. The decision-ontology SVG is deliberately absent - it
// carries a viewBox and scales on its own, and rebuilding its node tree at
// drag framerate is the one repaint here that actually costs something.
window.addEventListener('resize', () => {
  if (terminalState.data) {
    renderTradingChart(
      terminalState.data.market || {},
      terminalState.data.algorithm,
    );
    renderTradingLayers(terminalState.data);
    renderSecondAnalysis(terminalState.data.market || {});
  }
  if (terminalState.diagnostics) {
    renderTrainingMonitor((terminalState.diagnostics.flows || {}).training || {});
  }
});
document.querySelectorAll('[data-chart-mode]').forEach((button) => {
  button.addEventListener('click', () => {
    terminalState.chartMode = button.dataset.chartMode || 'seconds';
    document.querySelectorAll('[data-chart-mode]').forEach((item) => {
      item.classList.toggle('active', item === button);
    });
    if (terminalState.data) {
      renderTradingChart(terminalState.data.market || {}, terminalState.data.algorithm);
      renderTradingLayers(terminalState.data);
    }
  });
});
function animateTradingChart(timestamp) {
  if (
    terminalState.chartMode === 'seconds'
    && terminalState.data
    && timestamp - terminalState.chartLastPaint >= 120
  ) {
    terminalState.chartLastPaint = timestamp;
    renderTradingChart(terminalState.data.market || {}, terminalState.data.algorithm);
  }
  terminalState.chartAnimationFrame = requestAnimationFrame(animateTradingChart);
}
setInterval(() => {
  document.getElementById('terminal-clock').textContent = new Date().toLocaleTimeString('ko-KR', { hour12: false });
}, 1000);
setInterval(() => fetchMarketStream(), 1000);
setInterval(() => fetchMarketView(), 3000);
setInterval(() => fetchSystemDiagnostics(), 5000);
setInterval(() => fetchAssetSummary(), 15000);
setInterval(() => fetchGnnGraph(), 15000);
setInterval(() => fetchGnnState(), 1000);
fetchMarketView();
fetchAssetSummary();
fetchSystemDiagnostics();
terminalState.chartAnimationFrame = requestAnimationFrame(animateTradingChart);
