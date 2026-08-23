/*
 * Movable, resizable frame layout for the strategy terminal.
 *
 * Three levels, which is the smallest model that expresses what an operator
 * actually wants to do with a wall of panels:
 *
 *   shell   stacks LAYERS down the screen        (grip between layers = height)
 *   layer   lays COLUMNS out side by side        (grip between columns = width)
 *   column  stacks FRAMES down the column        (grip between frames = height)
 *
 * So a frame can be widened, heightened, stacked under a neighbour, or dragged
 * by its ⠿ handle into any other layer - the phone-homescreen gesture. While a
 * drag is in flight the real layout reflows around a placeholder, so the
 * preview IS the result rather than an approximation of it.
 *
 * Sizes are flex-grow weights, never pixels, so an arrangement means the same
 * thing on a 1080p laptop and a 1440p monitor. Nothing persists on its own: the
 * operator presses 저장. Dragging is cheap and reversible; overwriting the
 * arrangement every browser loads is not.
 */
(() => {
  'use strict';

  const SHELL = document.querySelector('.terminal-shell');
  if (!SHELL) return;

  // Default arrangement. Mirrors the CSS grid in strategy_terminal.css - if you
  // change one, change the other, or the no-JS fallback and 초기화 disagree.
  const KNOWN_PANELS = [
    ['ops', '.ops-overview', '통합 운영 관제'],
    ['own', '.live-owner-strip', '실시간 자동 트레이딩'],
    ['ast', '.asset-overview', '내 자산'],
    ['sel', '.selection-strip', '온톨로지 후보'],
    ['ins', '.instrument-hero', '종목·알고리즘'],
    ['sec', '.second-analysis-panel', '1초 체결·호가'],
    ['wrk', '.workspace-grid', '가격 차트·결정 근거'],
    ['trd', '.trade-layers', '전략별 거래 레이어'],
    ['exe', '.execution-panel', '주문·체결 과정'],
    ['gnn', '.gnn-model-panel', '학습·추론 GNN'],
    ['don', '.decision-ontology-panel', '결정 온톨로지'],
    ['dia', '.diagnostics-panel', '시스템 진행·학습'],
  ];
  const DEFAULT_LAYERS = [
    { height: 1.1, columns: [col(6, 'ops'), col(3, 'own'), col(3, 'ast')] },
    { height: 0.8, columns: [col(3, 'sel'), col(4, 'ins'), col(5, 'sec')] },
    { height: 1.25, columns: [col(7, 'wrk'), col(5, 'exe')] },
    { height: 1.0, columns: [col(12, 'trd')] },
    { height: 1.15, columns: [col(5, 'gnn'), col(7, 'don')] },
    { height: 1.35, columns: [col(12, 'dia')] },
  ];

  function col(width, ...keys) {
    return { width, frames: keys.map((key) => ({ key, height: 1 })) };
  }

  const SCHEMA = 'strategy_terminal_layout_v2';
  const CACHE_KEY = 'terminalFrameLayout';
  const ENDPOINT = '/api/account/layout';
  const MIN_FRAME_W = 130;
  const MIN_FRAME_H = 66;
  const DENSE_QUERY = window.matchMedia('(min-width: 1200px) and (min-height: 680px)');

  const header = SHELL.querySelector('.terminal-header');
  const footer = SHELL.querySelector('.terminal-footer');
  const originalOrder = Array.from(SHELL.children);
  const saveButton = document.getElementById('layout-save');
  const resetButton = document.getElementById('layout-reset');

  // Every child between header and footer is a frame, including sections this
  // file has never heard of: a panel added later must not silently disappear
  // from a managed layout, so it gets a generated key and its own column.
  const frames = new Map();
  const labels = new Map();
  originalOrder.forEach((element, index) => {
    if (element === header || element === footer) return;
    const known = KNOWN_PANELS.find(([, selector]) => element.matches(selector));
    const key = known ? known[0] : `panel-${index + 1}`;
    element.dataset.tlFrame = key;
    frames.set(key, element);
    const heading = element.querySelector('h2, h3');
    labels.set(key, (known && known[2]) || (heading && heading.textContent.trim()) || key);
  });

  let currentSpec = cloneSpec(DEFAULT_LAYERS);
  let managed = false;
  let dirty = false;
  let flashTimer = 0;
  let paintQueued = false;
  let observer = null;

  /* ---------------------------------------------------------------- build */

  function build(spec) {
    teardown();
    const placed = new Set();
    const layerNodes = [];

    spec.forEach((layerSpec) => {
      const layer = document.createElement('div');
      layer.className = 'tl-layer';
      layer.style.flexGrow = String(weight(layerSpec.height));
      (layerSpec.columns || []).forEach((columnSpec) => {
        const members = (columnSpec.frames || [])
          .filter((frame) => frames.has(frame.key) && !placed.has(frame.key));
        if (!members.length) return;
        const column = makeColumn(weight(columnSpec.width));
        members.forEach((frame) => {
          column.appendChild(dressFrame(frame.key, weight(frame.height)));
          placed.add(frame.key);
        });
        layer.appendChild(column);
      });
      if (layer.children.length) layerNodes.push(layer);
    });

    const orphans = Array.from(frames.keys()).filter((key) => !placed.has(key));
    if (orphans.length) {
      const layer = document.createElement('div');
      layer.className = 'tl-layer';
      layer.style.flexGrow = '1';
      orphans.forEach((key) => {
        const column = makeColumn(1);
        column.appendChild(dressFrame(key, 1));
        layer.appendChild(column);
      });
      layerNodes.push(layer);
    }

    layerNodes.forEach((layer) => insertBeforeFooter(layer));
    SHELL.classList.add('tl-managed');
    syncGrips();
    observeFrames();
  }

  function makeColumn(width) {
    const column = document.createElement('div');
    column.className = 'tl-col';
    column.style.flexGrow = String(width);
    return column;
  }

  function dressFrame(key, height) {
    const element = frames.get(key);
    element.style.flexGrow = String(height);
    if (!element.querySelector(':scope > .tl-frame-tools')) {
      // Sticky and zero-height: the handle stays reachable while the frame's
      // own content scrolls under it, without stealing a row of layout.
      const tools = document.createElement('div');
      tools.className = 'tl-frame-tools';
      const move = document.createElement('button');
      move.type = 'button';
      move.className = 'tl-move';
      move.title = `${labels.get(key)} · 드래그하여 다른 층·열로 이동`;
      move.setAttribute('aria-label', `${labels.get(key)} 프레임 이동`);
      move.textContent = '⠿';
      move.addEventListener('pointerdown', (event) => startMove(event, element));
      tools.appendChild(move);
      element.insertBefore(tools, element.firstChild);
    }
    return element;
  }

  function insertBeforeFooter(node) {
    if (footer && footer.parentNode === SHELL) SHELL.insertBefore(node, footer);
    else SHELL.appendChild(node);
  }

  function teardown() {
    if (observer) observer.disconnect();
    SHELL.querySelectorAll('.tl-grip, .tl-placeholder').forEach((node) => node.remove());
    SHELL.querySelectorAll('.tl-frame-tools').forEach((node) => node.remove());
    // Re-appending in the source order both restores the stacked document and
    // empties the layers, which are then safe to drop.
    originalOrder.forEach((element) => {
      element.style.flexGrow = '';
      SHELL.appendChild(element);
    });
    SHELL.querySelectorAll('.tl-layer, .tl-col').forEach((node) => node.remove());
    SHELL.classList.remove('tl-managed');
  }

  /* ---------------------------------------------------------------- grips */

  // Grips are derived from the tree, never bookkept: after any structural
  // change (a move, a dropped column) this rebuilds every splitter from what
  // the DOM actually contains, so no path can leave a stale one behind.
  function syncGrips() {
    SHELL.querySelectorAll('.tl-grip').forEach((grip) => grip.remove());
    const layers = childrenOf(SHELL, '.tl-layer');
    layers.forEach((layer, index) => {
      if (index) SHELL.insertBefore(makeGrip('y'), layer);
      const columns = childrenOf(layer, '.tl-col');
      columns.forEach((column, columnIndex) => {
        if (columnIndex) layer.insertBefore(makeGrip('x'), column);
        childrenOf(column, '[data-tl-frame]').forEach((frame, frameIndex) => {
          if (frameIndex) column.insertBefore(makeGrip('y'), frame);
        });
      });
    });
  }

  function childrenOf(parent, selector) {
    return Array.from(parent.children).filter((child) => child.matches(selector));
  }

  function makeGrip(axis) {
    const grip = document.createElement('div');
    grip.className = `tl-grip tl-grip-${axis}`;
    grip.setAttribute('role', 'separator');
    grip.setAttribute('tabindex', '0');
    grip.setAttribute('aria-orientation', axis === 'x' ? 'vertical' : 'horizontal');
    grip.title = axis === 'x'
      ? '드래그하여 좌우 폭 조절 (←/→ 키로 미세 조정)'
      : '드래그하여 위아래 높이 조절 (↑/↓ 키로 미세 조정)';
    grip.addEventListener('pointerdown', (event) => startDrag(event, axis, grip));
    grip.addEventListener('keydown', (event) => nudge(event, axis, grip));
    return grip;
  }

  /* ---------------------------------------------------------------- resize */

  /*
   * Pixels to weights. A flex-basis:0 item is laid out as its own padding and
   * border PLUS its share of the free space, so converting a pointer delta with
   * the naive size ratio leaves the boundary trailing the cursor by whatever
   * those edges add up to (~8% on these panels). Measure the edges, solve for
   * px-per-weight, and the splitter sits exactly under the pointer.
   */
  function axisMetrics(element, horizontal) {
    const style = window.getComputedStyle(element);
    const edges = horizontal
      ? ['paddingLeft', 'paddingRight', 'borderLeftWidth', 'borderRightWidth']
      : ['paddingTop', 'paddingBottom', 'borderTopWidth', 'borderBottomWidth'];
    const edge = edges.reduce((total, name) => total + (parseFloat(style[name]) || 0), 0);
    return { size: horizontal ? element.offsetWidth : element.offsetHeight, edge };
  }

  function resizePair(before, after, horizontal, targetBeforeSize) {
    const b = axisMetrics(before, horizontal);
    const a = axisMetrics(after, horizontal);
    const totalGrow = growOf(before) + growOf(after);
    const scale = (b.size - b.edge + a.size - a.edge) / totalGrow;
    if (!(scale > 0)) return false;
    const minimum = horizontal ? MIN_FRAME_W : MIN_FRAME_H;
    const low = Math.max(minimum, b.edge + 1);
    const high = b.size + a.size - Math.max(minimum, a.edge + 1);
    if (low >= high) return false;
    const target = clamp(targetBeforeSize, low, high);
    const growBefore = clamp((target - b.edge) / scale, 0.02, totalGrow - 0.02);
    before.style.flexGrow = growBefore.toFixed(4);
    after.style.flexGrow = (totalGrow - growBefore).toFixed(4);
    return true;
  }

  function startDrag(event, axis, grip) {
    if (event.button !== 0) return;
    const before = grip.previousElementSibling;
    const after = grip.nextElementSibling;
    if (!before || !after) return;
    event.preventDefault();

    const horizontal = axis === 'x';
    const origin = horizontal ? event.clientX : event.clientY;
    const beforeSize = horizontal ? before.offsetWidth : before.offsetHeight;
    const afterSize = horizontal ? after.offsetWidth : after.offsetHeight;
    const minimum = horizontal ? MIN_FRAME_W : MIN_FRAME_H;
    if (beforeSize + afterSize <= minimum * 2) return;

    capture(grip, event.pointerId);
    grip.classList.add('tl-active');
    document.body.classList.add('tl-dragging', `tl-dragging-${axis}`);

    const onMove = (moveEvent) => {
      const position = horizontal ? moveEvent.clientX : moveEvent.clientY;
      if (resizePair(before, after, horizontal, beforeSize + (position - origin))) schedulePaint();
    };
    const onEnd = () => {
      grip.removeEventListener('pointermove', onMove);
      grip.classList.remove('tl-active');
      document.body.classList.remove('tl-dragging', 'tl-dragging-x', 'tl-dragging-y');
      try {
        grip.releasePointerCapture(event.pointerId);
      } catch (error) {
        /* the pointer already left; capture is gone either way. */
      }
      markDirty();
      notifyResize();
    };

    grip.addEventListener('pointermove', onMove);
    grip.addEventListener('pointerup', onEnd, { once: true });
    grip.addEventListener('pointercancel', onEnd, { once: true });
  }

  function nudge(event, axis, grip) {
    const horizontal = axis === 'x';
    const keys = horizontal ? ['ArrowLeft', 'ArrowRight'] : ['ArrowUp', 'ArrowDown'];
    const direction = keys.indexOf(event.key);
    if (direction < 0) return;
    const before = grip.previousElementSibling;
    const after = grip.nextElementSibling;
    if (!before || !after) return;
    event.preventDefault();
    const step = (direction === 0 ? -1 : 1) * (event.shiftKey ? 48 : 16);
    const beforeSize = horizontal ? before.offsetWidth : before.offsetHeight;
    if (!resizePair(before, after, horizontal, beforeSize + step)) return;
    markDirty();
    notifyResize();
  }

  /* ------------------------------------------------------------------ move */

  const moveState = { active: false };

  function startMove(event, element) {
    if (event.button !== 0 || moveState.active) return;
    event.preventDefault();
    event.stopPropagation();
    const handle = event.currentTarget;
    capture(handle, event.pointerId);

    const column = element.parentElement;
    moveState.active = true;
    moveState.element = element;
    moveState.handle = handle;
    moveState.pointerId = event.pointerId;
    moveState.homeColumn = column;
    moveState.homeNext = nextFrame(element);
    moveState.grow = growOf(element);
    moveState.signature = null;

    // Grips are re-derived on drop, so drop them now: the preview only has to
    // reason about layers, columns and frames.
    SHELL.querySelectorAll('.tl-grip').forEach((grip) => grip.remove());
    document.body.classList.add('tl-moving');

    moveState.chip = document.createElement('div');
    moveState.chip.className = 'tl-drag-chip';
    moveState.chip.textContent = labels.get(element.dataset.tlFrame) || '프레임';
    document.body.appendChild(moveState.chip);
    positionChip(event);

    moveState.placeholder = document.createElement('div');
    moveState.placeholder.className = 'tl-placeholder';
    moveState.placeholder.innerHTML = '<span></span>';
    moveState.placeholder.firstChild.textContent = `${labels.get(element.dataset.tlFrame)} · 여기로 이동`;
    moveState.placeholder.style.flexGrow = String(moveState.grow);
    column.insertBefore(moveState.placeholder, element);
    element.classList.add('tl-ghosted');
    element.style.display = 'none';

    handle.addEventListener('pointermove', onMovePointer);
    handle.addEventListener('pointerup', finishMove, { once: true });
    handle.addEventListener('pointercancel', cancelMove, { once: true });
    window.addEventListener('keydown', onMoveKey, true);
  }

  function onMovePointer(event) {
    if (!moveState.active) return;
    positionChip(event);
    const target = dropTargetAt(event.clientX, event.clientY);
    if (!target || target.signature === moveState.signature) return;
    moveState.signature = target.signature;
    applyPreview(target);
  }

  function positionChip(event) {
    moveState.chip.style.transform = `translate(${event.clientX + 14}px, ${event.clientY + 14}px)`;
  }

  /*
   * Where would a drop land? Edge zones answer "beside", middles answer
   * "inside", which is the same vocabulary a homescreen uses: near the rim of a
   * column you create a neighbour, over its body you join the stack.
   */
  function dropTargetAt(x, y) {
    const layers = childrenOf(SHELL, '.tl-layer');
    if (!layers.length) return null;
    const layer = layers.find((node) => within(node.getBoundingClientRect(), y, 'y'))
      || nearest(layers, y, 'y');
    const layerBox = layer.getBoundingClientRect();
    const edgeY = clamp(layerBox.height * 0.14, 12, 42);
    if (y < layerBox.top + edgeY) return { type: 'layer', ref: layer, before: true, signature: `L<${index(layer)}` };
    if (y > layerBox.bottom - edgeY) return { type: 'layer', ref: layer, before: false, signature: `L>${index(layer)}` };

    const columns = childrenOf(layer, '.tl-col');
    if (!columns.length) return { type: 'layer', ref: layer, before: true, signature: `L<${index(layer)}` };
    const column = columns.find((node) => within(node.getBoundingClientRect(), x, 'x'))
      || nearest(columns, x, 'x');
    const columnBox = column.getBoundingClientRect();
    const edgeX = clamp(columnBox.width * 0.22, 22, 90);
    const columnId = `${index(layer)}:${index(column)}`;
    if (x < columnBox.left + edgeX) return { type: 'column', ref: column, before: true, signature: `C<${columnId}` };
    if (x > columnBox.right - edgeX) return { type: 'column', ref: column, before: false, signature: `C>${columnId}` };

    const members = childrenOf(column, '[data-tl-frame], .tl-placeholder')
      .filter((node) => node.style.display !== 'none');
    if (!members.length) return { type: 'column', ref: column, before: true, signature: `C<${columnId}` };
    const frame = members.find((node) => within(node.getBoundingClientRect(), y, 'y')) || nearest(members, y, 'y');
    const frameBox = frame.getBoundingClientRect();
    const before = y < frameBox.top + frameBox.height / 2;
    return { type: 'frame', ref: frame, before, signature: `F${before ? '<' : '>'}${columnId}:${index(frame)}` };
  }

  function within(box, value, axis) {
    return axis === 'x' ? value >= box.left && value <= box.right : value >= box.top && value <= box.bottom;
  }

  function nearest(nodes, value, axis) {
    let best = nodes[0];
    let bestDistance = Infinity;
    nodes.forEach((node) => {
      const box = node.getBoundingClientRect();
      const centre = axis === 'x' ? (box.left + box.right) / 2 : (box.top + box.bottom) / 2;
      const distance = Math.abs(centre - value);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = node;
      }
    });
    return best;
  }

  function index(node) {
    return Array.from(node.parentElement.children).indexOf(node);
  }

  function applyPreview(target) {
    const placeholder = moveState.placeholder;
    detachPlaceholder();
    SHELL.querySelectorAll('.tl-drop-layer').forEach((node) => node.classList.remove('tl-drop-layer'));

    if (target.type === 'frame') {
      const column = target.ref.parentElement;
      placeholder.style.flexGrow = String(averageGrow(childrenOf(column, '[data-tl-frame]')));
      column.insertBefore(placeholder, target.before ? target.ref : target.ref.nextSibling);
      column.closest('.tl-layer').classList.add('tl-drop-layer');
      return;
    }
    if (target.type === 'column') {
      const layer = target.ref.parentElement;
      const column = makeColumn(averageGrow(childrenOf(layer, '.tl-col')));
      column.classList.add('tl-temp');
      placeholder.style.flexGrow = '1';
      column.appendChild(placeholder);
      layer.insertBefore(column, target.before ? target.ref : target.ref.nextSibling);
      layer.classList.add('tl-drop-layer');
      return;
    }
    const layer = document.createElement('div');
    layer.className = 'tl-layer tl-temp';
    layer.style.flexGrow = String(averageGrow(childrenOf(SHELL, '.tl-layer')));
    const column = makeColumn(1);
    column.classList.add('tl-temp');
    placeholder.style.flexGrow = '1';
    column.appendChild(placeholder);
    layer.appendChild(column);
    SHELL.insertBefore(layer, target.before ? target.ref : target.ref.nextSibling);
    layer.classList.add('tl-drop-layer');
  }

  function detachPlaceholder() {
    const placeholder = moveState.placeholder;
    if (!placeholder || !placeholder.parentElement) return;
    const column = placeholder.parentElement;
    const layer = column.parentElement;
    placeholder.remove();
    if (column.classList.contains('tl-temp') && !column.children.length) column.remove();
    if (layer && layer.classList.contains('tl-temp') && !childrenOf(layer, '.tl-col').length) layer.remove();
  }

  function finishMove() {
    if (!moveState.active) return;
    const { element, placeholder } = moveState;
    const column = placeholder.parentElement;
    if (column) {
      element.style.flexGrow = placeholder.style.flexGrow || '1';
      column.insertBefore(element, placeholder);
      column.classList.remove('tl-temp');
      const layer = column.parentElement;
      if (layer) layer.classList.remove('tl-temp');
    }
    endMove();
    pruneEmpty();
    syncGrips();
    markDirty();
    notifyResize();
  }

  function cancelMove() {
    if (!moveState.active) return;
    const { element, homeColumn, homeNext } = moveState;
    if (homeColumn && homeColumn.isConnected) homeColumn.insertBefore(element, homeNext);
    endMove();
    pruneEmpty();
    syncGrips();
    notifyResize();
  }

  function onMoveKey(event) {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    cancelMove();
  }

  function endMove() {
    const { element, handle, placeholder, chip, pointerId } = moveState;
    detachPlaceholder();
    if (placeholder) placeholder.remove();
    if (chip) chip.remove();
    element.style.display = '';
    element.classList.remove('tl-ghosted');
    handle.removeEventListener('pointermove', onMovePointer);
    window.removeEventListener('keydown', onMoveKey, true);
    try {
      handle.releasePointerCapture(pointerId);
    } catch (error) {
      /* the pointer already left; capture is gone either way. */
    }
    document.body.classList.remove('tl-moving');
    SHELL.querySelectorAll('.tl-drop-layer').forEach((node) => node.classList.remove('tl-drop-layer'));
    moveState.active = false;
  }

  function pruneEmpty() {
    childrenOf(SHELL, '.tl-layer').forEach((layer) => {
      childrenOf(layer, '.tl-col').forEach((column) => {
        if (!childrenOf(column, '[data-tl-frame]').length) column.remove();
      });
      if (!childrenOf(layer, '.tl-col').length) layer.remove();
    });
  }

  function nextFrame(element) {
    let node = element.nextElementSibling;
    while (node && !node.matches('[data-tl-frame]')) node = node.nextElementSibling;
    return node;
  }

  function averageGrow(nodes) {
    if (!nodes.length) return 1;
    const total = nodes.reduce((sum, node) => sum + growOf(node), 0);
    return round(total / nodes.length);
  }

  /* --------------------------------------------------------- auto resizing */

  function observeFrames() {
    if (typeof ResizeObserver === 'undefined') return;
    // Charts, the GNN canvas and the ontology SVG all size themselves from
    // their box on a resize event. Watching the frames means content follows
    // any size change - a drag, a move, a restored layout - not just a window
    // resize.
    observer = observer || new ResizeObserver(() => schedulePaint());
    observer.disconnect();
    frames.forEach((element) => observer.observe(element));
  }

  function schedulePaint() {
    if (paintQueued) return;
    paintQueued = true;
    requestAnimationFrame(() => {
      paintQueued = false;
      notifyResize();
    });
  }

  function notifyResize() {
    window.dispatchEvent(new Event('resize'));
  }

  /* ------------------------------------------------------------ persistence */

  function readSpec() {
    return childrenOf(SHELL, '.tl-layer').map((layer) => ({
      height: round(growOf(layer)),
      columns: childrenOf(layer, '.tl-col').map((column) => ({
        width: round(growOf(column)),
        frames: childrenOf(column, '[data-tl-frame]').map((frame) => ({
          key: frame.dataset.tlFrame,
          height: round(growOf(frame)),
        })),
      })).filter((column) => column.frames.length),
    })).filter((layer) => layer.columns.length);
  }

  function markDirty() {
    dirty = true;
    if (!saveButton) return;
    window.clearTimeout(flashTimer);
    saveButton.classList.remove('saved', 'failed');
    saveButton.classList.add('dirty');
    saveButton.textContent = '레이아웃 저장 •';
  }

  function flash(state, label) {
    if (!saveButton) return;
    window.clearTimeout(flashTimer);
    saveButton.classList.remove('dirty', 'saved', 'failed');
    if (state) saveButton.classList.add(state);
    saveButton.textContent = label;
    flashTimer = window.setTimeout(() => {
      saveButton.classList.remove('saved', 'failed');
      saveButton.textContent = dirty ? '레이아웃 저장 •' : '레이아웃 저장';
      if (dirty) saveButton.classList.add('dirty');
    }, 2400);
  }

  function cacheLocally(payload) {
    try {
      window.localStorage.setItem(CACHE_KEY, JSON.stringify(payload));
    } catch (error) {
      /* private mode: the server copy is still the real one. */
    }
  }

  function readCache() {
    try {
      return JSON.parse(window.localStorage.getItem(CACHE_KEY) || 'null');
    } catch (error) {
      return null;
    }
  }

  async function save() {
    if (!managed) return;
    const payload = { schema: SCHEMA, layers: readSpec() };
    if (!payload.layers.length) return;
    currentSpec = cloneSpec(payload.layers);
    cacheLocally(payload);
    try {
      const response = await fetch(ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(String(response.status));
      dirty = false;
      flash('saved', '저장됨');
    } catch (error) {
      // The local cache already holds it, so a reload keeps the arrangement -
      // but say plainly that the server copy did not land.
      flash('failed', '저장 실패');
    }
  }

  async function reset() {
    if (dirty && !window.confirm('저장된 레이아웃을 지우고 기본 배치로 되돌립니다. 계속할까요?')) return;
    currentSpec = cloneSpec(DEFAULT_LAYERS);
    if (managed) build(currentSpec);
    try {
      window.localStorage.removeItem(CACHE_KEY);
    } catch (error) {
      /* nothing cached. */
    }
    dirty = false;
    try {
      await fetch(ENDPOINT, { method: 'DELETE' });
      flash('saved', '기본 배치');
    } catch (error) {
      flash('failed', '초기화 실패');
    }
    notifyResize();
  }

  async function loadSaved() {
    const cached = readCache();
    if (cached && Array.isArray(cached.layers) && cached.layers.length) {
      currentSpec = cloneSpec(cached.layers);
      if (managed) build(currentSpec);
    }
    try {
      const response = await fetch(ENDPOINT, { cache: 'no-store' });
      if (!response.ok) return;
      const stored = await response.json();
      if (!stored || !Array.isArray(stored.layers) || !stored.layers.length) return;
      currentSpec = cloneSpec(stored.layers);
      cacheLocally({ schema: SCHEMA, layers: currentSpec });
      if (managed) build(currentSpec);
      notifyResize();
    } catch (error) {
      /* offline or route missing: the cached or default layout stands. */
    }
  }

  /* --------------------------------------------------------------- lifecycle */

  function sync() {
    const shouldManage = DENSE_QUERY.matches && !document.body.classList.contains('classic-layout');
    if (shouldManage === managed) return;
    managed = shouldManage;
    if (managed) build(currentSpec);
    else teardown();
    [saveButton, resetButton].forEach((button) => {
      if (button) button.hidden = !managed;
    });
    notifyResize();
  }

  // A saved v1 layout (flat panels per layer) predates columns; every panel was
  // its own full-height column, so that is what it becomes.
  function cloneSpec(spec) {
    return spec.map((layer) => {
      const columns = Array.isArray(layer.columns) && layer.columns.length
        ? layer.columns
        : (layer.panels || []).map((panel) => ({ width: panel.width, frames: [{ key: panel.key, height: 1 }] }));
      return {
        height: weight(layer.height),
        columns: columns.map((column) => ({
          width: weight(column.width),
          frames: (column.frames || []).map((frame) => ({ key: frame.key, height: weight(frame.height) })),
        })).filter((column) => column.frames.length),
      };
    }).filter((layer) => layer.columns.length);
  }

  // Capture keeps the pointer stream on the handle once the cursor leaves its
  // 7 by 26 pixels. It is an optimisation, not a requirement: a pointer that
  // cannot be captured still drags, so a failure here must not abort the drag.
  function capture(element, pointerId) {
    try {
      element.setPointerCapture(pointerId);
    } catch (error) {
      /* no active pointer for this id. */
    }
  }

  function growOf(element) {
    const value = parseFloat(window.getComputedStyle(element).flexGrow);
    return Number.isFinite(value) && value > 0 ? value : 1;
  }

  function weight(value) {
    const number = Number(value);
    return Number.isFinite(number) ? clamp(number, 0.05, 24) : 1;
  }

  function clamp(value, low, high) {
    return Math.min(Math.max(value, low), high);
  }

  function round(value) {
    return Math.round(value * 10000) / 10000;
  }

  if (saveButton) saveButton.addEventListener('click', save);
  if (resetButton) resetButton.addEventListener('click', reset);
  // The density toggle lives in strategy_terminal.js and only flips a body
  // class; watching the class keeps the two files from having to know about
  // each other.
  new MutationObserver(sync).observe(document.body, { attributes: true, attributeFilter: ['class'] });
  DENSE_QUERY.addEventListener('change', sync);

  sync();
  loadSaved();
})();
