// Body coverage: the skeleton figure, the soft-tissue list and the structure x dataset matrix, from coverageData and
// structuresData (data.js). Clicking a structure lists its scans in the scan table (app.js: BoneHubTables.scans).
(function() {
    const SVG_NS = 'http://www.w3.org/2000/svg';
    // Ordinal blue ramp, light to dark (steps 250, 350, 450, 550, 700; validated: monotone, visible gaps, the light end
    // clears 2:1 on white); "no data" is a neutral gray.
    const RAMP = ['#84b9f3', '#519ced', '#257de0', '#1b64be', '#0e3d7e'];
    const NO_DATA = { fill: '#ededed', stroke: '#c2c2c2' };
    const KIND_LABELS = { any: 'Segmentation or mesh', segmentation: 'Segmentation', mesh: 'Mesh' };

    let mode = 'any';
    let matrixMode = 'scans';
    let groups = [];
    let groupIndex = new Map();
    let totalScans = 0;
    let tooltip = null;

    function formatNumber(value) {
        return Number(value).toLocaleString('en-US');
    }

    function compact(value) {
        if (value >= 10000) return `${Math.round(value / 1000)}k`;
        if (value >= 1000) return `${(value / 1000).toFixed(1).replace(/\.0$/, '')}k`;
        return String(value);
    }

    function countOf(groupId, kind) {
        const coverage = (coverageData || {})[groupId];
        if (!coverage) return 0;
        return kind === 'any' ? coverage.scans : (coverage.kinds[kind] || 0);
    }

    // Log-scale bins: bin k holds counts up to max^((k + 1) / RAMP.length).
    function binner(max) {
        const top = Math.max(1, max);
        const edges = RAMP.map((_, k) => Math.max(1, Math.round(Math.pow(top, (k + 1) / RAMP.length))));
        return {
            edges,
            color(count) {
                if (!count) return null;
                for (let k = 0; k < edges.length; k++) {
                    if (count <= edges[k]) return RAMP[k];
                }
                return RAMP[RAMP.length - 1];
            }
        };
    }

    function svg(tag, attrs, parent) {
        const el = document.createElementNS(SVG_NS, tag);
        Object.entries(attrs || {}).forEach(([key, value]) => el.setAttribute(key, value));
        if (parent) parent.appendChild(el);
        return el;
    }

    // ---- tooltip -------------------------------------------------------------------------------------------------

    function showTooltip(event, lines) {
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.className = 'viz-tooltip';
            tooltip.setAttribute('role', 'tooltip');
            document.body.appendChild(tooltip);
        }
        tooltip.innerHTML = '';
        lines.forEach((line, i) => {
            const p = document.createElement(i === 0 ? 'strong' : 'span');
            p.textContent = line;
            tooltip.appendChild(p);
        });
        tooltip.style.display = 'flex';
        const rect = tooltip.getBoundingClientRect();
        let x = event.clientX + 14;
        let y = event.clientY + 14;
        if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - 14;
        if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - 14;
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
    }

    function hideTooltip() {
        if (tooltip) tooltip.style.display = 'none';
    }

    function groupLines(group) {
        const coverage = (coverageData || {})[group.id];
        if (!coverage) {
            return [group.name, 'No scans in this database'];
        }
        const share = totalScans ? ` (${(100 * coverage.scans / totalScans).toFixed(1)}% of scans)` : '';
        const kinds = Object.entries(coverage.kinds).map(([k, n]) => `${KIND_LABELS[k] || k}: ${formatNumber(n)}`).join(' · ');
        const origins = Object.entries(coverage.origins).map(([o, n]) => `${o} ${formatNumber(n)}`).join(' · ');
        const providers = Object.keys(coverage.providers).length;
        return [
            group.name,
            `${formatNumber(coverage.scans)} scans${share}`,
            kinds,
            `From ${providers} dataset${providers === 1 ? '' : 's'}; origin: ${origins}`,
            'Click to list these scans'
        ];
    }

    function selectGroup(group) {
        const tables = window.BoneHubTables || {};
        const index = groupIndex.get(group.id);
        if (!tables.scans || index === undefined) return;
        tables.scans.setFilter(group.name, row => Array.isArray(row._groups) && row._groups.includes(index));
        const target = document.getElementById('scans');
        if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function makeInteractive(el, group) {
        el.setAttribute('tabindex', '0');
        el.setAttribute('role', 'button');
        el.setAttribute('aria-label', groupLines(group).slice(0, 2).join(': '));
        el.addEventListener('mousemove', event => showTooltip(event, groupLines(group)));
        el.addEventListener('mouseleave', hideTooltip);
        el.addEventListener('focus', () => {
            const box = el.getBoundingClientRect();
            showTooltip({ clientX: box.right, clientY: box.top }, groupLines(group));
        });
        el.addEventListener('blur', hideTooltip);
        el.addEventListener('click', () => selectGroup(group));
        el.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                selectGroup(group);
            }
        });
    }

    // ---- skeleton figure -----------------------------------------------------------------------------------------
    // Front view in anatomical position: the subject's right side is on the viewer's left. Sided shapes are drawn
    // for the right side and mirrored for the left. Each shape is ['fill' | 'stroke', svg tag, attributes].

    const RIB_WIDTH = [36, 46, 54, 60, 64, 66, 67, 66, 64, 60, 52, 42];

    function sidedShapes() {
        const shapes = {
            scapula: [['fill', 'path', { d: 'M 98 178 L 134 190 L 122 248 Z' }]],
            clavicle: [['stroke', 'path', { d: 'M 153 178 C 140 171, 120 176, 97 168', 'stroke-width': 6 }]],
            humerus: [
                ['fill', 'circle', { cx: 99, cy: 181, r: 10 }],
                ['stroke', 'path', { d: 'M 96 188 L 86 300', 'stroke-width': 12 }]
            ],
            radius: [['stroke', 'path', { d: 'M 81 310 L 69 410', 'stroke-width': 6 }]],
            ulna: [['stroke', 'path', { d: 'M 92 310 L 86 412', 'stroke-width': 6 }]],
            carpals: [['fill', 'rect', { x: 63, y: 416, width: 28, height: 14, rx: 5 }]],
            metacarpals: [0, 1, 2, 3, 4].map(k => ['stroke', 'path', {
                d: k === 0 ? 'M 64 431 L 55 450' : `M ${66 + k * 6} 433 L ${61 + k * 7.5} 461`, 'stroke-width': 3.5
            }]),
            fingers: [0, 1, 2, 3, 4].map(k => ['stroke', 'path', {
                d: k === 0 ? 'M 53 455 L 46 474' : `M ${61 + k * 7.5} 465 L ${59 + k * 8.4} ${501 - (k === 4 ? 7 : 0)}`, 'stroke-width': 3.5
            }]),
            hip: [['fill', 'path', {
                d: 'M 142 372 C 120 352, 98 356, 96 378 C 95 398, 110 412, 122 424 C 124 440, 134 452, 152 452 L 156 440 C 146 436, 144 426, 148 414 L 146 392 Z'
            }]],
            femur: [
                ['fill', 'circle', { cx: 137, cy: 437, r: 9 }],
                ['stroke', 'path', { d: 'M 128 446 L 140 592', 'stroke-width': 13 }]
            ],
            patella: [['fill', 'ellipse', { cx: 142, cy: 603, rx: 8, ry: 10 }]],
            tibia: [['stroke', 'path', { d: 'M 142 617 L 140 716', 'stroke-width': 11 }]],
            fibula: [['stroke', 'path', { d: 'M 127 621 L 128 714', 'stroke-width': 5 }]],
            tarsals: [['fill', 'rect', { x: 124, y: 720, width: 28, height: 13, rx: 5 }]],
            metatarsals: [0, 1, 2, 3, 4].map(k => ['stroke', 'path', { d: `M ${127 + k * 5.5} 737 L ${124 + k * 7} 756`, 'stroke-width': 3 }]),
            toes: [0, 1, 2, 3, 4].map(k => ['stroke', 'path', { d: `M ${124 + k * 7} 760 L ${123 + k * 7.4} ${773 - k}`, 'stroke-width': 3 }])
        };
        RIB_WIDTH.forEach((w, i) => {
            const n = i + 1;
            const y = 190 + i * 9.5;
            const x0 = 154 - Math.max(0, n - 7) * 6; // ribs 8-12 do not reach the sternum
            const outer = 160 - w;
            shapes[`rib${n}`] = [['stroke', 'path', {
                d: `M ${x0} ${y + Math.max(0, n - 7) * 4} C ${x0 - 14} ${y - 10}, ${outer} ${y - 8}, ${outer} ${y + 8} C ${outer} ${y + 16}, ${outer + 4} ${y + 22}, ${outer + 8} ${y + 26}`,
                'stroke-width': 3.2
            }]];
        });
        return shapes;
    }

    function midlineShapes() {
        const shapes = {
            skull: [
                ['fill', 'path', { d: 'M 160 14 C 200 14, 198 66, 190 86 L 182 92 C 176 82, 144 82, 138 92 L 130 86 C 122 66, 120 14, 160 14 Z' }],
                ['decor', 'ellipse', { cx: 147, cy: 60, rx: 8, ry: 9 }],
                ['decor', 'ellipse', { cx: 173, cy: 60, rx: 8, ry: 9 }],
                ['decor', 'path', { d: 'M 156.5 70 L 163.5 70 L 160 79 Z' }]
            ],
            mandible: [['fill', 'path', { d: 'M 133 90 Q 136 120 160 126 Q 184 120 187 90 L 179 93 Q 176 113 160 116 Q 144 113 141 93 Z' }]],
            sacrum: [['fill', 'path', { d: 'M 141 374 L 179 374 L 172 410 L 160 424 L 148 410 Z' }]],
            coccyx: [['fill', 'path', { d: 'M 155 427 L 165 427 L 160 439 Z' }]],
            sternum: [['outline', 'rect', { x: 154, y: 188, width: 12, height: 76, rx: 5, 'stroke-width': 3 }]]
        };
        for (let i = 1; i <= 7; i++) shapes[`C${i}`] = [['fill', 'rect', { x: 152, y: 128 + (i - 1) * 6.6, width: 16, height: 5, rx: 1.5 }]];
        for (let i = 1; i <= 12; i++) shapes[`T${i}`] = [['fill', 'rect', { x: 150, y: 176 + (i - 1) * 10, width: 20, height: 8, rx: 2 }]];
        for (let i = 1; i <= 5; i++) shapes[`L${i}`] = [['fill', 'rect', { x: 147, y: 298 + (i - 1) * 15, width: 26, height: 12, rx: 3 }]];
        return shapes;
    }

    // Draw order: back to front.
    const SIDED_ORDER = ['scapula', ...RIB_WIDTH.map((_, i) => `rib${i + 1}`), 'clavicle', 'humerus', 'radius', 'ulna', 'carpals',
        'metacarpals', 'fingers', 'hip', 'femur', 'patella', 'tibia', 'fibula', 'tarsals', 'metatarsals', 'toes'];
    const MIDLINE_ORDER = ['skull', 'mandible', ...[1, 2, 3, 4, 5, 6, 7].map(i => `C${i}`),
        ...[...Array(12).keys()].map(i => `T${i + 1}`), ...[1, 2, 3, 4, 5].map(i => `L${i}`), 'sacrum', 'coccyx', 'sternum'];

    function drawShapes(parent, specs, group, color) {
        const g = svg('g', { class: 'map-shape', 'data-group': group.id }, parent);
        specs.forEach(([paint, tag, attrs]) => {
            const el = svg(tag, attrs, g);
            if (paint === 'decor') {
                el.setAttribute('fill', '#ffffff');
                el.setAttribute('opacity', '0.85');
                el.setAttribute('pointer-events', 'none');
            } else if (paint === 'fill') {
                el.setAttribute('fill', color ? color : NO_DATA.fill);
                el.setAttribute('stroke', color ? '#ffffff' : NO_DATA.stroke);
                el.setAttribute('stroke-width', color ? 1 : 0.8);
            } else if (paint === 'outline') {
                el.setAttribute('fill', 'none');
                el.setAttribute('stroke', color || NO_DATA.stroke);
            } else {
                el.setAttribute('fill', 'none');
                el.setAttribute('stroke', color || '#d6d6d6');
                el.setAttribute('stroke-linecap', 'round');
            }
        });
        makeInteractive(g, group);
        return g;
    }

    function renderSkeleton() {
        const root = document.getElementById('skeletonMap');
        const legendRoot = document.getElementById('skeletonLegend');
        if (!root) return;
        root.innerHTML = '';
        const byId = new Map(groups.map(group => [group.id, group]));
        const counts = groups.filter(group => group.map).map(group => countOf(group.id, mode));
        const bins = binner(Math.max(0, ...counts));

        const figure = svg('svg', { viewBox: '0 0 320 784', class: 'skeleton-svg', role: 'group', 'aria-label': 'Skeleton coverage map' }, root);
        svg('text', { x: 14, y: 24, class: 'map-side-label' }, figure).textContent = 'Right';
        svg('text', { x: 306, y: 24, class: 'map-side-label', 'text-anchor': 'end' }, figure).textContent = 'Left';

        const sided = sidedShapes();
        const midline = midlineShapes();
        const right = svg('g', {}, figure);
        const left = svg('g', { transform: 'translate(320,0) scale(-1,1)' }, figure);
        const mid = svg('g', {}, figure);
        const draw = (parent, name, suffix) => {
            const group = byId.get(suffix ? `${name}-${suffix}` : name);
            if (group) drawShapes(parent, (suffix ? sided : midline)[name], group, bins.color(countOf(group.id, mode)));
        };
        MIDLINE_ORDER.filter(name => !name.startsWith('sternum')).forEach(name => draw(mid, name));
        SIDED_ORDER.forEach(name => {
            draw(right, name, 'R');
            draw(left, name, 'L');
        });
        draw(mid, 'sternum');
        figure.appendChild(mid); // the sternum outline and the spine stay on top of the ribs

        if (legendRoot) renderLegend(legendRoot, bins, Math.max(0, ...counts), 'scans');
    }

    function renderLegend(root, bins, max, unit) {
        root.innerHTML = '';
        const item = (color, text, isNoData) => {
            const span = document.createElement('span');
            span.className = 'legend-item';
            const swatch = document.createElement('i');
            swatch.className = 'legend-swatch' + (isNoData ? ' legend-swatch-empty' : '');
            swatch.style.background = color;
            span.appendChild(swatch);
            span.appendChild(document.createTextNode(text));
            root.appendChild(span);
        };
        item(NO_DATA.fill, 'none', true);
        let low = 1;
        bins.edges.forEach((edge, k) => {
            if (edge < low || low > max) return;
            item(RAMP[k], low === edge ? `${compact(edge)}` : `${compact(low)}–${compact(edge)}`);
            low = edge + 1;
        });
        const note = document.createElement('span');
        note.className = 'legend-note';
        note.textContent = unit;
        root.appendChild(note);
    }

    // ---- soft tissue list ----------------------------------------------------------------------------------------

    function renderSoftTissue() {
        const root = document.getElementById('softTissueBars');
        if (!root) return;
        root.innerHTML = '';
        const soft = groups.filter(group => group.row && group.kind !== 'bone');
        const max = Math.max(1, ...soft.map(group => countOf(group.id, mode)));
        let section = null;
        let list = null;
        soft.forEach(group => {
            if (group.section !== section) {
                section = group.section;
                const heading = document.createElement('h4');
                heading.className = 'soft-section';
                heading.textContent = section;
                root.appendChild(heading);
                list = document.createElement('div');
                list.className = 'soft-list';
                root.appendChild(list);
            }
            const count = countOf(group.id, mode);
            const row = document.createElement('button');
            row.type = 'button';
            row.className = 'soft-row' + (count ? '' : ' soft-row-empty');
            row.disabled = !count;
            const label = document.createElement('span');
            label.className = 'soft-label';
            label.textContent = group.name;
            const value = document.createElement('span');
            value.className = 'soft-value';
            value.textContent = count ? formatNumber(count) : '—';
            const track = document.createElement('span');
            track.className = 'soft-track';
            const fill = document.createElement('span');
            fill.className = 'soft-fill';
            fill.style.width = `${(100 * count / max).toFixed(1)}%`;
            track.appendChild(fill);
            row.appendChild(label);
            row.appendChild(value);
            row.appendChild(track);
            if (count) {
                row.addEventListener('mousemove', event => showTooltip(event, groupLines(group)));
                row.addEventListener('mouseleave', hideTooltip);
                row.addEventListener('click', () => selectGroup(group));
            }
            list.appendChild(row);
        });
    }

    // ---- structure x dataset matrix ------------------------------------------------------------------------------

    function renderMatrix() {
        const root = document.getElementById('coverageMatrix');
        const legendRoot = document.getElementById('matrixLegend');
        if (!root) return;
        root.innerHTML = '';
        const cards = new Map((typeof datasetCards !== 'undefined' ? datasetCards : []).map(card => [String(card.id), card]));
        const rows = groups.filter(group => group.row);
        const providers = new Set();
        rows.forEach(group => Object.keys(((coverageData || {})[group.id] || {}).providers || {}).forEach(id => providers.add(id)));
        const columns = [...providers].sort((a, b) => Number(a) - Number(b));
        if (columns.length === 0) {
            root.innerHTML = '<p class="overview-empty">No labels in this database yet.</p>';
            return;
        }

        const value = (group, id) => {
            const scans = (((coverageData || {})[group.id] || {}).providers || {})[id] || 0;
            if (matrixMode === 'scans') return scans;
            const card = cards.get(id);
            return card && card.scansWithLabels ? Math.round(100 * scans / card.scansWithLabels) : 0;
        };
        let max = 0;
        rows.forEach(group => columns.forEach(id => { max = Math.max(max, value(group, id)); }));
        const bins = matrixMode === 'scans' ? binner(max) : {
            edges: [20, 40, 60, 80, 100],
            color(v) { if (!v) return null; return RAMP[Math.min(RAMP.length - 1, Math.floor((v - 1) / (100 / RAMP.length)))]; }
        };

        const table = document.createElement('table');
        table.className = 'coverage-matrix';
        const caption = document.createElement('caption');
        caption.className = 'visually-hidden';
        caption.textContent = matrixMode === 'scans'
            ? 'Number of scans in which each dataset gave each structure group'
            : "Percentage of each dataset's labelled scans that have each structure group";
        table.appendChild(caption);
        const thead = document.createElement('thead');
        const head = document.createElement('tr');
        const corner = document.createElement('th');
        corner.className = 'matrix-corner';
        corner.textContent = 'Structure \\ dataset';
        head.appendChild(corner);
        columns.forEach(id => {
            const th = document.createElement('th');
            th.className = 'matrix-col';
            th.scope = 'col';
            const card = cards.get(id);
            th.title = card ? `${id}: ${card.name}` : id;
            const label = document.createElement('span');
            label.textContent = card ? `${id} · ${card.name}` : id;
            th.appendChild(label);
            head.appendChild(th);
        });
        thead.appendChild(head);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        let section = null;
        rows.forEach(group => {
            if (group.section !== section) {
                section = group.section;
                const tr = document.createElement('tr');
                tr.className = 'matrix-group-row';
                const th = document.createElement('th');
                th.colSpan = columns.length + 1;
                th.textContent = section;
                tr.appendChild(th);
                tbody.appendChild(tr);
            }
            const tr = document.createElement('tr');
            const th = document.createElement('th');
            th.scope = 'row';
            th.className = 'matrix-row';
            th.textContent = group.name;
            tr.appendChild(th);
            columns.forEach(id => {
                const td = document.createElement('td');
                const v = value(group, id);
                const color = bins.color(v);
                td.className = 'matrix-cell' + (v ? '' : ' matrix-empty');
                if (color) {
                    td.style.background = color;
                    td.style.color = RAMP.indexOf(color) >= 2 ? '#ffffff' : '#191919';
                    td.textContent = matrixMode === 'scans' ? compact(v) : `${v}%`;
                }
                const card = cards.get(id);
                const scans = (((coverageData || {})[group.id] || {}).providers || {})[id] || 0;
                const lines = [`${group.name} · ${card ? card.name : id}`, `${formatNumber(scans)} scans`];
                if (card && card.scansWithLabels) lines.push(`${Math.round(100 * scans / card.scansWithLabels)}% of the dataset's ${formatNumber(card.scansWithLabels)} labelled scans`);
                td.addEventListener('mousemove', event => showTooltip(event, lines));
                td.addEventListener('mouseleave', hideTooltip);
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        root.appendChild(table);
        if (legendRoot) {
            if (matrixMode === 'scans') renderLegend(legendRoot, bins, max, 'scans');
            else {
                legendRoot.innerHTML = '';
                renderLegend(legendRoot, { edges: bins.edges }, 100, "% of the dataset's labelled scans");
                legendRoot.querySelectorAll('.legend-item').forEach((item, i) => {
                    if (i > 0) item.lastChild.textContent = `≤ ${bins.edges[i - 1]}%`;
                });
            }
        }
    }

    function setUpToggle(rootId, onChange) {
        const root = document.getElementById(rootId);
        if (!root) return;
        root.querySelectorAll('button[data-value]').forEach(button => {
            button.addEventListener('click', () => {
                root.querySelectorAll('button[data-value]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
                onChange(button.dataset.value);
            });
        });
    }

    document.addEventListener('DOMContentLoaded', function() {
        if (typeof structuresData === 'undefined' || !structuresData || typeof coverageData === 'undefined') {
            const section = document.getElementById('coverage');
            if (section) section.style.display = 'none';
            const matrix = document.getElementById('matrix');
            if (matrix) matrix.style.display = 'none';
            return;
        }
        groups = structuresData.groups;
        groupIndex = new Map(groups.map((group, i) => [group.id, i]));
        totalScans = typeof scansData !== 'undefined' ? scansData.length : 0;
        renderSkeleton();
        renderSoftTissue();
        renderMatrix();
        setUpToggle('coverageKind', value => { mode = value; renderSkeleton(); renderSoftTissue(); });
        setUpToggle('matrixMode', value => { matrixMode = value; renderMatrix(); });
        window.addEventListener('scroll', hideTooltip, { passive: true });
    });
})();
