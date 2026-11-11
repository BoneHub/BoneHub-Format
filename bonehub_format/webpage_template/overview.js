// Renders a compact overview dashboard from scansData without coupling to app.js.
(function() {
    const SVG_NS = 'http://www.w3.org/2000/svg';

    const GENDER_FILL_CLASSES = {
        Male: 'fill-male',
        Female: 'fill-female'
    };

    // Population chart: only male and female subjects with a known age are plotted.
    const POPULATION_SERIES = [
        { key: 'Male', className: 'series-male' },
        { key: 'Female', className: 'series-female' }
    ];
    const POPULATION_BIN_WIDTH = 5;
    const POPULATION_OPEN_BIN_START = 90; // ages >= 90 share one open-ended bin
    const POPULATION_BAR_GAP = 2;

    function parseItemList(value) {
        if (value === null || value === undefined) {
            return [];
        }

        return String(value)
            .split(';')
            .map(item => item.trim())
            .filter(Boolean);
    }

    function normalizeGender(value) {
        const raw = String(value === null || value === undefined ? '' : value).trim().toUpperCase();
        if (raw === 'M' || raw === 'MALE') {
            return 'Male';
        }
        if (raw === 'F' || raw === 'FEMALE') {
            return 'Female';
        }
        if (raw === 'O' || raw === 'OTHER') {
            return 'Other';
        }
        if (raw) {
            return raw;
        }
        return 'Unknown';
    }

    function formatNumber(value, digits) {
        return new Intl.NumberFormat('en-US', {
            maximumFractionDigits: typeof digits === 'number' ? digits : 0,
            minimumFractionDigits: typeof digits === 'number' ? digits : 0
        }).format(value);
    }

    // The rows are scans; a subject with several scans has several rows, and is counted once.
    function uniqueSubjects(rows) {
        const subjects = new Map();
        rows.forEach(row => {
            const key = `${row['Dataset ID']}/${row['Subject ID']}`;
            if (!subjects.has(key)) {
                subjects.set(key, row);
            }
        });
        return [...subjects.values()];
    }

    function summarizeSubjects(data) {
        const scans = Array.isArray(data) ? data : [];
        const subjects = uniqueSubjects(scans);
        const totalScans = scans.length;
        const totalSubjects = subjects.length;
        const datasetIds = new Set();
        const genders = {};
        const ages = [];

        const segmentationItems = [];
        const meshItems = [];
        const nurbsItems = [];

        subjects.forEach(subject => {
            const gender = normalizeGender(subject['Gender']);
            genders[gender] = (genders[gender] || 0) + 1;

            const age = Number(subject['Age (years)']);
            if (!Number.isNaN(age) && age > 0) {
                ages.push(age);
            }
        });

        scans.forEach(scan => {
            if (scan['Dataset ID'] !== null && scan['Dataset ID'] !== undefined) {
                datasetIds.add(scan['Dataset ID']);
            }
            segmentationItems.push(...parseItemList(scan['Segmentation Available']));
            meshItems.push(...parseItemList(scan['Mesh Available']));
            nurbsItems.push(...parseItemList(scan['NURBS Available']));
        });

        const allBoneLabels = [...segmentationItems, ...meshItems, ...nurbsItems];
        const uniqueBoneLabels = new Set(allBoneLabels);

        const scansWithImage = scans.filter(scan => scan['Image Available'] === true || scan['Image Available'] === 'true').length;
        const scansWithSeg = scans.filter(scan => parseItemList(scan['Segmentation Available']).length > 0).length;
        const scansWithMesh = scans.filter(scan => parseItemList(scan['Mesh Available']).length > 0).length;
        const scansWithNurbs = scans.filter(scan => parseItemList(scan['NURBS Available']).length > 0).length;

        const boneFrequencies = {};
        allBoneLabels.forEach(label => {
            boneFrequencies[label] = (boneFrequencies[label] || 0) + 1;
        });

        const topBones = Object.entries(boneFrequencies)
            .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
            .slice(0, 8);

        let ageMin = null;
        let ageMax = null;
        let ageMean = null;
        if (ages.length > 0) {
            ageMin = Math.min(...ages);
            ageMax = Math.max(...ages);
            ageMean = ages.reduce((sum, value) => sum + value, 0) / ages.length;
        }

        return {
            totalSubjects,
            totalScans,
            totalDatasets: datasetIds.size,
            genders,
            ages,
            ageMin,
            ageMax,
            ageMean,
            ageKnownCount: ages.length,
            uniqueBoneLabelsCount: uniqueBoneLabels.size,
            scansWithImage,
            totalSegmentationMasks: segmentationItems.length,
            totalMeshes: meshItems.length,
            totalNurbs: nurbsItems.length,
            scansWithSeg,
            scansWithMesh,
            scansWithNurbs,
            topBones
        };
    }

    function createMetricCard(label, value, detail) {
        const card = document.createElement('article');
        card.className = 'overview-card';

        const labelEl = document.createElement('p');
        labelEl.className = 'overview-card-label';
        labelEl.textContent = label;

        const valueEl = document.createElement('p');
        valueEl.className = 'overview-card-value';
        valueEl.textContent = value;

        const detailEl = document.createElement('p');
        detailEl.className = 'overview-card-detail';
        detailEl.textContent = detail;

        card.appendChild(labelEl);
        card.appendChild(valueEl);
        card.appendChild(detailEl);

        return card;
    }

    function createBarRow(label, count, total, colorClass) {
        const row = document.createElement('div');
        row.className = 'overview-bar-row';

        const head = document.createElement('div');
        head.className = 'overview-bar-head';
        head.innerHTML = `<span>${label}</span><span>${count}</span>`;

        const track = document.createElement('div');
        track.className = 'overview-bar-track';

        const fill = document.createElement('div');
        fill.className = `overview-bar-fill ${colorClass || ''}`.trim();
        const ratio = total > 0 ? Math.max(0, Math.min(100, (count / total) * 100)) : 0;
        fill.style.width = `${ratio.toFixed(1)}%`;
        fill.setAttribute('aria-hidden', 'true');

        track.appendChild(fill);
        row.appendChild(head);
        row.appendChild(track);

        return row;
    }

    function renderOverview() {
        const metricsRoot = document.getElementById('overviewMetrics');
        const genderBarsRoot = document.getElementById('overviewGenderBars');
        const assetBarsRoot = document.getElementById('overviewAssetBars');
        const ageStatsRoot = document.getElementById('overviewAgeStats');
        const topBonesRoot = document.getElementById('overviewTopBones');

        if (!metricsRoot || !genderBarsRoot || !assetBarsRoot || !ageStatsRoot || !topBonesRoot) {
            return;
        }

        if (typeof scansData === 'undefined' || !Array.isArray(scansData) || scansData.length === 0) {
            metricsRoot.innerHTML = '<p class="overview-empty">No scan data available to summarize.</p>';
            return;
        }

        const summary = summarizeSubjects(scansData);

        metricsRoot.innerHTML = '';
        metricsRoot.appendChild(createMetricCard(
            'Subjects',
            formatNumber(summary.totalSubjects),
            `${formatNumber(summary.totalScans)} scans from ${formatNumber(summary.totalDatasets)} datasets`
        ));
        metricsRoot.appendChild(createMetricCard(
            'Unique Bone Labels',
            formatNumber(summary.uniqueBoneLabelsCount),
            'Across segmentation, mesh, and NURBS fields'
        ));
        metricsRoot.appendChild(createMetricCard(
            'Segmentation Masks',
            formatNumber(summary.totalSegmentationMasks),
            `${formatNumber(summary.scansWithSeg)} scans with at least one segmentation`
        ));
        metricsRoot.appendChild(createMetricCard(
            'Meshes',
            formatNumber(summary.totalMeshes),
            `${formatNumber(summary.scansWithMesh)} scans with at least one mesh`
        ));
        metricsRoot.appendChild(createMetricCard(
            'NURBS',
            formatNumber(summary.totalNurbs),
            `${formatNumber(summary.scansWithNurbs)} scans with at least one NURBS label`
        ));

        genderBarsRoot.innerHTML = '';
        const orderedGenders = Object.entries(summary.genders).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
        if (orderedGenders.length === 0) {
            genderBarsRoot.innerHTML = '<p class="overview-empty">No gender metadata available.</p>';
        } else {
            orderedGenders.forEach(([gender, count]) => {
                // Colors follow the gender (not its rank) so they match the population chart below.
                const styleClass = GENDER_FILL_CLASSES[gender] || 'fill-neutral';
                genderBarsRoot.appendChild(createBarRow(gender, count, summary.totalSubjects, styleClass));
            });
        }

        assetBarsRoot.innerHTML = '';
        const assets = [
            { label: 'Image', count: summary.scansWithImage, className: 'fill-image' },
            { label: 'Segmentation', count: summary.scansWithSeg, className: 'fill-primary' },
            { label: 'Mesh', count: summary.scansWithMesh, className: 'fill-secondary' },
            { label: 'NURBS', count: summary.scansWithNurbs, className: 'fill-accent' }
        ];
        assets.forEach(asset => {
            assetBarsRoot.appendChild(createBarRow(asset.label, asset.count, summary.totalScans, asset.className));
        });

        ageStatsRoot.innerHTML = '';
        const ageItems = [
            ['Known ages', `${formatNumber(summary.ageKnownCount)} / ${formatNumber(summary.totalSubjects)}`],
            ['Average age', summary.ageMean === null ? 'N/A' : `${formatNumber(summary.ageMean, 1)} years`],
            ['Age range', summary.ageMin === null || summary.ageMax === null ? 'N/A' : `${formatNumber(summary.ageMin)} - ${formatNumber(summary.ageMax)} years`]
        ];
        ageItems.forEach(([label, value]) => {
            const row = document.createElement('div');
            row.className = 'overview-list-row';
            row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
            ageStatsRoot.appendChild(row);
        });

        topBonesRoot.innerHTML = '';
        if (summary.topBones.length === 0) {
            topBonesRoot.innerHTML = '<li class="overview-empty">No bone labels available.</li>';
        } else {
            summary.topBones.forEach(([label, count]) => {
                const li = document.createElement('li');
                li.innerHTML = `<span>${label}</span><strong>${formatNumber(count)}</strong>`;
                topBonesRoot.appendChild(li);
            });
        }
    }

    function collectPopulation(data) {
        const people = [];
        uniqueSubjects(data).forEach(subject => {
            const gender = normalizeGender(subject['Gender']);
            const age = Number(subject['Age (years)']);
            if ((gender === 'Male' || gender === 'Female') && age > 0) {
                people.push({ age, gender });
            }
        });
        return people;
    }

    function binPopulation(people, width) {
        const binStart = age => Math.min(Math.floor(age / width) * width, POPULATION_OPEN_BIN_START);
        const lastStart = binStart(Math.max(...people.map(person => person.age)));

        const bins = [];
        for (let start = 0; start <= lastStart; start += width) {
            const label = start === POPULATION_OPEN_BIN_START ? `${start}+` : `${start}–${start + width - 1}`;
            bins.push({ label, Male: 0, Female: 0 });
        }
        people.forEach(person => {
            bins[binStart(person.age) / width][person.gender] += 1;
        });
        return bins;
    }

    function niceStep(maxValue, targetTicks) {
        const rough = maxValue / targetTicks;
        const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
        const residual = rough / magnitude;
        const nice = residual <= 1 ? 1 : residual <= 2 ? 2 : residual <= 5 ? 5 : 10;
        return Math.max(1, nice * magnitude);
    }

    function svgEl(tag, attrs, text) {
        const el = document.createElementNS(SVG_NS, tag);
        Object.entries(attrs).forEach(([name, value]) => el.setAttribute(name, value));
        if (text !== undefined) {
            el.textContent = text;
        }
        return el;
    }

    // Horizontal bar from x0 (square end) to x1 (4px rounded end); grows left or right.
    function barPath(x0, x1, y, height) {
        const dir = x1 >= x0 ? 1 : -1;
        const r = Math.min(4, height / 2, Math.abs(x1 - x0));
        const sweep = dir > 0 ? 1 : 0;
        return `M${x0},${y}H${x1 - dir * r}A${r},${r} 0 0 ${sweep} ${x1},${y + r}` +
            `V${y + height - r}A${r},${r} 0 0 ${sweep} ${x1 - dir * r},${y + height}H${x0}Z`;
    }

    function fillPopulationTooltip(tooltip, bin, plottedTotal) {
        tooltip.replaceChildren();

        const title = document.createElement('p');
        title.className = 'population-tooltip-title';
        title.textContent = `Age ${bin.label}`;
        tooltip.appendChild(title);

        POPULATION_SERIES.forEach(series => {
            const row = document.createElement('p');
            row.className = 'population-tooltip-row';
            const key = document.createElement('span');
            key.className = `population-key ${series.className}`;
            const value = document.createElement('strong');
            value.textContent = formatNumber(bin[series.key]);
            const label = document.createElement('span');
            label.textContent = series.key;
            row.append(key, value, label);
            tooltip.appendChild(row);
        });

        const binTotal = bin.Male + bin.Female;
        const foot = document.createElement('p');
        foot.className = 'population-tooltip-foot';
        foot.textContent = `${formatNumber(binTotal)} total · ${formatNumber((binTotal / plottedTotal) * 100, 1)}% of plotted`;
        tooltip.appendChild(foot);
    }

    function createSideLabel(series, count, x, y, anchor) {
        const label = svgEl('text', { class: 'population-side-label', x, y, 'text-anchor': anchor });
        label.appendChild(svgEl('tspan', {}, series.key));
        label.appendChild(svgEl('tspan', { class: 'population-side-count', dx: 6 }, formatNumber(count)));
        return label;
    }

    function drawPopulationPyramid(root, tooltip, bins, totals) {
        const width = Math.max(root.clientWidth, 280);
        const compact = width < 600;
        const rowHeight = compact ? 18 : 20;
        const margin = { top: 30, right: 12, bottom: 44, left: 48 };
        const plotWidth = width - margin.left - margin.right;
        const plotHeight = bins.length * rowHeight;
        const plotBottom = margin.top + plotHeight;
        const height = plotBottom + margin.bottom;
        const center = margin.left + plotWidth / 2;
        const halfGap = POPULATION_BAR_GAP / 2;
        const halfWidth = plotWidth / 2 - halfGap;
        const plottedTotal = totals.Male + totals.Female;

        const maxCount = Math.max(...bins.map(bin => Math.max(bin.Male, bin.Female)));
        const step = niceStep(maxCount, compact ? 2 : 4);
        const xMax = Math.ceil(maxCount / step) * step;
        // Keep single-subject bins visible; exact values live in the tooltip.
        const barLength = count => Math.max(2, (count / xMax) * halfWidth);
        const barHeight = rowHeight - POPULATION_BAR_GAP;

        const svg = svgEl('svg', {
            class: 'population-svg',
            width,
            height,
            role: 'group',
            'aria-label': 'Population pyramid: number of male and female subjects per 5-year age bin'
        });

        // Mirrored gridlines and tick labels around the center axis.
        svg.appendChild(svgEl('text', { class: 'population-tick', x: center, y: plotBottom + 18, 'text-anchor': 'middle' }, '0'));
        for (let value = step; value <= xMax; value += step) {
            const offset = halfGap + (value / xMax) * halfWidth;
            [center - offset, center + offset].forEach(x => {
                const lineX = Math.round(x) + 0.5;
                svg.appendChild(svgEl('line', { class: 'population-grid', x1: lineX, x2: lineX, y1: margin.top, y2: plotBottom }));
                svg.appendChild(svgEl('text', { class: 'population-tick', x: lineX, y: plotBottom + 18, 'text-anchor': 'middle' }, formatNumber(value)));
            });
        }

        svg.appendChild(svgEl('text', { class: 'population-axis-title', x: margin.left - 8, y: margin.top - 12, 'text-anchor': 'end' }, 'Age'));
        svg.appendChild(svgEl('text', { class: 'population-axis-title', x: center, y: height - 6, 'text-anchor': 'middle' }, 'Subjects'));
        svg.appendChild(createSideLabel(POPULATION_SERIES[0], totals.Male, center - 10, margin.top - 12, 'end'));
        svg.appendChild(createSideLabel(POPULATION_SERIES[1], totals.Female, center + 10, margin.top - 12, 'start'));

        function showAt(bin, anchorX, anchorY) {
            fillPopulationTooltip(tooltip, bin, plottedTotal);
            tooltip.hidden = false;
            const tipWidth = tooltip.offsetWidth;
            const tipHeight = tooltip.offsetHeight;
            let left = anchorX + 14;
            if (left + tipWidth > width) {
                left = anchorX - 14 - tipWidth;
            }
            tooltip.style.left = `${Math.max(0, left)}px`;
            tooltip.style.top = `${Math.max(0, anchorY - tipHeight - 10)}px`;
        }

        bins.forEach((bin, index) => {
            // Youngest bin at the bottom.
            const rowY = margin.top + (bins.length - 1 - index) * rowHeight;
            const barY = rowY + (rowHeight - barHeight) / 2;
            const group = svgEl('g', { class: 'population-bin' });
            group.appendChild(svgEl('rect', { class: 'population-wash', x: margin.left, y: rowY, width: plotWidth, height: rowHeight }));

            if (bin.Male > 0) {
                const x0 = center - halfGap;
                group.appendChild(svgEl('path', { class: 'population-bar series-male', d: barPath(x0, x0 - barLength(bin.Male), barY, barHeight) }));
            }
            if (bin.Female > 0) {
                const x0 = center + halfGap;
                group.appendChild(svgEl('path', { class: 'population-bar series-female', d: barPath(x0, x0 + barLength(bin.Female), barY, barHeight) }));
            }

            group.appendChild(svgEl('text', {
                class: 'population-tick',
                x: margin.left - 8,
                y: rowY + rowHeight / 2,
                'text-anchor': 'end',
                'dominant-baseline': 'middle'
            }, bin.label));

            const hit = svgEl('rect', {
                class: 'population-hit',
                x: 0,
                y: rowY,
                width: width - margin.right,
                height: rowHeight,
                tabindex: 0,
                role: 'img',
                'aria-label': `Age ${bin.label}: ${bin.Male} male, ${bin.Female} female`
            });
            const showAtPointer = event => {
                const rootBox = root.getBoundingClientRect();
                showAt(bin, event.clientX - rootBox.left, event.clientY - rootBox.top);
            };
            hit.addEventListener('pointermove', showAtPointer);
            hit.addEventListener('pointerdown', showAtPointer);
            hit.addEventListener('pointerleave', event => {
                // Touch keeps the tooltip until the next tap; see the pointerdown handler in renderPopulation.
                if (event.pointerType === 'mouse') {
                    tooltip.hidden = true;
                }
            });
            hit.addEventListener('focus', () => {
                // Keyboard focus anchors the tooltip to the row; pointer focus keeps it at the cursor.
                if (hit.matches(':focus-visible')) {
                    showAt(bin, center, rowY);
                }
            });
            hit.addEventListener('blur', () => {
                tooltip.hidden = true;
            });
            group.appendChild(hit);

            svg.appendChild(group);
        });

        const axisX = Math.round(center) + 0.5;
        svg.appendChild(svgEl('line', { class: 'population-baseline', x1: axisX, x2: axisX, y1: margin.top, y2: plotBottom }));

        tooltip.hidden = true;
        root.replaceChildren(svg, tooltip);
    }

    function renderPopulation() {
        const chartRoot = document.getElementById('overviewPopulation');
        if (!chartRoot) {
            return;
        }

        const subjects = typeof scansData !== 'undefined' && Array.isArray(scansData) ? scansData : [];
        const people = collectPopulation(subjects);
        if (people.length === 0) {
            chartRoot.innerHTML = '<p class="overview-empty">No subjects with both age and gender recorded.</p>';
            return;
        }

        const bins = binPopulation(people, POPULATION_BIN_WIDTH);
        const totals = { Male: 0, Female: 0 };
        people.forEach(person => {
            totals[person.gender] += 1;
        });

        const tooltip = document.createElement('div');
        tooltip.className = 'population-tooltip';

        let drawnWidth = 0;
        function draw() {
            drawPopulationPyramid(chartRoot, tooltip, bins, totals);
            drawnWidth = chartRoot.clientWidth;
        }

        document.addEventListener('pointerdown', event => {
            if (!chartRoot.contains(event.target)) {
                tooltip.hidden = true;
            }
        });

        new ResizeObserver(() => {
            if (chartRoot.clientWidth !== drawnWidth) {
                draw();
            }
        }).observe(chartRoot);

        draw();
    }

    document.addEventListener('DOMContentLoaded', renderOverview);
    document.addEventListener('DOMContentLoaded', renderPopulation);
})();
