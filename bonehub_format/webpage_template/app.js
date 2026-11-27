"use strict";
// The database's webpage: the overview, the scan table with its filters, and the data card of a scan. Drawn from
// `site` (data.js) and, for the cards, `siteDetails` (details/<dataset>.js, loaded when first needed); see
// update_data.py.

const KINDS = ["image", "segmentation", "mesh", "nurbs"];
const LABELED_KINDS = ["segmentation", "mesh", "nurbs"];
const KIND_NAMES = { image: "Image", segmentation: "Segmentation", mesh: "Mesh", nurbs: "NURBS" };
const SEX_NAMES = { M: "Male", F: "Female", O: "Other" };
const SEX_COLORS = { M: "var(--series-male)", F: "var(--series-female)", unknown: "var(--series-unknown)" };
const PAGE_SIZE = 25;

// ------------------------------------------------------------------------------------------------------ formatting

const fmt = (n) => Number(n).toLocaleString("en-US");
const fmtDecimal = (n, digits = 1) => Number(n).toLocaleString("en-US", { maximumFractionDigits: digits });

function compact(n) {
    if (n < 10000) return fmt(n);
    if (n < 1e6) return `${(n / 1e3).toFixed(n < 1e5 ? 1 : 0).replace(/\.0$/, "")}K`;
    return `${(n / 1e6).toFixed(1).replace(/\.0$/, "")}M`;
}

function humanSize(bytes) {
    if (bytes == null) return "";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let value = bytes;
    let unit = 0;
    while (value >= 1000 && unit < units.length - 1) {
        value /= 1000;
        unit += 1;
    }
    return `${value < 10 && unit ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

function esc(text) {
    return String(text ?? "").replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
}

const SIDES = { LEFT: "left", RIGHT: "right" };

function prettyLabel(name) {
    const words = name.split("_");
    let side = "";
    if (SIDES[words[words.length - 1]]) side = ` (${SIDES[words.pop()]})`;
    const text = words
        .map((word, i) => (/^[A-Z]?\d+$/.test(word) ? word : i === 0 ? word[0] + word.slice(1).toLowerCase() : word.toLowerCase()))
        .join(" ");
    return text + side;
}

function originName(origin) {
    if (origin === "source") return "From the source dataset";
    if (origin === "manual") return "Drawn by hand";
    const [kind, rest] = origin.split(/:(.*)/s);
    const [tool, version] = rest.split("@");
    return `${kind === "model" ? "Predicted by" : "Derived with"} ${tool} ${version}`;
}

const LABELS = site.labels.map((label) => label.name);
const PRETTY = LABELS.map(prettyLabel);
const SEARCH_TEXT = LABELS.map((label, i) => `${PRETTY[i]} ${label.replace(/_/g, " ")}`.toLowerCase());
const SCAN_INDEX = new Map(site.scans.map((scan, i) => [scan.scan, i]));
const subjectOf = (scan) => scan.scan.split("/").slice(0, 2).join("/");
const datasetOf = (scan) => site.datasets[scan.dataset];
const registryUrl = (id) => `${site.registry}#dataset=${id}`;
const fileUrl = (path) => `https://huggingface.co/datasets/${site.repo}/resolve/main/${path}?download=true`;
const folderUrl = (path) => `https://huggingface.co/datasets/${site.repo}/tree/main/${path}`;
const pad = (number) => String(number).padStart(2, "0");
const labelsOf = (scan, kind) => scan[kind] || [];

function fileName(kind, number) {
    if (kind === "image") return "image.nii.gz";
    if (kind === "segmentation") return `segmentation-${pad(number)}.nrrd`;
    return `${kind}-${pad(number)}.zip`;
}

/** The labels a scan has in files of the given kinds (all labelled kinds by default), without repeats. */
function scanLabels(scan, kinds = LABELED_KINDS) {
    if (kinds.length === 1) return labelsOf(scan, kinds[0]);
    return [...new Set(kinds.flatMap((kind) => labelsOf(scan, kind)))].sort((a, b) => a - b);
}

/** The labels whose name contains every word of a query (e.g. 'patella left'). */
function labelsMatching(query) {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    return LABELS.map((_, i) => i).filter((i) => words.every((word) => SEARCH_TEXT[i].includes(word)));
}

/** The parts of a label value (labelmap.py, R S M M part tissue side): structure RSMM, part, tissue, side. */
function codes(value) {
    return { structure: Math.floor(value / 1e5), part: Math.floor(value / 1000) % 100, tissue: Math.floor(value / 10) % 100, side: value % 10 };
}

/** The labels a label covers in the label map: itself and every label under it. A structure without a side covers
 * both sides; a group (e.g. Ribs) covers its members (Rib 1 .. Rib 12), a region (e.g. Lower extremity) its bones;
 * a part (Femur, proximal) or tissue (cortical) covers only that part or tissue. */
function labelsCovered(index) {
    const chosen = site.labels[index].value;
    if (chosen == null) return [index];
    const c = codes(chosen);
    const region = Math.floor(c.structure / 1000), family = Math.floor(c.structure / 100) % 10, member = c.structure % 100;
    return site.labels
        .map((label, i) => [i, label.value == null ? null : codes(label.value)])
        .filter(([, o]) => {
            if (o === null) return false;
            let sameStructure;
            if (member === 0 && family === 0) sameStructure = Math.floor(o.structure / 1000) === region && Math.floor(o.structure / 100) % 10 <= 6;
            else if (member === 0) sameStructure = Math.floor(o.structure / 100) === Math.floor(c.structure / 100);
            else sameStructure = o.structure === c.structure;
            return sameStructure && (!c.side || o.side === c.side) && (!c.part || o.part === c.part) && (!c.tissue || o.tissue === c.tissue);
        })
        .map(([i]) => i);
}

function subjectFacts(scan) {
    return [
        SEX_NAMES[scan.sex],
        scan.age != null ? `${scan.age} y` : null,
        scan.weight != null ? `${fmtDecimal(scan.weight)} kg` : null,
        scan.height != null ? `${fmtDecimal(scan.height)} cm` : null,
        scan.bmi != null ? `BMI ${fmtDecimal(scan.bmi)}` : null,
    ].filter(Boolean);
}

// ---------------------------------------------------------------------------------------------------------- tooltip

const tooltip = document.getElementById("tooltip");

function placeTooltip(x, y) {
    const gap = 14;
    const { width, height } = tooltip.getBoundingClientRect();
    tooltip.style.left = `${Math.max(8, Math.min(x + gap, window.innerWidth - width - 8))}px`;
    tooltip.style.top = `${y + gap + height > window.innerHeight ? y - height - gap : y + gap}px`;
}

function showTooltip(target, x, y) {
    tooltip.innerHTML = target.dataset.tip;
    tooltip.hidden = false;
    placeTooltip(x, y);
}

document.addEventListener("mouseover", (event) => {
    const target = event.target.closest("[data-tip]");
    if (target) showTooltip(target, event.clientX, event.clientY);
});
document.addEventListener("mousemove", (event) => {
    if (!tooltip.hidden) placeTooltip(event.clientX, event.clientY);
});
document.addEventListener("mouseout", (event) => {
    const target = event.target.closest("[data-tip]");
    if (target && !target.contains(event.relatedTarget)) tooltip.hidden = true;
});
document.addEventListener("focusin", (event) => {
    const target = event.target.closest("[data-tip]");
    if (!target) return;
    const box = target.getBoundingClientRect();
    showTooltip(target, box.left, box.bottom);
});
document.addEventListener("focusout", () => {
    tooltip.hidden = true;
});

// --------------------------------------------------------------------------------------------------------- overview

function barRows(rows, max) {
    // rows: [{label, value, valueText, tip}]
    return rows
        .map(
            (row) => `<div class="bar-label" title="${esc(row.label)}">${esc(row.label)}</div>
            <div class="bar-track" tabindex="0" data-tip="${esc(row.tip)}" aria-label="${esc(row.label)}: ${esc(row.valueText ?? fmt(row.value))}">
                <div class="bar" style="width:${max ? Math.max((row.value / max) * 85, row.value ? 0.6 : 0) : 0}%"></div>
                <span class="bar-value">${esc(row.valueText ?? fmt(row.value))}</span>
            </div>`
        )
        .join("");
}

function renderOverview() {
    const scans = site.scans;
    const subjects = new Map();
    for (const scan of scans) if (!subjects.has(subjectOf(scan))) subjects.set(subjectOf(scan), scan.sex);
    const sources = new Set(site.datasets.flatMap((d) => d.sources.map((s) => s[0])));
    let publishedBytes = 0;
    for (const scan of scans) for (const file of scan.files) if (file.published) publishedBytes += file.size || 0;
    const segmented = new Set(scans.flatMap((scan) => labelsOf(scan, "segmentation")));
    const kpis = [
        ["Subjects", fmt(subjects.size)],
        ["Scans", fmt(scans.length)],
        ["Datasets", fmt(site.datasets.length)],
        ["Source datasets", fmt(sources.size)],
        ["Segmented structures", fmt(segmented.size)],
        ["Published data", humanSize(publishedBytes)],
    ];
    document.getElementById("kpis").innerHTML = kpis
        .map(([label, value]) => `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div></div>`)
        .join("");

    const withKind = Object.fromEntries(KINDS.map((kind) => [kind, 0]));
    for (const scan of scans) for (const kind of new Set(scan.files.map((f) => f.kind))) withKind[kind] += 1;
    document.getElementById("chart-kinds").innerHTML = barRows(
        KINDS.map((kind) => ({
            label: KIND_NAMES[kind],
            value: withKind[kind],
            valueText: `${fmt(withKind[kind])} (${scans.length ? Math.round((withKind[kind] / scans.length) * 100) : 0}%)`,
            tip: `<strong>${KIND_NAMES[kind]}</strong><br>${fmt(withKind[kind])} of ${fmt(scans.length)} scans`,
        })),
        scans.length
    );

    const datasets = [...site.datasets].sort((a, b) => b.scans - a.scans);
    document.getElementById("chart-datasets").innerHTML = barRows(
        datasets.map((d) => ({
            label: d.name,
            value: d.scans,
            tip: `<strong>${esc(d.name)}</strong> (${esc(d.folder)})<br>${fmt(d.subjects)} subjects, ${fmt(d.scans)} scans<br>Sources: ${d.sources.map((s) => `${s[0]} ${esc(s[1])}`).join("; ")}`,
        })),
        datasets.length ? datasets[0].scans : 0
    );

    const sexes = { M: 0, F: 0, unknown: 0 };
    for (const sex of subjects.values()) sexes[sex === "M" || sex === "F" ? sex : "unknown"] += 1;
    const sexParts = [["M", "Male"], ["F", "Female"], ["unknown", "Unknown"]].filter(([key]) => sexes[key]);
    document.getElementById("chart-sex").innerHTML = `<div class="segmented" role="img" aria-label="${sexParts.map(([k, n]) => `${n}: ${sexes[k]}`).join(", ")}">${sexParts
        .map(([key, name]) => {
            const share = sexes[key] / subjects.size;
            const label = share > 0.12 ? `${Math.round(share * 100)}%` : "";
            return `<div class="segment${key === "unknown" ? " dark-text" : ""}" style="flex:${sexes[key]};background:${SEX_COLORS[key]}" tabindex="0" data-tip="<strong>${name}</strong><br>${fmt(sexes[key])} subjects (${Math.round(share * 100)}%)">${label}</div>`;
        })
        .join("")}</div>
        <div class="legend">${sexParts.map(([key, name]) => `<span class="legend-item"><span class="swatch" style="background:${SEX_COLORS[key]}"></span>${name} ${fmt(sexes[key])}</span>`).join("")}</div>`;

    const modalities = new Map();
    for (const scan of scans) modalities.set(scan.modality || "Unknown", (modalities.get(scan.modality || "Unknown") || 0) + 1);
    const modalityRows = [...modalities.entries()].sort((a, b) => b[1] - a[1]);
    document.getElementById("chart-modality").innerHTML = barRows(
        modalityRows.map(([name, count]) => ({ label: name, value: count, tip: `<strong>${esc(name)}</strong><br>${fmt(count)} scans` })),
        modalityRows.length ? modalityRows[0][1] : 0
    );

    const bins = Array.from({ length: 10 }, (_, i) => ({ from: i * 10, count: 0 }));
    let known = 0;
    for (const scan of scans) {
        if (scan.age == null) continue;
        known += 1;
        bins[Math.min(Math.floor(scan.age / 10), 9)].count += 1;
    }
    const maxBin = Math.max(1, ...bins.map((b) => b.count));
    document.getElementById("age-note").textContent = `Scans per age group (age known for ${fmt(known)} of ${fmt(scans.length)})`;
    const binName = (b) => (b.from === 90 ? "90 years or older" : `${b.from}–${b.from + 9} years`);
    document.getElementById("chart-age").innerHTML = bins
        .map(
            (b) => `<div class="column" tabindex="0" data-tip="<strong>${binName(b)}</strong><br>${fmt(b.count)} scans">
                <span class="column-value">${b.count ? compact(b.count) : ""}</span>
                <div class="column-bar" style="height:${(b.count / maxBin) * 85}%"></div></div>`
        )
        .join("");
    document.getElementById("chart-age").insertAdjacentHTML("afterend", `<div class="column-axis">${bins.map((b) => `<span>${b.from === 90 ? "90+" : b.from}</span>`).join("")}</div>`);

    renderStructures(scans);
}

function renderStructures(scans) {
    const sections = site.sections;
    const perLabel = new Array(LABELS.length).fill(0);
    const perSection = new Array(sections.length).fill(0);
    for (const scan of scans) {
        const found = new Set();
        for (const label of labelsOf(scan, "segmentation")) {
            perLabel[label] += 1;
            found.add(site.labels[label].section);
        }
        for (const section of found) perSection[section] += 1;
    }
    const labelsIn = sections.map(() => []);
    LABELS.forEach((_, i) => perLabel[i] && labelsIn[site.labels[i].section].push(i));
    const shown = sections.map((_, i) => i).filter((i) => perSection[i]);
    const total = perLabel.filter(Boolean).length;
    document.getElementById("chart-sections").innerHTML = barRows(
        shown.map((i) => ({
            label: sections[i],
            value: perSection[i],
            tip: `<strong>${esc(sections[i])}</strong><br>${fmt(perSection[i])} scans, ${labelsIn[i].length} structures`,
        })),
        Math.max(0, ...perSection)
    );
    document.getElementById("structure-summary").textContent = `All ${fmt(total)} structures`;

    const maxLabel = Math.max(1, ...perLabel);
    const groups = document.getElementById("structure-groups");
    groups.innerHTML = shown
        .map(
            (section) => `<details class="structure-group"><summary>${esc(sections[section])}<span class="count">${labelsIn[section].length}</span></summary><div class="bars">${labelsIn[section]
                .map((label) => {
                    const search = esc(SEARCH_TEXT[label]);
                    return `<div class="bar-label" data-search="${search}">${esc(PRETTY[label])}</div>
                    <div class="bar-track" data-search="${search}" data-tip="<strong>${esc(PRETTY[label])}</strong> (${esc(LABELS[label])})<br>${fmt(perLabel[label])} scans">
                        <div class="bar" style="width:${(perLabel[label] / maxLabel) * 85}%"></div><span class="bar-value">${fmt(perLabel[label])}</span></div>`;
                })
                .join("")}</div></details>`
        )
        .join("");
    document.getElementById("structure-search").addEventListener("input", (event) => {
        const words = event.target.value.toLowerCase().split(/\s+/).filter(Boolean);
        for (const group of groups.querySelectorAll(".structure-group")) {
            let visible = 0;
            for (const cell of group.querySelectorAll("[data-search]")) {
                const hit = words.every((word) => cell.dataset.search.includes(word));
                cell.style.display = hit ? "" : "none";
                if (hit) visible += 1;
            }
            group.style.display = visible ? "" : "none";
            group.open = words.length > 0 && visible > 0;
        }
    });
}

// ----------------------------------------------------------------------------------------------------------- filters

// Which scans have each label in files of each kind: postings[kind][label] = scan indices.
const postings = Object.fromEntries(LABELED_KINDS.map((kind) => [kind, LABELS.map(() => [])]));
site.scans.forEach((scan, i) => {
    for (const kind of LABELED_KINDS) for (const label of labelsOf(scan, kind)) postings[kind][label].push(i);
});

/** A mask of the scans that have any of these labels in files of these kinds. */
function scansWith(labels, kinds) {
    const mask = new Uint8Array(site.scans.length);
    for (const kind of kinds) for (const label of labels) for (const i of postings[kind][label]) mask[i] = 1;
    return mask;
}

const scanKinds = site.scans.map((scan) => new Set(scan.files.map((f) => f.kind)));
const publishedImage = site.scans.map((scan) => scan.files.some((f) => f.kind === "image" && f.published));
const downloadSize = site.scans.map((scan) => scan.files.reduce((sum, f) => sum + (f.published ? f.size || 0 : 0), 0));

// The facets: within a facet the ticked values are alternatives (any of them), except where a scan must have them all.
const FACETS = [
    { id: "dataset", title: "Dataset", values: (i) => [site.scans[i].dataset], name: (v) => site.datasets[v].name, all: false },
    { id: "kind", title: "Has all of", values: (i) => [...scanKinds[i]], name: (v) => KIND_NAMES[v], all: true, order: KINDS },
    { id: "access", title: "Access", values: (i) => (publishedImage[i] ? ["published"] : []), name: () => "Image can be downloaded", all: true },
    { id: "modality", title: "Modality", values: (i) => [site.scans[i].modality || ""], name: (v) => v || "Unknown", all: false },
    { id: "sex", title: "Sex", values: (i) => [site.scans[i].sex || ""], name: (v) => SEX_NAMES[v] || "Unknown", all: false, order: ["F", "M", "O", ""] },
    { id: "race", title: "Race", values: (i) => [site.scans[i].race || ""], name: (v) => v || "Unknown", all: false, optional: true },
    { id: "ethnicity", title: "Ethnicity", values: (i) => [site.scans[i].ethnicity || ""], name: (v) => v || "Unknown", all: false, optional: true },
];

const NUMERIC = [
    { id: "age", title: "Age", unit: "years" },
    { id: "weight", title: "Weight", unit: "kg" },
    { id: "height", title: "Height", unit: "cm" },
    { id: "bmi", title: "BMI", unit: "kg/m²" },
];

const SORTS = [
    { id: "scan", name: "Scan" },
    { id: "age", name: "Age" },
    { id: "weight", name: "Weight" },
    { id: "height", name: "Height" },
    { id: "bmi", name: "BMI" },
    { id: "structures", name: "Number of structures" },
    { id: "size", name: "Download size" },
];

const state = {
    checked: Object.fromEntries(FACETS.map((f) => [f.id, new Set()])),
    ranges: Object.fromEntries(NUMERIC.map((n) => [n.id, { min: null, max: null }])),
    structures: [], // chips: {text, labels}
    foundIn: "any",
    search: "",
    sort: { key: "scan", dir: 1 },
    page: 0,
};
let filtered = []; // indices into site.scans
let matchedLabels = new Set();

function facetOptions(facet) {
    const counts = new Map();
    site.scans.forEach((_, i) => {
        for (const value of facet.values(i)) counts.set(value, (counts.get(value) || 0) + 1);
    });
    let keys = [...counts.keys()];
    if (facet.order) keys = facet.order.filter((v) => counts.has(v));
    else if (facet.id === "dataset") keys.sort((a, b) => a - b);
    else keys.sort((a, b) => counts.get(b) - counts.get(a));
    return keys;
}

function renderFilters() {
    const groups = [];
    for (const facet of FACETS) {
        const options = facetOptions(facet);
        if (!options.length || (facet.optional && options.length === 1 && options[0] === "")) continue;
        groups.push(`<fieldset class="filter-group"><legend>${facet.title}</legend>${options
            .map((value) => `<label class="filter-option"><input type="checkbox" data-facet="${facet.id}" value="${esc(value)}"><span class="option-name">${esc(facet.name(value))}</span><span class="count" data-count="${facet.id}:${esc(value)}"></span></label>`)
            .join("")}</fieldset>`);
    }
    for (const numeric of NUMERIC) {
        if (!site.scans.some((scan) => scan[numeric.id] != null)) continue;
        groups.push(`<fieldset class="filter-group"><legend>${numeric.title} <span class="unit">(${numeric.unit})</span></legend><div class="filter-range">
            <input type="number" data-range="${numeric.id}" data-bound="min" min="0" placeholder="from" aria-label="${numeric.title} from">–<input type="number" data-range="${numeric.id}" data-bound="max" min="0" placeholder="to" aria-label="${numeric.title} to"></div></fieldset>`);
    }
    groups.push(`<fieldset class="filter-group"><legend>Structures</legend>
        <input type="text" id="structure-filter" list="structure-options" placeholder="e.g. femur, patella left" aria-label="Add a structure">
        <datalist id="structure-options">${PRETTY.map((name) => `<option value="${esc(name)}">`).join("")}</datalist>
        <label class="filter-select">in <select id="found-in" aria-label="Type of data the structures are in">
            <option value="any">any data</option><option value="segmentation">a segmentation</option><option value="mesh">a mesh</option><option value="nurbs">a NURBS surface</option>
        </select></label>
        <div class="chips" id="structure-chips"></div>
        <p class="filter-hint">Press Enter to add. A structure without a side covers both sides and all of its parts; a word covers every structure it names. A scan must have every structure added.</p></fieldset>`);
    const filters = document.getElementById("filters");
    filters.innerHTML = groups.join("");

    filters.addEventListener("change", (event) => {
        const input = event.target;
        if (input.dataset.facet) {
            const value = input.dataset.facet === "dataset" ? Number(input.value) : input.value;
            const checked = state.checked[input.dataset.facet];
            input.checked ? checked.add(value) : checked.delete(value);
            update();
        } else if (input.id === "found-in") {
            state.foundIn = input.value;
            update();
        }
    });
    filters.addEventListener("input", (event) => {
        const input = event.target;
        if (!input.dataset.range) return;
        state.ranges[input.dataset.range][input.dataset.bound] = input.value === "" ? null : Number(input.value);
        update();
    });
    const structureInput = document.getElementById("structure-filter");
    const addStructure = () => {
        const text = structureInput.value.trim();
        if (!text) return;
        const exact = PRETTY.findIndex((name) => name.toLowerCase() === text.toLowerCase());
        const labels = exact >= 0 ? labelsCovered(exact) : labelsMatching(text);
        structureInput.value = "";
        if (!labels.length) {
            structureInput.placeholder = `No structure matches '${text}'`;
            return;
        }
        structureInput.placeholder = "e.g. femur, patella left";
        state.structures.push({ text: exact >= 0 ? PRETTY[exact] : text, labels });
        renderChips();
        update();
    };
    structureInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            addStructure();
        }
    });
    structureInput.addEventListener("input", (event) => {
        if (event.inputType === "insertReplacementText" || PRETTY.includes(structureInput.value)) addStructure();
    });
}

function renderChips() {
    document.getElementById("structure-chips").innerHTML = state.structures
        .map((chip, i) => {
            const many = chip.labels.length > 1 ? ` (${chip.labels.length})` : "";
            const tip = chip.labels.slice(0, 30).map((l) => esc(PRETTY[l])).join(", ") + (chip.labels.length > 30 ? ", …" : "");
            return `<button type="button" class="chip" data-chip="${i}" data-tip="${esc(tip)}" aria-label="Remove ${esc(chip.text)}">${esc(chip.text)}${many} <span aria-hidden="true">×</span></button>`;
        })
        .join("");
}

function facetPasses(facet, i) {
    const checked = state.checked[facet.id];
    if (!checked.size) return true;
    const values = facet.values(i);
    if (facet.all) {
        for (const value of checked) if (!values.includes(value)) return false;
        return true;
    }
    return values.some((value) => checked.has(value));
}

function update(keepPage = false) {
    const kinds = state.foundIn === "any" ? LABELED_KINDS : [state.foundIn];
    const structureMasks = state.structures.map((chip) => scansWith(chip.labels, kinds));
    const query = state.search.trim().toLowerCase();
    const searchLabels = query.length >= 3 ? labelsMatching(query) : [];
    const searchMask = searchLabels.length ? scansWith(searchLabels, LABELED_KINDS) : null;
    matchedLabels = new Set([...state.structures.flatMap((chip) => chip.labels), ...searchLabels]);
    const otherPasses = (i) => {
        const scan = site.scans[i];
        for (const { id } of NUMERIC) {
            const { min, max } = state.ranges[id];
            if (min == null && max == null) continue;
            if (scan[id] == null || (min != null && scan[id] < min) || (max != null && scan[id] > max)) return false;
        }
        for (const mask of structureMasks) if (!mask[i]) return false;
        if (query) {
            const dataset = datasetOf(scan);
            const text = `${scan.scan} ${dataset.name} ${dataset.sources.map((s) => `${s[0]} ${s[1]}`).join(" ")} ${scan.modality || ""}`.toLowerCase();
            if (!text.includes(query) && !(searchMask && searchMask[i])) return false;
        }
        return true;
    };

    // An option's count: the scans there would be with it ticked as well.
    const counts = Object.fromEntries(FACETS.map((f) => [f.id, new Map()]));
    const add = (facet, i) => {
        for (const value of facet.values(i)) counts[facet.id].set(value, (counts[facet.id].get(value) || 0) + 1);
    };
    filtered = [];
    for (let i = 0; i < site.scans.length; i++) {
        if (!otherPasses(i)) continue;
        const failing = FACETS.filter((facet) => !facetPasses(facet, i));
        if (!failing.length) {
            filtered.push(i);
            for (const facet of FACETS) add(facet, i);
        } else if (failing.length === 1 && !failing[0].all) {
            add(failing[0], i);
        }
    }
    for (const element of document.querySelectorAll("[data-count]")) {
        const [facetId, raw] = element.dataset.count.split(/:(.*)/s);
        const value = facetId === "dataset" ? Number(raw) : raw;
        const count = counts[facetId].get(value) || 0;
        element.textContent = fmt(count);
        element.closest(".filter-option").classList.toggle("empty-option", count === 0);
    }
    sortFiltered();
    if (!keepPage) state.page = 0;
    renderRows();
}

const SORT_VALUE = {
    age: (i) => site.scans[i].age,
    weight: (i) => site.scans[i].weight,
    height: (i) => site.scans[i].height,
    bmi: (i) => site.scans[i].bmi,
    structures: (i) => scanLabels(site.scans[i]).length,
    size: (i) => downloadSize[i],
};

function sortFiltered() {
    const { key, dir } = state.sort;
    if (key === "scan") {
        filtered.sort((a, b) => dir * (a - b)); // the scans are in path order
        return;
    }
    const value = SORT_VALUE[key];
    filtered.sort((a, b) => {
        const va = value(a), vb = value(b);
        if (va == null || vb == null) return (va == null) - (vb == null) || a - b; // unknown values last
        return dir * (va - vb) || a - b;
    });
}

// ------------------------------------------------------------------------------------------------------------ table

function downloadButtons(scan) {
    const counts = {};
    for (const file of scan.files) counts[file.kind] = (counts[file.kind] || 0) + 1;
    return scan.files
        .map((file) => {
            const name = fileName(file.kind, file.number);
            const label = `${KIND_NAMES[file.kind]}${counts[file.kind] > 1 ? ` ${pad(file.number)}` : ""}`;
            const source = site.sources[file.source] || {};
            const from = `Source: ${file.source} ${esc(source.name || "")}`;
            if (file.published) {
                const tip = `<strong>${esc(name)}</strong> (${humanSize(file.size)})<br>${from}<br>License: ${esc(file.license)}`;
                return `<a class="dl" href="${fileUrl(`${scan.scan}/${name}`)}" download="${esc(name)}" data-tip="${esc(tip)}">${label}<span class="size">${humanSize(file.size)}</span></a>`;
            }
            const tip = `<strong>${esc(name)}</strong> is not in the repository: its license is ${esc(file.license || "unknown")}.<br>${from}: obtain it there.`;
            return `<a class="dl locked" href="${esc(source.url || registryUrl(file.source))}" target="_blank" rel="noopener noreferrer" data-tip="${esc(tip)}"><span aria-hidden="true">🔒</span> ${label}</a>`;
        })
        .join("");
}

function scanName(key) {
    return key.split("/").map((part) => `<span>${esc(part)}</span>`).join("");
}

function subjectCell(scan) {
    const facts = subjectFacts(scan);
    const origin = [scan.race, scan.ethnicity].filter(Boolean);
    if (!facts.length && !origin.length) return `<span class="muted">Unknown</span>`;
    return `<div>${esc(facts.join(" · "))}</div>${origin.length ? `<div class="muted">${esc(origin.join(" · "))}</div>` : ""}`;
}

/** The number of structures of a scan, and per type of data. */
function structureCounts(scan) {
    const kinds = LABELED_KINDS.filter((kind) => labelsOf(scan, kind).length);
    return { total: scanLabels(scan).length, perKind: kinds.map((kind) => `${KIND_NAMES[kind].toLowerCase()} ${labelsOf(scan, kind).length}`).join(" · ") };
}

function dataCell(scan) {
    const all = scanLabels(scan);
    if (!all.length) return `<span class="muted">No structures</span>`;
    const { perKind } = structureCounts(scan);
    let matched = "";
    if (matchedLabels.size) {
        const hits = all.filter((label) => matchedLabels.has(label));
        if (hits.length) {
            const kindsOf = (label) => LABELED_KINDS.filter((kind) => labelsOf(scan, kind).includes(label)).map((k) => KIND_NAMES[k].toLowerCase());
            const shown = hits.slice(0, 3).map((l) => `${esc(PRETTY[l])} <span class="muted">${kindsOf(l).join(", ")}</span>`);
            matched = `<div class="matched">${shown.join("<br>")}${hits.length > 3 ? `<br>and ${hits.length - 3} more` : ""}</div>`;
        }
    }
    return `<div>${fmt(all.length)} structures</div><div class="muted">${esc(perKind)}</div>${matched}`;
}

function renderRows() {
    const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    const rows = filtered.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
    document.getElementById("result-count").textContent =
        filtered.length === site.scans.length ? `${fmt(site.scans.length)} scans` : `${fmt(filtered.length)} of ${fmt(site.scans.length)} scans`;
    document.getElementById("scan-rows").innerHTML = rows.length
        ? rows
              .map((i) => {
                  const scan = site.scans[i];
                  return `<tr data-scan="${i}" tabindex="0" class="${i === openScan ? "selected" : ""}" aria-label="Scan ${esc(scan.scan)}">
                    <td><span class="scan-name">${scanName(scan.scan)}</span></td>
                    <td>${subjectCell(scan)}</td>
                    <td>${esc(scan.modality || "Unknown")}</td>
                    <td>${dataCell(scan)}</td>
                    <td><div class="downloads">${downloadButtons(scan)}</div></td>
                </tr>`;
              })
              .join("")
        : `<tr><td colspan="5" class="empty">No scan matches the filters.</td></tr>`;
    document.getElementById("pager").innerHTML = `<button type="button" data-page="${state.page - 1}" ${state.page === 0 ? "disabled" : ""}>Previous</button>
        <span>Page ${fmt(state.page + 1)} of ${fmt(pages)}</span>
        <button type="button" data-page="${state.page + 1}" ${state.page >= pages - 1 ? "disabled" : ""}>Next</button>`;
}

// -------------------------------------------------------------------------------------------------------- data card

let openScan = null; // index into site.scans

function loadDetails(folder) {
    window.siteDetails = window.siteDetails || {};
    if (window.siteDetails[folder]) return Promise.resolve(window.siteDetails[folder]);
    return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = `details/${folder}.js`;
        script.onload = () => (window.siteDetails[folder] ? resolve(window.siteDetails[folder]) : reject(new Error("empty")));
        script.onerror = () => reject(new Error(`details/${folder}.js could not be loaded`));
        document.head.appendChild(script);
    });
}

function facts(rows) {
    const shown = rows.filter(([, value]) => value != null && value !== "");
    if (!shown.length) return `<p class="muted">Not recorded.</p>`;
    return `<dl class="facts">${shown.map(([name, value]) => `<dt>${name}</dt><dd>${value}</dd>`).join("")}</dl>`;
}

function sourceName(id, details) {
    const source = details.sources.find((s) => s.id === id) || site.sources[id] || {};
    return `<a href="${registryUrl(id)}" target="_blank" rel="noopener noreferrer">${esc(source.name || `Source ${id}`)}</a> <span class="muted">(${id})</span>`;
}

function labelList(labels, truncated) {
    const names = labels.map((l) => `${esc(PRETTY[l])}${truncated.has(l) ? '<sup class="border-mark" title="Touches the image border">*</sup>' : ""}`);
    if (names.length <= 24) return `<p class="label-list">${names.join(", ")}</p>`;
    return `<details class="label-more"><summary>${names.slice(0, 12).join(", ")}, … <span class="more">show all ${names.length}</span></summary><p class="label-list">${names.join(", ")}</p></details>`;
}

function fileSection(scan, file, info, details, truncated) {
    const name = fileName(file.kind, file.number);
    const title = `${KIND_NAMES[file.kind]}${file.kind === "image" ? "" : ` ${pad(file.number)}`}`;
    const meta = facts([
        ["Source", sourceName(file.source, details)],
        ["ID in source", info && info.sourceScanId ? `<code>${esc(info.sourceScanId)}</code>` : null],
        ["Collection", info && info.subset ? esc(info.subset) : null],
        ["License", esc(file.license || "Unknown") + (file.published ? "" : ' <span class="badge">not published</span>')],
        ["Size", humanSize(file.size)],
    ]);
    const origins = info && info.origins ? Object.entries(info.origins) : [];
    const labels = origins
        .map(([origin, list]) => `<h5>${esc(originName(origin))} <span class="muted">(${list.length})</span></h5>${labelList(list, file.kind === "segmentation" ? truncated : new Set())}`)
        .join("");
    const link = file.published
        ? `<a class="dl" href="${fileUrl(`${scan.scan}/${name}`)}" download="${esc(name)}">Download<span class="size">${humanSize(file.size)}</span></a>`
        : "";
    return `<article class="file"><header class="file-header"><h4>${title} <code class="muted">${esc(name)}</code></h4>${link}</header>${meta}${labels}</article>`;
}

function cardHtml(scan, details) {
    const dataset = datasetOf(scan);
    const info = details.scans[scan.scan] || {};
    const truncated = new Set(info.truncated || []);
    const fileInfo = (file) => (info.files || []).find((f) => f.kind === file.kind && (f.number ?? null) === (file.number ?? null));
    const image = scan.files.find((f) => f.kind === "image");
    const size = info.imageSize;
    const spacing = info.imageSpacing;
    const fov = size && spacing ? size.slice(0, 3).map((n, k) => fmtDecimal(n * spacing[k], 0)).join(" × ") + " mm" : null;
    const subjectIds = Object.entries(info.subjectSources || {})
        .map(([id, value]) => `${sourceName(Number(id), details)}: <code>${esc(value)}</code>`)
        .join("<br>");
    const sources = details.sources
        .map((s) => {
            const licenses = Object.entries(s.licenses || {}).map(([kind, license]) => `${KIND_NAMES[kind]}: ${esc(license)}`).join(" · ");
            const links = [s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">Data</a>` : null,
                s.paper ? `<a href="${esc(s.paper)}" target="_blank" rel="noopener noreferrer">Paper</a>` : null,
                `<a href="${registryUrl(s.id)}" target="_blank" rel="noopener noreferrer">Registry entry</a>`].filter(Boolean).join(" · ");
            return `<div class="source"><div class="source-name">${esc(s.name)} <span class="muted">(${s.id})</span></div><div class="muted">${licenses}</div><div>${links}</div></div>`;
        })
        .join("");
    return `
        <section class="card-section">
            <div class="downloads">${downloadButtons(scan)}</div>
            <p class="card-link"><a href="${folderUrl(scan.scan)}" target="_blank" rel="noopener noreferrer">Open the folder on Hugging Face</a></p>
        </section>
        <section class="card-section">
            <h3>Subject</h3>
            ${facts([
                ["Sex", SEX_NAMES[scan.sex]],
                ["Age at scan", scan.age != null ? `${scan.age} years` : null],
                ["Weight", scan.weight != null ? `${fmtDecimal(scan.weight)} kg` : null],
                ["Height", scan.height != null ? `${fmtDecimal(scan.height)} cm` : null],
                ["BMI", scan.bmi != null ? fmtDecimal(scan.bmi) : null],
                ["Race", scan.race ? esc(scan.race) : null],
                ["Ethnicity", scan.ethnicity ? esc(scan.ethnicity) : null],
                ["Postmortem", info.postmortem ? "Yes" : null],
                ["ID in source", subjectIds || null],
                ["Remarks", info.subjectRemarks ? esc(info.subjectRemarks) : null],
            ])}
        </section>
        <section class="card-section">
            <h3>Imaging</h3>
            ${facts([
                ["Modality", esc(scan.modality || "Unknown")],
                ["Image size", size ? `${size.map(fmt).join(" × ")} voxels` : image ? null : "No image"],
                ["Voxel spacing", spacing ? `${spacing.map((v) => fmtDecimal(v, 3)).join(" × ")} mm` : null],
                ["Field of view", fov],
                ["Days from first scan", info.daysFromBaseline != null ? fmt(info.daysFromBaseline) : null],
                ["Registered to", info.registeredTo ? esc(info.registeredTo) : null],
                ["Remarks", info.remarks ? esc(info.remarks) : null],
            ])}
        </section>
        <section class="card-section">
            <h3>Data</h3>
            ${(({ total, perKind }) => (total ? `<p class="card-summary">${fmt(total)} structures <span class="muted">(${esc(perKind)})</span></p>` : `<p class="muted">No structures</p>`))(structureCounts(scan))}
            ${truncated.size ? `<p class="muted"><sup class="border-mark">*</sup> touches the image border, so probably cut off by the field of view</p>` : ""}
            ${scan.files.map((file) => fileSection(scan, file, fileInfo(file), details, truncated)).join("")}
        </section>
        <section class="card-section">
            <h3>Dataset</h3>
            ${facts([
                ["Name", `${esc(dataset.name)} <span class="muted">(${esc(dataset.folder)})</span>`],
                ["Description", details.dataset.description ? esc(details.dataset.description) : null],
            ])}
            <h4 class="sources-heading">Sources</h4>
            ${sources}
        </section>`;
}

function openCard(index, { focus = true } = {}) {
    const scan = site.scans[index];
    if (!scan) return;
    openScan = index;
    const card = document.getElementById("card");
    const dataset = datasetOf(scan);
    document.getElementById("card-dataset").textContent = `${dataset.name} · ${dataset.folder}`;
    document.getElementById("card-title").textContent = scan.scan;
    const body = document.getElementById("card-body");
    body.innerHTML = `<p class="muted">Loading…</p>`;
    card.hidden = false;
    document.body.classList.add("card-open");
    history.replaceState(null, "", `#scan=${encodeURIComponent(scan.scan)}`);
    for (const row of document.querySelectorAll("#scan-rows tr[data-scan]")) row.classList.toggle("selected", Number(row.dataset.scan) === index);
    updateCardNavigation();
    loadDetails(dataset.folder)
        .then((details) => {
            if (openScan !== index) return;
            body.innerHTML = cardHtml(scan, details);
            body.scrollTop = 0;
        })
        .catch((error) => {
            if (openScan === index) body.innerHTML = `<p class="error">The details of this scan could not be loaded (${esc(error.message)}).</p>`;
        });
    if (focus) document.getElementById("card-close").focus({ preventScroll: true });
}

function closeCard() {
    const card = document.getElementById("card");
    if (card.hidden) return;
    const previous = openScan;
    card.hidden = true;
    openScan = null;
    document.body.classList.remove("card-open");
    history.replaceState(null, "", location.pathname + location.search);
    for (const row of document.querySelectorAll("#scan-rows tr.selected")) row.classList.remove("selected");
    const row = document.querySelector(`#scan-rows tr[data-scan="${previous}"]`);
    if (row) row.focus({ preventScroll: true });
}

function updateCardNavigation() {
    const position = filtered.indexOf(openScan);
    document.getElementById("card-previous").disabled = position <= 0;
    document.getElementById("card-next").disabled = position < 0 || position >= filtered.length - 1;
}

function stepCard(step) {
    const position = filtered.indexOf(openScan);
    const next = filtered[position + step];
    if (position < 0 || next === undefined) return;
    const page = Math.floor((position + step) / PAGE_SIZE);
    if (page !== state.page) {
        state.page = page;
        renderRows();
    }
    openCard(next, { focus: false });
    const row = document.querySelector(`#scan-rows tr[data-scan="${next}"]`);
    if (row) row.scrollIntoView({ block: "nearest" });
}

// -------------------------------------------------------------------------------------------------------------- CSV

function csvCell(value) {
    const text = String(value ?? "");
    return /[",\n;]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function exportCsv() {
    const header = ["scan", "dataset", "dataset_name", "sources", "sex", "age", "weight", "height", "bmi", "race", "ethnicity",
        "modality", ...LABELED_KINDS.map((k) => `${k}_labels`), ...KINDS.map((k) => `${k}_files`)];
    const lines = [header.join(",")];
    for (const i of filtered) {
        const scan = site.scans[i];
        const dataset = datasetOf(scan);
        const labels = LABELED_KINDS.map((kind) => labelsOf(scan, kind).map((l) => LABELS[l]).join(" "));
        const urls = KINDS.map((kind) =>
            scan.files
                .filter((f) => f.kind === kind)
                .map((f) => (f.published ? fileUrl(`${scan.scan}/${fileName(kind, f.number)}`) : `restricted (source ${f.source})`))
                .join(" ")
        );
        lines.push(
            [scan.scan, dataset.folder, dataset.name, dataset.sources.map((s) => s[0]).join(" "), scan.sex, scan.age, scan.weight,
                scan.height, scan.bmi, scan.race, scan.ethnicity, scan.modality, ...labels, ...urls]
                .map(csvCell)
                .join(",")
        );
    }
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "scans.csv";
    link.click();
    URL.revokeObjectURL(link.href);
}

// ------------------------------------------------------------------------------------------------------------ setup

function initBrowser() {
    renderFilters();
    const sortKey = document.getElementById("sort-key");
    sortKey.innerHTML = SORTS.map((s) => `<option value="${s.id}">${s.name}</option>`).join("");
    sortKey.addEventListener("change", () => {
        state.sort.key = sortKey.value;
        sortFiltered();
        state.page = 0;
        renderRows();
    });
    const direction = document.getElementById("sort-direction");
    direction.addEventListener("click", () => {
        state.sort.dir = -state.sort.dir;
        const ascending = state.sort.dir === 1;
        direction.textContent = ascending ? "↑" : "↓";
        direction.title = ascending ? "Ascending" : "Descending";
        direction.setAttribute("aria-label", direction.title);
        sortFiltered();
        state.page = 0;
        renderRows();
    });
    document.getElementById("scan-search").addEventListener("input", (event) => {
        state.search = event.target.value;
        update();
    });
    document.getElementById("clear-filters").addEventListener("click", () => {
        for (const set of Object.values(state.checked)) set.clear();
        for (const range of Object.values(state.ranges)) range.min = range.max = null;
        Object.assign(state, { structures: [], foundIn: "any", search: "" });
        for (const input of document.querySelectorAll("#filters input")) input.type === "checkbox" ? (input.checked = false) : (input.value = "");
        document.getElementById("found-in").value = "any";
        document.getElementById("scan-search").value = "";
        renderChips();
        update();
    });
    document.getElementById("export-csv").addEventListener("click", exportCsv);
    document.getElementById("pager").addEventListener("click", (event) => {
        const button = event.target.closest("button[data-page]");
        if (!button || button.disabled) return;
        state.page = Number(button.dataset.page);
        renderRows();
        document.getElementById("scan-table").scrollIntoView({ block: "start", behavior: "smooth" });
    });
    const rows = document.getElementById("scan-rows");
    rows.addEventListener("click", (event) => {
        if (event.target.closest("a, button")) return; // the download links act on their own
        const row = event.target.closest("tr[data-scan]");
        if (row) openCard(Number(row.dataset.scan));
    });
    rows.addEventListener("keydown", (event) => {
        const row = event.target.closest("tr[data-scan]");
        if (row && event.target === row && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            openCard(Number(row.dataset.scan));
        }
    });
    document.getElementById("card-close").addEventListener("click", closeCard);
    document.getElementById("card-previous").addEventListener("click", () => stepCard(-1));
    document.getElementById("card-next").addEventListener("click", () => stepCard(1));
    document.addEventListener("keydown", (event) => {
        if (document.getElementById("card").hidden) return;
        if (event.key === "Escape") {
            closeCard();
            return;
        }
        if (event.target.closest("input, select, textarea")) return;
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            stepCard(event.key === "ArrowDown" ? 1 : -1);
        }
    });
    document.addEventListener("click", (event) => {
        const chip = event.target.closest(".chip[data-chip]");
        if (!chip) return;
        state.structures.splice(Number(chip.dataset.chip), 1);
        tooltip.hidden = true;
        renderChips();
        update();
    });
    update();
    openLinkedCard();
    window.addEventListener("hashchange", openLinkedCard);
}

/** Open the card a link names (#scan=ds-0001/sub-000001/scan-01), so that a card can be shared. */
function openLinkedCard() {
    const linked = new URLSearchParams(location.hash.slice(1)).get("scan");
    if (linked && SCAN_INDEX.has(linked) && SCAN_INDEX.get(linked) !== openScan) openCard(SCAN_INDEX.get(linked));
}

function renderFooter() {
    const repo = `https://huggingface.co/datasets/${site.repo}`;
    document.getElementById("footer-text").innerHTML = `Data as of ${esc(site.updated)} · Each file keeps the license of its source dataset
        (<a href="${repo}/blob/main/LICENSE.md" target="_blank" rel="noopener noreferrer">LICENSE.md</a>); cite the source datasets you use.`;
}

renderOverview();
initBrowser();
renderFooter();
