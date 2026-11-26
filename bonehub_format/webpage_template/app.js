"use strict";
// The database's webpage: the overview figures and the scan browser, drawn from `site` (data.js, see update_data.py).

const KINDS = ["image", "segmentation", "mesh", "nurbs"];
const LABELED_KINDS = ["segmentation", "mesh", "nurbs"];
const KIND_NAMES = { image: "Image", segmentation: "Segmentation", mesh: "Mesh", nurbs: "NURBS" };
const SEX_NAMES = { M: "Male", F: "Female", O: "Other" };
const SEX_COLORS = { M: "var(--series-male)", F: "var(--series-female)", unknown: "var(--series-unknown)" };
const PAGE_SIZE = 25;
// the fields of a scan of site.scans
const KEY = 0, DATASET = 1, SEX = 2, AGE = 3, MODALITY = 4, RACE = 5, ETHNICITY = 6, FILES = 10;
const LABEL_FIELD = { segmentation: 7, mesh: 8, nurbs: 9 };
// the fields of a file of a scan
const F_KIND = 0, F_NUMBER = 1, F_SIZE = 2, F_PUBLISHED = 3, F_SOURCE = 4, F_LICENSE = 5;

const fmt = (n) => Number(n).toLocaleString("en-US");

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

const PRETTY = site.labels.map(prettyLabel);
const SEARCH_TEXT = site.labels.map((label, i) => `${PRETTY[i]} ${label.replace(/_/g, " ")}`.toLowerCase());
const subjectOf = (scan) => scan[KEY].split("/").slice(0, 2).join("/");
const registryUrl = (id) => `${site.registry}#dataset=${id}`;
const fileUrl = (path) => `https://huggingface.co/datasets/${site.repo}/resolve/main/${path}?download=true`;
const folderUrl = (path) => `https://huggingface.co/datasets/${site.repo}/tree/main/${path}`;
const pad = (number) => String(number).padStart(2, "0");

function fileName(kind, number) {
    if (kind === "image") return "image.nii.gz";
    if (kind === "segmentation") return `segmentation-${pad(number)}.nrrd`;
    return `${kind}-${pad(number)}.zip`;
}

/** The labels a scan has in files of the given kinds (all labelled kinds by default), without repeats. */
function scanLabels(scan, kinds = LABELED_KINDS) {
    if (kinds.length === 1) return scan[LABEL_FIELD[kinds[0]]];
    return [...new Set(kinds.flatMap((kind) => scan[LABEL_FIELD[kind]]))].sort((a, b) => a - b);
}

/** The labels whose name contains every word of a query (e.g. 'patella left'). */
function labelsMatching(query) {
    const words = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    return site.labels.map((_, i) => i).filter((i) => words.every((word) => SEARCH_TEXT[i].includes(word)));
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
    for (const scan of scans) if (!subjects.has(subjectOf(scan))) subjects.set(subjectOf(scan), scan[SEX]);
    const sources = new Set(site.datasets.flatMap((d) => d.sources.map((s) => s[0])));
    let publishedBytes = 0;
    let files = 0;
    let restricted = 0;
    for (const scan of scans) {
        for (const file of scan[FILES]) {
            files += 1;
            if (file[F_PUBLISHED]) publishedBytes += file[F_SIZE] || 0;
            else restricted += 1;
        }
    }
    const segmented = new Set(scans.flatMap((scan) => scan[LABEL_FIELD.segmentation]));
    const kpis = [
        ["Subjects", compact(subjects.size), "people, each stored once"],
        ["Scans", compact(scans.length), `${fmt(files)} data files`],
        ["Datasets", fmt(site.datasets.length), `made from ${fmt(sources.size)} source datasets`],
        ["Structures segmented", fmt(segmented.size), `in ${new Set([...segmented].map((i) => site.labelSection[i])).size} parts of the body`],
        ["Data to download", humanSize(publishedBytes), restricted ? `${fmt(restricted)} restricted files not included` : "every file is published"],
    ];
    document.getElementById("kpis").innerHTML = kpis
        .map(([label, value, detail]) => `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-detail">${detail}</div></div>`)
        .join("");

    // What the scans hold
    const withKind = Object.fromEntries(KINDS.map((kind) => [kind, 0]));
    for (const scan of scans) for (const kind of new Set(scan[FILES].map((f) => f[F_KIND]))) withKind[kind] += 1;
    document.getElementById("chart-kinds").innerHTML = barRows(
        KINDS.map((kind) => ({
            label: KIND_NAMES[kind],
            value: withKind[kind],
            valueText: `${fmt(withKind[kind])} (${scans.length ? Math.round((withKind[kind] / scans.length) * 100) : 0}%)`,
            tip: `<strong>${KIND_NAMES[kind]}</strong><br>${fmt(withKind[kind])} of ${fmt(scans.length)} scans`,
        })),
        scans.length
    );

    // Datasets
    document.getElementById("datasets-note").textContent = `Scans per dataset; ${site.datasets.length} datasets made from ${sources.size} source datasets of the BoneHub registry`;
    const datasets = [...site.datasets].sort((a, b) => b.scans - a.scans);
    document.getElementById("chart-datasets").innerHTML = barRows(
        datasets.map((d) => ({
            label: d.name,
            value: d.scans,
            tip: `<strong>${esc(d.folder)}: ${esc(d.name)}</strong><br>${fmt(d.subjects)} subjects, ${fmt(d.scans)} scans<br>Sources: ${d.sources.map((s) => `${s[0]} ${esc(s[1])}`).join("; ")}`,
        })),
        datasets.length ? datasets[0].scans : 0
    );

    // Sex of the subjects: one segmented bar
    const sexes = { M: 0, F: 0, unknown: 0 };
    for (const sex of subjects.values()) sexes[sex === "M" || sex === "F" ? sex : "unknown"] += 1;
    const sexParts = [["M", "Male"], ["F", "Female"], ["unknown", "Not known"]].filter(([key]) => sexes[key]);
    document.getElementById("chart-sex").innerHTML = `<div class="segmented" role="img" aria-label="${sexParts.map(([k, n]) => `${n}: ${sexes[k]}`).join(", ")}">${sexParts
        .map(([key, name]) => {
            const share = sexes[key] / subjects.size;
            const label = share > 0.12 ? `${Math.round(share * 100)}%` : "";
            return `<div class="segment${key === "unknown" ? " dark-text" : ""}" style="flex:${sexes[key]};background:${SEX_COLORS[key]}" tabindex="0" data-tip="<strong>${name}</strong><br>${fmt(sexes[key])} subjects (${Math.round(share * 100)}%)">${label}</div>`;
        })
        .join("")}</div>
        <div class="legend">${sexParts.map(([key, name]) => `<span class="legend-item"><span class="swatch" style="background:${SEX_COLORS[key]}"></span>${name} ${fmt(sexes[key])}</span>`).join("")}</div>`;

    // Age at the scan: a histogram of 10-year bins
    const bins = Array.from({ length: 10 }, (_, i) => ({ from: i * 10, count: 0 }));
    let known = 0;
    for (const scan of scans) {
        if (scan[AGE] == null) continue;
        known += 1;
        bins[Math.min(Math.floor(scan[AGE] / 10), 9)].count += 1;
    }
    const maxBin = Math.max(1, ...bins.map((b) => b.count));
    document.getElementById("age-note").textContent = `Scans per 10 years of age; ${fmt(known)} scans with a known age, ${fmt(scans.length - known)} without`;
    const binName = (b) => (b.from === 90 ? "90 or older" : `${b.from}–${b.from + 9} years`);
    document.getElementById("chart-age").innerHTML = bins
        .map(
            (b) => `<div class="column" tabindex="0" data-tip="<strong>${binName(b)}</strong><br>${fmt(b.count)} scans">
                <span class="column-value">${b.count ? compact(b.count) : ""}</span>
                <div class="column-bar" style="height:${(b.count / maxBin) * 85}%"></div></div>`
        )
        .join("");
    document.getElementById("chart-age").insertAdjacentHTML("afterend", `<div class="column-axis">${bins.map((b) => `<span>${b.from === 90 ? "90+" : b.from}</span>`).join("")}</div>`);

    // Imaging modality
    const modalities = new Map();
    for (const scan of scans) modalities.set(scan[MODALITY] || "not known", (modalities.get(scan[MODALITY] || "not known") || 0) + 1);
    const modalityRows = [...modalities.entries()].sort((a, b) => b[1] - a[1]);
    document.getElementById("chart-modality").innerHTML = barRows(
        modalityRows.map(([name, count]) => ({ label: name, value: count, tip: `<strong>${esc(name)}</strong><br>${fmt(count)} scans` })),
        modalityRows.length ? modalityRows[0][1] : 0
    );

    renderStructures(scans);
}

function renderStructures(scans) {
    // segmentations only: what is segmented in the images
    const perLabel = new Array(site.labels.length).fill(0);
    const perSection = new Array(site.sections.length).fill(0);
    for (const scan of scans) {
        const sections = new Set();
        for (const label of scan[LABEL_FIELD.segmentation]) {
            perLabel[label] += 1;
            sections.add(site.labelSection[label]);
        }
        for (const section of sections) perSection[section] += 1;
    }
    const labelsOf = site.sections.map(() => []);
    site.labels.forEach((name, i) => perLabel[i] && labelsOf[site.labelSection[i]].push(i));
    const shown = site.sections.map((name, i) => i).filter((i) => perSection[i]);
    const segmentedCount = perLabel.filter(Boolean).length;
    document.getElementById("sections-note").textContent = `Scans with a segmentation of each part of the body; ${segmentedCount} structures in all`;
    document.getElementById("chart-sections").classList.add("bars-two-columns");
    document.getElementById("chart-sections").innerHTML = barRows(
        shown.map((i) => ({
            label: site.sections[i],
            value: perSection[i],
            tip: `<strong>${esc(site.sections[i])}</strong><br>${fmt(perSection[i])} scans, ${labelsOf[i].length} structures`,
        })),
        Math.max(0, ...perSection)
    );
    document.getElementById("structure-summary").textContent = `Show all ${segmentedCount} structures`;

    const maxLabel = Math.max(1, ...perLabel);
    const groups = document.getElementById("structure-groups");
    groups.className = "structure-groups";
    groups.innerHTML = shown
        .map(
            (section) => `<details class="structure-group" data-section="${section}"><summary>${esc(site.sections[section])}<span class="count">${labelsOf[section].length} structures</span></summary><div class="bars">${labelsOf[section]
                .map((label) => {
                    const search = esc(SEARCH_TEXT[label]);
                    return `<div class="bar-label" data-search="${search}">${esc(PRETTY[label])}</div>
                    <div class="bar-track" data-search="${search}" data-tip="<strong>${esc(PRETTY[label])}</strong> (${esc(site.labels[label])})<br>${fmt(perLabel[label])} scans">
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

// ------------------------------------------------------------------------------------------------------ scan browser

// Which scans have each label in files of each kind: postings[kind][label] = scan indices.
const postings = Object.fromEntries(LABELED_KINDS.map((kind) => [kind, site.labels.map(() => [])]));
site.scans.forEach((scan, i) => {
    for (const kind of LABELED_KINDS) for (const label of scan[LABEL_FIELD[kind]]) postings[kind][label].push(i);
});

/** A mask of the scans that have any of these labels in files of these kinds. */
function scansWith(labels, kinds) {
    const mask = new Uint8Array(site.scans.length);
    for (const kind of kinds) for (const label of labels) for (const i of postings[kind][label]) mask[i] = 1;
    return mask;
}

const scanKinds = site.scans.map((scan) => new Set(scan[FILES].map((f) => f[F_KIND])));
const publishedImage = site.scans.map((scan) => scan[FILES].some((f) => f[F_KIND] === "image" && f[F_PUBLISHED]));

// The facets: each checks scans by their values; within a facet the checked values are alternatives (any of them),
// except the kinds of data, which a scan must all have.
const FACETS = [
    { id: "dataset", title: "Dataset", values: (i) => [site.scans[i][DATASET]], name: (v) => site.datasets[v].name, all: false },
    { id: "kind", title: "Has all of", values: (i) => [...scanKinds[i]], name: (v) => KIND_NAMES[v], all: true, order: KINDS },
    { id: "access", title: "Access", values: (i) => (publishedImage[i] ? ["published"] : []), name: () => "Image can be downloaded", all: true },
    { id: "modality", title: "Modality", values: (i) => [site.scans[i][MODALITY] || ""], name: (v) => v || "not known", all: false },
    { id: "sex", title: "Sex", values: (i) => [site.scans[i][SEX] || ""], name: (v) => SEX_NAMES[v] || "not known", all: false, order: ["F", "M", "O", ""] },
    { id: "race", title: "Race", values: (i) => [site.scans[i][RACE] || ""], name: (v) => v || "not known", all: false, optional: true },
    { id: "ethnicity", title: "Ethnicity", values: (i) => [site.scans[i][ETHNICITY] || ""], name: (v) => v || "not known", all: false, optional: true },
];

const state = {
    checked: Object.fromEntries(FACETS.map((f) => [f.id, new Set()])),
    ageMin: null,
    ageMax: null,
    structures: [], // chips: {text, labels}
    foundIn: "any",
    search: "",
    sort: { key: "scan", dir: 1 },
    page: 0,
};
let filtered = [];
let matchedLabels = null; // labels the structure filter and search look for, to show in the rows

function facetOptions(facet) {
    const values = new Map();
    site.scans.forEach((_, i) => {
        for (const value of facet.values(i)) values.set(value, (values.get(value) || 0) + 1);
    });
    let keys = [...values.keys()];
    if (facet.order) keys = facet.order.filter((v) => values.has(v));
    else if (facet.id === "dataset") keys.sort((a, b) => a - b);
    else keys.sort((a, b) => values.get(b) - values.get(a));
    return keys;
}

function renderFilters() {
    const groups = [];
    for (const facet of FACETS) {
        const options = facetOptions(facet);
        facet.options = options;
        if (!options.length || (facet.optional && options.length === 1 && options[0] === "")) continue;
        groups.push(`<div class="filter-group" data-facet="${facet.id}"><h3>${facet.title}</h3>${options
            .map((value) => `<label class="filter-option"><input type="checkbox" data-facet="${facet.id}" value="${esc(value)}">${esc(facet.name(value))}<span class="count" data-count="${facet.id}:${esc(value)}"></span></label>`)
            .join("")}</div>`);
    }
    groups.push(`<div class="filter-group"><h3>Age (years)</h3><div class="filter-range"><input type="number" id="age-min" min="0" max="120" placeholder="from" aria-label="Age from">–<input type="number" id="age-max" min="0" max="120" placeholder="to" aria-label="Age to"></div></div>`);
    groups.push(`<div class="filter-group"><h3>Structures</h3>
        <input type="text" id="structure-filter" list="structure-options" placeholder="e.g. femur, patella left" aria-label="Add a structure">
        <datalist id="structure-options">${PRETTY.map((name) => `<option value="${esc(name)}">`).join("")}</datalist>
        <label class="filter-select">in <select id="found-in" aria-label="Kind of data the structures are in">
            <option value="any">any data</option><option value="segmentation">a segmentation</option><option value="mesh">a mesh</option><option value="nurbs">a NURBS surface</option>
        </select></label>
        <div class="chips" id="structure-chips"></div>
        <p class="filter-hint">Press Enter to add. Scans must have every structure added, a word covers all the structures it names.</p></div>`);
    const filters = document.getElementById("filters");
    filters.innerHTML = groups.join("");

    filters.addEventListener("change", (event) => {
        const input = event.target;
        if (input.dataset.facet) {
            const facet = FACETS.find((f) => f.id === input.dataset.facet);
            const value = facet.id === "dataset" ? Number(input.value) : input.value;
            if (input.checked) state.checked[facet.id].add(value);
            else state.checked[facet.id].delete(value);
            update();
        } else if (input.id === "found-in") {
            state.foundIn = input.value;
            update();
        }
    });
    const age = (id, key) =>
        document.getElementById(id).addEventListener("input", (event) => {
            state[key] = event.target.value === "" ? null : Number(event.target.value);
            update();
        });
    age("age-min", "ageMin");
    age("age-max", "ageMax");
    const structureInput = document.getElementById("structure-filter");
    const addStructure = () => {
        const text = structureInput.value.trim();
        if (!text) return;
        const exact = PRETTY.findIndex((name) => name.toLowerCase() === text.toLowerCase());
        const labels = exact >= 0 ? [exact] : labelsMatching(text);
        structureInput.value = "";
        if (!labels.length) {
            structureInput.placeholder = `No structure is called '${text}'`;
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
    // picking a name from the list
    structureInput.addEventListener("input", (event) => {
        if (event.inputType === "insertReplacementText" || PRETTY.some((name) => name === structureInput.value)) addStructure();
    });
}

function renderChips() {
    document.getElementById("structure-chips").innerHTML = state.structures
        .map((chip, i) => {
            const many = chip.labels.length > 1 ? ` (${chip.labels.length})` : "";
            const tip = chip.labels.slice(0, 30).map((l) => esc(PRETTY[l])).join(", ") + (chip.labels.length > 30 ? ", ..." : "");
            return `<button type="button" class="chip" data-chip="${i}" data-tip="${esc(tip)}" aria-label="Remove ${esc(chip.text)}">${esc(chip.text)}${many} ×</button>`;
        })
        .join("");
}

document.addEventListener("click", (event) => {
    const chip = event.target.closest(".chip[data-chip]");
    if (!chip) return;
    state.structures.splice(Number(chip.dataset.chip), 1);
    tooltip.hidden = true;
    renderChips();
    update();
});

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
    const n = site.scans.length;
    const kinds = state.foundIn === "any" ? LABELED_KINDS : [state.foundIn];
    // the other filters, as masks or tests of a scan
    const structureMasks = state.structures.map((chip) => scansWith(chip.labels, kinds));
    const query = state.search.trim().toLowerCase();
    const searchLabels = query.length >= 3 ? labelsMatching(query) : [];
    const searchMask = searchLabels.length ? scansWith(searchLabels, LABELED_KINDS) : null;
    matchedLabels = new Set([...state.structures.flatMap((chip) => chip.labels), ...searchLabels]);
    const otherPasses = (i) => {
        const scan = site.scans[i];
        if (state.ageMin != null && (scan[AGE] == null || scan[AGE] < state.ageMin)) return false;
        if (state.ageMax != null && (scan[AGE] == null || scan[AGE] > state.ageMax)) return false;
        for (const mask of structureMasks) if (!mask[i]) return false;
        if (query) {
            const dataset = site.datasets[scan[DATASET]];
            const text = `${scan[KEY]} ${dataset.name} ${dataset.folder} ${dataset.sources.map((s) => `${s[0]} ${s[1]}`).join(" ")} ${scan[MODALITY] || ""}`.toLowerCase();
            if (!text.includes(query) && !(searchMask && searchMask[i])) return false;
        }
        return true;
    };

    // The count of an option is the number of scans there would be with it ticked as well: for a facet whose values
    // are alternatives, the scans that pass every other filter; for one whose values a scan must all have, the scans
    // that pass every filter.
    const counts = Object.fromEntries(FACETS.map((f) => [f.id, new Map()]));
    const add = (facet, i) => {
        for (const value of facet.values(i)) counts[facet.id].set(value, (counts[facet.id].get(value) || 0) + 1);
    };
    filtered = [];
    for (let i = 0; i < n; i++) {
        if (!otherPasses(i)) continue;
        const failing = FACETS.filter((facet) => !facetPasses(facet, i));
        if (!failing.length) {
            filtered.push(site.scans[i]);
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

    const sorter = SORTERS[state.sort.key];
    filtered.sort((a, b) => state.sort.dir * sorter(a, b));
    if (!keepPage) state.page = 0;
    renderRows();
}

const SORTERS = {
    scan: (a, b) => a[KEY].localeCompare(b[KEY], "en", { numeric: true }),
    dataset: (a, b) => site.datasets[a[DATASET]].name.localeCompare(site.datasets[b[DATASET]].name) || SORTERS.scan(a, b),
    sex: (a, b) => (a[SEX] || "~").localeCompare(b[SEX] || "~") || SORTERS.scan(a, b),
    age: (a, b) => (a[AGE] ?? 999) - (b[AGE] ?? 999) || SORTERS.scan(a, b),
    modality: (a, b) => (a[MODALITY] || "~").localeCompare(b[MODALITY] || "~") || SORTERS.scan(a, b),
    structures: (a, b) => scanLabels(a).length - scanLabels(b).length || SORTERS.scan(a, b),
};

function downloadButtons(scan) {
    const counts = {};
    for (const file of scan[FILES]) counts[file[F_KIND]] = (counts[file[F_KIND]] || 0) + 1;
    return scan[FILES]
        .map((file) => {
            const [kind, number, size, published, sourceId, license] = file;
            const name = fileName(kind, number);
            const label = `${KIND_NAMES[kind]}${counts[kind] > 1 ? ` ${number}` : ""}`;
            const source = site.sources[sourceId] || {};
            const from = `Source: ${sourceId} ${esc(source.name || "")}`;
            if (published) {
                const tip = `<strong>${esc(name)}</strong> (${humanSize(size)})<br>${from}<br>License: ${esc(license)}`;
                return `<a class="dl" href="${fileUrl(`${scan[KEY]}/${name}`)}" download="${esc(name)}" data-tip="${esc(tip)}">${label}<span class="size">${humanSize(size)}</span></a>`;
            }
            const tip = `<strong>${esc(name)}</strong> is not in the repository: its license is ${esc(license || "not known")}.<br>${from}: obtain it there.`;
            const href = source.url || registryUrl(sourceId);
            return `<a class="dl locked" href="${esc(href)}" target="_blank" rel="noopener noreferrer" data-tip="${esc(tip)}">🔒 ${label}</a>`;
        })
        .join("");
}

function structuresCell(scan) {
    const all = scanLabels(scan);
    if (!all.length) return "–";
    const tip = LABELED_KINDS.filter((kind) => scan[LABEL_FIELD[kind]].length)
        .map((kind) => {
            const labels = scan[LABEL_FIELD[kind]];
            const names = labels.slice(0, 25).map((l) => esc(PRETTY[l])).join(", ");
            return `<strong>${KIND_NAMES[kind]} (${labels.length})</strong>: ${names}${labels.length > 25 ? ", ..." : ""}`;
        })
        .join("<br>");
    let matched = "";
    if (matchedLabels && matchedLabels.size) {
        const hits = all.filter((label) => matchedLabels.has(label));
        if (hits.length) {
            const kindsOf = (label) => LABELED_KINDS.filter((kind) => scan[LABEL_FIELD[kind]].includes(label)).map((k) => KIND_NAMES[k].toLowerCase());
            const shown = hits.slice(0, 3).map((l) => `${esc(PRETTY[l])} <span class="kinds">${kindsOf(l).join(", ")}</span>`);
            matched = `<div class="matched">${shown.join("<br>")}${hits.length > 3 ? `<br>and ${hits.length - 3} more` : ""}</div>`;
        }
    }
    return `<span class="structures" tabindex="0" data-tip="${esc(tip)}">${all.length}</span>${matched}`;
}

function renderRows() {
    const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    const rows = filtered.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
    document.getElementById("result-count").textContent =
        filtered.length === site.scans.length ? `${fmt(site.scans.length)} scans` : `${fmt(filtered.length)} of ${fmt(site.scans.length)} scans`;
    document.getElementById("scan-rows").innerHTML = rows.length
        ? rows
              .map((scan) => {
                  const dataset = site.datasets[scan[DATASET]];
                  const sources = dataset.sources
                      .map((s) => `<a href="${registryUrl(s[0])}" target="_blank" rel="noopener noreferrer" data-tip="${esc(`<strong>${s[0]}</strong> ${esc(s[1])}<br>Its entry in the BoneHub registry`)}">${s[0]}</a>`)
                      .join(" · ");
                  return `<tr>
                    <td><a class="scan-key" href="${folderUrl(scan[KEY])}" target="_blank" rel="noopener noreferrer" data-tip="Open the scan's folder on Hugging Face">${esc(scan[KEY])}</a></td>
                    <td><div class="dataset-name">${esc(dataset.name)}</div><div class="dataset-sources">sources ${sources}</div></td>
                    <td>${SEX_NAMES[scan[SEX]] || "–"}</td>
                    <td class="num">${scan[AGE] ?? "–"}</td>
                    <td>${esc(scan[MODALITY] || "–")}</td>
                    <td class="structures-cell">${structuresCell(scan)}</td>
                    <td><div class="downloads">${downloadButtons(scan)}</div></td>
                </tr>`;
              })
              .join("")
        : `<tr><td colspan="7" class="empty">No scan matches the filters.</td></tr>`;
    document.getElementById("pager").innerHTML = `<button type="button" data-page="${state.page - 1}" ${state.page === 0 ? "disabled" : ""}>Previous</button>
        <span>Page ${fmt(state.page + 1)} of ${fmt(pages)}</span>
        <button type="button" data-page="${state.page + 1}" ${state.page >= pages - 1 ? "disabled" : ""}>Next</button>`;
    for (const th of document.querySelectorAll("#scan-table th[data-sort]")) {
        th.classList.toggle("sorted-asc", th.dataset.sort === state.sort.key && state.sort.dir === 1);
        th.classList.toggle("sorted-desc", th.dataset.sort === state.sort.key && state.sort.dir === -1);
    }
}

function csvCell(value) {
    const text = String(value ?? "");
    return /[",\n;]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function exportCsv() {
    const header = ["scan", "dataset", "dataset_name", "sources", "sex", "age", "race", "ethnicity", "modality",
        ...LABELED_KINDS.map((k) => `${k}_labels`), ...KINDS.map((k) => `${k}_files`)];
    const lines = [header.join(",")];
    for (const scan of filtered) {
        const dataset = site.datasets[scan[DATASET]];
        const labels = LABELED_KINDS.map((kind) => scan[LABEL_FIELD[kind]].map((l) => site.labels[l]).join(" "));
        const urls = KINDS.map((kind) =>
            scan[FILES]
                .filter((f) => f[F_KIND] === kind)
                .map((f) => (f[F_PUBLISHED] ? fileUrl(`${scan[KEY]}/${fileName(kind, f[F_NUMBER])}`) : `restricted (source ${f[F_SOURCE]})`))
                .join(" ")
        );
        lines.push(
            [scan[KEY], dataset.folder, dataset.name, dataset.sources.map((s) => s[0]).join(" "), scan[SEX] || "", scan[AGE] ?? "",
                scan[RACE] || "", scan[ETHNICITY] || "", scan[MODALITY] || "", ...labels, ...urls]
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

function initBrowser() {
    renderFilters();
    document.getElementById("scan-search").addEventListener("input", (event) => {
        state.search = event.target.value;
        update();
    });
    document.getElementById("clear-filters").addEventListener("click", () => {
        for (const set of Object.values(state.checked)) set.clear();
        Object.assign(state, { ageMin: null, ageMax: null, structures: [], foundIn: "any", search: "" });
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
    for (const th of document.querySelectorAll("#scan-table th[data-sort]")) {
        th.addEventListener("click", () => {
            const key = th.dataset.sort;
            state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : 1 };
            update(true);
        });
    }
    update();
}

function renderFooter() {
    const repo = `https://huggingface.co/datasets/${site.repo}`;
    document.getElementById("footer-text").innerHTML = `Data as of ${esc(site.updated)}. Each file keeps the license of its source dataset: see
        <a href="${repo}/blob/main/LICENSE.md" target="_blank" rel="noopener noreferrer">LICENSE.md</a>, and cite the source datasets you use.`;
}

renderOverview();
initBrowser();
renderFooter();
