"use strict";
// The database's webpage: the overview figures and the scan browser, drawn from `site` (data.js, see update_data.py).

const KINDS = ["image", "segmentation", "mesh", "nurbs"];
const KIND_NAMES = { image: "Image", segmentation: "Segmentation", mesh: "Mesh", nurbs: "NURBS" };
const SEX_NAMES = { M: "Male", F: "Female", O: "Other" };
const SEX_COLORS = { M: "var(--series-male)", F: "var(--series-female)", unknown: "var(--series-unknown)" };
const PAGE_SIZE = 25;
// the fields of a scan of site.scans
const KEY = 0, DATASET = 1, SEX = 2, AGE = 3, MODALITY = 4, LABELS = 5, FILES = 6;
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
        .map((word, i) => (/^[A-Z]?\d+$|^[CTLS]\d+$/.test(word) ? word : i === 0 ? word[0] + word.slice(1).toLowerCase() : word.toLowerCase()))
        .join(" ");
    return text + side;
}

const subjectOf = (scan) => scan[KEY].split("/").slice(0, 2).join("/");
const registryUrl = (id) => `${site.registry}#dataset=${id}`;
const fileUrl = (path) => `https://huggingface.co/datasets/${site.repo}/resolve/main/${path}?download=true`;
const folderUrl = (path) => `https://huggingface.co/datasets/${site.repo}/tree/main/${path}`;

function fileName(kind, number) {
    if (kind === "image") return "image.nii.gz";
    if (kind === "segmentation") return `segmentation-${number}.nrrd`;
    return `${kind}-${number}.zip`;
}

// ---------------------------------------------------------------------------------------------------------- tooltip

const tooltip = document.getElementById("tooltip");

function placeTooltip(x, y) {
    const pad = 14;
    const { width, height } = tooltip.getBoundingClientRect();
    tooltip.style.left = `${Math.min(x + pad, window.innerWidth - width - 8)}px`;
    tooltip.style.top = `${y + pad + height > window.innerHeight ? y - height - pad : y + pad}px`;
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

function barRows(rows, max, options = {}) {
    // rows: [{label, value, valueText, tip}]
    return rows
        .map(
            (row) => `<div class="bar-label" title="${esc(row.label)}">${esc(row.label)}</div>
            <div class="bar-track" tabindex="0" data-tip="${esc(row.tip)}" aria-label="${esc(row.label)}: ${esc(row.valueText ?? fmt(row.value))}">
                <div class="bar" style="width:${max ? Math.max((row.value / max) * (options.scale ?? 85), row.value ? 0.6 : 0) : 0}%"></div>
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
    const segmented = new Set(scans.flatMap((scan) => scan[LABELS]));

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
    const perLabel = new Array(site.labels.length).fill(0);
    const perSection = new Array(site.sections.length).fill(0);
    for (const scan of scans) {
        const sections = new Set();
        for (const label of scan[LABELS]) {
            perLabel[label] += 1;
            sections.add(site.labelSection[label]);
        }
        for (const section of sections) perSection[section] += 1;
    }
    const labelsOf = site.sections.map(() => []);
    site.labels.forEach((name, i) => labelsOf[site.labelSection[i]].push(i));
    const shown = site.sections.map((name, i) => i).filter((i) => perSection[i]);
    document.getElementById("sections-note").textContent = `Scans with a segmentation of each part of the body; ${site.labels.length} structures in all`;
    document.getElementById("chart-sections").classList.add("bars-two-columns");
    const maxSection = Math.max(0, ...perSection);
    document.getElementById("chart-sections").innerHTML = barRows(
        shown.map((i) => ({
            label: site.sections[i],
            value: perSection[i],
            tip: `<strong>${esc(site.sections[i])}</strong><br>${fmt(perSection[i])} scans, ${labelsOf[i].length} structures`,
        })),
        maxSection
    );
    document.getElementById("structure-summary").textContent = `Show all ${site.labels.length} structures`;

    const maxLabel = Math.max(1, ...perLabel);
    const groups = document.getElementById("structure-groups");
    groups.className = "structure-groups";
    groups.innerHTML = shown
        .map(
            (section) => `<details class="structure-group" data-section="${section}"><summary>${esc(site.sections[section])}<span class="count">${labelsOf[section].length} structures</span></summary><div class="bars">${labelsOf[section]
                .map((label) => {
                    const name = prettyLabel(site.labels[label]);
                    return `<div class="bar-label" data-search="${esc(`${name} ${site.labels[label]}`.toLowerCase())}">${esc(name)}</div>
                    <div class="bar-track" data-search="${esc(`${name} ${site.labels[label]}`.toLowerCase())}" data-tip="<strong>${esc(name)}</strong> (${esc(site.labels[label])})<br>${fmt(perLabel[label])} scans">
                        <div class="bar" style="width:${(perLabel[label] / maxLabel) * 85}%"></div><span class="bar-value">${fmt(perLabel[label])}</span></div>`;
                })
                .join("")}</div></details>`
        )
        .join("");
    document.getElementById("structure-search").addEventListener("input", (event) => {
        const query = event.target.value.trim().toLowerCase();
        for (const group of groups.querySelectorAll(".structure-group")) {
            let visible = 0;
            for (const cell of group.querySelectorAll("[data-search]")) {
                const hit = !query || cell.dataset.search.includes(query);
                cell.style.display = hit ? "" : "none";
                if (hit) visible += 1;
            }
            group.style.display = visible ? "" : "none";
            group.open = Boolean(query) && visible > 0;
        }
    });
}

// ------------------------------------------------------------------------------------------------------ scan browser

const state = {
    datasets: new Set(),
    modalities: new Set(),
    sexes: new Set(),
    kinds: new Set(),
    published: false,
    ageMin: null,
    ageMax: null,
    structures: [],
    search: "",
    sort: { key: "scan", dir: 1 },
    page: 0,
};
let filtered = [];

function scanKinds(scan) {
    return new Set(scan[FILES].map((f) => f[F_KIND]));
}

function checkboxGroup(title, name, options) {
    // options: [{value, label, count}]
    return `<div class="filter-group"><h3>${title}</h3>${options
        .map(
            (o) => `<label class="filter-option"><input type="checkbox" name="${name}" value="${esc(o.value)}">${esc(o.label)}<span class="count">${fmt(o.count)}</span></label>`
        )
        .join("")}</div>`;
}

function renderFilters() {
    const count = (fn) => {
        const counts = new Map();
        for (const scan of site.scans) for (const value of fn(scan)) counts.set(value, (counts.get(value) || 0) + 1);
        return counts;
    };
    const byDataset = count((s) => [s[DATASET]]);
    const byModality = count((s) => [s[MODALITY] || ""]);
    const bySex = count((s) => [s[SEX] || ""]);
    const byKind = count((s) => scanKinds(s));
    const publishedImages = site.scans.filter((s) => s[FILES].some((f) => f[F_KIND] === "image" && f[F_PUBLISHED])).length;
    const html = [
        checkboxGroup("Dataset", "dataset", site.datasets.map((d, i) => ({ value: i, label: d.name, count: byDataset.get(i) || 0 }))),
        checkboxGroup("Data in the scan", "kind", KINDS.filter((k) => byKind.get(k)).map((k) => ({ value: k, label: KIND_NAMES[k], count: byKind.get(k) }))),
        `<div class="filter-group"><h3>Access</h3><label class="filter-option"><input type="checkbox" name="published">Image can be downloaded<span class="count">${fmt(publishedImages)}</span></label></div>`,
        checkboxGroup("Modality", "modality", [...byModality.entries()].sort((a, b) => b[1] - a[1]).map(([m, c]) => ({ value: m, label: m || "not known", count: c }))),
        checkboxGroup("Sex", "sex", ["F", "M", "O", ""].filter((s) => bySex.get(s)).map((s) => ({ value: s, label: SEX_NAMES[s] || "not known", count: bySex.get(s) }))),
        `<div class="filter-group"><h3>Age (years)</h3><div class="filter-range"><input type="number" id="age-min" min="0" max="120" placeholder="from" aria-label="Age from">–<input type="number" id="age-max" min="0" max="120" placeholder="to" aria-label="Age to"></div></div>`,
        `<div class="filter-group"><h3>Segmented structures</h3><input type="text" id="structure-filter" list="structure-options" placeholder="Add a structure" aria-label="Add a structure">
            <datalist id="structure-options">${site.labels.map((l) => `<option value="${esc(prettyLabel(l))}">`).join("")}</datalist><div class="chips" id="structure-chips"></div></div>`,
    ];
    const filters = document.getElementById("filters");
    filters.innerHTML = html.join("");
    filters.addEventListener("change", (event) => {
        const input = event.target;
        const sets = { dataset: state.datasets, modality: state.modalities, sex: state.sexes, kind: state.kinds };
        if (sets[input.name]) {
            const value = input.name === "dataset" ? Number(input.value) : input.value;
            if (input.checked) sets[input.name].add(value);
            else sets[input.name].delete(value);
        } else if (input.name === "published") {
            state.published = input.checked;
        }
        update();
    });
    const age = (id, key) =>
        document.getElementById(id).addEventListener("input", (event) => {
            state[key] = event.target.value === "" ? null : Number(event.target.value);
            update();
        });
    age("age-min", "ageMin");
    age("age-max", "ageMax");
    const byName = new Map(site.labels.map((l, i) => [prettyLabel(l).toLowerCase(), i]));
    const structureInput = document.getElementById("structure-filter");
    structureInput.addEventListener("change", () => {
        const label = byName.get(structureInput.value.trim().toLowerCase());
        if (label !== undefined && !state.structures.includes(label)) {
            state.structures.push(label);
            renderChips();
            update();
        }
        structureInput.value = "";
    });
}

function renderChips() {
    document.getElementById("structure-chips").innerHTML = state.structures
        .map((label) => `<button type="button" class="chip" data-label="${label}" aria-label="Remove ${esc(prettyLabel(site.labels[label]))}">${esc(prettyLabel(site.labels[label]))} ×</button>`)
        .join("");
}

document.addEventListener("click", (event) => {
    const chip = event.target.closest(".chip[data-label]");
    if (!chip) return;
    state.structures = state.structures.filter((label) => label !== Number(chip.dataset.label));
    renderChips();
    update();
});

function matches(scan, searchLabels, query) {
    if (state.datasets.size && !state.datasets.has(scan[DATASET])) return false;
    if (state.modalities.size && !state.modalities.has(scan[MODALITY] || "")) return false;
    if (state.sexes.size && !state.sexes.has(scan[SEX] || "")) return false;
    if (state.kinds.size) {
        const kinds = scanKinds(scan);
        for (const kind of state.kinds) if (!kinds.has(kind)) return false;
    }
    if (state.published && !scan[FILES].some((f) => f[F_KIND] === "image" && f[F_PUBLISHED])) return false;
    if (state.ageMin != null && (scan[AGE] == null || scan[AGE] < state.ageMin)) return false;
    if (state.ageMax != null && (scan[AGE] == null || scan[AGE] > state.ageMax)) return false;
    if (state.structures.length) {
        const labels = new Set(scan[LABELS]);
        for (const label of state.structures) if (!labels.has(label)) return false;
    }
    if (query) {
        const dataset = site.datasets[scan[DATASET]];
        const text = `${scan[KEY]} ${dataset.name} ${dataset.folder} ${dataset.sources.map((s) => `${s[0]} ${s[1]}`).join(" ")} ${scan[MODALITY] || ""}`.toLowerCase();
        if (!text.includes(query) && !scan[LABELS].some((label) => searchLabels.has(label))) return false;
    }
    return true;
}

const SORTERS = {
    scan: (a, b) => a[KEY].localeCompare(b[KEY], "en", { numeric: true }),
    dataset: (a, b) => site.datasets[a[DATASET]].name.localeCompare(site.datasets[b[DATASET]].name) || SORTERS.scan(a, b),
    sex: (a, b) => (a[SEX] || "~").localeCompare(b[SEX] || "~") || SORTERS.scan(a, b),
    age: (a, b) => (a[AGE] ?? 999) - (b[AGE] ?? 999) || SORTERS.scan(a, b),
    modality: (a, b) => (a[MODALITY] || "~").localeCompare(b[MODALITY] || "~") || SORTERS.scan(a, b),
    structures: (a, b) => a[LABELS].length - b[LABELS].length || SORTERS.scan(a, b),
};

function update(keepPage = false) {
    const query = state.search.trim().toLowerCase();
    const searchLabels = new Set();
    if (query.length > 2) site.labels.forEach((l, i) => (prettyLabel(l).toLowerCase().includes(query) || l.toLowerCase().includes(query)) && searchLabels.add(i));
    filtered = site.scans.filter((scan) => matches(scan, searchLabels, query));
    const sorter = SORTERS[state.sort.key];
    filtered.sort((a, b) => state.sort.dir * sorter(a, b));
    if (!keepPage) state.page = 0;
    renderRows();
}

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
    const labels = scan[LABELS];
    if (!labels.length) return "–";
    const names = labels.slice(0, 40).map((l) => esc(prettyLabel(site.labels[l])));
    const more = labels.length > 40 ? `<br>and ${labels.length - 40} more` : "";
    return `<span class="structures" tabindex="0" data-tip="${esc(names.join(", ") + more)}">${labels.length}</span>`;
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
                    <td class="num">${structuresCell(scan)}</td>
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
    const header = ["scan", "dataset", "dataset_name", "sources", "sex", "age", "modality", "structures", ...KINDS.map((k) => `${k}_files`)];
    const lines = [header.join(",")];
    for (const scan of filtered) {
        const dataset = site.datasets[scan[DATASET]];
        const urls = KINDS.map((kind) =>
            scan[FILES]
                .filter((f) => f[F_KIND] === kind)
                .map((f) => (f[F_PUBLISHED] ? fileUrl(`${scan[KEY]}/${fileName(kind, f[F_NUMBER])}`) : `restricted (source ${f[F_SOURCE]})`))
                .join(" ")
        );
        lines.push(
            [scan[KEY], dataset.folder, dataset.name, dataset.sources.map((s) => s[0]).join(" "), scan[SEX] || "", scan[AGE] ?? "", scan[MODALITY] || "",
                scan[LABELS].map((l) => site.labels[l]).join(" "), ...urls]
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
        for (const set of [state.datasets, state.modalities, state.sexes, state.kinds]) set.clear();
        Object.assign(state, { published: false, ageMin: null, ageMax: null, structures: [], search: "" });
        for (const input of document.querySelectorAll("#filters input")) input.type === "checkbox" ? (input.checked = false) : (input.value = "");
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
