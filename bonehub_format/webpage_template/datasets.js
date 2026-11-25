// The source dataset cards (from datasetCards in data.js) and the copy buttons of the code snippets.
(function() {
    const KIND_NAMES = { image: 'Images', segmentation: 'Segmentations', mesh: 'Meshes', nurbs: 'NURBS' };
    const LICENSE_URLS = {
        'CC0-1.0': 'https://creativecommons.org/publicdomain/zero/1.0/',
        'MIT': 'https://opensource.org/license/mit',
        'Apache-2.0': 'https://www.apache.org/licenses/LICENSE-2.0',
        'GPL-3.0-only': 'https://www.gnu.org/licenses/gpl-3.0.html',
        'ODbL-1.0': 'https://opendatacommons.org/licenses/odbl/1-0/',
        'ODC-By-1.0': 'https://opendatacommons.org/licenses/by/1-0/',
        'PDDL-1.0': 'https://opendatacommons.org/licenses/pddl/1-0/'
    };

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = text;
        return node;
    }

    function formatNumber(value) {
        return Number(value || 0).toLocaleString('en-US');
    }

    function licenseUrl(license) {
        if (!license || license === 'restricted') return null;
        const cc = /^CC-(BY(?:-NC)?(?:-SA|-ND)?)-(\d\.\d)$/.exec(license);
        if (cc) return `https://creativecommons.org/licenses/${cc[1].toLowerCase()}/${cc[2]}/`;
        return LICENSE_URLS[license] || null;
    }

    function licenseNode(license) {
        if (!license) return el('span', 'license-unknown', 'not given');
        if (license === 'restricted') return el('span', 'license-restricted', 'Restricted');
        const url = licenseUrl(license);
        const node = el(url ? 'a' : 'span', 'license-link', license.replace(/-/g, ' ').replace('CC ', 'CC '));
        if (url) {
            node.href = url;
            node.target = '_blank';
            node.rel = 'noopener noreferrer';
        }
        return node;
    }

    function badge(kind, text, icon) {
        const node = el('span', `status-badge status-${kind}`);
        const mark = el('span', 'status-icon', icon);
        mark.setAttribute('aria-hidden', 'true');
        node.appendChild(mark);
        node.appendChild(document.createTextNode(text));
        return node;
    }

    async function copyText(text, button) {
        try {
            await navigator.clipboard.writeText(text);
            const before = button.textContent;
            button.textContent = 'Copied';
            setTimeout(() => { button.textContent = before; }, 1500);
        } catch (error) {
            window.prompt('Copy:', text);
        }
    }

    function linkButton(href, text) {
        const a = el('a', 'card-link', text);
        a.href = href;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        return a;
    }

    function renderCard(card) {
        const article = el('article', 'dataset-card');
        article.dataset.search = [card.id, card.name, card.description, card.country, Object.keys(card.modalities || {}).join(' ')].join(' ').toLowerCase();

        const head = el('header', 'dataset-card-head');
        head.appendChild(el('span', 'dataset-id', `ds-${String(card.id).padStart(4, '0')}`));
        head.appendChild(el('h3', 'dataset-name', card.name));
        article.appendChild(head);

        const chips = el('div', 'dataset-chips');
        [card.country, card.year, card.version ? `v${card.version}` : null].filter(Boolean).forEach(value => chips.appendChild(el('span', 'chip', value)));
        Object.entries(card.modalities || {}).forEach(([modality, count]) => chips.appendChild(el('span', 'chip chip-modality', `${modality} · ${formatNumber(count)}`)));
        if (chips.childElementCount) article.appendChild(chips);

        const stats = el('dl', 'dataset-stats');
        [['Subjects', card.subjects], ['Scans', card.scans], ['Scans with its data', card.scansProvided]].forEach(([label, value]) => {
            const item = el('div', 'dataset-stat');
            item.appendChild(el('dt', null, label));
            item.appendChild(el('dd', null, formatNumber(value)));
            stats.appendChild(item);
        });
        article.appendChild(stats);
        if (card.scans === 0 && card.scansProvided > 0) {
            article.appendChild(el('p', 'dataset-note', 'All its subjects live in other datasets’ folders: it adds its data to their scans.'));
        }

        const kinds = Object.entries(card.kinds || {});
        if (kinds.length) {
            const table = el('table', 'license-table');
            const caption = el('caption', 'visually-hidden', `Licenses of ${card.name}`);
            table.appendChild(caption);
            kinds.forEach(([kind, info]) => {
                const tr = el('tr');
                tr.appendChild(el('th', null, KIND_NAMES[kind] || kind));
                const license = el('td');
                license.appendChild(licenseNode(info.license));
                tr.appendChild(license);
                const status = el('td', 'license-status');
                if (info.published) status.appendChild(badge('published', `${formatNumber(info.published)} published`, '✓'));
                if (info.withheld) status.appendChild(badge('withheld', `${formatNumber(info.withheld)} not published`, '🔒'));
                tr.appendChild(status);
                table.appendChild(tr);
            });
            article.appendChild(table);
        }

        if (card.description) {
            const description = el('p', 'dataset-description', card.description);
            article.appendChild(description);
            if (card.description.length > 220) {
                description.classList.add('clamped');
                const more = el('button', 'text-button', 'More');
                more.type = 'button';
                more.addEventListener('click', () => {
                    const clamped = description.classList.toggle('clamped');
                    more.textContent = clamped ? 'More' : 'Less';
                });
                article.appendChild(more);
            }
        }
        if (card.remarks) article.appendChild(el('p', 'dataset-remarks', card.remarks));

        if (card.subsets && Object.keys(card.subsets).length) {
            const details = el('details', 'dataset-subsets');
            details.appendChild(el('summary', null, `${Object.keys(card.subsets).length} collections with terms of their own`));
            const list = el('ul');
            Object.entries(card.subsets).forEach(([key, subset]) => {
                const li = el('li');
                li.appendChild(el('span', 'subset-name', subset.name || key));
                const license = subset.license || {};
                const values = [...new Set(Object.values(license).filter(Boolean))];
                if (values.length) {
                    li.appendChild(document.createTextNode(' · '));
                    values.forEach((value, i) => {
                        if (i) li.appendChild(document.createTextNode(', '));
                        li.appendChild(licenseNode(value));
                    });
                }
                list.appendChild(li);
            });
            details.appendChild(list);
            article.appendChild(details);
        }

        const actions = el('div', 'dataset-actions');
        if (card.scansProvided) {
            const show = el('button', 'btn-secondary btn-small', 'Show scans');
            show.type = 'button';
            show.addEventListener('click', () => {
                const tables = window.BoneHubTables || {};
                if (!tables.scans) return;
                tables.scans.setFilter(`data of ${card.name} (${card.id})`, row => Array.isArray(row._providers) && row._providers.includes(card.id));
                const target = document.getElementById('scans');
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
            actions.appendChild(show);
        }
        if (card.citation) {
            const cite = el('button', 'btn-secondary btn-small', 'Copy BibTeX');
            cite.type = 'button';
            cite.addEventListener('click', () => copyText(card.citation, cite));
            actions.appendChild(cite);
        }
        if (card.url) actions.appendChild(linkButton(card.url, 'Source'));
        if (card.paper) actions.appendChild(linkButton(card.paper.startsWith('http') ? card.paper : `https://doi.org/${card.paper}`, 'Paper'));
        actions.appendChild(linkButton(card.registry, 'Registry'));
        article.appendChild(actions);
        return article;
    }

    function renderCards() {
        const root = document.getElementById('datasetCards');
        if (!root) return;
        const cards = typeof datasetCards !== 'undefined' ? datasetCards : [];
        root.innerHTML = '';
        if (!cards.length) {
            root.appendChild(el('p', 'overview-empty', 'No dataset metadata available.'));
            return;
        }
        cards.forEach(card => root.appendChild(renderCard(card)));
        const filter = document.getElementById('datasetCardFilter');
        const count = document.getElementById('datasetCardCount');
        const update = () => {
            const term = filter ? filter.value.trim().toLowerCase() : '';
            let shown = 0;
            root.querySelectorAll('.dataset-card').forEach(node => {
                const visible = !term || node.dataset.search.includes(term);
                node.hidden = !visible;
                shown += visible ? 1 : 0;
            });
            if (count) count.textContent = `${shown} of ${cards.length} datasets`;
        };
        if (filter) filter.addEventListener('input', update);
        update();
    }

    function setUpCopyButtons() {
        document.querySelectorAll('.code-block').forEach(block => {
            const code = block.querySelector('code');
            const button = block.querySelector('.copy-button');
            if (code && button) button.addEventListener('click', () => copyText(code.textContent, button));
        });
    }

    document.addEventListener('DOMContentLoaded', function() {
        renderCards();
        setUpCopyButtons();
    });
})();
