/* Which labels a label covers, for the database's webpage: the same rules as covers() in bonehub_format/labelmap.py,
 * read from the label map's tables (structures.json, see bonehub_format/body_map.py).
 *
 * makeCovers(labelmap) returns covers(a, b) for label names a and b, where labelmap holds
 *   fields:        label -> [node, part, tissue, side] (null where the label has none)
 *   nodes:         node -> {parent, alsoUnder, members, inPart, partBroader}
 *   tissueBroader: tissue -> the tissues it lies in
 */

function makeCovers(labelmap) {
    const fields = labelmap.fields, nodes = labelmap.nodes, tissueBroader = labelmap.tissueBroader || {};
    const node = (name) => nodes[name] || {};
    const isAggregate = (name) => Array.isArray(node(name).members);

    function treeLine(name) {
        const line = [];
        for (let n = name; n; n = node(n).parent || null) line.push(n);
        return line;
    }

    const ancestorsOf = new Map();
    function ancestors(name) {
        if (!ancestorsOf.has(name)) {
            const out = new Set(), todo = [name];
            while (todo.length) {
                const n = todo.pop();
                if (out.has(n) || !(n in nodes)) continue;
                out.add(n);
                if (node(n).parent) todo.push(node(n).parent);
                todo.push(...(node(n).alsoUnder || []));
            }
            ancestorsOf.set(name, out);
        }
        return ancestorsOf.get(name);
    }

    function within(broader, outer, inner) {
        if (outer === inner) return true;
        const todo = [inner], seen = new Set();
        while (todo.length) {
            const x = todo.pop();
            if (seen.has(x)) continue;
            seen.add(x);
            for (const y of broader[x] || []) {
                if (y === outer) return true;
                todo.push(y);
            }
        }
        return false;
    }
    const partWithin = (name, outer, inner) => within(node(name).partBroader || {}, outer, inner);
    const tissueWithin = (outer, inner) => outer === null || outer === inner || (inner !== null && within(tissueBroader, outer, inner));

    function partOk(a, b) {
        const [aNode, aPart] = a, [bNode, bPart] = b;
        if (aPart === null) return true;
        if (aNode === bNode) return bPart !== null && partWithin(bNode, aPart, bPart);
        const line = treeLine(bNode);
        const at = line.indexOf(aNode);
        if (at < 0) return false; // a part covers only what lies under its node in the tree
        if (bPart !== null && partWithin(bNode, aPart, bPart)) return true;
        const child = node(line[at - 1]);
        return child.inPart != null && partWithin(aNode, aPart, child.inPart);
    }

    const withQualifiers = (member, of) => {
        const m = fields[member];
        return [m[0], m[1], of[2] ?? m[2], of[3] ?? m[3]];
    };

    const memo = new Map();
    function coversKey(a, b) {
        const key = a.join("|") + "#" + b.join("|");
        if (a.join("|") === b.join("|")) return true;
        if (memo.has(key)) return memo.get(key);
        memo.set(key, false); // guards against a cycle while this answer is worked out
        let result = false;
        if (isAggregate(a[0]) && a[1] === null) result = node(a[0]).members.some((m) => coversKey(withQualifiers(m, a), b));
        if (!result) {
            if (isAggregate(b[0]) && b[1] === null && !ancestors(b[0]).has(a[0]))
                result = node(b[0]).members.every((m) => coversKey(a, withQualifiers(m, b)));
            else if (ancestors(b[0]).has(a[0]))
                result = partOk(a, b) && tissueWithin(a[2], b[2]) && (a[3] === null || a[3] === b[3]);
        }
        memo.set(key, result);
        return result;
    }

    return (a, b) => (a in fields && b in fields ? coversKey(fields[a], fields[b]) : a === b);
}

if (typeof module !== "undefined") module.exports = { makeCovers };
