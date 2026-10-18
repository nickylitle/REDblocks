"""Renaming a Value (name box, editor.html 3217/3224) does not update the '$name' text in the fields it is plugged into
(field-socket wires are text, 1879-1881), nor '$name' in expressions. The wire the user drew vanishes and the model
stops compiling ('undefined parameter'). Duplicate/paste does rename refs (3528), so the machinery exists."""
import asyncio
from common import main, load, export, MIN1D, OUT

async def s(pg):
    # build it the way the UI does: a Value plugged into the spring's k field writes '$k' into the field text (no S.link)
    await pg.evaluate("""() => { S = newState(); S.meta.dim = '1d';
        addNode('value', {name:'k', value:'1000 N/m'}, 40, 40); addNode('value', {name:'m', value:'1 kg'}, 40, 200);
        const b = addNode('body', {name:'mass', dof:'translation', mass:'$m', x0:'1 mm'}, 400, 40);
        const e = addNode('element', {name:'spring', type:'spring', fields:{k:'$k'}, terms:[{coef:'1'}]}, 700, 40);
        S.links.push({from:{node:b.id, key:'body'}, to:{node:e.id, key:'t0'}, field:''});
        const o = addNode('output', {name:'F', quantity:'force', reduce:'max', unit:'N'}, 1000, 40);
        S.links.push({from:{node:e.id, key:'q.force'}, to:{node:o.id, key:'src'}, field:''}); render(); }""")
    await pg.wait_for_timeout(300)
    before = await pg.evaluate("async () => (await API.call('check', (await API.call('export_yaml', toDoc())).text)).ok")
    await pg.evaluate("""() => { const k = byName('k', 'p'); const el = world.querySelector(`.node[data-id="${k.id}"] input.nm`);
        el.value = 'stiff'; el.dispatchEvent(new Event('input', {bubbles:true})); el.dispatchEvent(new Event('change', {bubbles:true})); }""")
    await pg.wait_for_timeout(800)
    r = await pg.evaluate("""async () => { const e = byName('spring', 'e'); const y = (await API.call('export_yaml', toDoc())).text;
        const c = await API.call('check', y); return {field: e.data.fields.k, names: S.nodes.filter(n => n.kind === 'value').map(n => n.data.name), ok: c.ok, err: c.error || ''}; }""")
    await pg.screenshot(path=OUT + "bug_11_rename.png")
    print("before rename compiles:", before, "| after rename: value names", r["names"], " spring.k =", r["field"], " compiles:", r["ok"], r["err"][:120])
    ok = r["field"] == "$stiff" and r["ok"]
    print("PASS" if ok else "FAIL", "references follow a rename\n   observed: spring.k =", r["field"], "\n   expected: spring.k = $stiff and the model still compiles")
    assert ok
asyncio.run(main(s))
