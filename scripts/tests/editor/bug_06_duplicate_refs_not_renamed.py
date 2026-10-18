"""Duplicate / paste (editor.html 3537-3539) renames $references only in value/P/mass/inertia/x0/v0/expr/demand/capacity/limit,
args and fields. 2D/3D body fields (angle0, omega0, at/com/v0 vectors), element/load `const` and term coefficients, control fields other
than `expr` (cond, then, else, kp, setpoint...), model inputs, part/esc/thermal fields, pulse fields and opt/goal ranges keep
pointing at the ORIGINAL value, so a duplicated module is silently wired to the old parameters."""
import asyncio
from common import main, export, check, OUT

async def s(pg):
    await pg.evaluate("""() => { S = newState(); S.meta.dim = '2d';
        const k = addNode('value', {name:'gain', value:'2'}, 40, 40);
        const a = addNode('value', {name:'a0', value:'10 deg'}, 40, 200);
        const b = addNode('body', {name:'arm', dof:'free', mass:'1 kg', inertia:'1 kg m^2', angle0:'$a0', omega0:''}, 400, 40);
        const c = addNode('ctl', {name:'sig', ctype:'expr', expr:'= $gain * 3'}, 400, 300);
        const e = addNode('element', {name:'sp', type:'spring', fields:{k:'1 N/m'}, terms:[{coef:'$gain', co:'x'}], const:'$gain', mk:'coords'}, 700, 40);
        S.links.push({from:{node:b.id, key:'body'}, to:{node:e.id, key:'t0'}, field:''});
        render(); setSel(S.nodes.map(n => n.id)); duplicateSel(); }""")
    await pg.wait_for_timeout(400)
    r = await pg.evaluate("""() => { const byN = (nm, ns) => byName(nm, ns);
        const names = S.nodes.map(n => n.kind + ':' + n.data.name);
        const b2 = byN('arm_2', 'b'), c2 = byN('sig_2', 'k'), e2 = byN('sp_2', 'e');
        return {names, body_angle0: b2 && b2.data.angle0, ctl_expr: c2 && c2.data.expr, el_coef: e2 && e2.data.terms[0].coef, el_const: e2 && e2.data.const}; }""")
    print(r)
    ok = check(r["body_angle0"] == "$a0_2", r["body_angle0"], "$a0_2", "2D body angle0 follows the renamed value")
    ok &= check(r["ctl_expr"] == "= $gain_2 * 3", r["ctl_expr"], "= $gain_2 * 3", "control expr follows the renamed value")
    ok &= check(r["el_coef"] == "$gain_2", r["el_coef"], "$gain_2", "element term coefficient follows the renamed value")
    ok &= check(r["el_const"] == "$gain_2", r["el_const"], "$gain_2", "element const follows the renamed value")
    assert ok, "BUG: duplicate leaves references pointing at the originals"
asyncio.run(main(s))
