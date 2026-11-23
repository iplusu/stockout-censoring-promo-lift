"""P6c: manuscript change map (instructions sec. 6: identify each manuscript table, figure, or sentence that changes)
from the git diff of latex/main.tex between the pre-correction commit and the corrected commit; appends the
P7-stage manuscript statistics to before_after.csv (section 'manuscript_location').
Writes out/manuscript_changes.csv."""
import difflib, os, re, subprocess
import pandas as pd

H = os.path.dirname(os.path.abspath(__file__)); TEX = os.path.join(os.path.dirname(H), 'latex')
OLD, NEW = 'd0acc5d', os.environ.get('NEW_REV', 'HEAD')
old = subprocess.run(['git', '-C', TEX, 'show', f'{OLD}:main.tex'], capture_output=True, text=True, check=True).stdout.splitlines()
new = subprocess.run(['git', '-C', TEX, 'show', f'{NEW}:main.tex'], capture_output=True, text=True, check=True).stdout.splitlines()
NUM = re.compile(r'(?<![A-Za-z_{])[-+]?\d[\d,]*\.?\d*(?:\\%|%)?')


def where(lines, i):
    sec = lab = env = ''
    for j in range(i, -1, -1):
        l = lines[j]
        if not env and re.search(r'\\begin\{(table|figure)', l): env = re.search(r'\\begin\{(table|figure)', l).group(1)
        if not lab and env:
            pass
        m = re.search(r'\\(sub)*section\*?\{([^}]*)\}', l)
        if m: sec = m.group(2); break
    if env:
        for j in range(i, min(i + 60, len(lines))):
            m = re.search(r'\\label\{([^}]*)\}', lines[j])
            if m: lab = m.group(1); break
            if re.search(r'\\end\{(table|figure)', lines[j]): break
    else:
        for j in range(i, max(i - 40, -1), -1):
            m = re.search(r'\\label\{(sec:[^}]*|app:[^}]*)\}', lines[j])
            if m: lab = m.group(1); break
    return sec, lab, env or ('caption' if '\\caption' in (lines[i] if i < len(lines) else '') else 'text')


rows = []
sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        continue
    o, n = ' '.join(old[i1:i2]), ' '.join(new[j1:j2])
    sec, lab, kind = where(new, j1 if j2 > j1 else max(j1 - 1, 0))
    on, nn = NUM.findall(o), NUM.findall(n)
    removed = [x for x in on if x not in nn]; added = [x for x in nn if x not in on]
    rows.append(dict(change=tag, section=sec, label=lab, kind=kind, old_lines=f'{i1 + 1}-{i2}', new_lines=f'{j1 + 1}-{j2}',
                     numbers_removed=' '.join(removed), numbers_added=' '.join(added),
                     old_text=o[:1500], new_text=n[:1500]))
M = pd.DataFrame(rows)
figs = [('fig_a4_sensitivity.pdf', 'Figure 4 (fig:a4_sensitivity)', 'regenerated from corrected E2 (p5_e2/corrected) + all-attempt failure panel', 'p7_figures/make_figures.py'),
        ('fig_hourly_profile.pdf', 'hour-of-day profile figure', 'regenerated: mask fix + corrected M (249 cells)', 'p7_figures/make_figures.py'),
        ('fig_decomposition.pdf', 'decomposition figure', 'regenerated from corrected decomp_cells (249 cells)', 'p7_figures/make_figures.py'),
        ('fig_dose_response.pdf', 'dose-response figure', 'regenerated from corrected pending_final_dose_wildci.csv', 'p7_figures/make_figures.py'),
        ('fig_controlled_sim.pdf', 'Figure 5 (fig:controlled_sim)', 'new (controlled simulation, protocol v1.0)', 'controlled_sim/figure5.py')]
M = pd.concat([M, pd.DataFrame([dict(change='figure', section='', label=l, kind='figure', old_text=f, new_text=d, numbers_removed='', numbers_added=s)
                                for f, l, d, s in figs])], ignore_index=True)
M.to_csv(os.path.join(H, 'out', 'manuscript_changes.csv'), index=False)
print('manuscript change blocks', len(M), M.kind.value_counts().to_dict())

B = pd.read_csv(os.path.join(H, 'out', 'before_after.csv'))
B = B[B.section != 'manuscript_location']
add = M[(M.numbers_removed.fillna('') != '') | (M.numbers_added.fillna('') != '')]
B = pd.concat([B, pd.DataFrame(dict(section='manuscript_location', item=add.label.fillna('') + ' | ' + add.section.fillna('') + ' | ' + add.kind + ' | new lines ' + add.new_lines.fillna(''),
                                    before=add.numbers_removed, after=add.numbers_added,
                                    comparison_basis='numbers as printed in the manuscript before (d0acc5d) and after; sources: before_after rows above and p7_figures/*.py; '
                                                     'empirical statistics on 249 valid cells unless stated, E2 A4 on valid fits with all-attempt failures reported',
                                    note='full sentences in manuscript_changes.csv'))], ignore_index=True)
B.to_csv(os.path.join(H, 'out', 'before_after.csv'), index=False)
print('before_after rows', len(B), '(manuscript_location', len(add), ')')
