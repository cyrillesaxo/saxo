import json,sys
from pathlib import Path
p=Path(sys.argv[1]); s=json.load((p/'results/summary.json').open())
def ck(n,c):
    assert c,n; print('PASS',n)
ck('two language pairs',len(s['language_pairs'])==2)
ck('large sample',s['n_examples']>10000)
ck('boundary-only cohort',s['quadrants']['S0_Bplus']['n']>100)
ck('dimensions distinct',abs(s['semantic_boundary_spearman'])<.95)
ck('typed rule stricter',s['rules']['typed_clear_2d']['coverage']<=s['rules']['semantic_clear_1d']['coverage'])
ck('typed false-success no worse',s['rules']['typed_clear_2d']['false_success_rate']<=s['rules']['semantic_clear_1d']['false_success_rate'])
ck('automatic estimator not claimed',not s['claims']['automatic_boundary_estimator_validated'])
ck('full RC not claimed',not s['claims']['full_rc_validated'])
print('STB-0 tests passed')
