import sys, json, os
# usage: table.py TAG [engine]   prints the score table for out_still/TAG/metrics.json
tag = sys.argv[1]; eng = sys.argv[2] if len(sys.argv) > 2 else 'keyframes'
d = json.load(open(os.path.expanduser(f'~/gdg-ideathon/out_still/{tag}/metrics.json')))
f = lambda x: '   -  ' if x is None else '%6.2f' % x
print('%-18s %6s %6s %7s %6s %6s %7s %7s' % ('clip', 'leak%', 'reveal', 'gen', 'bg', 'flick', 'score', ''))
for c, v in d.items():
    if c == 'MEAN_SCORE': print('MEAN', v); continue
    m = v[eng]; print('%-18s %s %s %s %s %7.3f %7.2f' % (c, f(m['leak_pct']), f(m['reveal_mae']), f(m['gen_mae']), f(m['bg_mae']), m['flicker'], m['score']))
