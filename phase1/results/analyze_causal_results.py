import json
from collections import Counter

with open('causal_discovery_findings.json', 'r') as f:
    data = json.load(f)

te = data['transfer_entropy']
print(f'Total significant relationships: {te["significant_count"]}')
print(f'\nLag distribution in top 20:')
lags = [p['lag'] for p in te['significant_pairs']]
lag_counts = Counter(lags)
for lag in sorted(lag_counts.keys()):
    print(f'  Lag {lag}: {lag_counts[lag]} relationships')

print(f'\nEffect direction:')
increases = sum(1 for p in te['significant_pairs'] if p['excitation_strength'] > 1.0)
decreases = sum(1 for p in te['significant_pairs'] if p['excitation_strength'] < 1.0)
print(f'  Increases: {increases}')
print(f'  Decreases: {decreases}')

print(f'\nStrongest effects:')
sorted_by_strength = sorted(te['significant_pairs'],
                           key=lambda x: abs(x['excitation_strength'] - 1.0),
                           reverse=True)[:5]
for p in sorted_by_strength:
    print(f"  {p['source']} -> {p['target']}: {p['excitation_strength']:.3f} at lag {p['lag']}")
