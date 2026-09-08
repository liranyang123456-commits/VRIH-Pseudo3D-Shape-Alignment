from pathlib import Path
import re

paper = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2.tex').read_text(encoding='utf-8')
resp = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials/response2.tex').read_text(encoding='utf-8')

patterns = [
    r'it is worth noting', r'it should be noted', r'it is important to note',
    r'plays a (?:crucial|pivotal|vital) role', r'testament to',
    r'delve into', r'a tapestry of', r'beacon of', r'foster\w*',
    r'harness\w*', r'underscores the importance', r'paramount importance',
    r'strikes a delicate balance', r'game-changer', r'groundbreaking',
    r'revolutioniz\w*', r'seamlessly integrates', r'pave[s]? the way',
    r'in a nutshell', r'notably,', r'importantly,', r'furthermore,',
    r'moreover,', r'additionally,', r'in summary,', r'to summarize,'
]

print('=== SCANNING PAPER FOR AI PATTERNS ===')
for pat in patterns:
    matches = list(re.finditer(r'\b' + pat, paper, re.I))
    if matches:
        print(f'Pattern "{pat}" matched {len(matches)} times:')
        for m in matches[:5]:
            snippet = paper[max(0, m.start()-40):min(len(paper), m.end()+40)].replace('\n', ' ')
            print(f'   ...{snippet}...')

print('\n=== SCANNING RESPONSE FOR AI PATTERNS ===')
for pat in patterns:
    matches = list(re.finditer(r'\b' + pat, resp, re.I))
    if matches:
        print(f'Pattern "{pat}" matched {len(matches)} times:')
        for m in matches[:5]:
            snippet = resp[max(0, m.start()-40):min(len(resp), m.end()+40)].replace('\n', ' ')
            print(f'   ...{snippet}...')
