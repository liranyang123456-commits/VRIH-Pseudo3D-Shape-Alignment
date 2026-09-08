from pathlib import Path
import re

paper = Path(r'E:/elsarticle-template-TMI_Revised/revision2_materials/VRIH_Paper_R2.tex').read_text(encoding='utf-8')
bib_start = paper.find(r'\begin{thebibliography}')
bib_end = paper.find(r'\end{thebibliography}')
bib_text = paper[bib_start:bib_end]

items = re.findall(r'\\bibitem\{([^}]+)\}(.*?)(?=\\bibitem|\Z)', bib_text, re.S)
print(f"Total bibitems to verify: {len(items)}\n")

for tag, raw in items:
    text = ' '.join(raw.split())
    # Extract authors (before first dot)
    first_dot = text.find('.')
    authors = text[:first_dot] if first_dot > 0 else 'UNKNOWN'
    
    # Extract year
    year_match = re.search(r'\b(19\d\d|20\d\d)\b', text)
    year = year_match.group(0) if year_match else 'NO_YEAR'
    
    # Extract DOI or URL
    doi_match = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', text)
    doi = doi_match.group(0) if doi_match else 'NO_DOI'
    
    # Check journal/venue
    has_venue = any(v in text.lower() for v in ['ieee', 'trans', 'cvpr', 'eccv', 'iccv', 'neurips', 'aaai', 'nature', 'medical', 'springer', 'elsevier', 'science', 'comput', 'vrih', 'media', 'inffus', 'patcog'])
    
    # Check where cited in text
    cites = list(re.finditer(r'\\cite\{[^}]*\b' + tag + r'\b[^}]*\}', paper[:bib_start]))
    
    print(f"[{tag:2s}] ({year}) DOI: {doi[:32]:32s} | Cited: {len(cites):2d}x | Venue OK: {str(has_venue):5s} | {authors[:35]}...")
