---
name: arxiv
description: Search and read arXiv papers by keyword, author, category, or ID via the arXiv API.
---

# arXiv Research

Search arXiv and read papers using its public API (no key). Run queries with
`Bash`/`curl`; read paper content with `WebFetch` or the firecrawl scrape tool.

## Quick reference

| Action | How |
|--------|-----|
| Search | `curl -s "https://export.arxiv.org/api/query?search_query=all:QUERY&max_results=5"` |
| Specific paper | `curl -s "https://export.arxiv.org/api/query?id_list=2402.03300"` |
| Read abstract | `WebFetch https://arxiv.org/abs/2402.03300` |
| Read full paper | firecrawl scrape / `WebFetch` on `https://arxiv.org/pdf/2402.03300` |

## Searching

The API returns Atom XML — parse it with a short python filter for clean output:

```bash
curl -s "https://export.arxiv.org/api/query?search_query=all:GRPO+reinforcement+learning&max_results=5&sortBy=submittedDate&sortOrder=descending" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns={'a':'http://www.w3.org/2005/Atom'}
for i,e in enumerate(ET.parse(sys.stdin).getroot().findall('a:entry',ns)):
    aid=e.find('a:id',ns).text.split('/abs/')[-1]
    title=' '.join(e.find('a:title',ns).text.split())
    authors=', '.join(a.find('a:name',ns).text for a in e.findall('a:author',ns))
    pub=e.find('a:published',ns).text[:10]
    cats=', '.join(c.get('term') for c in e.findall('a:category',ns))
    summ=' '.join(e.find('a:summary',ns).text.split())[:200]
    print(f'{i+1}. [{aid}] {title}')
    print(f'   {authors}')
    print(f'   {pub} | {cats}')
    print(f'   {summ}...')
    print(f'   PDF: https://arxiv.org/pdf/{aid}\n')
"
```

## Query syntax

| Prefix | Field | Example |
|--------|-------|---------|
| `all:` | all fields | `all:transformer+attention` |
| `ti:` | title | `ti:large+language+models` |
| `au:` | author | `au:vaswani` |
| `abs:` | abstract | `abs:reinforcement+learning` |
| `cat:` | category | `cat:cs.AI` |

Booleans: `+` = AND (default), `+OR+`, `+ANDNOT+`; exact phrase `ti:"chain+of+thought"`;
combine `au:hinton+AND+cat:cs.LG`.

Sorting/paging: `sortBy=relevance|lastUpdatedDate|submittedDate`,
`sortOrder=ascending|descending`, `start=<offset>`, `max_results=<n>` (default 10).

```bash
# latest 10 in cs.AI
curl -s "https://export.arxiv.org/api/query?search_query=cat:cs.AI&sortBy=submittedDate&sortOrder=descending&max_results=10"
```

## Fetching specific papers

```bash
curl -s "https://export.arxiv.org/api/query?id_list=2402.03300"                 # one
curl -s "https://export.arxiv.org/api/query?id_list=2402.03300,1706.03762"      # many
```

## Reading content

- **Abstract + metadata (fast):** `WebFetch` on `https://arxiv.org/abs/<id>`.
- **Full paper:** scrape `https://arxiv.org/pdf/<id>` with the firecrawl scrape
  tool (PDF → markdown) or `WebFetch`. Large papers: read the abstract and
  section headings first, then fetch only the sections you need.

## Common categories

`cs.AI` `cs.LG` `cs.CL` (NLP) `cs.CV` `cs.RO` `stat.ML` `math.OC` `q-bio` `eess.AS`.

When summarizing findings from papers, pair this with the `grounded-citations`
skill so every claim carries its arXiv id / URL.
