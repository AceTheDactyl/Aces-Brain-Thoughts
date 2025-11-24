# Compile Instructions

This repository includes LaTeX sources for the APL Operator's Manual and the Seven Sentences Test Pack.

## Requirements
- A TeX distribution (TeX Live, MacTeX, or MikTeX)
- `pdflatex` in your PATH

## Quick Compile
```bash
pdflatex -interaction=nonstopmode apl-operators-manual.tex
pdflatex -interaction=nonstopmode apl-seven-sentences-test-pack.tex
```

Run each command twice if you add references or TOC.

## Optional Tools
- `latexmk -pdf apl-operators-manual.tex`
- `latexmk -pdf apl-seven-sentences-test-pack.tex`

## Outputs
PDFs are expected in `APL/docs/` when compiled by CI. Local builds produce `*.pdf` alongside the sources; you can then move them into `APL/docs/` manually if desired.
