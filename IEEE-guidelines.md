# IEEE Quick Authoring Guidelines (distilled)

A short checklist distilled from the IEEE conference LaTeX template. Keep this nearby while writing.

## Document skeleton

- Keep `\documentclass[conference]{IEEEtran}` and `\maketitle`.
- Fill in: Title, Authors, Affiliations, Abstract, Keywords.
- Leave placeholders for Acknowledgment and References.

## Abstract & Keywords

- Abstract: single short paragraph summarizing problem, methods, results, contribution.
- Keywords: 3–6 terms, comma-separated.

## Abbreviations & Acronyms

- Define on first use (even if defined in the abstract).
- Do not use abbreviations in the title or headings unless unavoidable.
- Common acronyms (IEEE, SI, MKS, CGS, rms) may be used without definition.

## Units

- Use SI units (MKS) as primary; avoid mixing SI and CGS.
- If using English units, put them in parentheses as a secondary unit (e.g., 5 cm (1.97 in)).
- Be consistent: either spell out all units or use abbreviations, not both.
- Use a leading zero for decimals (0.25, not .25).
- Prefer `cm\textsuperscript{3}` over `cc`.

## Equations

- Number consecutively and reference with `\eqref{}`.
- Use `align` or `IEEEeqnarray`, not `eqnarray`.
- Define all symbols before or right after the equation.
- Use the solidus `/`, `\exp()`, or exponents for compactness when appropriate.

## Figures & Tables

- Place at top or bottom of columns; large ones may span both columns.
- Figure captions go below figures; table captions go above tables.
- Refer to figures as `Fig.~\ref{}`.
- Axis labels: use words (e.g., "Magnetization (A/m)") and include units in parentheses.
- Use 8 pt font for figure labels where needed.

## LaTeX tips

- Use soft cross-references (`\ref`, `\eqref`) instead of hard-coded numbers.
- Put `\label` after the caption (not before).
- Avoid `\nonumber` inside `{array}`.
- If using BibTeX, include the `.bib` files with your submission.

## Authors & Affiliations

- List authors left-to-right, then move to next line for more.
- Keep affiliations concise (university, city, country, email is optional).
- Do not format authors in columns or group solely by affiliation.

## Headings

- Use component heads (e.g., Acknowledgment, References) for those sections.
- Use hierarchical text heads for organization (do not add subheads unless needed).

## References

- Number citations consecutively in square brackets: `[1]`.
- Use `\bibliographystyle{IEEEtran}` and include `.bib` file(s).
- For unpublished work use "unpublished"; for accepted use "in press".
- Include full author list unless there are six or more authors.

## Common editorial tips (quick)

- "data" is plural.
- Use `\mu_{0}` (zero) not letter `o` for physical constants subscripts.
- Punctuation around quotation marks follows American English rules.
- Prefer "alternatively" over "alternately" unless truly alternating.
- No period after "et" in "et al.".
- `i.e.` = that is; `e.g.` = for example.

## Submission checklist

- Remove all template instructional text (this has been done in the `.tex` template).
- Ensure figures are embedded or included and compile without missing files.
- Include `.bib` files if using BibTeX and any necessary style files.
- Verify page limits and formatting requirements of the target conference.

## Acknowledgment

Bullet points from original text:

- Spelling without "e" after "g" (Acknowledgment)
- Avoid stilted expressions like "one of us (R. B. G.) thanks ..."
- Use "R. B. G. thanks ..." instead
- Place sponsor acknowledgments in the unnumbered footnote on the first page
