# Responsible AI and qualitative translation review

## Risks to examine

- **Bias:** occupational, gender, regional, and dialectal terms may be translated
  into stereotyped or less specific forms.
- **Hallucination and omission:** decoder-only prompting may add fluent details or
  omit names, numbers, negation, and formatting.
- **Cultural misrepresentation:** idioms, honorifics, food, place names, and
  culturally bound references may be rendered literally or offensively.
- **Sensitive content:** medical, legal, financial, identity, and personal text
  can cause harm when mistranslated or sent to an external service.

## Review protocol

Inspect the saved test samples for named entities, numbers, negation, idioms,
gendered occupations, and sensitive-domain sentences. Record the source, reference,
each model output, error category, and whether a human reviewer would accept it.
Do not treat BLEU or chrF as a safety metric.

## Mitigations

Use licensed and representative data, document dataset limitations, audit by
language variety and subgroup, preserve named entities and numeric expressions,
apply terminology constraints for high-risk domains, keep a human in the loop,
redact or avoid unnecessary personal data, and route uncertain or sensitive cases
to qualified translators. Outputs must be labelled as machine-generated and
users must be able to correct them.
