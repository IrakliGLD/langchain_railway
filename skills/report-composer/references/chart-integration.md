# Chart Integration Rules

Charts are deterministic evidence exhibits. The model may request a supported
visual purpose but must never create chart values.

Supported purposes are trend, comparison, composition, relationship, forecast,
and table. Code selects the final chart family, performs calculations, validates
units and periods, and produces the chart payload.

Each chart must:

- have a stable chart identifier;
- belong to exactly one report section;
- use evidence references available in the frozen manifest;
- use only axis and series field names present in those table items;
- answer a stated analytical question;
- carry a concise evidence-grounded title and caption;
- agree with the values and period discussed in its section.

Use a table when a chart would obscure exact values. Omit unsupported or
decorative chart requests rather than fabricating an exhibit.
