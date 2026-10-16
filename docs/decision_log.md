# Decision log

- D1: Primary dataset is CarDekho Car details v3 (Indian market, has owner and seller_type).
- D2: UK audi.csv is only an optional secondary generalisation check, never mixed into training.
- D3: De-duplicate before any train/test split (1,202 exact duplicate rows found).
- D4: "torque" is a candidate to drop (inconsistent units).
