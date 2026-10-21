# Ensemble API Reference: `frogchem.ensemble`

The `frogchem.ensemble` module provides utilities for parsing conformer populations, matching file paths to conformer identities, and performing Boltzmann ensemble weighting without external dependencies.

---

## Functions

### `load_checkpoint_populations(checkpoint_path, protocol=None, energy_term='G', temp=298.15)`

Loads Boltzmann populations from JSON, CSV, or TSV checkpoint files.

**Supported Formats:**

1. **Hierarchical JSON** (from `ensemble_analyzer` or automated screening pipelines):
   ```json
   { "1": { "energies": { "2": { "Pop": 75.0 } } } }
   ```
2. **Flat Dictionary JSON**:
   ```json
   { "1": 75.0, "2": 25.0 }
   ```
3. **CSV/TSV Table**:
   ```text
   conf,pop
   1,75.0
   2,25.0
   ```

**Parameters:**

* `checkpoint_path` (`str | Path`): Path to the checkpoint or population file.
* `protocol` (`str | int | None`, optional): Specific protocol number to read from nested JSON. Default: `None` (uses highest/latest protocol).
* `energy_term` (`str`, optional): Energy term to use if computing populations from raw energies (`'G'` or `'E'`). Default: `'G'`.
* `temp` (`float`, optional): Temperature in Kelvin for Boltzmann reweighting. Default: `298.15`.

**Returns:**

* `dict[int, float]`: Dictionary mapping conformer index (`int`) to its percentage population ($0.0 \text{ to } 100.0$).

**Example:**

```python
from frogchem.ensemble import load_checkpoint_populations

pops = load_checkpoint_populations("checkpoint.json", energy_term="G")
print("Loaded populations:", pops)
```

**Output atteso:**
```text
Loaded populations: {1: 45.2, 2: 30.1, 3: 24.7}
```

---

### `extract_conf_and_protocol(file_path)`

Extracts the conformer number and protocol number from a calculation output path, directory structure, or filename label.

**Supported Patterns:**

* Directory hierarchy: `.../conf_<ID>/protocol_<X>/*.(out|log)`
* Filename convention: `conf_1_p2.out`, `1_p2_orca.out`, `conf_3.log`
* Trajectory labels: `... - Conf 4`

**Parameters:**

* `file_path` (`str | Path`): File path or string identifier.

**Returns:**

* `tuple[int | None, int | None]`: `(conf_number, protocol_number)` as integers, or `None` if not detected.

**Example:**

```python
from frogchem.ensemble import extract_conf_and_protocol

conf, proto = extract_conf_and_protocol("project/conf_03/protocol_1/opt.out")
print(f"Parsed: Conformer {conf}, Protocol {proto}")
```

**Output atteso:**
```text
Parsed: Conformer 3, Protocol 1
```
