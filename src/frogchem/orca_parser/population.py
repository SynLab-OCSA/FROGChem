import re
import logging

logger = logging.getLogger(__name__)


def parse_charges(text, regex_key, reg):
    """Atomic charges {idx: {'atom', 'charge'}} from the LAST matching population block.

    Multi-step jobs (optimisations, scans) print one block per geometry: the last one
    belongs to the final geometry.
    """
    try:
        block_match = None
        for block_match in re.finditer(reg[regex_key], text):
            pass
        if not block_match:
            return None

        block = block_match.group(1)

        # two-letter elements are printed without a space before the colon ('  32 Br:   -0.138');
        # open-shell blocks ('... AND SPIN POPULATIONS') add the spin population after the charge
        pattern = r'\s+(\d+)\s+([a-zA-Z]+)\s*:\s+(-?\d+\.\d+)(?:[ \t]+(-?\d+\.\d+))?'
        matches = re.findall(pattern, block)

        charges = {}
        for idx, atom, chg, spin in matches:
            charges[int(idx)] = {'atom': atom, 'charge': float(chg)}
            if spin:
                charges[int(idx)]['spin'] = float(spin)
        return charges

    except Exception as e:
        logger.warning("Error parsing %s: %s", regex_key, e)
        return None