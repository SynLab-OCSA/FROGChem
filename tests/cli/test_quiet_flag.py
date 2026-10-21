"""-q silences banner and info messages on stderr; data still go to stdout."""
import subprocess
import sys


def _run(*args):
    return subprocess.run([sys.executable, "-m", "frogchem.cli.calc_er", "er", "1.5", *args],
                          capture_output=True, text=True, check=True)


def test_quiet_flag_silences_stderr():
    loud, quiet = _run(), _run("-q")
    assert "FROG-ER" in loud.stderr
    assert quiet.stderr.strip() == ""
    assert "92.63" in quiet.stdout and quiet.stdout == loud.stdout
