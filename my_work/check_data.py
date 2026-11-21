# Verifica che tutti i dataset della minischool si carichino
from pathlib import Path
import spikeinterface.full as si

# repo/my_work/check_data.py -> Documents/neuralnet2026/data/school_neuralnet_2026
ROOT = Path(__file__).resolve().parents[2] / "data" / "school_neuralnet_2026"

rec = si.load(ROOT / "data" / "dandiset-000939-90s")             # recording salvata da SI
print(rec)

glx = si.read_spikeglx(ROOT / "data" / "sub-001_ses-001_g0_t0", stream_name="imec0.ap")  # segnale AP, non il canale SYNC
print(glx)

res = ROOT / "results" / "sub-A3702_ses-191126-3600s"
for a in sorted(p for p in res.glob("analyzer_*") if p.is_dir()):   # salta gli .zip originali
    an = si.load_sorting_analyzer(a, load_extensions=False)          # solo metadati, veloce
    print(f"{a.name}: {an.get_num_units()} units")