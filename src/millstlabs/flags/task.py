"""Ground-truth task, deterministic paired trial assignments, and paper observables."""
import base64
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def catalog_images(path, cfg):
    """A user-supplied raster catalog can replace the explicitly approximate flags."""
    catalog = json.loads(Path(path).read_text())
    images = {}
    for spec in catalog["flags"]:
        if "pixels" in spec:
            arr = np.asarray(spec["pixels"])
            if arr.shape != (cfg.height, cfg.width, 3) or np.any(arr < 0) or np.any(arr > 255):
                raise ValueError("Catalog pixels must be height x width x RGB bytes")
            arr = arr.astype(np.uint8)
        else:
            colors = [tuple(bytes.fromhex(c.lstrip("#"))) for c in spec["colors"]]
            if any(len(c) != 3 for c in colors):
                raise ValueError("Use six-digit RGB colors")
            vertical = spec["orientation"] == "vertical"
            arr = np.empty((cfg.height, cfg.width, 3), np.uint8)
            for y in range(cfg.height):
                for x in range(cfg.width):
                    j = x*len(colors)//cfg.width if vertical else y*len(colors)//cfg.height
                    arr[y, x] = colors[j]
                    if spec.get("triangle") and x/cfg.width <= .5*(1-abs(2*(y+.5)/cfg.height-1)):
                        arr[y, x] = tuple(bytes.fromhex(spec["triangle"].lstrip("#")))
        name = spec["country"]
        if name in images:
            raise ValueError("Duplicate country in catalog")
        images[name] = arr
    if len(images) < 2:
        raise ValueError("Need at least two country labels")
    return catalog, images


@dataclass
class Trial:
    seed: int
    country: str
    crops: list[list[int]]  # Private assignment metadata; never shown to a policy/model.
    contacts: list[list[int]]

    @property
    def hash(self):
        return digest(asdict(self))


def generate_trial(seed, cfg, countries):
    world, crop_rng, contacts = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    return Trial(seed, str(world.choice(countries)),
                 [[int(crop_rng.integers(cfg.width-cfg.crop_width+1)),
                   int(crop_rng.integers(cfg.height-cfg.crop_height+1))] for _ in range(cfg.population)],
                 [contacts.choice(cfg.population, 2, replace=False).tolist()
                  for _ in range(cfg.interaction_rounds*cfg.population)])


def prepare_trials(cfg, images):
    if cfg.trial_manifest:
        data = json.loads(Path(cfg.trial_manifest).read_text())
        trials = {k: [Trial(**r) for r in data[k]] for k in ["train", "evaluation"]}
    else:
        trials = {name: [generate_trial(start+i, cfg, list(images)) for i in range(count)]
                  for name, start, count in [("train", cfg.train_seed, cfg.train_trials),
                                             ("evaluation", cfg.evaluation_seed, cfg.evaluation_trials)]}
    for name, count in [("train", cfg.train_trials), ("evaluation", cfg.evaluation_trials)]:
        if len(trials[name]) != count:
            raise ValueError(f"Manifest needs {count} {name} trials")
        for t in trials[name]:
            if t.country not in images or len(t.crops) != cfg.population:
                raise ValueError("Invalid trial country or population")
            if len(t.contacts) != cfg.interaction_rounds*cfg.population:
                raise ValueError("Manifest contact schedule length differs from the frozen protocol")
            for x, y in t.crops:
                if not (0 <= x <= cfg.width-cfg.crop_width and 0 <= y <= cfg.height-cfg.crop_height):
                    raise ValueError("Crop outside canvas")
            if any(a == b or not 0 <= min(a, b) <= max(a, b) < cfg.population for a, b in t.contacts):
                raise ValueError("Invalid directed contact")
    if {t.hash for t in trials["train"]} & {t.hash for t in trials["evaluation"]}:
        raise ValueError("Training and evaluation trials overlap")
    if {t.seed for t in trials["train"]} & {t.seed for t in trials["evaluation"]}:
        raise ValueError("Training and evaluation seed namespaces overlap")
    def evidence_hash(t):
        return digest({"country": t.country, "crops": t.crops})
    if {evidence_hash(t) for t in trials["train"]} & {evidence_hash(t) for t in trials["evaluation"]}:
        raise ValueError("Training and evaluation private evidence assignments overlap")
    return trials


def crops_for(trial, images, cfg):
    return [images[trial.country][y:y+cfg.crop_height, x:x+cfg.crop_width].copy() for x, y in trial.crops]


def image_url(crop, scale):
    # Sensor rendering, not generated artwork. PNG contains no country/position metadata.
    im = Image.fromarray(crop).resize((crop.shape[1]*scale, crop.shape[0]*scale), Image.Resampling.NEAREST)
    stream = io.BytesIO()
    im.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def endpoint(initial, final, country, cfg):
    def majority(labels):
        counts = Counter(label for label in labels if label != "__invalid__")
        return sorted(counts, key=lambda k: (-counts[k], k))[0] if counts else "__invalid__"
    counts = Counter(label for label in final if label != "__invalid__")
    top = majority(final)
    mass = counts[top]/len(final)
    if mass >= cfg.consensus_threshold:
        category = "correct_consensus" if top == country else "wrong_consensus"
    elif sum(c/len(final) >= cfg.polarization_threshold for c in counts.values()) >= 2:
        category = "polarization"
    else:
        category = "fragmentation"
    initial_accuracy = sum(x == country for x in initial)/len(initial)
    truth_mass = sum(x == country for x in final)/len(final)
    return {"initial_accuracy": initial_accuracy, "initial_majority_correct": majority(initial) == country,
            "terminal_truth_mass": truth_mass, "social_uplift": truth_mass-initial_accuracy,
            "final_majority_correct": top == country, "endpoint": category,
            "top_mass": mass, "country_counts": dict(counts)}
