"""
PlantGuard — disease metadata and basic management recommendations.

Maps each PlantVillage class to a human-readable name, the crop, the
disease type, and a short, non-prescriptive management note shown in the
app output (proposal Stage 5: "basic treatment info").

NOTE: recommendations are educational summaries of common agronomic
practice, not professional agricultural advice. Always confirm with a
local extension officer before applying chemical treatments.
"""
from __future__ import annotations


def _pretty(raw: str) -> tuple[str, str]:
    """Return (crop, disease) nicely formatted from a PlantVillage label."""
    crop, _, disease = raw.partition("___")
    crop = crop.replace("_", " ").replace("(maize)", "(Maize)").strip()
    disease = disease.replace("_", " ").strip()
    if disease.lower() == "healthy":
        disease = "Healthy"
    return crop, disease


# Short management notes keyed by a disease keyword (lower-case match).
_TREATMENTS = {
    "healthy": "No disease detected. Maintain regular monitoring, balanced "
               "nutrition, and good field hygiene.",
    "scab": "Remove fallen leaves, prune for airflow, and apply a labelled "
            "fungicide (e.g. captan) at green-tip and petal-fall stages.",
    "black rot": "Prune and destroy infected canes/mummies; apply protectant "
                 "fungicides early in the season and avoid overhead watering.",
    "rust": "Remove nearby alternate hosts where possible and apply a "
            "labelled fungicide preventively in humid conditions.",
    "powdery mildew": "Improve airflow, avoid excess nitrogen, and use sulphur "
                      "or potassium-bicarbonate sprays at first signs.",
    "leaf spot": "Remove infected debris, rotate crops, and apply a protectant "
                 "fungicide; avoid wetting foliage during irrigation.",
    "gray leaf spot": "Use resistant hybrids, rotate crops, and apply foliar "
                      "fungicide if disease pressure is high.",
    "blight": "Remove infected tissue promptly, ensure good drainage, and apply "
              "a labelled fungicide; avoid working plants when wet.",
    "early blight": "Mulch to limit soil splash, rotate crops, and apply "
                    "chlorothalonil/mancozeb-type protectants on a schedule.",
    "late blight": "Act fast — destroy infected plants, avoid overhead water, "
                   "and apply protectant fungicides; this disease spreads rapidly.",
    "esca": "Avoid pruning wounds in wet weather; remove and destroy severely "
            "affected vines. No reliable chemical cure exists.",
    "haunglongbing": "Citrus greening has no cure. Remove infected trees and "
                     "control the psyllid vector to limit spread.",
    "citrus greening": "Remove infected trees and manage the psyllid vector; "
                       "there is no curative treatment.",
    "bacterial spot": "Use certified disease-free seed, copper-based sprays, and "
                      "avoid overhead irrigation to limit bacterial spread.",
    "leaf mold": "Lower humidity and improve ventilation in greenhouses; apply "
                 "labelled fungicide and use resistant cultivars.",
    "septoria": "Remove lower infected leaves, mulch, rotate crops, and apply "
                "protectant fungicide early.",
    "spider mites": "These are pests, not a pathogen. Increase humidity, spray "
                    "water, and use miticides or predatory mites if severe.",
    "target spot": "Improve airflow, remove infected leaves, and apply a "
                   "labelled fungicide preventively.",
    "yellow leaf curl virus": "Virus spread by whiteflies. Control whiteflies, "
                              "remove infected plants, and use resistant varieties.",
    "mosaic virus": "No cure. Remove infected plants, disinfect tools, and "
                    "control aphid vectors; use resistant seed.",
    "leaf scorch": "Remove infected leaves, improve airflow, and apply a "
                   "labelled fungicide; renovate beds after harvest.",
    "leaf blight": "Remove infected foliage, ensure airflow, and apply "
                   "protectant fungicide in humid weather.",
}


def _treatment_for(disease: str) -> str:
    d = disease.lower()
    if d == "healthy":
        return _TREATMENTS["healthy"]
    # most specific keyword first
    for key in sorted(_TREATMENTS, key=len, reverse=True):
        if key in d:
            return _TREATMENTS[key]
    return ("Consult a local agricultural extension officer for an accurate "
            "diagnosis and an appropriate, locally-approved treatment plan.")


def get_disease_info(raw_label: str) -> dict:
    """Return a dict with crop, disease, status, type and recommendation."""
    crop, disease = _pretty(raw_label)
    is_healthy = disease == "Healthy"
    return {
        "raw": raw_label,
        "crop": crop,
        "disease": disease,
        "status": "Healthy" if is_healthy else "Diseased",
        "recommendation": _treatment_for(disease),
    }
