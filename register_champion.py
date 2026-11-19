"""Register both evaluated models and point the @champion / @challenger aliases
at whichever one evaluate.py's decision says should win."""
import json

import mlflow
from mlflow import MlflowClient

from config import OUT_DIR, REGISTERED_MODEL, TRACKING_URI


def main():
    decision = json.loads((OUT_DIR / "decision.json").read_text())
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    versions = {}
    # register the current champion first, then the challenger
    for role in ("champion", "challenger"):
        m = decision["models"][role]
        mv = mlflow.register_model(m["model_uri"], REGISTERED_MODEL)
        client.set_model_version_tag(REGISTERED_MODEL, mv.version, "hf_model_id", m["hf_id"])
        client.set_model_version_tag(REGISTERED_MODEL, mv.version, "macro_f1",
                                     f"{m['metrics']['macro_f1']:.4f}")
        client.set_model_version_tag(REGISTERED_MODEL, mv.version, "source_run_id", m["run_id"])
        versions[role] = mv.version

    winner = decision["winner"]
    loser = "challenger" if winner == "champion" else "champion"
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", versions[winner])
    client.set_registered_model_alias(REGISTERED_MODEL, "challenger", versions[loser])
    client.update_registered_model(
        REGISTERED_MODEL,
        description=f"AG News topic classifier. Current champion: {decision['models'][winner]['hf_id']}",
    )

    print(f"{REGISTERED_MODEL} v{versions[winner]} ({decision['models'][winner]['short']}) -> @champion")
    print(f"{REGISTERED_MODEL} v{versions[loser]} ({decision['models'][loser]['short']}) -> @challenger")


if __name__ == "__main__":
    main()
