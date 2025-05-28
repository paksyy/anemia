import os, io, contextlib, logging, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.set_stderrthreshold("error")
except Exception:
    pass
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import tensorflow as tf
tf.get_logger().setLevel("ERROR")

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import cloudpickle
import pandas as pd

def augment_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["sex", "red", "green", "blue"]].copy()
    X["rg_ratio"] = X["red"]   / (X["green"] + 1e-6)
    X["rb_ratio"] = X["red"]   / (X["blue"]  + 1e-6)
    X["gb_ratio"] = X["green"] / (X["blue"]  + 1e-6)
    return X

if __name__ == "__main__":
    with open("anemia_pipeline_final.joblib", "rb") as f, \
         contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        pipe = cloudpickle.load(f)

    sample = pd.DataFrame({"sex":[0], "red":[.450693], "green":[.298506], "blue":[.250801]})
    X_pred = augment_features(sample)

    proba = pipe.predict_proba(X_pred)
    prob_anemia = float(proba[0]) if proba.ndim == 1 else float(proba[0][1])
    diagnosis   = "Anemia" if prob_anemia >= 0.5 else "No Anemia"

    print(f"{prob_anemia:.4f}", diagnosis)
