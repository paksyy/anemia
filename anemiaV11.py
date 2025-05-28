import argparse, random
import multiprocessing as mp
from functools import partial
import joblib
from pathlib import Path
import time

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as K
import cloudpickle

tf.get_logger().setLevel('ERROR')

from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline

from deap import base, creator, tools, algorithms

# ------------------  SEED  ------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ------------------  HIPER-PARÁMETROS  ------------------
ACTIVATIONS = ["relu", "tanh", "selu", "elu", "gelu", "leaky_relu"]
OPTIMIZERS  = ["adam", "rmsprop", "nadam"]

# ------------------  CARGA Y LIMPIEZA  ------------------
def load_dataset(csv_path: Path):
    print(f"\n📂 Cargando dataset de {csv_path}...")
    df = pd.read_csv(csv_path)
    print("✅ Primeras filas:")
    print(df.head(3))
    
    print("\n🧹 Limpiando datos y creando ratios…")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "%Red Pixel":   "red",
        "%Green pixel": "green",
        "%Blue pixel":  "blue",
        "Hb":           "hb",
        "Sex":          "sex",
        "Anaemic":      "target",
    })
    df["sex"]    = LabelEncoder().fit_transform(df["sex"])
    df["target"] = df["target"].map({"Yes": 1, "No": 0}).astype(int)
    df["rg_ratio"] = df["red"]   / (df["green"] + 1e-6)
    df["rb_ratio"] = df["red"]   / (df["blue"]  + 1e-6)
    df["gb_ratio"] = df["green"] / (df["blue"]  + 1e-6)
    
    print("✅ Columnas finales:", df.columns.tolist())
    X = df[["sex","red","green","blue","rg_ratio","rb_ratio","gb_ratio"]]
    y = df["target"].values
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)

# ------------------  MODEL KERAS  ------------------
def build_model(n_hidden, n_neurons, lr, act, dropout, opt, input_dim):
    print(f"\n🧠 Construyendo modelo con: {n_hidden} capas ocultas, {n_neurons} neuronas, lr={lr:.2e}, activación={act}, dropout={dropout}, optimizador={opt}")
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(
            n_neurons,
            activation=act,
            kernel_regularizer=regularizers.l2(1e-4)
        ))
        model.add(BatchNormalization())
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    optimizer_instance = keras.optimizers.get(opt)
    optimizer_instance.learning_rate = lr
    model.compile(optimizer=optimizer_instance,
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
    print("✅ Modelo construido")
    return model

# ------------------  GA SETUP  ------------------
LOW = [1, 4, -5, 0, 0, 0, 8, 10] # n_hidden, n_neurons, log_lr, act_idx, opt_idx, dropout_x10, batch_size, epochs
UP  = [4, 128, -2, len(ACTIVATIONS)-1, len(OPTIMIZERS)-1, 6, 128, 100]

def setup_toolbox():
    if hasattr(creator, "FitnessMax"): del creator.FitnessMax
    if hasattr(creator, "Individual"): del creator.Individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    tb = base.Toolbox()
    tb.register("n_hidden",  random.randint, LOW[0], UP[0])
    tb.register("n_neurons", random.randint, LOW[1], UP[1])
    tb.register("log_lr",    random.uniform, LOW[2], UP[2])
    tb.register("act",       random.randint, LOW[3], UP[3])
    tb.register("opt",       random.randint, LOW[4], UP[4])
    tb.register("drop",      random.randint, LOW[5], UP[5])
    tb.register("batch",     random.randint, LOW[6], UP[6])
    tb.register("epoch",     random.randint, LOW[7], UP[7])

    tb.register("individual", tools.initCycle, creator.Individual,
                (tb.n_hidden, tb.n_neurons, tb.log_lr, tb.act,
                 tb.opt, tb.drop, tb.batch, tb.epoch), n=1)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxUniform, indpb=0.5)
    tb.register("mutate", tools.mutUniformInt, low=LOW, up=UP, indpb=0.3)
    tb.register("select", tools.selTournament, tournsize=3)
    return tb

# ------------------  EVALUATOR  ------------------
def evaluate_individual(ind, X, y, preprocess):
    print(f"\n🔍 Evaluando individuo: {ind}")
    nh, nn, log_lr, act_idx, opt_idx, dr, bs, ep = ind
    lr = 10**log_lr
    dropout = dr/10.0
    dim = X.shape[1]
    print(f"📊 Parámetros modelo:\n - Capas ocultas: {nh}\n - Neuronas/capa: {nn}\n - LR: {lr:.2e}\n - Activación: {ACTIVATIONS[act_idx]}\n - Optimizador: {OPTIMIZERS[opt_idx]}\n - Dropout: {dropout}\n - Batch: {bs}\n - Épocas: {ep}")
    model = build_model(int(nh), int(nn), lr,
                        ACTIVATIONS[int(act_idx)], dropout,
                        OPTIMIZERS[int(opt_idx)], dim)
    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=5,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_auc", mode="max",
                          factor=0.5, patience=3, verbose=0)
    ]
    pipe = Pipeline([
        ("prep", preprocess),
        ("smote", BorderlineSMOTE(random_state=SEED)),
        ("net", KerasClassifier(
            model=model,
            batch_size=int(bs),
            epochs=int(ep),
            class_weight="balanced",
            callbacks=callbacks,
            loss="binary_crossentropy",
            verbose=0))
    ])
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scores = []
    for i, (tr, val) in enumerate(rkf.split(X, y)):
        pipe_clone = Pipeline(steps=pipe.steps)
        pipe_clone.fit(X.iloc[tr], y[tr])
        y_prob = pipe_clone.predict_proba(X.iloc[val])[:,1]
        score = roc_auc_score(y[val], y_prob)
        scores.append(score)
        print(f"   Fold {i+1}: AUC = {score:.4f}")
    mean_score = float(np.mean(scores))
    print(f"🎯 AUC media CV: {mean_score:.4f}")
    K.clear_session()
    return (mean_score,)

# ------------------  EVOLUCIÓN  ------------------
def evolve(csv: Path, pop_size=40, n_gen=25, cxpb=0.5, mutpb=0.3):
    print("\n🚀 Iniciando proceso evolutivo")
    print(f"⚙️ Población: {pop_size}, Generaciones: {n_gen}, cxpb: {cxpb}, mutpb: {mutpb}")
    X_train, X_test, y_train, y_test = load_dataset(csv)
    preprocess = ColumnTransformer([
        ("scale", StandardScaler(), ["red","green","blue","rg_ratio","rb_ratio","gb_ratio"])
    ], remainder="passthrough")

    toolbox = setup_toolbox()
    toolbox.register("evaluate", partial(evaluate_individual,
                                         X=X_train, y=y_train,
                                         preprocess=preprocess))
    n_cpu = min(mp.cpu_count(), 4)
    with mp.Pool(n_cpu) as pool:
        toolbox.register("map", pool.map)
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        logbook = tools.Logbook()
        logbook.header = ["gen","nevals"] + stats.fields

        # generación 0
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses): ind.fitness.values = fit
        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(invalid), **record)
        print(logbook.stream)

        best_prev = record['max']; no_imp=0
        # evoluciona
        for gen in range(1, n_gen+1):
            print(f"\n🌱 Generación {gen}")
            offspring = tools.selTournament(pop, len(pop), tournsize=3)
            offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses): ind.fitness.values = fit
            hof.update(offspring)
            pop[:] = offspring
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid), **record)
            print(logbook.stream)
            print(f"📝 Mejor gen {gen}: AUC={round(record['max'],4)}")
            if record['max'] <= best_prev:
                no_imp +=1
            else:
                best_prev = record['max']; no_imp=0
            if no_imp>=5:
                print(f"🔻 Sin mejora en {no_imp} gens, deteniendo.")
                break

    # resultado final
    best = hof[0]
    print("\n🏆 Mejor individuo encontrado:")
    print(f" - Capas ocultas: {int(best[0])}")
    print(f" - Neuronas/capa: {int(best[1])}")
    print(f" - Tasa aprendizaje: {10**best[2]:.2e}")
    print(f" - Activación: {ACTIVATIONS[int(best[3])]}")
    print(f" - Optimizador: {OPTIMIZERS[int(best[4])]}")
    print(f" - Dropout: {best[5]/10.0}")
    print(f" - Batch size: {int(best[6])}")
    print(f" - Épocas: {int(best[7])}")
    print(f"📈 AUC CV: {round(best.fitness.values[0],4)}")

    # evaluación test y reporte
    K.clear_session()
    final_model = build_model(int(best[0]), int(best[1]), 10**best[2],
                              ACTIVATIONS[int(best[3])], best[5]/10.0,
                              OPTIMIZERS[int(best[4])], X_train.shape[1])
    
    final_callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=5,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_auc", mode="max",
                          factor=0.5, patience=3, verbose=0)
    ]

    final_pipe = Pipeline([
        ("prep", preprocess),
        ("smote", BorderlineSMOTE(random_state=SEED)),
        ("net", KerasClassifier(
            model=final_model,
            batch_size=int(best[6]),
            epochs=int(best[7]),
            class_weight="balanced",
            loss="binary_crossentropy",
            callbacks=final_callbacks,
            verbose=0))
    ])

    final_pipe.fit(X_train, y_train)

    print("\n💾 Serializando pipeline completo con cloudpickle…")
    with open("anemia_pipeline_final.joblib", "wb") as f:
        cloudpickle.dump(final_pipe, f)
    print("✅ Pipeline guardado correctamente.")

    y_prob = final_pipe.predict_proba(X_test)[:,1]

    auc_test = round(roc_auc_score(y_test, y_prob),4)
    y_pred = (y_prob>=0.5).astype(int)
    print(f"\n📊 Rendimiento en test: AUC={auc_test}")
    print("\n📝 Classification report:\n", classification_report(y_test,y_pred, target_names=["No Anemia","Anemia"]))
    print("\n📊 Confusion matrix:\n", confusion_matrix(y_test,y_pred))

# ------------------  CLI  ------------------
def main():
    try:
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ap = argparse.ArgumentParser(description="GA-tuned NN for anemia")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gen", type=int, default=25)
    ap.add_argument("--cxpb", type=float, default=0.5)
    ap.add_argument("--mutpb", type=float, default=0.3)
    args = ap.parse_args()
    evolve(args.csv, pop_size=args.pop, n_gen=args.gen, cxpb=args.cxpb, mutpb=args.mutpb)

if __name__ == "__main__":
    main()
