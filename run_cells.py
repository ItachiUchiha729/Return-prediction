"""
Run the minimum cell chain needed to:
  1. Fix xgb.joblib (0 bytes → properly saved)
  2. Generate figures/monthly_feature_distribution_all_features.png
  3. Generate figures/top_feature_bin_plots_train_4bins.png
"""

import matplotlib
matplotlib.use("Agg")

import sys, warnings, time, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, norm as sp_norm
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")


def main():
    # ── Paths ──────────────────────────────────────────────────────────────
    _project_root = Path(__file__).resolve().parent
    _submit_root  = _project_root / "submission_files"
    for p in [str(_project_root), str(_submit_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from src.config import FEATURE_DATA_DIR

    # ══════════════════════════════════════════════════════════════════════
    # CELL 1 — Fast reload
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("CELL 1 — Loading normalised train / test data")
    print("=" * 70)
    df_train_norm = pd.read_parquet(FEATURE_DATA_DIR / "feature_together" / "features_norm_train.parquet")
    df_test_norm  = pd.read_parquet(FEATURE_DATA_DIR / "feature_together" / "features_norm_test.parquet")

    target_col   = "target_variable"
    feature_cols = [c for c in df_train_norm.columns
                    if c not in {"Id", target_col}
                    and df_train_norm[c].dtype != object]

    print(f"✓ train: {df_train_norm.shape}  test: {df_test_norm.shape}  ({len(feature_cols)} features)")

    # ══════════════════════════════════════════════════════════════════════
    # CELL 27 — ResidualMLP + train_resmlp + DEVICE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CELL 27 — ResidualMLP / train_resmlp / DEVICE")
    print("=" * 70)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"Device: {DEVICE}")

    class ResBlock(nn.Module):
        def __init__(self, d, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, d), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d, d),
            )
            self.act = nn.ReLU()
        def forward(self, x):
            return self.act(x + self.net(x))

    class ResidualMLP(nn.Module):
        def __init__(self, in_dim, hidden, n_blocks, dropout):
            super().__init__()
            self.proj   = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
            self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
            self.head   = nn.Linear(hidden, 1)
        def forward(self, x):
            return self.head(self.blocks(self.proj(x))).squeeze(-1)

    def train_resmlp(X_tr, y_tr, X_va, y_va, hidden, n_blocks, dropout,
                     lr=1e-3, max_epochs=60, patience=8, batch_size=4096,
                     stop_on="ic", device="cpu", sample_weight=None):
        model = ResidualMLP(X_tr.shape[1], hidden, n_blocks, dropout).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        Xt = torch.tensor(X_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
        Xv = torch.tensor(X_va, dtype=torch.float32, device=device)
        yv = torch.tensor(y_va, dtype=torch.float32, device=device)
        n  = len(Xt)
        if sample_weight is not None:
            wt = torch.tensor(sample_weight, dtype=torch.float32, device=device)
            wt = wt / wt.mean()
        else:
            wt = None
        val_loss_fn = nn.MSELoss()
        best_metric = float("inf") if stop_on == "loss" else float("-inf")
        best_state, wait = None, 0
        for _ in range(max_epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx  = perm[start: start + batch_size]
                pred = model(Xt[idx])
                if wt is not None:
                    loss = (wt[idx] * F.mse_loss(pred, yt[idx], reduction="none")).mean()
                else:
                    loss = F.mse_loss(pred, yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pv_t     = model(Xv)
                val_loss = float(val_loss_fn(pv_t, yv).item())
                pv       = pv_t.cpu().numpy()
            val_ic = float(spearmanr(y_va, pv).correlation)
            metric   = val_loss if stop_on == "loss" else val_ic
            improved = (metric < best_metric) if stop_on == "loss" else (metric > best_metric)
            if improved:
                best_metric = metric
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_state)
        return model, val_ic

    # ══════════════════════════════════════════════════════════════════════
    # CELL 37 — Load 2015 OOS features
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CELL 37 — Load 2015 OOS features")
    print("=" * 70)

    CACHE_NORM_2015 = FEATURE_DATA_DIR / "feature_together" / "features_norm_oos2015.parquet"

    if CACHE_NORM_2015.exists():
        print("✓ Cached normalized 2015 features found — loading directly.")
        df_2015_norm = pd.read_parquet(CACHE_NORM_2015)
    else:
        print("Building normalized 2015 features from raw cache...")
        CACHE_RAW_2015 = FEATURE_DATA_DIR / "feature_together" / "features_raw_oos2015.parquet"
        from feature_src.nan_handler import clean_feature_panel

        df_raw_2015 = pd.read_parquet(CACHE_RAW_2015)
        train_cutoff = df_train_norm.index.min()
        df_2015_clean = clean_feature_panel(
            df_raw_2015, drop_target_nan=True, warmup_cutoff=train_cutoff, verbose=False
        )
        clip_params = joblib.load(FEATURE_DATA_DIR / "clip_params.joblib")
        df_2015_clipped = df_2015_clean.copy()
        for col, p in clip_params["features"].items():
            if col in df_2015_clipped.columns:
                df_2015_clipped[col] = df_2015_clipped[col].clip(lower=p["clip_lo"], upper=p["clip_hi"])
        tp = clip_params["target"]
        df_2015_clipped["target_variable"] = df_2015_clipped["target_variable"].clip(
            lower=tp["clip_lo"], upper=tp["clip_hi"]
        )
        norm_params_saved = joblib.load(FEATURE_DATA_DIR / "norm_params.joblib")
        stage1 = norm_params_saved["stage1_params"]
        df_2015_s1 = df_2015_clipped.copy()
        for col in feature_cols:
            mu, std = stage1[col]["mean"], stage1[col]["std"]
            df_2015_s1[col] = (df_2015_s1[col] - mu) / std if std > 1e-12 else 0.0

        def _cross_sectional_rank_norm(df, cols):
            df_out = df.copy()
            for col in cols:
                ranked = df_out.groupby(df_out.index)[col].rank(method="average", na_option="keep")
                counts = df_out.groupby(df_out.index)[col].transform("count")
                df_out[col] = sp_norm.ppf((ranked - 0.5) / counts)
            return df_out

        df_2015_norm = _cross_sectional_rank_norm(df_2015_s1, feature_cols)
        df_2015_norm.to_parquet(CACHE_NORM_2015)
        print(f"✓ Cached normalized 2015 features → {CACHE_NORM_2015.name}")

    print(f"✓ 2015 data ready: {df_2015_norm.shape}  "
          f"({df_2015_norm.index.min().date()} → {df_2015_norm.index.max().date()})")
    X_2015 = df_2015_norm[feature_cols].values
    y_2015 = df_2015_norm[target_col].values

    # ══════════════════════════════════════════════════════════════════════
    # CELL 38 — MDV63 sample weights
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CELL 38 — MDV63 sample weights")
    print("=" * 70)
    from feature_src.daily_data_loader import load_daily_data

    _daily_train_dir = _project_root / "raw_data"  / "data_daily"
    _daily_oos_dir   = _project_root / "oos_data"  / "daily_data"

    print("Loading daily data for MDV63 weights...")
    t0 = time.time()
    _daily_train = load_daily_data(daily_dir=_daily_train_dir)
    _daily_oos   = load_daily_data(daily_dir=_daily_oos_dir)
    _daily_all   = pd.concat([_daily_train, _daily_oos], ignore_index=True)
    _daily_all["Date"] = pd.to_datetime(_daily_all["Date"])

    _daily_mdv = (
        _daily_all[["Date", "Id", "MDV_63"]]
        .drop_duplicates(subset=["Date", "Id"], keep="last")
        .copy()
    )
    print(f"  Loaded {len(_daily_mdv):,} unique (Date, Id) rows in {time.time()-t0:.1f}s")

    def _make_weights(norm_df):
        df_r = norm_df.reset_index()[["Date", "Id"]].copy()
        df_r["Date"] = pd.to_datetime(df_r["Date"])
        merged = df_r.merge(_daily_mdv, on=["Date", "Id"], how="left")
        assert len(merged) == len(df_r), \
            f"Merge expanded rows: {len(merged)} vs {len(df_r)}"
        mdv = merged["MDV_63"].values.astype(float)
        med = np.nanmedian(mdv[mdv > 0])
        mdv = np.where(np.isnan(mdv) | (mdv <= 0), med, mdv)
        return np.sqrt(mdv)

    w_train_full = _make_weights(pd.concat([df_train_norm, df_test_norm]).sort_index())
    w_2015       = _make_weights(df_2015_norm)
    print(f"✓ w_train_full {w_train_full.shape}  w_2015 {w_2015.shape}")

    # ── Hardcoded best params from HP search ──────────────────────────────
    best_rf_params     = {"n_estimators": 200, "max_depth": 5,
                          "min_samples_leaf": 50, "max_features": "sqrt"}
    best_xgb_params    = {"n_estimators": 500, "max_depth": 3,
                          "learning_rate": 0.01, "reg_lambda": 1.0}
    best_mlp_params    = {"hidden_layer_sizes": (256, 128, 64),
                          "alpha": 1e-4, "learning_rate_init": 1e-3}
    best_resmlp_params = {"hidden": 128, "n_blocks": 2, "dropout": 0.2}

    # ══════════════════════════════════════════════════════════════════════
    # CELL 39 — Refit all 5 models + save
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CELL 39 — Refit all 5 models on 2010–2014 + save")
    print("=" * 70)
    import shutil
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    import xgboost as xgb_lib

    df_train_full = pd.concat([df_train_norm, df_test_norm]).sort_index()
    X_tr_full = df_train_full[feature_cols].values
    y_tr_full = df_train_full[target_col].values
    print(f"2010–2014 set: {df_train_full.shape}")

    print("\nFitting Ridge (weighted)...", flush=True); t0 = time.time()
    ridge_2015 = Ridge(alpha=1000)
    ridge_2015.fit(X_tr_full, y_tr_full, sample_weight=w_train_full)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("Fitting RF (weighted)...", flush=True); t0 = time.time()
    rf_2015 = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    rf_2015.fit(X_tr_full, y_tr_full, sample_weight=w_train_full)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("Fitting XGBoost (weighted)...", flush=True); t0 = time.time()
    xgb_2015 = xgb_lib.XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1,
                                      verbosity=0, eval_metric="rmse")
    xgb_2015.fit(X_tr_full, y_tr_full, sample_weight=w_train_full)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("Fitting MLP (unweighted)...", flush=True); t0 = time.time()
    mlp_2015 = MLPRegressor(**best_mlp_params, max_iter=300, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=15,
                             solver="adam", random_state=42)
    mlp_2015.fit(X_tr_full, y_tr_full)
    print(f"  Done in {time.time()-t0:.1f}s  ({mlp_2015.n_iter_} iters)")

    print("Fitting ResidualMLP (weighted)...", flush=True); t0 = time.time()
    _tr_mask = df_train_full.index.year < 2014
    _va_mask = df_train_full.index.year == 2014
    X_tr_nn  = df_train_full.loc[_tr_mask, feature_cols].values.astype(np.float32)
    y_tr_nn  = df_train_full.loc[_tr_mask, target_col].values.astype(np.float32)
    X_va_nn  = df_train_full.loc[_va_mask, feature_cols].values.astype(np.float32)
    y_va_nn  = df_train_full.loc[_va_mask, target_col].values.astype(np.float32)
    w_tr_nn  = w_train_full[np.asarray(_tr_mask)]

    resmlp_2015, _val_ic = train_resmlp(
        X_tr_nn, y_tr_nn, X_va_nn, y_va_nn,
        **best_resmlp_params,
        max_epochs=100, patience=12, batch_size=4096,
        stop_on="loss", device=DEVICE,
        sample_weight=w_tr_nn,
    )
    print(f"  Done in {time.time()-t0:.1f}s  (val IC={_val_ic:+.4f})")

    MODEL_SAVE_DIR = _submit_root / "model_src" / "saved_models"
    MODEL_SAVE_DIR.mkdir(exist_ok=True)

    joblib.dump(ridge_2015,              MODEL_SAVE_DIR / "ridge.joblib")
    joblib.dump(rf_2015,                 MODEL_SAVE_DIR / "rf.joblib")
    joblib.dump(xgb_2015,                MODEL_SAVE_DIR / "xgb.joblib")
    joblib.dump(mlp_2015,                MODEL_SAVE_DIR / "mlp.joblib")
    torch.save(resmlp_2015.state_dict(), MODEL_SAVE_DIR / "resmlp_state.pt")
    joblib.dump(best_resmlp_params,      MODEL_SAVE_DIR / "resmlp_params.joblib")
    joblib.dump({"feature_cols": feature_cols, "target_col": target_col},
                MODEL_SAVE_DIR / "feature_info.joblib")
    shutil.copy(FEATURE_DATA_DIR / "clip_params.joblib",  MODEL_SAVE_DIR / "clip_params.joblib")
    shutil.copy(FEATURE_DATA_DIR / "norm_params.joblib",  MODEL_SAVE_DIR / "norm_params.joblib")
    print(f"\n✓ All models saved → {MODEL_SAVE_DIR}")

    # ── diversity_weights helper ───────────────────────────────────────────
    def diversity_weights(pred_matrix, names):
        n_m      = pred_matrix.shape[1]
        corr_mat = np.eye(n_m)
        for i in range(n_m):
            for j in range(i + 1, n_m):
                c = spearmanr(pred_matrix[:, i], pred_matrix[:, j]).correlation
                corr_mat[i, j] = corr_mat[j, i] = abs(c)
        avg_corr = np.array([(corr_mat[i].sum() - 1) / (n_m - 1) for i in range(n_m)])
        raw_w    = 1.0 / (1.0 + avg_corr)
        weights  = raw_w / raw_w.sum()
        return weights, avg_corr, pd.DataFrame(corr_mat, index=names, columns=names)

    # ══════════════════════════════════════════════════════════════════════
    # CELL 40 — OOS 2015 predictions + ensemble weights saved
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CELL 40 — OOS 2015 predictions + ensemble weights")
    print("=" * 70)

    def _daily_ic(y_true, y_pred, dates):
        recs = []
        for d in sorted(dates.unique()):
            m = np.asarray(dates == d)
            yd, pd_ = y_true[m], y_pred[m]
            if len(yd) > 10:
                recs.append({"date": d, "SpearmanIC": spearmanr(yd, pd_).correlation})
        return pd.DataFrame(recs)

    resmlp_2015.eval()
    with torch.no_grad():
        _resmlp_pred = resmlp_2015(
            torch.tensor(X_2015.astype(np.float32), device=DEVICE)
        ).cpu().numpy()

    preds_2015 = {
        "Ridge":    ridge_2015.predict(X_2015),
        "RF":       rf_2015.predict(X_2015),
        "XGB":      xgb_2015.predict(X_2015),
        "MLP":      mlp_2015.predict(X_2015),
        "ResidMLP": _resmlp_pred,
    }

    mat_2015_5 = np.column_stack(list(preds_2015.values()))
    w2015_5, _, _ = diversity_weights(mat_2015_5, list(preds_2015.keys()))
    pred_ens5_2015 = mat_2015_5 @ w2015_5

    mat_2015_3 = np.column_stack([preds_2015[k] for k in ["Ridge", "RF", "XGB"]])
    w2015_3, _, _ = diversity_weights(mat_2015_3, ["Ridge", "RF", "XGB"])
    pred_ens3_2015 = mat_2015_3 @ w2015_3

    joblib.dump(
        {"w5": w2015_5, "names_5": list(preds_2015.keys()),
         "w3": w2015_3, "names_3": ["Ridge", "RF", "XGB"]},
        MODEL_SAVE_DIR / "ensemble_weights.joblib",
    )
    print(f"✓ Ensemble weights saved")

    all_preds_2015 = {**preds_2015,
                      "Ens-5 (Div)":               pred_ens5_2015,
                      "Ens-3 (Div, Ridge+RF+XGB)":  pred_ens3_2015}

    print(f"\n{'Model':38s}  {'R²':>8s}  {'IC':>8s}  {'Hit%':>6s}")
    print("-" * 68)
    for name, pred in all_preds_2015.items():
        r2   = r2_score(y_2015, pred)
        ic   = spearmanr(y_2015, pred).correlation
        hitp = (_daily_ic(y_2015, pred, df_2015_norm.index)["SpearmanIC"] > 0).mean() * 100
        print(f"  {name:36s}  {r2:+.6f}  {ic:+.4f}  {hitp:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # CELL 45 — Feature importance: MDI
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CELL 45 — Feature importance (MDI)")
    print("=" * 70)
    mdi = pd.Series(rf_2015.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 10 features by MDI:")
    print(mdi.head(10).to_string())

    # ══════════════════════════════════════════════════════════════════════
    # CELL 51 — Feature time series (saves PNG)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CELL 51 — Feature time series (monthly aggregates, saves PNG)")
    print("=" * 70)

    feat_dir    = FEATURE_DATA_DIR / "feature_together"
    FIGURES_DIR = _submit_root / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw feature data for all 5 years...")
    df_raw_all = pd.concat([
        pd.read_parquet(feat_dir / "features_raw_train.parquet"),
        pd.read_parquet(feat_dir / "features_raw_test.parquet"),
        pd.read_parquet(feat_dir / "features_raw_oos2015.parquet"),
    ]).sort_index()

    feat_ts_cols = [c for c in df_raw_all.columns if c not in {"Id", "target_variable"}]
    print(f"  {len(feat_ts_cols)} features")

    df_raw_all.index = pd.DatetimeIndex(df_raw_all.index)
    monthly = df_raw_all[feat_ts_cols].resample("ME").agg(
        ["mean",
         lambda x: x.quantile(0.01),
         lambda x: x.quantile(0.99),
         "min", "max"]
    )
    monthly.columns = ["_".join(c) if isinstance(c, tuple) else c for c in monthly.columns]
    monthly.columns = [
        c.replace("<lambda_0>", "p01").replace("<lambda_1>", "p99")
        for c in monthly.columns
    ]

    n_feats = len(feat_ts_cols)
    n_cols  = 3
    n_rows  = (n_feats + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = axes.flatten()

    for ax_i, feat in enumerate(feat_ts_cols):
        ax   = axes[ax_i]
        mn   = monthly[f"{feat}_mean"]
        p01  = monthly[f"{feat}_p01"]
        p99  = monthly[f"{feat}_p99"]
        fmin = monthly[f"{feat}_min"]
        fmax = monthly[f"{feat}_max"]
        ax.plot(mn.index,  mn.values,  color="#2c3e50", linewidth=2.0, label="Mean")
        ax.plot(p01.index, p01.values, color="#3498db", linewidth=1.0, linestyle="--", label="1st pct")
        ax.plot(p99.index, p99.values, color="#e74c3c", linewidth=1.0, linestyle="--", label="99th pct")
        ax.fill_between(fmin.index, fmin.values, fmax.values, color="#bdc3c7", alpha=0.3, label="Min–Max")
        ax.axvspan(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31"),
                   alpha=0.08, color="#f39c12", label="2014 OOS")
        ax.axvspan(pd.Timestamp("2015-01-01"), pd.Timestamp("2015-12-31"),
                   alpha=0.08, color="#e74c3c", label="2015 OOS")
        ax.set_title(feat, fontsize=10, fontweight="bold")
        ax.set_ylabel("Feature value", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        if ax_i == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    for ax_i in range(n_feats, len(axes)):
        axes[ax_i].set_visible(False)

    plt.suptitle("Feature Time Series — Monthly Aggregates (2010–2015)\n"
                 "Mean (dark), 1st/99th pct (dashed), Min/Max shaded; orange=2014 OOS, red=2015 OOS",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _fig_path = FIGURES_DIR / "monthly_feature_distribution_all_features.png"
    fig.savefig(_fig_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved → {_fig_path}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # CELL 52 — Feature bin plots (saves PNG)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CELL 52 — Feature bin plots (top 8 MDI features vs target)")
    print("=" * 70)

    df_raw_train   = pd.read_parquet(feat_dir / "features_raw_train.parquet")
    target_col_raw = "target_variable"
    top8_features  = list(mdi.head(8).index)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax_i, feat in enumerate(top8_features):
        ax  = axes[ax_i]
        sub = df_raw_train[[feat, target_col_raw]].dropna()
        try:
            sub = sub.copy()
            sub["bin"] = pd.qcut(sub[feat], 4, labels=["Q1", "Q2", "Q3", "Q4"],
                                 duplicates="drop")
        except Exception:
            ax.set_visible(False)
            continue
        grouped = sub.groupby("bin", observed=True)[target_col_raw].agg(["mean", "sem"])
        colors  = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        ax.bar(grouped.index, grouped["mean"] * 1e4,
               yerr=grouped["sem"] * 1e4 * 1.96,
               color=colors, alpha=0.8, edgecolor="white",
               capsize=4, error_kw={"linewidth": 1.2})
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{feat}\n(MDI rank #{list(mdi.index).index(feat)+1})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Feature Quartile", fontsize=9)
        ax.set_ylabel("Mean Target (bps)" if ax_i % 4 == 0 else "", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ic = spearmanr(sub[feat], sub[target_col_raw]).correlation
        ax.text(0.05, 0.95, f"IC={ic:+.4f}", transform=ax.transAxes,
                fontsize=9, va="top", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    plt.suptitle("Feature Bin Plots — Top 8 Features (by MDI) vs Target Variable\n"
                 "Training data 2010–2013; Q4−Q1 spread shows feature signal; ±1.96 SE",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _fig_path2 = FIGURES_DIR / "top_feature_bin_plots_train_4bins.png"
    fig.savefig(_fig_path2, dpi=150, bbox_inches="tight")
    print(f"✓ Saved → {_fig_path2}")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL CHECK
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL CHECK")
    print("=" * 70)
    for fpath in sorted(MODEL_SAVE_DIR.iterdir()):
        size   = fpath.stat().st_size
        status = "OK" if size > 0 else "*** 0 BYTES ***"
        print(f"  [{status}] {fpath.name} ({size:,} bytes)")
    print()
    for fpath in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  [OK] {fpath.name} ({fpath.stat().st_size:,} bytes)")
    print("\n✓ All done.")


if __name__ == "__main__":
    main()
