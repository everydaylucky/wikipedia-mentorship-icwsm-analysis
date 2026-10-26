"""
build_dataset.py: M7 PSM dataset and data dictionary

Module: M7 (PLAN.md M7; CODE_PLAN.md section 5, M7)
Rules: DATA_PROCESSING R7.2 (pre-question variables use only records from before the question), R7.8. Issues: I-01 to I-05, I-08, I-29, I-33, I-37

1. What it does
    Joins M5 (sample, treatment, outcomes), M3 (edits, logs, abuse filter) and M6 (text features, labels, embeddings)
    on the key mentee_uid. One row per newcomer. The covariates come in 6 blocks. All of them use only information
    from before Q, the time of the first question.
        E (edit history, 32 columns):
            n_edits                    number of edits before Q
            n_edits_7d / n_edits_1d    edits in the 168 / 24 hours before Q (I-08: the paper version used date differences, which really gives 8 days and 48 hours)
            ns0_mainspace, ns2_userpage, ns3_usertalk, ns4_wp, ns118_draft  edits in each namespace; n_unique_ns
            mainspace_ratio, draft_ratio
            avg_sizediff, std_sizediff (population standard deviation, same as np.std in the paper version), max_sizediff (largest absolute value), neg_sizediff_ratio
            hours_since_last_edit (-1 if no edits), active_span_hours, account_age_hours (Q minus registration time)
            n_reverts                  edits made before Q that were already reverted before Q (R7.8; the paper version simply counted the tag present at crawl time)
            revert_rate_pre = n_reverts / n_edits
            tag_visualeditor, tag_mobile, tag_newcomer_task, tag_editcheck_newref, tag_discussion  number of edits with each tag (counts, I-04)
            log_create                 page-creation log entries before Q (create/create; I-03: the paper version also counted account creation, newusers/create)
            log_thanks                 thanks log entries before Q (thanks/thank, sent or received by the person)
            n_abuse, n_abuse_warn      abuse filter hits before Q, and how many of them have a result containing "warn"
            q_weekday, q_hour_utc, q_is_weekend
          Two columns are dropped: n_ai_reverts (always 0 in the paper version, I-01) and tag_mw_reverted (identical to n_reverts, I-02).
        Qtext (37 columns), Qpersp (8 columns), Qtype (5 columns): from M6.
        emb20: the first 20 PCA components of the question embedding. PCA is fitted only on people who have an embedding (I-37).
        temporal: year-month dummies (the reference group is the first month, 2021-05, I-29) + q_has_article_context (the person was editing a page when asking, R4.3).
        M (mentor side, 6 columns, used only in the ablation): for this mentor (the owner of the talk page), before Q: number of questions received,
            share of them already answered before Q, mean reply delay in hours (only replies that happened before Q),
            questions and reply rate in the last 30 days, and days as a mentor.
    Missing values: in the propensity score model they are filled with the median, and a missing indicator (*_missing) is added.
    The data dictionary gives the missing rate of every column.

2. Why it is designed this way
    - Every merge is a join on the key (I-33). Nothing depends on row order.
    - Built-in guard against future-information leakage: every E-block variable uses only records with ts < Q (each function cuts at Q).
      Mentor-side variables use only questions and replies from before Q.
    - We also export an npz in the paper's layout (same key names), so M8 can reuse the whole analysis code in legacy_parity.

3. Inputs
    data/interim/m5/sample.parquet, lag_info.json; data/interim/m3/mentee_edits/, mentee_logs.parquet,
    mentee_abuse.parquet, mentee_profile.parquet; data/interim/m6/question_features.parquet, emb_q_sample.npy;
    data/interim/m2/questions.parquet; data/raw/m3/revert_times*.jsonl.

4. Outputs (data/analytic/)
    psm_dataset.parquet, data_dictionary.yaml, psm_dataset_legacy_layout.npz

8. How to check
    checks/m7_check.ipynb (variables computed by hand for each of the 11 fixed sample newcomers); tests/test_m7.py.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml
from sklearn.decomposition import PCA

import config
from pipeline.common.io import write_table
from pipeline.common.manifest import check_upstream, write_manifest
from pipeline.common.runlog import base_parser, setup_logging
from pipeline.m5_sample.build_sample import revert_lookup
from pipeline.m5_sample.outcomes import OC_KEYS
from pipeline.m6_text.text_features import QTEXT_COLS

OUT = config.ANALYTIC
E_COLS = ["n_edits", "n_edits_7d", "n_edits_1d", "ns0_mainspace", "ns2_userpage", "ns3_usertalk", "ns4_wp", "ns118_draft",
          "n_unique_ns", "mainspace_ratio", "draft_ratio", "avg_sizediff", "std_sizediff", "max_sizediff",
          "neg_sizediff_ratio", "hours_since_last_edit", "active_span_hours", "account_age_hours", "n_reverts",
          "revert_rate_pre", "tag_visualeditor", "tag_mobile", "tag_newcomer_task", "tag_editcheck_newref",
          "tag_discussion", "log_create", "log_thanks", "n_abuse", "n_abuse_warn", "q_weekday", "q_hour_utc",
          "q_is_weekend"]
QTYPE_COLS = ["q_substantive", "q_referent", "q_appraisal", "q_normative", "q_own_work"]
QPERSP_COLS = ["q_persp_toxicity", "q_persp_severe_toxicity", "q_persp_identity_attack", "q_persp_insult",
               "q_persp_profanity", "q_persp_threat", "q_persp_sexually_explicit", "q_persp_flirtation"]
M_COLS = ["m_prior_mentee_count", "m_prior_reply_rate", "m_prior_avg_response_hours", "m_recent30d_mentee_count",
          "m_recent30d_reply_rate", "m_days_active"]
TAGS = {"tag_visualeditor": "visualeditor", "tag_mobile": "mobile edit", "tag_newcomer_task": "newcomer task",
        "tag_editcheck_newref": "editcheck-newreference", "tag_discussion": "discussiontools-added-comment"}


def block_edit_history(s: pd.DataFrame, log) -> pd.DataFrame:
    """E block (see section 1). For each person, use only edits, logs and abuse filter records with ts < Q."""
    uids = list(s.mentee_uid)
    e = ds.dataset(config.INTERIM / "m3" / "mentee_edits").to_table(
        columns=["uid", "revid", "timestamp", "ns", "sizediff", "tags"], filter=ds.field("uid").isin(uids)).to_pandas()
    e["ts"] = pd.to_datetime(e.timestamp, utc=True)
    logs = pd.read_parquet(config.INTERIM / "m3" / "mentee_logs.parquet")
    logs = logs[logs.uid.isin(uids)].assign(ts=lambda d: pd.to_datetime(d.timestamp, utc=True))
    abuse = pd.read_parquet(config.INTERIM / "m3" / "mentee_abuse.parquet")
    abuse = abuse[abuse.uid.isin(uids)].assign(ts=lambda d: pd.to_datetime(d.timestamp, utc=True))
    prof = pd.read_parquet(config.INTERIM / "m3" / "mentee_profile.parquet").set_index("mentee_uid")
    rv = revert_lookup()
    eg, lg, ag = dict(tuple(e.groupby("uid"))), dict(tuple(logs.groupby("uid"))), dict(tuple(abuse.groupby("uid")))
    rows = []
    for r in s.itertuples():
        Q = pd.Timestamp(r.q_ts)
        pre = eg.get(r.mentee_uid, e.iloc[0:0])
        pre = pre[pre.ts < Q].sort_values("ts")
        n = len(pre)
        sd = pre.sizediff.fillna(0).to_numpy(dtype=float)
        hrs = (Q - pre.ts).dt.total_seconds().to_numpy() / 3600
        ns = pre.ns.to_numpy()
        tagged = pre.tags.map(lambda t: "mw-reverted" in list(t)).to_numpy()
        rts = np.array([rv.get(int(x), np.datetime64("NaT")) for x in pre.revid], dtype="datetime64[s]")
        reverted = tagged & (np.isnat(rts) | (rts < np.datetime64(Q.tz_localize(None), "s")))
        L = lg.get(r.mentee_uid, logs.iloc[0:0])
        L = L[L.ts < Q]
        A = ag.get(r.mentee_uid, abuse.iloc[0:0])
        A = A[A.ts < Q]
        reg = pd.to_datetime(prof.registration.get(r.mentee_uid), utc=True, errors="coerce")
        ta = {k: int(pre.tags.map(lambda t, v=v: v in list(t)).sum()) for k, v in TAGS.items()}
        rows.append({"mentee_uid": r.mentee_uid, "n_edits": n, "n_edits_7d": int((hrs <= 168).sum()),
                     "n_edits_1d": int((hrs <= 24).sum()), "ns0_mainspace": int((ns == 0).sum()),
                     "ns2_userpage": int((ns == 2).sum()), "ns3_usertalk": int((ns == 3).sum()),
                     "ns4_wp": int((ns == 4).sum()), "ns118_draft": int((ns == 118).sum()),
                     "n_unique_ns": int(np.unique(ns).size), "mainspace_ratio": (ns == 0).sum() / n if n else 0.0,
                     "draft_ratio": (ns == 118).sum() / n if n else 0.0, "avg_sizediff": float(sd.mean()) if n else 0.0,
                     "std_sizediff": float(sd.std()) if n > 1 else 0.0,
                     "max_sizediff": float(np.abs(sd).max()) if n else 0.0,
                     "neg_sizediff_ratio": float((sd < 0).sum() / n) if n else 0.0,
                     "hours_since_last_edit": float(hrs[-1]) if n else -1.0,
                     "active_span_hours": float((pre.ts.iloc[-1] - pre.ts.iloc[0]).total_seconds() / 3600) if n > 1 else 0.0,
                     "account_age_hours": float((Q - reg).total_seconds() / 3600) if pd.notna(reg) else -1.0,
                     "n_reverts": int(reverted.sum()), "revert_rate_pre": float(reverted.sum() / n) if n else 0.0, **ta,
                     "log_create": int(((L.type == "create") & (L.action == "create")).sum()),
                     "log_thanks": int(((L.type == "thanks") & (L.action == "thank")).sum()),
                     "n_abuse": len(A), "n_abuse_warn": int(A.result.fillna("").str.lower().str.contains("warn").sum()),
                     "q_weekday": Q.weekday(), "q_hour_utc": Q.hour, "q_is_weekend": int(Q.weekday() >= 5)})
    log.info("E block: %d rows", len(rows))
    return pd.DataFrame(rows)


def block_mentor(s: pd.DataFrame) -> pd.DataFrame:
    """M block (ablation only): questions and replies on the talk page owner's page before Q (only replies that happened before Q count)."""
    q = pd.read_parquet(config.INTERIM / "m2" / "questions.parquet",
                        columns=["page_mentor_uid", "q_ts", "has_mentor_reply", "mentor_first_reply_ts"])
    q = q.dropna(subset=["page_mentor_uid"]).assign(t=lambda d: pd.to_datetime(d.q_ts, utc=True),
                                                    rt=lambda d: pd.to_datetime(d.mentor_first_reply_ts, utc=True))
    per = pd.read_parquet(config.INTERIM / "m1" / "mentor_periods.parquet")
    first_join = pd.to_datetime(per.groupby("mentor_uid").start.min(), utc=True)
    G = dict(tuple(q.groupby("page_mentor_uid")))
    rows = []
    for r in s.itertuples():
        Q = pd.Timestamp(r.q_ts)
        g = G.get(r.page_mentor_uid)
        if g is None:
            rows.append({"mentee_uid": r.mentee_uid, **{c: 0.0 for c in M_COLS}}); continue
        prior = g[g.t < Q]
        replied = prior.rt.notna() & (prior.rt < Q)
        recent = prior[prior.t >= Q - pd.Timedelta(days=30)]
        rep30 = recent.rt.notna() & (recent.rt < Q)
        lag = ((prior.rt - prior.t).dt.total_seconds() / 3600)[replied]
        fj = first_join.get(r.page_mentor_uid)
        rows.append({"mentee_uid": r.mentee_uid, "m_prior_mentee_count": len(prior),
                     "m_prior_reply_rate": float(replied.mean()) if len(prior) else 0.0,
                     "m_prior_avg_response_hours": float(lag.mean()) if len(lag) else 0.0,
                     "m_recent30d_mentee_count": len(recent), "m_recent30d_reply_rate": float(rep30.mean()) if len(recent) else 0.0,
                     "m_days_active": float((Q - fj).total_seconds() / 86400) if fj is not None and pd.notna(fj) else 0.0})
    return pd.DataFrame(rows)


def block_embedding(s: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float]:
    """emb20: PCA is fitted only on people who have an embedding (I-37). People without an embedding get NaN."""
    qf = pd.read_parquet(config.INTERIM / "m6" / "question_features.parquet")
    E = np.load(config.INTERIM / "m6" / "emb_q_sample.npy")
    pos = {rv: i for i, rv in enumerate(qf.first_question_revid)}
    full = np.stack([E[pos[rv]] for rv in s.first_question_revid])
    ok = ~np.isnan(full).any(axis=1)
    pca = PCA(n_components=20, random_state=config.SEED).fit(full[ok])
    out = np.full((len(s), 20), np.nan)
    out[ok] = pca.transform(full[ok])
    return (pd.DataFrame(out, columns=[f"emb_pc{i + 1}" for i in range(20)]).assign(mentee_uid=s.mentee_uid.values),
            np.nan_to_num(full, nan=0.0).astype(np.float32), float(pca.explained_variance_ratio_.sum()))


def block_temporal(s: pd.DataFrame) -> pd.DataFrame:
    """Year-month dummies (reference group = the earliest month, 2021-05) + whether the person was editing a page when asking."""
    ym = pd.to_datetime(s.q_ts, utc=True).dt.strftime("%Y-%m")
    months = sorted(ym.unique())
    d = pd.DataFrame({f"ym_{m}": (ym == m).astype(float).values for m in months[1:]})
    d["q_has_article_context"] = s.editing_page.fillna(False).astype(float).values
    d["mentee_uid"] = s.mentee_uid.values
    return d


def impute(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Fill with the median + add a missing indicator (only for columns that have missing values)."""
    df = df.copy()
    ind = []
    for c in cols:
        if df[c].isna().any():
            df[f"{c}_missing"] = df[c].isna().astype(float)
            ind.append(f"{c}_missing")
            df[c] = df[c].fillna(df[c].median())
    return df, ind


def data_dictionary(df: pd.DataFrame, blocks: dict[str, list[str]], pca_var: float) -> dict:
    """For each column: block, missing rate, type (0/1, count, continuous), source module."""
    out = {}
    for b, cols in blocks.items():
        for c in cols:
            x = df[c]
            kind = "binary" if set(pd.unique(x.dropna())) <= {0, 1} else (
                "count" if pd.api.types.is_integer_dtype(x) or (x.dropna() % 1 == 0).all() else "continuous")
            out[c] = {"block": b, "missing_rate": round(float(x.isna().mean()), 4), "type": kind,
                      "source": {"E": "M3 (mentee_edits/logs/abuse) + M5", "Qtext": "M6 text_features",
                                 "Qpersp": "M6 Perspective", "Qtype": "M6 LLM", "emb20": f"M6 embeddings, PCA var {pca_var:.3f}",
                                 "temporal": "M2/M5", "M": "M2 + M1"}.get(b, b)}
    return out


def main(argv: list[str] | None = None) -> None:
    base_parser("M7: PSM dataset and data dictionary.").parse_args(argv)
    log = setup_logging("m7")
    t0 = time.time()
    check_upstream("m7", ["m5", "m6"])
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_parquet(config.INTERIM / "m5" / "sample.parquet")
    s = s[s.in_sample | s.in_sample_unrestricted].sort_values(["q_ts", "first_question_revid"]).reset_index(drop=True)
    s["page_mentor_uid"] = pd.to_numeric(s.page_mentor_uid, errors="coerce")
    E = block_edit_history(s, log)
    qf = pd.read_parquet(config.INTERIM / "m6" / "question_features.parquet")
    Q = s[["mentee_uid", "first_question_revid"]].merge(qf, on="first_question_revid", how="left")
    EM, emb_full, pca_var = block_embedding(s)
    T = block_temporal(s)
    M = block_mentor(s)
    df = s.merge(E, on="mentee_uid").merge(Q.drop(columns=["first_question_revid"]), on="mentee_uid") \
        .merge(EM, on="mentee_uid").merge(T, on="mentee_uid").merge(M, on="mentee_uid")
    assert len(df) == len(s) and df.mentee_uid.is_unique
    ym_cols = [c for c in T.columns if c.startswith("ym_")]
    blocks = {"E": E_COLS, "Qtext": QTEXT_COLS, "Qpersp": QPERSP_COLS, "Qtype": QTYPE_COLS,
              "emb20": [f"emb_pc{i + 1}" for i in range(20)], "temporal": ym_cols + ["q_has_article_context"], "M": M_COLS}
    dd = data_dictionary(df, blocks, pca_var)
    imp, ind = impute(df, [c for b in ("Qpersp", "Qtype", "emb20") for c in blocks[b]])
    for c in [c for b in blocks.values() for c in b]:
        if imp[c].nunique() <= 1:
            log.warning("constant column (kept; it has no effect on the PS model: after standardization it is all 0): %s", c)
    write_table(OUT / "psm_dataset.parquet", imp)
    (OUT / "data_dictionary.yaml").write_text(yaml.safe_dump({"blocks": {k: v for k, v in blocks.items()},
                                                              "missing_indicators": ind, "columns": dd},
                                                             allow_unicode=True, sort_keys=False))
    # npz in the paper's layout (same key names, so M8 can reuse legacy_parity). Missing indicators (one per block is
    # enough, because values in the same block are always missing together) go at the end of the temporal block,
    # and their names are added to temporal_cols too.
    seen, keep_ind = set(), []
    for c in ind:
        sig = tuple(imp[c].to_numpy())
        if sig not in seen:
            seen.add(sig)
            keep_ind.append(c)
    tcols = blocks["temporal"] + keep_ind
    info = json.loads((config.INTERIM / "m5" / "lag_info.json").read_text())
    X = lambda cols: imp[cols].to_numpy(dtype=float)     # noqa: E731
    save = {"cid_order": imp.first_question_revid.to_numpy(), "mentee_ids": imp.mentee_uid.astype(str).to_numpy(),
            "mentor_ids": imp.page_mentor_uid.astype("Int64").astype(str).to_numpy(),
            "q_year_month": pd.to_datetime(imp.q_ts, utc=True).dt.strftime("%Y-%m").to_numpy(),
            "q_has_article_context": X(["q_has_article_context"]).ravel(),
            "mentor_reply_rate": X(["m_prior_reply_rate"]).ravel(),
            "in_sample": imp.in_sample.to_numpy(bool), "in_sample_unrestricted": imp.in_sample_unrestricted.to_numpy(bool),
            "late_fetch": imp.late_fetch.fillna(False).to_numpy(bool), "q_ts": imp.q_ts.astype(str).to_numpy(),
            "y_treat": imp.treat.to_numpy(float), "y_treat_48h": imp.treat_48h.to_numpy(float),
            "y_treat_mentor": imp.treat_mentor.to_numpy(float),
            "X_E": X(E_COLS), "X_Qtext": X(QTEXT_COLS), "X_Qpersp": X(QPERSP_COLS), "X_Qtype": X(QTYPE_COLS),
            "X_emb20": X(blocks["emb20"]), "X_emb_full": emb_full, "X_temporal": X(tcols), "X_M": X(M_COLS),
            "E_cols": np.array(E_COLS), "Qtext_cols": np.array(QTEXT_COLS), "Qpersp_cols": np.array(QPERSP_COLS),
            "Qtype_cols": np.array(QTYPE_COLS), "M_cols": np.array(M_COLS), "temporal_cols": np.array(tcols),
            "pseudo_lag_info": np.array([info["median"], info["p25"], info["p75"], 0.0, info["n_multi"]])}
    for k in OC_KEYS:
        save[f"oc_{k}"] = imp[f"oc_{k}"].to_numpy(float)
        for v in ("median", "p25", "p75", "zero", "multi"):
            save[f"oc_{k}__{v}"] = imp[f"oc_{k}__{v}"].to_numpy(float)
    from pipeline.m5_sample.outcomes import ELIGIBLE_WINDOWS
    for w in ELIGIBLE_WINDOWS:
        save[f"eligible_{w}d"] = imp[f"oc_eligible_{w}d"].to_numpy(float)
    # text_complete: LLM labels, Perspective and embedding are all present (nothing was imputed).
    # in_paper: the person is also in the paper sample (matched on the revid of the first question).
    save["text_complete"] = (imp.has_llm.fillna(False) & imp.has_persp.fillna(False) & imp.has_emb_q.fillna(False)).to_numpy(bool)
    from pipeline.common.io import iter_jsonl
    s7 = {int(r["conversation_id"]): r["revid"] for r in iter_jsonl(config.EXTERNAL["paper_s7"])}
    with np.load(config.EXTERNAL["paper_npz"], allow_pickle=True) as z:
        paper_rev = {int(s7[int(c)]) for c in z["cid_order"] if s7.get(int(c))}
    save["in_paper"] = imp.first_question_revid.isin(paper_rev).to_numpy(bool)
    np.savez_compressed(OUT / "psm_dataset_legacy_layout.npz", **save)
    counts = {"rows": len(imp), "treated": int(imp.treat.sum()), "missing_indicators": ind, "pca_var": round(pca_var, 4),
              "n_covariates": sum(len(v) for k, v in blocks.items() if k != "M")}
    log.info("done: %s", counts)
    write_manifest("m7", inputs=[config.INTERIM / "m5" / "sample.parquet", config.INTERIM / "m6" / "question_features.parquet"],
                   outputs=[OUT / "psm_dataset.parquet", OUT / "psm_dataset_legacy_layout.npz"], params={}, counts=counts,
                   started=t0)


if __name__ == "__main__":
    main(sys.argv[1:])
