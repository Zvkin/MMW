from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from scipy.stats import chi2_contingency
from kmodes.kmodes import KModes
from sklearn.decomposition import PCA
import json
import os
import traceback

app = Flask(__name__, static_folder='static')
CORS(app)

# ── serve the frontend ──────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ── main analysis endpoint ───────────────────────────────────────────────────

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # ── 1. parse incoming form data ──────────────────────────────────────
        csv_file   = request.files.get('csv')
        config_raw = request.form.get('config')

        if not csv_file or not config_raw:
            return jsonify({'error': 'Missing CSV file or configuration.'}), 400

        config = json.loads(config_raw)

        K             = int(config.get('k', 3))
        domains       = config.get('domains', [])
        answer_key    = config.get('answer_key', {})
        dist_cats     = config.get('distractor_categories', [])

        # ── 2. load CSV ──────────────────────────────────────────────────────
        df = pd.read_csv(csv_file)

        # tolerate various student_id column names
        id_col = None
        for c in df.columns:
            if c.strip().lower() in ('student_id', 'id', 'studentid', 'student id'):
                id_col = c
                break
        if id_col:
            df = df.set_index(id_col)

        # normalise column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        # ── 3. map responses to error types ─────────────────────────────────
        error_df = pd.DataFrame(index=df.index)
        for qcol, mapping in answer_key.items():
            qcol_norm = qcol.strip().lower()
            if qcol_norm in df.columns:
                error_df[qcol_norm] = df[qcol_norm].astype(str).str.strip().str.upper().map(
                    {k.upper(): v for k, v in mapping.items()}
                ).fillna('blank')
            else:
                error_df[qcol_norm] = 'blank'

        cat_matrix = error_df.replace({'correct': 'none', 'blank': 'none'})

        # ── Multi-label categorical encoding for K-Modes ─────────────────────
        # Instead of collapsing each domain to one dominant error label, we
        # encode every distractor category per domain as a binary presence
        # indicator AND a frequency-band label, preserving overlapping error
        # behaviours within each student's response pattern.
        from collections import Counter as _Counter

        error_profile_rows = {}
        for _dom in domains:
            _dname = _dom['name']
            _cols  = [f"q{i}" for i in range(int(_dom['start']), int(_dom['end']) + 1)]
            _cols  = [c for c in _cols if c in error_df.columns]
            _safe  = _dname.replace(' ', '_').lower()
            n_items = max(len(_cols), 1)

            for idx in error_df.index:
                if idx not in error_profile_rows:
                    error_profile_rows[idx] = {}

                _errs = [v for v in error_df.loc[idx, _cols]
                         if v not in ('correct', 'blank')]
                _counts = _Counter(_errs)

                for _cat in dist_cats:
                    _freq  = _counts.get(_cat, 0)
                    _prop  = _freq / n_items

                    # Binary presence indicator: 'yes' / 'no'
                    error_profile_rows[idx][f"{_safe}__{_cat}__present"] = (
                        'yes' if _freq > 0 else 'no'
                    )

                    # Frequency-band label: none / low / moderate / high
                    if _prop == 0:
                        _band = 'none'
                    elif _prop <= 0.2:
                        _band = 'low'
                    elif _prop <= 0.5:
                        _band = 'moderate'
                    else:
                        _band = 'high'
                    error_profile_rows[idx][f"{_safe}__{_cat}__freq_band"] = _band

        _ep_df = pd.DataFrame.from_dict(error_profile_rows, orient='index')

        # Keep only columns with variance (single-value columns add no signal)
        _valid_cols = [c for c in _ep_df.columns if _ep_df[c].nunique() > 1]
        cat_matrix_kmodes = _ep_df[_valid_cols] if _valid_cols else _ep_df

        n_students = len(cat_matrix)

        # ── 4. build numerical feature matrix ───────────────────────────────
        num_features = {}
        domain_col_map = {}

        for domain in domains:
            dname = domain['name']
            cols  = [f"q{i}" for i in range(int(domain['start']), int(domain['end']) + 1)]
            cols  = [c for c in cols if c in error_df.columns]
            domain_col_map[dname] = cols
            safe  = dname.replace(' ', '_').lower()

            for etype in dist_cats:
                num_features[f"{safe}__{etype}_rate"] = (
                    error_df[cols] == etype
                ).sum(axis=1) / max(len(cols), 1)

            num_features[f"{safe}__accuracy"] = (
                error_df[cols] == 'correct'
            ).sum(axis=1) / max(len(cols), 1)

        num_matrix = pd.DataFrame(num_features, index=error_df.index)

        # 🔥 IMPROVED K-MEANS: Merge behavioral features alongside domain features
        behavioral_rows = []
        for idx in error_df.index:
            row = error_df.loc[idx]
            vals = list(row)
            errors = [v for v in vals if v not in ('correct', 'blank')]
            total = len(vals)
            total_err = len(errors)

            consistency = (max([errors.count(x) for x in set(errors)]) / total_err) if total_err else 0
            diversity = len(set(errors))

            feat_dict = {
                'behavior__total_error_rate': total_err / max(total, 1),
                'behavior__consistency': consistency,
                'behavior__diversity': diversity
            }
            # Dynamically handle distractor categories instead of hardcoding
            for cat in dist_cats:
                feat_dict[f'behavior__{cat}_ratio'] = errors.count(cat) / max(total_err, 1)

            behavioral_rows.append(feat_dict)

        beh_df = pd.DataFrame(behavioral_rows, index=error_df.index)
        num_matrix = pd.concat([num_matrix, beh_df], axis=1)

        scaler            = StandardScaler()
        num_matrix_scaled = scaler.fit_transform(num_matrix)

        # ── 5. K-modes ───────────────────────────────────────────────────────
        km_modes  = KModes(n_clusters=K, init='Huang', n_init=20, random_state=42)
        labels_modes = km_modes.fit_predict(cat_matrix_kmodes.values)

        # ── 6. K-means ───────────────────────────────────────────────────────
        km_means  = KMeans(n_clusters=K, init='k-means++', n_init=20, random_state=42)
        labels_means = km_means.fit_predict(num_matrix_scaled)

        # ── 6b. PCA SCATTER ──────────────────────────────────────────────────
        pca = PCA(n_components=2)
        pts = pca.fit_transform(num_matrix_scaled)

        scatter = []
        for i, (idx, p) in enumerate(zip(error_df.index, pts)):
            scatter.append({
                'x': float(p[0]),
                'y': float(p[1]),
                'kmeans': int(labels_means[i]),
                'kmodes': int(labels_modes[i]),
                'id': str(idx)
            })

        # ── 7. elbow data (K = 2..6) ─────────────────────────────────────────
        max_k = min(7, n_students - 1)
        elbow_k      = list(range(2, max_k))
        elbow_modes  = []
        elbow_means  = []
        for k in elbow_k:
            _km = KModes(n_clusters=k, init='Huang', n_init=10, random_state=42)
            _km.fit(cat_matrix_kmodes.values)
            elbow_modes.append(float(_km.cost_))

            _km2 = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
            _km2.fit(num_matrix_scaled)
            elbow_means.append(float(_km2.inertia_))

        # ── 8. silhouette ────────────────────────────────────────────────────
        def encode_cat_matrix(df):
            encoded_cols = {}
            for col in df.columns:
                if df[col].nunique() > 1:
                    le = LabelEncoder()
                    encoded_cols[col] = le.fit_transform(df[col])
            if not encoded_cols:
                return pd.DataFrame(np.zeros((len(df), 1)))
            return pd.DataFrame(encoded_cols, index=df.index)

        enc     = encode_cat_matrix(cat_matrix_kmodes)
        sil_km  = float(silhouette_score(num_matrix_scaled, labels_means)) \
                  if len(set(labels_means)) > 1 else 0.0
        sil_kmo = float(silhouette_score(enc, labels_modes, metric='hamming')) \
                  if len(set(labels_modes)) > 1 else 0.0

        db_km  = float(davies_bouldin_score(num_matrix_scaled, labels_means)) \
                 if len(set(labels_means)) > 1 else 0.0
        db_kmo = float(davies_bouldin_score(enc, labels_modes)) \
                 if len(set(labels_modes)) > 1 else 0.0

        # ── 9. ARI stability (10 runs) ───────────────────────────────────────
        ari_means_list = []
        ari_modes_list = []
        for seed in range(10):
            t1 = KMeans(n_clusters=K, init='k-means++', n_init=10,
                        random_state=seed).fit_predict(num_matrix_scaled)
            ari_means_list.append(float(adjusted_rand_score(labels_means, t1)))

            t2 = KModes(n_clusters=K, init='Huang', n_init=10,
                        random_state=seed).fit_predict(cat_matrix_kmodes.values)
            ari_modes_list.append(float(adjusted_rand_score(labels_modes, t2)))

        ari_means_mean = float(np.mean(ari_means_list))
        ari_means_std  = float(np.std(ari_means_list))
        ari_modes_mean = float(np.mean(ari_modes_list))
        ari_modes_std  = float(np.std(ari_modes_list))

        # ── 10. chi-square per question ──────────────────────────────────────
        chi_results = []
        for col in cat_matrix.columns:
            ct = pd.crosstab(labels_modes, cat_matrix[col])
            if ct.shape[1] > 1:
                chi2, p, dof, _ = chi2_contingency(ct)
                chi_results.append({
                    'question': col.upper(),
                    'chi2': round(float(chi2), 3),
                    'p_value': float(f"{p:.2e}") if p < 0.0001 else round(float(p), 4),
                    'significant': int(p < 0.05) 
                })

        sig_count = sum(1 for r in chi_results if r['significant'])

        # ── 11. cluster profiles ─────────────────────────────────────────────
        from collections import Counter

        kmodes_profiles = []
        for c in range(K):
            members = error_df[labels_modes == c]
            all_err = [v for v in members.values.flatten()
                       if v not in ('correct', 'blank')]
            counts  = Counter(all_err)
            total   = len(all_err) if all_err else 1

            breakdown = {k: round(v / total * 100, 1)
                         for k, v in sorted(counts.items(), key=lambda x: -x[1])}

            # Multi-label profile: per distractor category, compute prevalence
            # (% of cluster members who exhibit it >= once) and mean frequency
            multi_label_profile = {}
            _all_q_cols = list(error_df.columns)
            for _cat in dist_cats:
                _present_count = int(
                    (members[_all_q_cols] == _cat).any(axis=1).sum()
                )
                _prevalence = round(_present_count / max(len(members), 1) * 100, 1)
                _mean_freq  = round(
                    (members[_all_q_cols] == _cat).sum(axis=1).mean(), 2
                )
                multi_label_profile[_cat] = {
                    'prevalence_pct': _prevalence,
                    'mean_item_count': _mean_freq
                }

            dominant = max(counts, key=counts.get) if counts else 'none'

            kmodes_profiles.append({
                'cluster': int(c),
                'size': int(len(members)),
                'dominant_error': dominant,
                'breakdown': breakdown,
                'multi_label_profile': multi_label_profile
            })

        centroid_df = pd.DataFrame(
            scaler.inverse_transform(km_means.cluster_centers_),
            columns=num_matrix.columns
        ).round(3)

        kmeans_profiles = []
        for c in range(K):
            row = centroid_df.iloc[c].to_dict()
            size = int((labels_means == c).sum())

            acc_cols = [col for col in row if col.endswith('__accuracy')]
            weakest  = min(acc_cols, key=lambda x: row[x]) if acc_cols else 'N/A'
            weakest_domain = weakest.replace('__accuracy', '').replace('_', ' ').title()

            # 🔥 NEW: dominant error (highest rate overall)
            error_cols = [col for col in row if col.endswith('_rate')]
            dominant_error_col = max(error_cols, key=lambda x: row[x]) if error_cols else None
            dominant_error = dominant_error_col.split('__')[1].replace('_',' ') if dominant_error_col else 'none'
            dominant_error_val = row[dominant_error_col] if dominant_error_col else 0

            # 🔥 NEW: severity level
            severity = (
                'High' if dominant_error_val > 0.6 else
                'Moderate' if dominant_error_val > 0.3 else
                'Low'
            )

            # 🔥 NEW: cluster name
            cluster_name = f"{dominant_error.title()}-Dominant, Weak in {weakest_domain}"

            # 🔥 NEW: summary text
            summary = (
                f"Students show high {dominant_error} errors with "
                f"low performance in {weakest_domain}."
            )

            kmeans_profiles.append({
                'cluster': int(c),
                'size': size,
                'weakest_domain': weakest_domain,
                'dominant_error': dominant_error,
                'severity': severity,
                'cluster_name': cluster_name,
                'summary': summary,
                'centroid': {
                    k: round(v, 3)
                    for k, v in row.items()
                    if not k.startswith('behavior__')
                }
            })

        # ── 11b. plain-language cluster summaries for teachers ──────────────
        # Pair each kmodes group (error type) with its matching kmeans group
        # (weakest domain) to produce one card per group.
        SEVERITY_EMOJI = {'High': '🔴', 'Moderate': '🟡', 'Low': '🟢'}

        plain_language_clusters = []
        for c in range(K):
            kmo = next((p for p in kmodes_profiles if p['cluster'] == c), None)
            kme = next((p for p in kmeans_profiles if p['cluster'] == c), None)
            if not kmo or not kme:
                continue

            size    = kmo['size']
            pct     = round(size / n_students * 100)
            dom_err = kmo['dominant_error'].replace('_', ' ').title()
            weak_d  = kme['weakest_domain']
            severity= kme['severity']
            emoji   = SEVERITY_EMOJI.get(severity, '🟡')

            # Friendly group name based on dominant error
            group_name = f"Group {c + 1} — {dom_err} Mistakes"

            # What they get wrong sentence
            # Top 2 errors from breakdown
            top_errs = list(kmo['breakdown'].keys())[:2]
            top_errs_fmt = ' and '.join(
                e.replace('_', ' ').title() for e in top_errs
            ) if top_errs else dom_err

            # Pull significant questions for this cluster from chi results
            sig_qs_for_cluster = [
                r['question'] for r in chi_results if r['significant'] == 1
            ]
            sig_qs_note = (
                f" Questions {', '.join(sig_qs_for_cluster[:3])} show the clearest differences between groups."
                if sig_qs_for_cluster else ""
            )

            what_wrong = (
                f"Students in this group mostly make {top_errs_fmt} mistakes. "
                f"These errors show up most often in the {weak_d} topic.{sig_qs_note}"
            )

            # Urgency sentence
            if severity == 'High':
                urgency = (
                    f"This group needs the most attention — their error rate is high."
                )
            elif severity == 'Moderate':
                urgency = (
                    f"This group has a moderate level of mistakes — worth addressing soon."
                )
            else:
                urgency = (
                    f"This group is doing relatively well — a light review should be enough."
                )

            # One-sentence classroom action
            classroom_action = (
                f"Run a short focused activity on {dom_err.lower()} mistakes "
                f"using {weak_d} examples."
            )

            plain_language_clusters.append({
                'group_number': c + 1,
                'group_name': group_name,
                'size': size,
                'pct': pct,
                'dominant_error': dom_err,
                'weakest_domain': weak_d,
                'severity': severity,
                'severity_emoji': emoji,
                'what_wrong': what_wrong,
                'urgency': urgency,
                'classroom_action': classroom_action,
            })

        # ── 12. cross-tabulation ──────────────────────────────────────────────
        cross = pd.crosstab(labels_modes, labels_means).values.tolist()

        compound_profiles = []
        cross_np = np.array(cross)
        for i in range(K):
            for j in range(K):
                count = cross_np[i][j]
                if count == 0:
                    continue
                kmode_p  = next(p for p in kmodes_profiles if p['cluster'] == i)
                kmeans_p = next(p for p in kmeans_profiles if p['cluster'] == j)
                error    = kmode_p['dominant_error'].replace('_', ' ')
                domain   = kmeans_p['weakest_domain']
                is_dominant = count == cross_np[i].max() and count == cross_np[:,j].max()
                compound_profiles.append({
                    'kmodes_cluster': i,
                    'kmeans_cluster': j,
                    'count': int(count),

                    # existing
                    'name': f"{error.title()} errors — weak in {domain}",
                    'dominant': int(is_dominant),

                    # 🔥 NEW: add multi-parameter info
                    'dominant_error': error,
                    'weak_domain': domain,

                    # get full breakdown
                    'error_breakdown': kmode_p['breakdown'],

                    # get centroid behavior
                    'centroid': kmeans_p['centroid']
                })

        # ── 12b. cross-tab summary ───────────────────────────────────────────
        # For each kmodes cluster, find which error type dominates which domain
        # by pairing it with the kmeans cluster it overlaps most with.
        cross_tab_summary = []
        for i in range(K):
            kmode_p = next((p for p in kmodes_profiles if p['cluster'] == i), None)
            if not kmode_p:
                continue

            # Find the kmeans cluster this kmodes cluster overlaps most with
            row_counts = cross_np[i]
            best_j = int(np.argmax(row_counts))
            kmeans_p = next((p for p in kmeans_profiles if p['cluster'] == best_j), None)

            dom_err   = kmode_p['dominant_error'].replace('_', ' ').title()
            weak_dom  = kmeans_p['weakest_domain'] if kmeans_p else 'Unknown'
            size      = kmode_p['size']
            pct       = round(size / n_students * 100, 1)

            # Top 2 error types with their percentages from breakdown
            top_errs = list(kmode_p['breakdown'].items())[:2]
            top_errs_str = ', '.join(
                f"{e.replace('_',' ').title()} ({v}%)" for e, v in top_errs
            )

            # Which specific questions are significant for this cluster
            # (chi-square significant questions where this cluster's error is dominant)
            sig_qs = [
                r['question'] for r in chi_results
                if r['significant'] == 1
            ]
            sig_qs_str = ', '.join(sig_qs[:5]) if sig_qs else 'none'

            cross_tab_summary.append({
                'cluster': i + 1,
                'size': size,
                'pct': pct,
                'dominant_error': dom_err,
                'weakest_domain': weak_dom,
                'top_errors': top_errs_str,
                'significant_questions': sig_qs_str,
                'plain': (
                    f"Group {i+1} ({size} students, {pct}%): mainly makes "
                    f"{dom_err} mistakes, struggles most in {weak_dom}. "
                    f"Top errors: {top_errs_str}."
                )
            })

        # ── 13. per-domain error rate breakdown for chart ─────────────────────
        domain_chart = []
        for domain in domains:
            dname = domain['name']
            safe  = dname.replace(' ', '_').lower()
            entry = {'domain': dname}
            for etype in dist_cats:
                col = f"{safe}__{etype}_rate"
                if col in num_matrix.columns:
                    entry[etype] = round(float(num_matrix[col].mean()), 3)
            entry['accuracy'] = round(
                float(num_matrix.get(f"{safe}__accuracy",
                      pd.Series([0])).mean()), 3)
            domain_chart.append(entry)

        # ── 14. student-level results ─────────────────────────────────────────
        student_results = []
        ids = list(error_df.index) if error_df.index.name else \
              [f"S{str(i+1).zfill(3)}" for i in range(n_students)]
        for i, sid in enumerate(ids):
            student_results.append({
                'id': str(sid),
                'kmodes_cluster': int(labels_modes[i]),
                'kmeans_cluster': int(labels_means[i]),
            })

        # ── 15. overall_summary + insight ────────────────────────────────────
        # Clustering quality rating (silhouette-based)
        avg_sil = (sil_km + sil_kmo) / 2
        if avg_sil >= 0.5:
            quality_label = 'Strong'
            quality_note  = f'Students show clearly distinct mistake patterns. About {round((1 - avg_sil) * 50 + 50)}% of students fit their group well — the groups are easy to tell apart.'
        elif avg_sil >= 0.25:
            quality_label = 'Moderate'
            quality_note  = f'Most students fit well into their group, though roughly {round(avg_sil * 30)}% share traits with more than one group.'
        else:
            quality_label = 'Weak'
            quality_note  = f'The groups overlap quite a bit — only about {round(avg_sil * 100)}% separation between groups. Use as a rough guide only.'

        # Stability rating (ARI-based, average of both methods)
        avg_ari = (ari_means_mean + ari_modes_mean) / 2
        if avg_ari >= 0.75:
            stability_label = 'High'
            stability_note  = f'The same groups appear every time the analysis runs ({round(avg_ari*100)}% agreement across runs) — you can trust these results.'
        elif avg_ari >= 0.40:
            stability_label = 'Moderate'
            stability_note  = f'The groups are mostly stable ({round(avg_ari*100)}% agreement across runs), with minor differences each time.'
        else:
            stability_label = 'Low'
            stability_note  = f'The groups shift noticeably between runs ({round(avg_ari*100)}% agreement) — treat them as a starting point only.'

        # Statistical significance summary
        chi_total = len(chi_results)
        sig_ratio = sig_count / chi_total if chi_total > 0 else 0
        if sig_ratio >= 0.7:
            chi_label = 'High'
            chi_note  = (f'{sig_count} out of {chi_total} questions had clearly different mistake patterns across groups — the groups are meaningful.')
        elif sig_ratio >= 0.3:
            chi_label = 'Moderate'
            chi_note  = (f'{sig_count} out of {chi_total} questions showed noticeable differences between groups — a partial signal.')
        else:
            chi_label = 'Low'
            chi_note  = (f'Only {sig_count} out of {chi_total} questions showed differences between groups — the groups may look very similar on most questions.')

        # Dominant error pattern across ALL clusters (most common dominant_error)
        all_dominant_errors = [p['dominant_error'] for p in kmodes_profiles
                               if p['dominant_error'] != 'none']
        from collections import Counter as _C2
        if all_dominant_errors:
            dom_err_counts = _C2(all_dominant_errors)
            global_dominant_error = dom_err_counts.most_common(1)[0][0]
        else:
            global_dominant_error = 'mixed'

        # Weakest domain overall (lowest mean accuracy across all clusters)
        weakest_domain_counts = _C2(
            [p['weakest_domain'] for p in kmeans_profiles]
        )
        global_weakest_domain = weakest_domain_counts.most_common(1)[0][0] \
            if kmeans_profiles else 'Unknown'

        # Largest cluster summary
        largest_cluster = max(kmodes_profiles, key=lambda p: p['size'])
        largest_pct = round(largest_cluster['size'] / n_students * 100, 1)

        # Assemble overall_summary object
        overall_summary = {
            'clustering_quality': {
                'label': quality_label,
                'score': round(avg_sil, 3),
                'note': quality_note
            },
            'stability': {
                'label': stability_label,
                'score': round(avg_ari, 3),
                'note': stability_note
            },
            'statistical_significance': {
                'label': chi_label,
                'significant_count': sig_count,
                'total_questions': chi_total,
                'note': chi_note
            },
            'dominant_error_pattern': global_dominant_error,
            'weakest_domain': global_weakest_domain,
            'n_students': n_students,
            'k': K,
            'plain_language_clusters': plain_language_clusters,
            'cross_tab_summary': cross_tab_summary
        }

        # ── plain-language insight (what / struggle / action) ────────────────
        err_label = global_dominant_error.replace('_', ' ').title()
        dom_label = global_weakest_domain

        # WHAT IS HAPPENING
        if quality_label == 'Strong' and stability_label in ('High', 'Moderate'):
            what = (
                f"Your {n_students} students fall into {K} clearly distinct groups "
                f"based on their mistake patterns — and the same groups appear "
                f"every time the analysis runs, so you can act on these with confidence."
            )
        elif quality_label == 'Moderate' or stability_label == 'Moderate':
            what = (
                f"Your {n_students} students split into {K} groups with noticeably "
                f"different mistake patterns. Most students fit their group well, "
                f"though a few sit between groups — use this as a helpful guide "
                f"rather than a definitive list."
            )
        else:
            what = (
                f"Your {n_students} students show {K} rough groupings of mistake "
                f"patterns, but there is quite a bit of overlap between groups. "
                f"Use these as a broad starting point for planning, not as "
                f"exact labels for individual students."
            )

        # WHAT STUDENTS STRUGGLE WITH
        if global_dominant_error != 'mixed':
            struggle = (
                f"{largest_pct}% of your students are making the same type of mistake: "
                f'"{err_label}." '
                f"This comes up most in the {dom_label} topic. "
                f"It is the single biggest pattern worth addressing first."
            )
        else:
            struggle = (
                f"There is no single mistake that dominates across all groups. "
                f"However, the {dom_label} topic has the lowest scores overall, "
                f"making it the best place to start when planning a review."
            )

        # WHAT ACTION TO TAKE
        if chi_label == 'High':
            confidence_note = (
                f"The mistake patterns are clearly different between groups, "
                f"so it is worth teaching each group differently."
            )
        elif chi_label == 'Moderate':
            confidence_note = (
                f"Some questions show clear differences between groups. "
                f"Focus on those questions first when planning group lessons."
            )
        else:
            confidence_note = (
                f"The differences between groups are small for most questions, "
                f"so a whole-class review may work just as well as group-specific lessons."
            )

        if quality_label == 'Strong' and stability_label in ('High', 'Moderate'):
            action = (
                f'Split students into their {K} groups and run a focused review '
                f'on "{err_label}" mistakes in the {dom_label} topic. '
                f"{confidence_note}"
            )
        elif quality_label in ('Strong', 'Moderate'):
            action = (
                f'Use the {K} groups as a guide and plan a review session '
                f'targeting "{err_label}" mistakes in the {dom_label} topic. '
                f"{confidence_note}"
            )
        else:
            action = (
                f'Start with a whole-class review on "{err_label}" mistakes '
                f"in the {dom_label} topic, then check whether students need "
                f"group-specific follow-up. {confidence_note}"
            )

        # ── cross-tab breakdown paragraph ────────────────────────────────────
        cross_tab_lines = []
        for entry in cross_tab_summary:
            cross_tab_lines.append(entry['plain'])
        cross_tab_paragraph = (
            "Here is what the cross-tabulation shows for each group:\n"
            + "\n".join(f"  • {line}" for line in cross_tab_lines)
        )

        insight = f"{what}\n\n{struggle}\n\n{cross_tab_paragraph}\n\n{action}"

        # ── 16. assemble response ─────────────────────────────────────────────
        return jsonify({
            'overall_summary': overall_summary,
            'insight': insight,
            'n_students': n_students,
            'k': K,
            'domains': [d['name'] for d in domains],
            'distractor_categories': dist_cats,
            'elbow': {
                'k_values': elbow_k,
                'kmodes_costs': elbow_modes,
                'kmeans_inertias': elbow_means
            },
            'validation': {
                'silhouette_kmeans': round(sil_km, 4),
                'silhouette_kmodes': round(sil_kmo, 4),
                'davies_bouldin_kmeans': round(db_km, 4),
                'davies_bouldin_kmodes': round(db_kmo, 4),
                'ari_kmeans_mean': round(ari_means_mean, 4),
                'ari_kmeans_std': round(ari_means_std, 4),
                'ari_kmodes_mean': round(ari_modes_mean, 4),
                'ari_kmodes_std': round(ari_modes_std, 4),
                'chi_square': chi_results,
                'chi_significant_count': sig_count,
                'chi_total': len(chi_results)
            },
            'kmodes_profiles': kmodes_profiles,
            'kmeans_profiles': kmeans_profiles,
            'cross_tab': cross,
            'compound_profiles': compound_profiles,
            'domain_chart': domain_chart,
            'students': student_results,
            'kmeans_labels': labels_means.tolist(),
            'kmodes_labels': labels_modes.tolist(),
            'scatter': scatter
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/parse-answer-key', methods=['POST'])
def parse_answer_key():
    try:
        csv_file = request.files.get('answer_key_csv')
        if not csv_file:
            return jsonify({'error': 'No file uploaded'}), 400

        df = pd.read_csv(csv_file)
        df.columns = [c.strip().lower() for c in df.columns]

        required = {'question', 'a', 'b', 'c', 'd'}
        if not required.issubset(set(df.columns)):
            return jsonify({
                'error': f'CSV must have columns: question, A, B, C, D (and optionally domain). Found: {list(df.columns)}'
            }), 400

        has_domain_col = 'domain' in df.columns

        answer_key = {}
        distractor_categories = set()
        # domainName -> list of 1-based question numbers
        domain_map = {}

        for q_num, (_, row) in enumerate(df.iterrows(), start=1):
            q = str(row['question']).strip().lower()
            mapping = {
                'A': str(row['a']).strip(),
                'B': str(row['b']).strip(),
                'C': str(row['c']).strip(),
                'D': str(row['d']).strip(),
            }
            answer_key[q] = mapping
            for v in mapping.values():
                if v and v != 'correct':
                    distractor_categories.add(v)

            # collect domain membership
            if has_domain_col:
                domain_name = str(row['domain']).strip()
                if domain_name and domain_name.lower() not in ('', 'nan', 'none'):
                    domain_map.setdefault(domain_name, []).append(q_num)

        # build domain list with start/end item numbers
        domains_out = []
        for dname, q_nums in domain_map.items():
            domains_out.append({
                'name': dname,
                'start': min(q_nums),
                'end': max(q_nums)
            })

        return jsonify({
            'answer_key': answer_key,
            'distractor_categories': sorted(distractor_categories),
            'item_count': len(answer_key),
            'domains': domains_out,          # populated when domain column present
            'has_domain_col': has_domain_col
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)