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

        # ── 15. assemble response ─────────────────────────────────────────────
        return jsonify({
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
                'error': f'CSV must have columns: question, A, B, C, D. Found: {list(df.columns)}'
            }), 400

        answer_key = {}
        distractor_categories = set()

        for _, row in df.iterrows():
            q = str(row['question']).strip().lower()
            mapping = {
                'A': str(row['a']).strip(),
                'B': str(row['b']).strip(),
                'C': str(row['c']).strip(),
                'D': str(row['d']).strip(),
            }
            answer_key[q] = mapping
            for v in mapping.values():
                if v != 'correct':
                    distractor_categories.add(v)

        return jsonify({
            'answer_key': answer_key,
            'distractor_categories': sorted(distractor_categories),
            'item_count': len(answer_key)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)