from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from scipy.stats import chi2_contingency
from kmodes.kmodes import KModes
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

        # config shape:
        # {
        #   "k": 3,
        #   "domains": [
        #     {"name": "Financial Math", "start": 1, "end": 10},
        #     ...
        #   ],
        #   "answer_key": {
        #     "q1": {"A": "off_by_one", "B": "sign_error", "C": "correct", "D": "decimal"},
        #     ...
        #   },
        #   "distractor_categories": ["off_by_one", "sign_error", "decimal"]
        # }

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
                error_df[qcol_norm] = df[qcol_norm].str.strip().str.upper().map(
                    {k.upper(): v for k, v in mapping.items()}
                ).fillna('blank')
            else:
                error_df[qcol_norm] = 'blank'

        # cat_matrix for display, chi-square, and cluster profiles (none = correct/blank)
        cat_matrix = error_df.replace({'correct': 'none', 'blank': 'none'})

        # K-modes matrix: only the error-type columns, correct/blank rows dropped per cell.
        # Strategy: build a per-student error-rate vector across all error types per domain,
        # then discretize into the dominant error type per domain — but ONLY among wrong answers.
        # Students with no errors in a domain get 'none' for that domain (rare edge case).
        # This means K-modes clusters purely on WHAT type of error, not HOW MANY were correct.
        from collections import Counter as _Counter
        error_profile_rows = {}
        for _dom in domains:
            _dname = _dom['name']
            _cols  = [f"q{i}" for i in range(int(_dom['start']), int(_dom['end']) + 1)]
            _cols  = [c for c in _cols if c in error_df.columns]
            _safe  = _dname.replace(' ', '_').lower()
            for idx in error_df.index:
                if idx not in error_profile_rows:
                    error_profile_rows[idx] = {}
                # Only look at wrong answers — ignore correct and blank entirely
                _errs = [v for v in error_df.loc[idx, _cols]
                         if v not in ('correct', 'blank')]
                _dominant = _Counter(_errs).most_common(1)[0][0] if _errs else 'none'
                error_profile_rows[idx][f"{_safe}__dominant_error"] = _dominant

        _ep_df = pd.DataFrame.from_dict(error_profile_rows, orient='index')

        # Only keep domain columns that have at least 2 distinct error types across students
        # (drops domains where everyone either aced it or all made the same error)
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

        scaler            = StandardScaler()
        num_matrix_scaled = scaler.fit_transform(num_matrix)

        # ── 5. K-modes ───────────────────────────────────────────────────────
        km_modes  = KModes(n_clusters=K, init='Huang', n_init=20, random_state=42)
        labels_modes = km_modes.fit_predict(cat_matrix_kmodes.values)

        # ── 6. K-means ───────────────────────────────────────────────────────
        km_means  = KMeans(n_clusters=K, init='k-means++', n_init=20, random_state=42)
        labels_means = km_means.fit_predict(num_matrix_scaled)

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
        # Each column needs its own LabelEncoder; drop columns with only 1 unique value
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

        # Davies-Bouldin Index (lower = better; complements silhouette)
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
                    'significant': int(p < 0.05)  # int instead of bool
                })

        sig_count = sum(1 for r in chi_results if r['significant'])

        # ── 11. cluster profiles ─────────────────────────────────────────────
        from collections import Counter

        kmodes_profiles = []
        for c in range(K):
            members = error_df[labels_modes == c]
            # Only count actual errors, exclude correct and blank
            all_err = [v for v in members.values.flatten()
                       if v not in ('correct', 'blank')]
            counts  = Counter(all_err)
            total   = len(all_err) if all_err else 1
            dominant = max(counts, key=counts.get) if counts else 'none'
            breakdown = {k: round(v / total * 100, 1)
                         for k, v in sorted(counts.items(), key=lambda x: -x[1])}
            kmodes_profiles.append({
                'cluster': int(c),
                'size': int(len(members)),
                'dominant_error': dominant,
                'breakdown': breakdown
            })

        # K-means centroids
        centroid_df = pd.DataFrame(
            scaler.inverse_transform(km_means.cluster_centers_),
            columns=num_matrix.columns
        ).round(3)

        kmeans_profiles = []
        for c in range(K):
            row = centroid_df.iloc[c].to_dict()
            size = int((labels_means == c).sum())
            # find weakest domain by accuracy
            acc_cols = [col for col in row if col.endswith('__accuracy')]
            weakest  = min(acc_cols, key=lambda x: row[x]) if acc_cols else 'N/A'
            weakest_domain = weakest.replace('__accuracy', '').replace('_', ' ').title()
            kmeans_profiles.append({
                'cluster': int(c),
                'size': size,
                'weakest_domain': weakest_domain,
                'centroid': {k: round(v, 3) for k, v in row.items()}
            })

        # ── 12. cross-tabulation ──────────────────────────────────────────────
        cross = pd.crosstab(labels_modes, labels_means).values.tolist()

        # Auto-generate compound profile names
        compound_profiles = []
        cross_np = np.array(cross)
        for i in range(K):
            for j in range(K):
                count = cross_np[i][j]
                if count == 0:
                    continue
                # get dominant error type from K-modes profile
                kmode_p  = next(p for p in kmodes_profiles if p['cluster'] == i)
                kmeans_p = next(p for p in kmeans_profiles if p['cluster'] == j)
                error    = kmode_p['dominant_error'].replace('_', ' ')
                domain   = kmeans_p['weakest_domain']
                is_dominant = count == cross_np[i].max() and count == cross_np[:,j].max()
                compound_profiles.append({
                    'kmodes_cluster': i,
                    'kmeans_cluster': j,
                    'count': int(count),
                    'name': f"{error.title()} errors — weak in {domain}",
                    'dominant': int(is_dominant)
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
            'students': student_results
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

        # expected columns: question, A, B, C, D
        # e.g:
        # question,A,B,C,D
        # q1,off_by_one,sign_error,correct,decimal
        # q2,correct,decimal,sign_error,off_by_one

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