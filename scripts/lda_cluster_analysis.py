import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, cdist
from scipy.special import rel_entr  # KL divergence
import matplotlib.pyplot as plt
import seaborn as sns

# ============ INTRA-CLUSTER: TOPIC COHERENCE ============
# Measures how semantically related top words are WITHIN each topic

def coherence_analysis(lda_model, texts, vectorizer):
    """
    Compute coherence scores for sklearn LDA.
    Coherence: 0-1 scale. Higher = top words are semantically related.
    Uses co-document frequency of top words (manual calculation).
    """
    texts_tokenized = [t.split() for t in texts]
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = []
    for topic_id in range(lda_model.n_components):
        top_indices = lda_model.components_[topic_id].argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)
    
    # Compute co-document frequency coherence (simplified C_V)
    def compute_topic_coherence(topic_words, texts_tokenized):
        """
        Compute coherence as average pairwise co-occurrence of top words.
        Higher = words appear together more often.
        """
        if len(topic_words) < 2:
            return 0.0
        
        # Count documents containing each pair of words
        coherence_scores = []
        for i, word1 in enumerate(topic_words):
            for word2 in topic_words[i+1:]:
                # Count docs with both words
                co_occur = sum(1 for text in texts_tokenized 
                              if word1 in text and word2 in text)
                # Normalize by docs containing word1 (conditional probability)
                occur_word1 = sum(1 for text in texts_tokenized if word1 in text)
                if occur_word1 > 0:
                    coherence_scores.append(co_occur / occur_word1)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    per_topic_coherence = np.array([
        compute_topic_coherence(topics[i], texts_tokenized) 
        for i in range(len(topics))
    ])
    
    overall_coherence = np.mean(per_topic_coherence)
    
    print("=== INTRA-CLUSTER: TOPIC COHERENCE ===")
    print(f"Overall Model Coherence: {overall_coherence:.3f}")
    print(f"  (0 = incoherent; 0.4+ = decent; 0.6+ = good; 0.75+ = excellent)\n")
    
    coherence_df = pd.DataFrame({
        'topic_id': range(lda_model.n_components),
        'coherence': per_topic_coherence
    }).sort_values('coherence', ascending=False)
    
    print("Per-Topic Coherence (sorted by strength):")
    print(coherence_df.to_string(index=False))
    
    return overall_coherence, per_topic_coherence, coherence_df

# ============ INTER-CLUSTER: TOPIC DISTANCE ============
# Measures how DISTINCT topics are from each other

def topic_distance_matrix(lda_model, distance_metric='cosine'):
    """
    Compute pairwise distances between topics.
    - Cosine distance: 0 = identical, 1 = orthogonal
    - Hellinger distance: 0 = identical, 1 = completely different
    """
    # Get topic distributions (components)
    components = lda_model.components_  # shape: (n_topics, n_words)
    
    # Normalize to probability distributions (sum to 1 per topic)
    components_norm = components / components.sum(axis=1, keepdims=True)
    
    if distance_metric == 'cosine':
        # Cosine distance between normalized components
        distances = cdist(components_norm, components_norm, metric='cosine')
        
    elif distance_metric == 'hellinger':
        # Hellinger distance: sqrt(0.5 * sum((sqrt(p) - sqrt(q))^2))
        def hellinger(p, q):
            return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        
        n_topics = components_norm.shape[0]
        distances = np.zeros((n_topics, n_topics))
        for i in range(n_topics):
            for j in range(i, n_topics):
                d = hellinger(components_norm[i], components_norm[j])
                distances[i, j] = d
                distances[j, i] = d
    
    elif distance_metric == 'kl':
        # KL divergence (asymmetric, so we symmetrize)
        def sym_kl(p, q):
            return (np.sum(rel_entr(p, q)) + np.sum(rel_entr(q, p))) / 2
        
        n_topics = components_norm.shape[0]
        distances = np.zeros((n_topics, n_topics))
        for i in range(n_topics):
            for j in range(i, n_topics):
                d = sym_kl(components_norm[i], components_norm[j])
                distances[i, j] = d
                distances[j, i] = d
    
    return distances

def inter_cluster_analysis(lda_model):
    """Compute and visualize topic distinctiveness."""
    print("\n=== INTER-CLUSTER: TOPIC DISTINCTIVENESS ===\n")
    
    # Compute multiple distance metrics
    cosine_dist = topic_distance_matrix(lda_model, 'cosine')
    hellinger_dist = topic_distance_matrix(lda_model, 'hellinger')
    
    # Summary statistics
    # Remove diagonal (self-distance = 0)
    cosine_offdiag = cosine_dist[np.triu_indices_from(cosine_dist, k=1)]
    hellinger_offdiag = hellinger_dist[np.triu_indices_from(hellinger_dist, k=1)]
    
    print(f"Cosine Distance (0=identical, 1=orthogonal):")
    print(f"  Mean: {cosine_offdiag.mean():.3f}")
    print(f"  Min: {cosine_offdiag.min():.3f} (most similar topics)")
    print(f"  Max: {cosine_offdiag.max():.3f} (most distinct topics)")
    
    print(f"\nHellinger Distance (0=identical, 1=completely different):")
    print(f"  Mean: {hellinger_offdiag.mean():.3f}")
    print(f"  Min: {hellinger_offdiag.min():.3f} (most similar topics)")
    print(f"  Max: {hellinger_offdiag.max():.3f} (most distinct topics)")
    
    # Find closest topic pairs
    print(f"\n=== Most Similar Topic Pairs (Closest to Redundancy) ===")
    pairs = []
    for i in range(lda_model.n_components):
        for j in range(i+1, lda_model.n_components):
            pairs.append((i, j, cosine_dist[i, j]))
    
    closest = sorted(pairs, key=lambda x: x[2])[:5]
    for t1, t2, dist in closest:
        print(f"  Topic {t1} ↔ {t2}: cosine={dist:.3f} (similarity={1-dist:.3f})")
    
    return cosine_dist, hellinger_dist

def visualize_topic_distance(cosine_dist, hellinger_dist, topic_labels=None):
    """Create heatmaps of topic distances."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cosine distance
    sns.heatmap(cosine_dist, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': 'Distance'}, square=True)
    axes[0].set_title('Topic Distances (Cosine)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Topic ID')
    axes[0].set_ylabel('Topic ID')
    
    # Hellinger distance
    sns.heatmap(hellinger_dist, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=axes[1], cbar_kws={'label': 'Distance'}, square=True)
    axes[1].set_title('Topic Distances (Hellinger)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Topic ID')
    axes[1].set_ylabel('Topic ID')
    
    plt.tight_layout()
    plt.savefig(r"C:\Thesis\results\topic_distances.png", dpi=300, bbox_inches='tight')
    print("\n✓ Saved to: C:\\Thesis\\results\\topic_distances.png")
    plt.show()

# ============ COMBINED SUMMARY ============

def lda_cluster_quality_report(lda_model, doc_topics, texts, vectorizer):
    """Generate comprehensive intra + inter cluster report."""
    
    print("=" * 70)
    print("LDA CLUSTER QUALITY REPORT")
    print("=" * 70)
    
    # Intra-cluster
    overall_coh, per_topic_coh, coh_df = coherence_analysis(lda_model, texts, vectorizer)
    
    # Inter-cluster
    cosine_dist, hellinger_dist = inter_cluster_analysis(lda_model)
    
    # Visualization
    visualize_topic_distance(cosine_dist, hellinger_dist)
    
    # Summary table
    print("\n=== COMBINED QUALITY MATRIX ===")
    summary_df = pd.DataFrame({
        'topic_id': range(lda_model.n_components),
        'coherence': per_topic_coh,
        'min_distance_to_other': [cosine_dist[i, [j for j in range(lda_model.n_components) if j != i]].min() 
                                   for i in range(lda_model.n_components)]
    }).sort_values('coherence', ascending=False)
    
    print(summary_df.to_string(index=False))
    print("\n  Coherence: higher is better (>0.5 good, >0.65 excellent)")
    print("  Min Distance: higher is better (>0.3 well-separated, <0.15 potentially redundant)")
    
    return overall_coh, per_topic_coh, cosine_dist, hellinger_dist