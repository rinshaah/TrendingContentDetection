import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('../Dataset/cleaned_dataset_5000.csv')

# Rename column properly
data.rename(columns={"Post Content": "post_content"}, inplace=True)

# Ensure platform column clean
data['Platform'] = data['Platform'].str.strip()

# Convert timestamp
data['Post Timestamp'] = pd.to_datetime(data['Post Timestamp'], errors='coerce')
data = data.dropna(subset=['Post Timestamp'])

# Calculate recency
current_time = pd.Timestamp.now()
data['recency_minutes'] = (current_time - data['Post Timestamp']).dt.total_seconds() / 60

# Ensure description exists
if 'description' not in data.columns:
    data['description'] = data['post_content']


# ---------- TRENDING PIPELINE FUNCTION ----------

def trending_pipeline(platform_name, seed):

    df = data.copy()
    round_num = 1

    while len(df) > 10:

        print(f"{platform_name} - Clustering Round {round_num}")

        # Select features
        features = df[['Likes','Shares','Comments','Engagement Rate','recency_minutes']]

        # Scale
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        # KMeans
        kmeans = KMeans(n_clusters=3, random_state=seed+round_num)
        df['cluster'] = kmeans.fit_predict(scaled)

        # Convert scaled array to DataFrame
        scaled_df = pd.DataFrame(scaled, columns=features.columns)

        # Create weighted score (your priority logic)
        df['score'] = (
        0.5 * scaled_df['Comments'] +
        -0.3 * scaled_df['recency_minutes'] +
         0.1 * scaled_df['Shares'] +
        0.05 * scaled_df['Likes'] +
        0.05 * scaled_df['Engagement Rate']
            )

        # Rank clusters
        cluster_avg = df.groupby('cluster')['score'].mean().sort_values(ascending=False)

        trend_labels = {}
        trend_labels[cluster_avg.index[0]] = "Highly Trending"
        trend_labels[cluster_avg.index[1]] = "Moderately Trending"
        trend_labels[cluster_avg.index[2]] = "Low Trending"

        df['trend'] = df['cluster'].map(trend_labels)

        # Save each round result
        df[['post_content','trend','Likes','Shares','Comments']].to_csv(
            f'../Dataset/{platform_name}_round{round_num}.csv',
            index=False
        )

        # Keep highly trending only
        df = df[df['trend']=="Highly Trending"].copy()

        round_num += 1

        if len(df) <= 10:
            break

    # Final Top10
    top10 = df.head(10).copy()
    top10['platform'] = platform_name
    top10['trend'] = "Highly Trending"

    return top10

# ---------- RUN FOR EACH PLATFORM ----------
twitter = trending_pipeline("Twitter",22)
facebook = trending_pipeline("Facebook",30)
instagram = trending_pipeline("Instagram",42)

# Combine results
final = pd.concat([twitter, facebook, instagram], ignore_index=True)

# Apply formatting
final['post_content'] = final['post_content'].str.capitalize()
# final['description'] = final['description'].apply(smart_format)

# Save JSON
final.to_json(
    '../frontend/trending.json',
    orient='records',
    indent=2
)

print("✅ Top 10 trending posts saved successfully!")