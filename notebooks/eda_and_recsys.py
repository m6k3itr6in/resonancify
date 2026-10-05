import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.compose import ColumnTransformer

    import matplotlib.pyplot as plt

    df = pd.read_csv('data/raw/tracks.csv', engine='pyarrow')

    df.shape
    return MinMaxScaler, NearestNeighbors, df, np, plt


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(MinMaxScaler, NearestNeighbors, df, np, plt):
    df_drop = df.dropna(subset=['artists', 'track_name', 'album_name']).reset_index(drop=True)

    feature_cols = [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo'
    ]

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df_drop[feature_cols])

    knn = NearestNeighbors(n_neighbors=50, metric='cosine')

    knn.fit(scaled_features)

    distance, indices = knn.kneighbors(scaled_features)
    sort_distance = np.sort(distance[:, -1])

    plt.figure(figsize=(10, 6))
    plt.plot(sort_distance, color='royalblue', lw=2)
    plt.title('График расстояний до 50-го соседа')
    plt.xlabel('Объекты')
    plt.ylabel('Косинусное расстояние')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    return df_drop, knn, scaled_features


@app.cell
def _(df_drop, knn, scaled_features):
    def recommend(track_name, df, knn, scaled_features, top_k=5, max_popularity=None):
        track_idx = df[df['track_name'] == track_name].index[0]

        vector = scaled_features[track_idx].reshape(1, -1)

        distance, indices = knn.kneighbors(vector, n_neighbors = 50)

        recommended_indices = indices[0][1:]
        recommendations = df.iloc[recommended_indices]

        return recommendations.drop_duplicates(subset=['artists']).head(5)

    recommend('505', df_drop, knn, scaled_features)
    return


if __name__ == "__main__":
    app.run()
