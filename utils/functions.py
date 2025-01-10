def search_song(song_name, popularity_range, year_range, hot_songs_only):
    """
    Search for a song and create a DataFrame with the relevant song details.
    """
    query = song_name
    results = spotify.search(q=query, type='track', limit=15)  # Limit to 50 for better result matching

    if not results['tracks']['items']:
        return None

    tracks_info = []
    
    for track in results['tracks']['items']:
        artist_id = track['artists'][0]['id']
        artist_info = spotify.artist(artist_id)
        genres = artist_info['genres']
        release_year = int(track['album']['release_date'].split('-')[0])  # Get the year as an integer

            
            # Store track data
        track_info = {
           
            'spotify_title': track['name'],
            'spotify_artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'release_date': track['album']['release_date'],
            'year': release_year,  # Add the year to the track_info

            'popularity': track['popularity'],
            'duration_ms': track['duration_ms'],
            'explicit': track['explicit'],
            'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None,
            'genres': genres
        }

        track_info['popularity'] = int(track_info['popularity'])

        # Filter based on popularity and year range
        if popularity_range and not (popularity_range[0] <= track_info['popularity'] <= popularity_range[1]):
            continue
        if year_range and not (year_range[0] <= track_info['year'] <= year_range[1]):
            continue
        if hot_songs_only and track_info['popularity'] < 50:
            continue  # Skip songs with low popularity if hot_songs_only is checked

        tracks_info.append(track_info)

    return pd.DataFrame(tracks_info)




def get_recommendations(selected_track_info, n_recommendations=10):
    """
    Main recommendation function that takes a song's details and provides song recommendations.
    """
    # 1. Build a dtaframe with the correct columns used in scaling and modeling
    user_song_data = {
        'release_date': int(selected_track_info['release_date'].split('-')[0]),
        'popularity': selected_track_info['popularity'],
        'duration_ms': selected_track_info['duration_ms'],
        'explicit': selected_track_info['explicit']
    }

    # Add the genres column
    # Load the columns used in spotify_numerical.csv and remove the first 4 columns
    genres_columns = pd.read_csv('../data/6_spotify_numerical_scaled.csv').columns[4:-1]

    # Add the genres column to user_song_data, if the genre is in the list, add 1, if not, add 0
    for genre in genres_columns:
        if genre in selected_track_info['genres']:
            user_song_data[genre] = 1
        else:
            user_song_data[genre] = 0

    # Convert the dictionary to a DataFrame
    user_song_df = pd.DataFrame([user_song_data])

    # 2. Scale the features and predict cluster
    user_song_scaled = spotify_scaler.transform(user_song_df)
    # add the columms
    user_song_scaled_df = pd.DataFrame(user_song_scaled, columns=user_song_df.columns)
    
    # predict the cluster
    cluster = kmeans_model.predict(user_song_scaled_df)[0]

    # 3. Load clustered dataset and filter for the same cluster
    clustered_df = pd.read_csv('../data/8_spotify_million_tracks_clustered.csv')
    cluster_songs = clustered_df[clustered_df['kmeans_cluster'] == cluster]

    # 4. Get recommendations excluding the input song
    recommendations = cluster_songs[
        (cluster_songs['spotify_title'] != selected_track_info['spotify_title']) |
        (cluster_songs['spotify_artist'] != selected_track_info['spotify_artist'])
    ]
    recommendations = recommendations.nlargest(n_recommendations, 'popularity')

    return recommendations[['spotify_title', 'spotify_artist', 'popularity', 'album_cover']]


