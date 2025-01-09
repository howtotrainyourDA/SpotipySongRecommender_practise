import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import joblib

# Load models and scaler only once
spotify_scaler = joblib.load('../scaler/spotify_scaler.pkl')
kmeans_model = joblib.load('../models/kmeans_model.pkl')


# Initialize session state for page management and data persistence
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'selected_song' not in st.session_state:
    st.session_state.selected_song = None
if 'tracks_info' not in st.session_state:
    st.session_state.tracks_info = None
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
if 'rejected_songs' not in st.session_state:
    st.session_state.rejected_songs = set()
if 'current_recommendations' not in st.session_state:
    st.session_state.current_recommendations = None

# Set your Spotify credentials
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

# Initialize Spotify client
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))


# Page navigation functions
def on_find_songs_click():
    st.session_state.page = 'validation'

def on_use_this_click(song):
    # Store selected song details in session state
    st.session_state.selected_song = song
    st.session_state.page = 'recommendations'

def on_new_song_click():
    st.session_state.page = 'landing'

def reject_song(song_name):
    # Add the rejected song to our set
    st.session_state.rejected_songs.add(song_name)
    # Get new recommendation (you'll need to implement get_new_recommendation())
    new_rec = get_new_recommendation(st.session_state.selected_song, st.session_state.rejected_songs)
    
    # Replace the rejected song with the new one
    for i, rec in enumerate(st.session_state.current_recommendations):
        if rec['name'] == song_name:
            st.session_state.current_recommendations[i] = new_rec
            break

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
            
            # Store track data
        track_info = {
            'original_title': track['title'],
            'original_artist': track['artist'],
            'spotify_title': track['name'],
            'spotify_artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'release_date': track['album']['release_date'],
            'popularity': track['popularity'],
            'duration_ms': track['duration_ms'],
            'explicit': track['explicit'],
            'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None,
            'genres': genres
        }

        # Filter based on popularity and year range
        if not (popularity_range[0] <= track_info['popularity'] <= popularity_range[1]):
            continue
        if not (year_range[0] <= track_info['year'] <= year_range[1]):
            continue
        if hot_songs_only and track_info['popularity'] < 50:
            continue  # Skip songs with low popularity if hot_songs_only is checked

        tracks_info.append(track_info)

    return pd.DataFrame(tracks_info)

def get_recommendations(selected_track_info, n_recommendations=10):
    """
    Main recommendation function that takes a song's details and provides song recommendations.
    """
    # 1. Use the selected track info directly (already passed as a dict)
    user_song_df = pd.DataFrame([selected_track_info])

    # 2. Scale the features and predict cluster
    scaled_features = spotify_scaler.transform(user_song_df[['popularity', 'duration_ms', 'explicit']])
    cluster = kmeans_model.predict(scaled_features)[0]

    # 3. Load clustered dataset and filter for the same cluster
    clustered_df = pd.read_csv('../data/8_spotify_million_tracks_clustered.csv')
    cluster_songs = clustered_df[clustered_df['cluster'] == cluster]

    # 4. Get recommendations excluding the input song
    recommendations = cluster_songs[
        (cluster_songs['spotify_title'] != selected_track_info['spotify_title']) |
        (cluster_songs['spotify_artist'] != selected_track_info['spotify_artist'])
    ]
    recommendations = recommendations.nlargest(n_recommendations, 'popularity')

    return recommendations[['spotify_title', 'spotify_artist', 'popularity', 'album_cover']]




#  Start with the first page (Landing page)
if st.session_state.page == 'landing':
    st.title('Spotify Song Recommender')
    st.write('Find your next favorite song based on your music taste!')
    
    # User inputs
    search_query = st.text_input('Enter a song or artist name')
    
    # Popularity filter (0-100)
    popularity = st.slider('Select popularity', 0, 100, (0, 100), step=1)
    
    # Year filter (1900 - 2024)
    year = st.slider('Select year range', 1900, 2024, (1900, 2024), step=1)

    # Checkbox for hot songs only
    hot_songs_only = st.checkbox('Use hot songs only!')

    def search_and_go():
        if search_query:
            # Call search_song with the input query and filters applied
            tracks_info = search_song(search_query, popularity, year, hot_songs_only)
            
            if tracks_info is not None:
                st.session_state.tracks_info = tracks_info  # Store the search results in session state
                st.session_state.page = 'validation'  # Move to validation page
            else:
                st.write("No songs found. Try searching with different terms.")

    st.button('Find Songs', on_click=search_and_go)

# Check for validation page
elif st.session_state.page == 'validation':
    st.title('Select Your Song')

    # Fetch the song info from the session state (tracks_info is set in 'landing' page)
    tracks_info = st.session_state.get('tracks_info', None)

    def go_to_recommendations(song):
        st.session_state.selected_song = song
        st.session_state.page = 'recommendations'

    if tracks_info is not None:
        # Iterate over tracks_info (which is a DataFrame)
        for idx, row in tracks_info.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{row['spotify_title']} - {row['spotify_artist']}")
            with col2:
                # Button to let user choose whether the song matches
                st.button('Use This', 
                        key=f"select_{idx}_{row['spotify_title']}",  # Make the key unique by including the index
                        on_click=go_to_recommendations,
                        args=(row.to_dict(),))  # Pass the row as a dictionary (song details)

        
        # After showing the options, ask if the user has found the song
        user_input = st.text_input('Enter the song or artist name to search again:')
        
        if user_input:
            # Call the search_song function to update tracks_info with new results
            tracks_info = search_song(user_input, (0, 100), (1900, 2024), hot_songs_only=False)
            if tracks_info is not None:
                st.session_state.tracks_info = tracks_info  # Store new results in session state
                st.session_state.page = 'validation'  # Stay on validation page
            else:
                st.write("No songs found. Try searching with different terms.")
                
    # Button to move to recommendations page
    st.button('Next', on_click=lambda: st.session_state.update({'page': 'recommendations'}))

# Check for recommendations page
# Recommendations Page (Page 3)
elif st.session_state.page == 'recommendations':
    st.title('Song Recommendations')

    # Get the selected song from session state
    selected_song = st.session_state.selected_song
    if selected_song:
        st.write(f"You selected: {selected_song['spotify_title']} by {selected_song['spotify_artist']}")

        # Call get_recommendations to get the recommended songs
        recommendations = get_recommendations(selected_song, n_recommendations=5)

        # Display recommendations
        for idx, row in recommendations.iterrows():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{row['spotify_title']} - {row['spotify_artist']}")
            with col2:
                st.image(row['album_cover'], use_column_width=True)
    
    # Button to go back to the landing page
    st.button('Back to Search', on_click=lambda: st.session_state.update({'page': 'landing'}))

